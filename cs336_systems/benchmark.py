import torch
import timeit
import argparse
import functools
from contextlib import nullcontext
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.checkpointing import checkpointed_forward
from cs336_systems.utils import BenchmarkReporter, BenchmarkRow, add_block_nvtx_ranges, nvtx_range, stats_ms
# from cs336_basics.nn_utils import cross_entropy_from_logits
from cs336_basics.optimizer import AdamW

MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

# MODEL_SPECS_RTX4060 = {
#     "small":  dict(d_model=384, d_ff=1536, num_layers=6,  num_heads=6),   # 384/6=64
#     "medium": dict(d_model=512, d_ff=2048, num_layers=8,  num_heads=8),   # 512/8=64
#     "large":  dict(d_model=640, d_ff=2560, num_layers=10, num_heads=10),  # 640/10=64
#     "xl":     dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),  # 768/12=64
#     "2.7b":   dict(d_model=768, d_ff=3072, num_layers=16, num_heads=12),  # 768/12=64
# }

# Given hyperparameters (e.g., number of layers), initialize a model.
def initialize_model(args, device: torch.device) -> torch.nn.Module:
    spec = MODEL_SPECS[args.model_size]
    model = BasicsTransformerLM(vocab_size=args.vocab_size, 
                                context_length=args.context_length, 
                                d_model=spec["d_model"], 
                                num_layers=spec["num_layers"],
                                num_heads=spec["num_heads"],
                                d_ff=spec["d_ff"],
                                rope_theta=args.rope_theta).to(device)
    model.train()
    return model

# Generate a random batch of data. 
def generate_data(args, device: torch.device) -> torch.tensor:
    return torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device, dtype=torch.long)

# Run forward
def forward(model, x, autocast_ctx):
    with autocast_ctx:
        logits = model(x)
    
    return logits

# Run backward
def backward(model, logits):
    loss = logits.float().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()

# Timestamp that works on both devices: a recorded CUDA event on GPU, wall clock on CPU.
def mark(device):
    if device.type == "cuda":
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    return timeit.default_timer()

def elapsed_ms(t0, t1):
    if isinstance(t0, torch.cuda.Event):
        return t0.elapsed_time(t1)
    return (t1 - t0) * 1e3

# time forward-only, forward-and-backward, and full training steps that include the optimizer step.
# Phases not run in the given mode come back as None.
def time(model, x, optimizer, device, autocast_ctx, mode, nvtx_enabled: bool = False):
    b_time = o_time = None
    # Forward-only is inference: don't build the autograd graph / keep activations around.
    grad_ctx = torch.no_grad() if mode == "forward" else nullcontext()

    t0 = mark(device)
    with nvtx_range("forward", nvtx_enabled), grad_ctx:
        logits = forward(model, x, autocast_ctx)
    t1 = mark(device)
    if mode != "forward":
        with nvtx_range("backward", nvtx_enabled):
            backward(model, logits)
        t2 = mark(device)
        if mode == "train":
            with nvtx_range("optimizer", nvtx_enabled):
                optimizer.step()
            t3 = mark(device)

    if device.type == "cuda":
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    f_time = elapsed_ms(t0, t1)
    if mode != "forward":
        b_time = elapsed_ms(t1, t2)
        if mode == "train":
            o_time = elapsed_ms(t2, t3)
    return f_time, b_time, o_time

# Run 𝑤 warm-up steps (before you start measuring time), then time the execution of 𝑛
# steps (either only forward, forward and backward, or forward and backward with optimizer
# step, depending on an argument). For timing, you can use the Python timeit module (e.g.,
# either using the timeit function, or using timeit.default_timer(), which gives you the
# system’s highest resolution clock, thus a better default for benchmarking than
# time.time()).
def benchmark(args, device, autocast_ctx):
    model = initialize_model(args, device)
    x =  generate_data(args, device)
    optimizer = AdamW(model.parameters()) if args.mode == "train" else None
    if args.nvtx and device.type == "cuda":
        add_block_nvtx_ranges(model.layers)
    if args.checkpoint_group > 0 or args.checkpoint_recursive:
        # Instance attribute shadows BasicsTransformerLM.forward, so model(x) checkpoints.
        model.forward = functools.partial(checkpointed_forward, model,
                                          group_size=args.checkpoint_group,
                                          recursive=args.checkpoint_recursive)
    if args.compile:
        # Compile last so it wraps whatever forward we ended up with. Compilation happens on
        # the first call, i.e. during warmup, so make sure --num_warmups >= 1.
        model = torch.compile(model)
    
    with nvtx_range("benchmark", args.nvtx):
        # warm up (time() already synchronizes at the end of each step)
        with nvtx_range("warmup", args.nvtx):
            for _ in range(args.num_warmups):
                time(model, x, optimizer, device, autocast_ctx, args.mode)

        # Memory is only measured over the measured steps. Things that persist from warmup
        # (weights, AdamW state created on the first optimizer step) are still counted,
        # since they are live memory; one-off warmup spikes are not.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            if args.memory_profile:
                # Start recording memory history.
                torch.cuda.memory._record_memory_history(max_entries=1_000_000)
            
        with nvtx_range("num_measure", args.nvtx):
            f_times, b_times, o_times = [], [], []
            for _ in range(args.num_measures):
                with nvtx_range("step", args.nvtx):
                    f_time, b_time, o_time = time(model, x, optimizer, device, autocast_ctx, args.mode, args.nvtx)

                f_times.append(f_time)
                if b_time is not None:
                    b_times.append(b_time)
                if o_time is not None:
                    o_times.append(o_time)

    peak_mem_mib = None
    if device.type == "cuda":
        peak_mem_mib = torch.cuda.max_memory_allocated() / 1024**2
        if args.memory_profile:
            # Save a pickle file to be loaded by pytorch.org/memory_viz, then stop recording.
            torch.cuda.memory._dump_snapshot(args.memory_profile)
            torch.cuda.memory._record_memory_history(enabled=None)

    return f_times, b_times, o_times, peak_mem_mib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=list(MODEL_SPECS.keys()), default="small")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=10_000)

    p.add_argument("--num_warmups", type=int, default=5)
    p.add_argument("--num_measures", type=int, default=10)
    # forward: forward only (no grad), forward_backward: + backward, train: + optimizer step
    p.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="train")

    # Mixed precision hook
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")

    # Activation checkpointing: checkpoint every K consecutive blocks (0 = off),
    # or nest checkpoints recursively (overrides --checkpoint-group)
    p.add_argument("--checkpoint-group", type=int, default=0)
    p.add_argument("--checkpoint-recursive", action="store_true")

    # torch_compile (b): compile the whole model with torch.compile
    p.add_argument("--compile", action="store_true")

    # Model misc
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    p.add_argument("--rmsnorm-eps", type=float, default=1e-5)

    # Determinism / matmul settings
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-tf32", action="store_true")

    # Profiling
    p.add_argument("--nvtx", action="store_true")
    # Path for a memory snapshot of the measured steps (use a small --num_measures, e.g. 1-2)
    p.add_argument("--memory-profile", type=str, default=None)

    # Reporting
    p.add_argument("--out-jsonl", type=str, default=None)
    p.add_argument("--out-md", type=str, default=None)

    args = p.parse_args()
    
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    
    
    # Autocast context (no-op on CPU)
    if device.type == "cuda" and args.amp != "none":
        s = args.amp.lower()
        amp_dtype = torch.float32
        if s in ("fp32", "float32"):
            amp_dtype = torch.float32
        if s in ("fp16", "float16"):
            amp_dtype = torch.float16
        if s in ("bf16", "bfloat16"):
            amp_dtype = torch.bfloat16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    reporter = None
    if args.out_jsonl:
        reporter = BenchmarkReporter(
            jsonl_path=args.out_jsonl,
            md_path=args.out_md,
            title="#### Forward / Backward timing"
        )

    f_times_ms, b_times_ms, o_times_ms, peak_mem_mib = benchmark(args, device, autocast_ctx)

    tok_per_step = args.batch_size * args.context_length

    ckpt = "recursive" if args.checkpoint_recursive else args.checkpoint_group
    print(f"[device={device}] amp={args.amp} mode={args.mode} checkpoint={ckpt} compile={args.compile}")
    print(f"model={args.model_size} B={args.batch_size} S={args.context_length} V={args.vocab_size}")
    print(f"warmup={args.num_warmups} measure={args.num_measures}")
    if peak_mem_mib is not None:
        print(f"peak memory: {peak_mem_mib:.1f} MiB")
    if args.memory_profile and device.type == "cuda":
        print(f"memory snapshot: {args.memory_profile}")

    for phase, times_ms in [("forward", f_times_ms), ("backward", b_times_ms), ("optimizer", o_times_ms)]:
        if not times_ms:  # phase not run in this mode
            continue
        mean, std = stats_ms(times_ms)
        tok_s = tok_per_step / (mean / 1e3) if mean > 0 else float("inf")
        print(f"{phase:<9}: mean={mean:.3f} ms, std={std:.3f} ms, tok/s={tok_s:.1f}")

        if reporter is not None:
            reporter.add(BenchmarkRow(
                model_size=args.model_size, batch_size=args.batch_size,
                context_length=args.context_length, vocab_size=args.vocab_size,
                amp=args.amp, mode=args.mode, device=str(device), phase=phase,
                mean_ms=mean, std_ms=std, tok_s=tok_s,
                num_warmups=args.num_warmups, num_measures=args.num_measures,
                peak_mem_mib=peak_mem_mib, checkpoint=str(ckpt), compile=args.compile,
            ))

if __name__ == "__main__":
    main()