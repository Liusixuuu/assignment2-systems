import torch
import timeit
import argparse
from contextlib import nullcontext
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.utils import BenchmarkReporter, BenchmarkRow, nvtx_range, stats_ms
# from cs336_basics.nn_utils import cross_entropy_from_logits
from cs336_basics.optimizer import AdamW

MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
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

# time forward-only, forward-and-backward, and full training steps that include the optimizer step.
def time(model, x, optimizer, device, autocast_ctx, nvtx_enabled: bool = False):
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        mid = torch.cuda.Event(enable_timing=True)
        bwd   = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with nvtx_range("forward", nvtx_enabled):
            logits = forward(model, x, autocast_ctx)
        mid.record()
        with nvtx_range("backward", nvtx_enabled):
            backward(model, logits)
        bwd.record()
        if optimizer != None:
            with nvtx_range("optimizer", nvtx_enabled):
                optimizer.step()
        end.record()
        
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        
        return start.elapsed_time(mid), mid.elapsed_time(bwd), bwd.elapsed_time(end)
    
    start = timeit.default_timer()
    logits = forward(model, x, autocast_ctx)
    mid = timeit.default_timer()
    backward(model, logits)
    bwd = timeit.default_timer()
    if optimizer is not None:
        optimizer.step()
    end = timeit.default_timer()
    return (mid - start) * 1e3, (bwd - mid) * 1e3, (end - bwd) * 1e3

# Run 𝑤 warm-up steps (before you start measuring time), then time the execution of 𝑛
# steps (either only forward, forward and backward, or forward and backward with optimizer
# step, depending on an argument). For timing, you can use the Python timeit module (e.g.,
# either using the timeit function, or using timeit.default_timer(), which gives you the
# system’s highest resolution clock, thus a better default for benchmarking than
# time.time()).
def benchmark(args, device, autocast_ctx) -> float:
    model = initialize_model(args, device)
    x =  generate_data(args, device)
    optimizer = AdamW(model.parameters())
    
    with nvtx_range("benchmark", args.nvtx):
        # warm up
        with nvtx_range("warmup", args.nvtx):
            for _ in range(args.num_warmups):
                time(model, x, optimizer, device, autocast_ctx)
                if device.type == "cuda":
                    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
            
        with nvtx_range("num_measure", args.nvtx):
            f_times, b_times, o_times = [], [], []
            for _ in range(args.num_measures):
                with nvtx_range("step", args.nvtx):
                    f_time, b_time, o_time = time(model, x, optimizer, device, autocast_ctx)

                f_times.append(f_time)
                b_times.append(b_time)
                o_times.append(o_time)
    
            return f_times, b_times, o_times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=list(MODEL_SPECS.keys()), default="small")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=10_000)

    p.add_argument("--num_warmups", type=int, default=5)
    p.add_argument("--num_measures", type=int, default=10)

    # Mixed precision hook
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")

    # Model misc
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    p.add_argument("--rmsnorm-eps", type=float, default=1e-5)

    # Determinism / matmul settings
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-tf32", action="store_true")

    # Profiling
    p.add_argument("--nvtx", action="store_true")

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

    f_times_ms, b_times_ms, o_times_ms = benchmark(args, device, autocast_ctx)

    f_mean, f_std = stats_ms(f_times_ms)
    b_mean, b_std = stats_ms(b_times_ms)
    o_mean, o_std = stats_ms(o_times_ms)

    tok_per_step = args.batch_size * args.context_length
    f_tok_s = tok_per_step / (f_mean / 1e3) if f_mean > 0 else float("inf")
    b_tok_s = tok_per_step / (b_mean / 1e3) if b_mean > 0 else float("inf")
    o_tok_s = tok_per_step / (o_mean / 1e3) if o_mean > 0 else float("inf")

    print(f"[device={device}] amp={args.amp}")
    print(f"model={args.model_size} B={args.batch_size} S={args.context_length} V={args.vocab_size}")
    print(f"warmup={args.num_warmups} measure={args.num_measures}")
    print(f"forward  : mean={f_mean:.3f} ms, std={f_std:.3f} ms, tok/s={f_tok_s:.1f}")
    print(f"backward : mean={b_mean:.3f} ms, std={b_std:.3f} ms, tok/s={b_tok_s:.1f}")
    print(f"optimizer: mean={o_mean:.3f} ms, std={o_std:.3f} ms, tok/s={o_tok_s:.1f}")

    if reporter is not None:
        for phase, mean, std, tok_s in [
            ("forward", f_mean, f_std, f_tok_s),
            ("backward", b_mean, b_std, b_tok_s),
            ("optimizer", o_mean, o_std, o_tok_s),
        ]:
            reporter.add(BenchmarkRow(
                model_size=args.model_size, batch_size=args.batch_size,
                context_length=args.context_length, vocab_size=args.vocab_size,
                amp=args.amp, device=str(device), phase=phase,
                mean_ms=mean, std_ms=std, tok_s=tok_s,
                num_warmups=args.num_warmups, num_measures=args.num_measures,
            ))


if __name__ == "__main__":
    main()