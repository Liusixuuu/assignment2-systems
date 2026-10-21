import argparse
import itertools
import json
import os
import timeit

import torch
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.utils import stats_ms

# Attention implementations to benchmark. Each entry builds a function that takes (Q, K, V)
# and returns the output. Builders are called once per config (see run_config).
ATTENTION_IMPLS = {
    "pytorch": lambda: scaled_dot_product_attention,
    # dynamic=False: compile a kernel specialized to this config's shapes, instead of the
    # dynamic-shape kernel dynamo switches to after it sees a second shape.
    "compiled": lambda: torch.compile(scaled_dot_product_attention, dynamic=False),
}

D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def mem_mib(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1024**2
    return None


# One forward and one backward pass, each timed separately. Synchronize after each pass so
# the timer measures the kernels themselves, not just their launches.
# Also returns the memory in use right before backward: the inputs plus everything autograd
# saved during forward for the backward pass.
def step(attn, q, k, v, grad_out, device):
    q.grad = k.grad = v.grad = None

    t0 = timeit.default_timer()
    out = attn(q, k, v)
    sync(device)
    t1 = timeit.default_timer()

    mem_before_bwd = mem_mib(device)

    t2 = timeit.default_timer()
    out.backward(grad_out)
    sync(device)
    t3 = timeit.default_timer()

    return (t1 - t0) * 1e3, (t3 - t2) * 1e3, mem_before_bwd


def benchmark_config(attn, batch_size, seq_len, d_model, args, device, dtype):
    # No head dimension: Q, K, V are (batch, seq_len, d_model).
    shape = (batch_size, seq_len, d_model)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    # Fixed upstream gradient, allocated once so it isn't part of the timed backward.
    grad_out = torch.randn(shape, device=device, dtype=dtype)

    for _ in range(args.num_warmups):
        step(attn, q, k, v, grad_out, device)

    f_times, b_times = [], []
    mem_before_bwd = None
    for _ in range(args.num_measures):
        f_time, b_time, mem_before_bwd = step(attn, q, k, v, grad_out, device)
        f_times.append(f_time)
        b_times.append(b_time)

    return f_times, b_times, mem_before_bwd


def run_config(impl, seq_len, d_model, args, device, dtype):
    # Dynamo caches compiled code per function, and after 8 recompiles (one per new shape)
    # silently falls back to eager. Reset so every config gets its own fresh compile; the
    # compile itself happens during warmup and is not timed.
    torch._dynamo.reset()
    attn = ATTENTION_IMPLS[impl]()
    row = dict(impl=impl, batch_size=args.batch_size, d_model=d_model, seq_len=seq_len,
               dtype=args.dtype, fwd_mean_ms=None, fwd_std_ms=None, bwd_mean_ms=None,
               bwd_std_ms=None, mem_before_bwd_mib=None, status="ok")
    try:
        f_times, b_times, mem = benchmark_config(attn, args.batch_size, seq_len, d_model, args, device, dtype)
        row["fwd_mean_ms"], row["fwd_std_ms"] = stats_ms(f_times)
        row["bwd_mean_ms"], row["bwd_std_ms"] = stats_ms(b_times)
        row["mem_before_bwd_mib"] = mem
    except torch.OutOfMemoryError:
        row["status"] = "OOM"
    # The tensors of benchmark_config are freed once its frame (and the exception) are gone,
    # so the next config starts from an empty cache.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_md(rows, path):
    cols = list(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("#### Attention: forward / backward timing\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row[c]) for c in cols) + " |\n")


def main():
    p = argparse.ArgumentParser()
    # Pass several (e.g. --impls pytorch compiled) to get both in one comparison table.
    p.add_argument("--impls", choices=list(ATTENTION_IMPLS.keys()), nargs="+", default=["pytorch"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--d-models", type=int, nargs="+", default=D_MODELS)
    p.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")

    p.add_argument("--num_warmups", type=int, default=5)
    p.add_argument("--num_measures", type=int, default=100)

    p.add_argument("--seed", type=int, default=42)

    # Reporting
    p.add_argument("--out-jsonl", type=str, default=None)
    p.add_argument("--out-md", type=str, default=None)

    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    print(f"[device={device}] impls={args.impls} B={args.batch_size} dtype={args.dtype} "
          f"warmup={args.num_warmups} measure={args.num_measures}")

    rows = []
    for d_model, seq_len, impl in itertools.product(args.d_models, args.seq_lens, args.impls):
        row = run_config(impl, seq_len, d_model, args, device, dtype)
        rows.append(row)
        if row["status"] == "OOM":
            print(f"{impl:<9} d={d_model:<4} S={seq_len:<6}: OOM")
        else:
            print(f"{impl:<9} d={d_model:<4} S={seq_len:<6}: fwd={row['fwd_mean_ms']:.3f} ms, "
                  f"bwd={row['bwd_mean_ms']:.3f} ms, mem_before_bwd={fmt(row['mem_before_bwd_mib'])} MiB")

        if args.out_jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
            with open(args.out_jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")

    if args.out_md:
        write_md(rows, args.out_md)


if __name__ == "__main__":
    main()