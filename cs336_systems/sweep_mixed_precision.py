import argparse
import subprocess
import sys

from cs336_systems.benchmark import MODEL_SPECS


# Problem (benchmarking_mixed_precision) (c): run benchmark.py for every model size,
# with and without BF16 autocast, appending all rows into one JSONL / Markdown table.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-sizes", nargs="+", choices=list(MODEL_SPECS.keys()), default=list(MODEL_SPECS.keys()))
    p.add_argument("--amps", nargs="+", choices=["none", "bf16", "fp16"], default=["none", "bf16"])
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num_warmups", type=int, default=5)
    p.add_argument("--num_measures", type=int, default=10)
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--out-jsonl", type=str, default="results/mixed_precision.jsonl")
    p.add_argument("--out-md", type=str, default="results/mixed_precision.md")
    args = p.parse_args()

    failed = []
    for size in args.model_sizes:
        for amp in args.amps:
            cmd = [
                sys.executable, "-m", "cs336_systems.benchmark",
                "--model-size", size,
                "--context-length", str(args.context_length),
                "--batch-size", str(args.batch_size),
                "--num_warmups", str(args.num_warmups),
                "--num_measures", str(args.num_measures),
                "--amp", amp,
                "--out-jsonl", args.out_jsonl,
                "--out-md", args.out_md,
            ]
            if args.allow_tf32:
                cmd.append("--allow-tf32")

            print(f"\n=== model={size} amp={amp} ===", flush=True)
            # Each config runs in its own process so an OOM doesn't kill the sweep
            # and GPU memory is fully released between runs.
            if subprocess.run(cmd).returncode != 0:
                failed.append((size, amp))

    if failed:
        print(f"\nFailed (likely OOM): {failed}")
    print(f"\nResults: {args.out_jsonl}" + (f", {args.out_md}" if args.out_md else ""))


if __name__ == "__main__":
    main()

# uv run python -m cs336_systems.sweep_mixed_precision
# uv run python -m cs336_systems.sweep_mixed_precision --model-sizes small medium --context-length 256
