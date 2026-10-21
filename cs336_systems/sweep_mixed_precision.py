import argparse
import os
import subprocess
import sys

from cs336_systems.benchmark import MODEL_SPECS


# Problem (benchmarking_mixed_precision) (c): run benchmark.py for every model size,
# with and without BF16 autocast, appending all rows into one JSONL / Markdown table.
# Problem (memory_profiling) (b)/(c): same sweep over context lengths and modes; peak
# memory ends up in the peak_mem_mib column, and --snapshot-dir saves one
# memory_viz snapshot per config.
# Problem (gradient_checkpointing) (b): --checkpoint-groups sweeps blocks per checkpoint
# (0 = no checkpointing).
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-sizes", nargs="+", choices=list(MODEL_SPECS.keys()), default=list(MODEL_SPECS.keys()))
    p.add_argument("--amps", nargs="+", choices=["none", "bf16", "fp16"], default=["none", "bf16"])
    p.add_argument("--context-lengths", nargs="+", type=int, default=[512])
    p.add_argument("--modes", nargs="+", choices=["forward", "forward_backward", "train"], default=["train"])
    p.add_argument("--checkpoint-groups", nargs="+", type=int, default=[0])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num_warmups", type=int, default=5)
    p.add_argument("--num_measures", type=int, default=10)
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--snapshot-dir", type=str, default=None)
    p.add_argument("--out-jsonl", type=str, default="results/mixed_precision.jsonl")
    p.add_argument("--out-md", type=str, default="results/mixed_precision.md")
    args = p.parse_args()

    if args.snapshot_dir:
        os.makedirs(args.snapshot_dir, exist_ok=True)

    failed = []
    for size in args.model_sizes:
        for context_length in args.context_lengths:
            for mode in args.modes:
                for amp in args.amps:
                    for group in args.checkpoint_groups:
                        cmd = [
                            sys.executable, "-m", "cs336_systems.benchmark",
                            "--model-size", size,
                            "--context-length", str(context_length),
                            "--batch-size", str(args.batch_size),
                            "--num_warmups", str(args.num_warmups),
                            "--num_measures", str(args.num_measures),
                            "--amp", amp,
                            "--mode", mode,
                            "--checkpoint-group", str(group),
                            "--out-jsonl", args.out_jsonl,
                            "--out-md", args.out_md,
                        ]
                        if args.allow_tf32:
                            cmd.append("--allow-tf32")
                        if args.snapshot_dir:
                            name = f"{size}_S{context_length}_{mode}_{amp}_ckpt{group}.pickle"
                            cmd += ["--memory-profile", os.path.join(args.snapshot_dir, name)]

                        print(f"\n=== model={size} S={context_length} mode={mode} amp={amp} ckpt={group} ===", flush=True)
                        # Each config runs in its own process so an OOM doesn't kill the sweep
                        # and GPU memory is fully released between runs.
                        if subprocess.run(cmd).returncode != 0:
                            failed.append((size, context_length, mode, amp, group))

    if failed:
        print(f"\nFailed (likely OOM): {failed}")
    print(f"\nResults: {args.out_jsonl}" + (f", {args.out_md}" if args.out_md else ""))


if __name__ == "__main__":
    main()

# uv run python -m cs336_systems.sweep_mixed_precision
# uv run python -m cs336_systems.sweep_mixed_precision --model-sizes small medium --context-lengths 256
# memory_profiling (b)/(c), with snapshots for (a)/(e):
# uv run python -m cs336_systems.sweep_mixed_precision --model-sizes xl --context-lengths 128 2048 \
#     --modes forward train --num_measures 2 --snapshot-dir results/memory_snapshots \
#     --out-jsonl results/memory.jsonl --out-md results/memory.md
# gradient_checkpointing (b): peak memory vs. blocks per checkpoint
# uv run python -m cs336_systems.sweep_mixed_precision --model-sizes xl --context-lengths 2048 \
#     --modes train --amps none --checkpoint-groups 0 1 2 4 8 --num_measures 2 \
#     --out-jsonl results/checkpointing.jsonl --out-md results/checkpointing.md
