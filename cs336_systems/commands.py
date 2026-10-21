    # uv run python cs336_systems/benchmark.py \
    # --model-size small \
    # --context-length 128 \
    # --batch-size 4 \
    # --num_warmups 0 \
    # --num_measures 10 \
    # --amp none

    # memory_profiling (a): single snapshot for pytorch.org/memory_viz (forward-only / full train step)
    # uv run python -m cs336_systems.benchmark --model-size xl --context-length 128 \
    # --mode forward --num_measures 1 --memory-profile results/xl_S128_forward.pickle
    #
    # memory_profiling (f): Nsight memory trace with per-block NVTX ranges, skipping warmup
    # NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --pytorch=autograd-nvtx \
    # --capture-range=nvtx --nvtx-capture=num_measure -o results/xl_mem \
    # python -m cs336_systems.benchmark --model-size xl --mode train --num_measures 1 --nvtx


    # pytorch_attention (a): full sweep, B=8, d in {16,32,64,128} x S in {256,...,16384}, 100 fwd/bwd passes
    # uv run python -m cs336_systems.benchmark_attention --impls pytorch \
    # --out-md results/pytorch_attention.md --out-jsonl results/pytorch_attention.jsonl
    #
    # pytorch_attention: quick smoke test on a couple of small configs
    # uv run python -m cs336_systems.benchmark_attention --d-models 16 --seq-lens 256 1024 \
    # --num_warmups 1 --num_measures 5

    # torch_compile (a): compiled vs. uncompiled attention, same sweep as pytorch_attention
    # uv run python -m cs336_systems.benchmark_attention --impls pytorch compiled \
    # --out-md results/torch_compile_attention.md --out-jsonl results/torch_compile_attention.jsonl
    #
    # torch_compile (b): vanilla vs. compiled Transformer, forward-only and full train step
    # for mode in forward forward_backward train; do
    #   for c in "" "--compile"; do
    #     uv run python -m cs336_systems.benchmark --model-size small --context-length 512 --mode $mode $c \
    #     --out-jsonl results/torch_compile_model.jsonl --out-md results/torch_compile_model.md
    #   done
    # done
