import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# Problem (benchmarking_mixed_precision) (a): inspect dtypes under autocast.
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amp", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--in-features", type=int, default=32)
    p.add_argument("--out-features", type=int, default=5)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: run this on a GPU (CPU autocast uses different op rules).")

    device = torch.device("cuda")
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16

    model = ToyModel(args.in_features, args.out_features).to(device)  # params stay FP32
    x = torch.randn(args.batch_size, args.in_features, device=device)
    y = torch.randint(0, args.out_features, (args.batch_size,), device=device)

    # Record each submodule's output dtype via forward hooks
    out_dtypes = {}
    for name in ("fc1", "ln", "fc2"):
        getattr(model, name).register_forward_hook(
            lambda mod, inp, out, name=name: out_dtypes.__setitem__(name, out.dtype)
        )

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        param_dtypes = {n: p.dtype for n, p in model.named_parameters()}
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    loss.backward()
    grad_dtypes = {n: p.grad.dtype for n, p in model.named_parameters()}

    print(f"autocast dtype = {amp_dtype}")
    print(f"params (inside autocast) : {param_dtypes}")
    print(f"fc1 output               : {out_dtypes['fc1']}")
    print(f"ln output                : {out_dtypes['ln']}")
    print(f"logits (fc2 output)      : {logits.dtype}")
    print(f"loss                     : {loss.dtype}")
    print(f"grads                    : {grad_dtypes}")


if __name__ == "__main__":
    main()

# uv run python cs336_systems/toy_model_dtypes.py --amp fp16
# uv run python cs336_systems/toy_model_dtypes.py --amp bf16
