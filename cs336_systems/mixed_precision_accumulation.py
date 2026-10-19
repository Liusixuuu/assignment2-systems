import torch

# Problem (mixed_precision_accumulation): sum 0.01 one thousand times (exact answer: 10.0).

# 1) FP32 accumulator, FP32 values
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print("fp32 acc + fp32 vals          :", s, s.dtype)

# 2) FP16 accumulator, FP16 values
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("fp16 acc + fp16 vals          :", s, s.dtype)

# 3) FP32 accumulator, FP16 values (implicit upcast on +=)
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("fp32 acc + fp16 vals          :", s, s.dtype)

# 4) FP32 accumulator, FP16 values explicitly cast to FP32
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print("fp32 acc + fp16->fp32 vals    :", s, s.dtype)

# uv run python cs336_systems/mixed_precision_accumulation.py
