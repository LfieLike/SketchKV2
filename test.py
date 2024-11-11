import torch

# 创建一个示例张量
tensor = torch.randn(3, 4)

# 按列（dim=0）计算 L2 范数
l2_norm_dim0 = torch.norm(tensor, p=2, dim=0)

# 按行（dim=1）计算 L2 范数
l2_norm_dim1 = torch.norm(tensor, p=2, dim=1)

print(f"L2 Norm along dimension 0: {l2_norm_dim0}")
print(f"L2 Norm along dimension 1: {l2_norm_dim1}")
