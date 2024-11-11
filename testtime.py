import torch
import time

# 设置输入张量和索引张量的维度
batch_size = 10000
input_dim = 10000
index_dim = 5000

# 创建输入张量和索引张量
input_tensor = torch.randn((1, 32, 990, 128))
index_tensor = torch.randint(0, 990, (1, 32, 222)).unsqueeze(-1).expand(-1, -1, -1, input_tensor.shape[-1])

# 在 CPU 上测试 torch.gather 的耗时
start_time = time.time()
output_cpu = torch.gather(input_tensor, 2, index_tensor)
end_time = time.time()
cpu_time = end_time - start_time

# 将输入张量和索引张量移动到 GPU
input_tensor_gpu = input_tensor.cuda()
index_tensor_gpu = index_tensor.cuda()

# 在 GPU 上测试 torch.gather 的耗时
start_time = time.time()
output_gpu = torch.gather(input_tensor_gpu, 2, index_tensor_gpu)
torch.cuda.synchronize()  # 确保 GPU 操作完成
end_time = time.time()
gpu_time = end_time - start_time

print(f"CPU time: {cpu_time:.6f} seconds")
print(f"GPU time: {gpu_time:.6f} seconds")