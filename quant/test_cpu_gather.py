import torch
import cpu_gather

# 示例输入
input_tensor = torch.rand(10, 5)
index_tensor = torch.tensor([0, 2, 1, 3, 4, 0, 1, 2, 3, 4], dtype=torch.int64)

# 使用 C++ 的 threaded_gather
output_cpp = my_gather.threaded_gather(input_tensor, index_tensor)
print(output_cpp)
