import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
def draw(tensor):
    data = tensor.cpu().detach().numpy()
    # 计算数据的均值和标准差
    mu, std = np.mean(data), np.std(data)

    # 绘制直方图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    n, bins, patches = plt.hist(data.flatten(), bins=200, density=True, alpha=0.6, color='g', label='Histogram')

    # 添加高斯分布拟合曲线
    xmin, xmax = plt.xlim()  # 获取直方图的最小和最大X轴值
    x = np.linspace(xmin, xmax, 100)  # 在这个范围内生成100个点
    p_norm = norm.pdf(x, mu, std)  # 计算这些点的高斯分布密度
    plt.plot(x, p_norm, 'k', linewidth=2, label='Gaussian Fit: $\mu=%.2f$, $\sigma=%.2f$' % (mu, std))

    # 添加拉普拉斯分布拟合曲线
    p_laplace = laplace.pdf(x, loc=mu, scale=std/np.sqrt(2))  # 拉普拉斯分布使用mu作为位置参数，std/sqrt(2)作为尺度参数
    plt.plot(x, p_laplace, 'r--', linewidth=2, label='Laplace Fit: $\mu=%.2f$, $b=%.2f$' % (mu, std/np.sqrt(2)))

    # 标题和标签
    plt.title('Probability Distribution with Gaussian and Laplace Fits')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()  # 显示图例

    # 保存图像
    plt.savefig('histogram_with_fits.png')  # 保存为PNG文件
    plt.close()  # 关闭图形窗口
def draw_matrix(matrix1,matrix2):
    # 创建一个图形窗口
    matrix1 = matrix1[:120,:120]
    matrix2 = matrix2[:120,:120]
    batch_size = 1
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    # pool = nn.Pool2d(60)
    pool = nn.AvgPool2d(batch_size,batch_size, 0)

    # 应用最大池化
    matrix1 = pool(matrix1.unsqueeze(0).unsqueeze(0)).squeeze()*(batch_size*batch_size)  # unsqueeze to add batch and channel dimensions
    matrix2 = pool(matrix2.unsqueeze(0).unsqueeze(0)).squeeze()*(batch_size*batch_size)
    # 可视化第一个矩阵
    ax[0].imshow(matrix1.cpu().detach(), cmap='viridis')
    ax[0].set_title('Matrix 1')
    ax[0].axis('off')  # 关闭坐标轴

    # 可视化第二个矩阵
    ax[1].imshow(matrix2.cpu().detach(), cmap='viridis')
    ax[1].set_title('Matrix 2')
    ax[1].axis('off')  # 关闭坐标轴

    plt.savefig('matrix.png')  # 保存为PNG文件
    plt.close() 
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # print(rotate_half_matrix(x) - torch.cat((-x2, x1), dim=-1))
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # print(cos.shape)
    return q_embed, k_embed
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
def quant_mock(tensor):

    # 最大值和最小值
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    min_val = tensor.min(dim=-1, keepdim=True)[0]

    # 归一化并量化
    scale = 15 / (max_val - min_val)
    quantized = torch.floor((tensor - min_val) * scale)

    # 反量化
    dequantized = quantized / scale + min_val
    return dequantized
class train_attn(nn.Module):
    def __init__(self,k_dim,v_dim,num_heads,lowrank_dim):
        super().__init__()
        self.v_dim = v_dim
        self.k_dim = k_dim
        in_channel = 32 
        out_channel = 16
        self.encoder_k = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1),  # 输出: [batch, 16, 128]
        )
        # 解码器
        self.decoder_k = nn.Sequential( 
            nn.Conv2d(in_channels=out_channel, out_channels=in_channel, kernel_size=1), # 输出: [batch, 16, 128]
        )
        self.encoder_v = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1),  # 输出: [batch, 16, 128]
        )
        # 解码器
        self.decoder_v = nn.Sequential( 
            nn.Conv2d(in_channels=out_channel, out_channels=in_channel, kernel_size=1), # 输出: [batch, 16, 128]
        )
        self.rotary_emb = LlamaRotaryEmbedding(dim = v_dim)

    def cal_attn(self,query_states,key_states,value_states):
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.v_dim)
        attn_weights = attn_weights + self.mask[...,-attn_weights.shape[-1]:, -attn_weights.shape[-1]:]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_weights
    def forward(self,query_states,key_states,value_states):
        bsz, num_heads,q_len,head_dim = key_states.shape

        key_states = self.encoder_k(key_states)
        key_states = quant_mock(key_states)+key_states-key_states.detach()
        key_states = self.decoder_k(key_states)
        
        # key_states = key_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        value_states = self.encoder_v(value_states)
        value_states = quant_mock(value_states)+value_states-value_states.detach()
        value_states = self.decoder_v(value_states)
        return key_states,value_states
