import triton
import triton.language as tl

# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,  # Pointers to matrices A, B, and C
    M, N, K,              # Dimensions of matrices
    stride_am, stride_ak, # Strides of A matrix
    stride_bk, stride_bn, # Strides of B matrix
    stride_cm, stride_cn, # Strides of C matrix
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for N dimension
    BLOCK_SIZE_K: tl.constexpr   # Block size for K dimension
):

    # Block index for the current program instance
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate the start index of the block in matrix A and B
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N

    # Create accumulators for the result
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Iterate over K dimension in chunks of BLOCK_SIZE_K
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a block of A and B into shared memory
        a = tl.load(A_ptr + (block_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_ak)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk + (block_n + tl.arange(0, BLOCK_SIZE_N)) * stride_bn)
        
        # Perform the matrix multiplication for this block
        acc += tl.dot(a, b)

    # Store the result in C
    c = acc.to(tl.float32)
    tl.store(C_ptr + (block_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm + (block_n + tl.arange(0, BLOCK_SIZE_N)) * stride_cn, c)

# Function to call the Triton kernel
def matmul_triton(A, B, C):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape

    # Define block size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 16

    # Launch the kernel
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

# Example usage
import torch

# Define matrices A, B, and C
A = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
B = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
C = torch.zeros((1024, 1024), device='cuda', dtype=torch.float32)

# Perform matrix multiplication using Triton
matmul_triton(A, B, C)

# Verify correctness
print(torch.allclose(C, torch.matmul(A, B)))
