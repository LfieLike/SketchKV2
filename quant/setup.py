from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = "32"
    common_nvcc_args = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=" + nvcc_threads,
        "-gencode=arch=compute_80,code=sm_80"
    ]
    return nvcc_extra_args + common_nvcc_args

ext_modules = [
    CUDAExtension(
        name='cuda_my_quant',
        sources=['csrc/my_quant_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_score',
        sources=['csrc/my_score_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_key_dequant',
        sources=['csrc/my_key_dequant_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_scatter_add',
        sources=['csrc/my_scatter_add_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_scatter_add_float',
        sources=['csrc/my_scatter_add_kernel_float.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_resduial_quant',
        sources=['csrc/my_resduial_quant_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_value_quant',
        sources=['csrc/my_value_quant_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    ),
    CUDAExtension(
        name='cuda_my_value_dequant',
        sources=['csrc/my_value_dequant_kernel.cu'],
        extra_compile_args={
            "cxx": ["-g", "-O3"],
            "nvcc": append_nvcc_threads([])
        }
    )
]

setup(
    name='combined_cuda_extensions',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=["torch"]
)
