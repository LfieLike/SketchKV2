from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='threaded_gather',
    ext_modules=[
        CppExtension(
            'threaded_gather',
            ['csrc/threaded_gather.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
