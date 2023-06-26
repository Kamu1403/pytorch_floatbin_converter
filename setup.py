from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='utils_cuda',
    ext_modules=[
        CUDAExtension('utils_cuda', [
            'utils_cuda.cpp',
            'utils_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
