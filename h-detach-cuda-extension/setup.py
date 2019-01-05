from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='h_detach',
    ext_modules=[
        CUDAExtension('h_detach_cuda', [
            'lstm_h_detach_cuda.cpp',
            'h_detach.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })