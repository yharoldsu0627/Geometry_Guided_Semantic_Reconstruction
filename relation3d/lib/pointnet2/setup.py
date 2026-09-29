# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import glob
import os.path as osp

this_dir = osp.dirname(osp.abspath(__file__))

if torch.version.cuda is None:
    raise RuntimeError(
        f'pointnet2 requires a CUDA-enabled PyTorch build, but got '
        f'torch {torch.__version__} with torch.version.cuda=None. '
        'Please remove any CPU-only PyTorch package from the current environment '
        'and reinstall the CUDA build before compiling extensions.'
    )

if CUDA_HOME is None:
    raise RuntimeError(
        'pointnet2 requires CUDA_HOME to be set. '
        'Please make sure the CUDA toolkit is installed and nvcc is available.'
    )

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
