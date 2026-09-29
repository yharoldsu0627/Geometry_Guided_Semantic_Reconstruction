#python3 setup.py install
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import os
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)


if torch.version.cuda is None:
    raise RuntimeError(
        f'attention_rpe_ops requires a CUDA-enabled PyTorch build, but got '
        f'torch {torch.__version__} with torch.version.cuda=None. '
        'Please remove any CPU-only PyTorch package from the current environment '
        'and reinstall the CUDA build before compiling extensions.'
    )

if CUDA_HOME is None:
    raise RuntimeError(
        'attention_rpe_ops requires CUDA_HOME to be set. '
        'Please make sure the CUDA toolkit is installed and nvcc is available.'
    )

setup(
    name='attention_rpe_ops',
    ext_modules=[
        CUDAExtension('attention_rpe_ops_cuda', [
            'src/attention_rpe_api.cpp',
            'src/attention/attention_cuda.cpp',
            'src/attention/attention_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
