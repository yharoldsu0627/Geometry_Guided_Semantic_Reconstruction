import ctypes
import os
import os.path as osp
import sys


def _python_version_dir():
    return f'python{sys.version_info.major}.{sys.version_info.minor}'


def _current_prefix():
    return sys.prefix or os.environ.get('CONDA_PREFIX')


def current_torch_lib_dirs():
    prefix = _current_prefix()
    py_dir = _python_version_dir()
    return [osp.join(prefix, 'lib', py_dir, 'site-packages', 'torch', 'lib')]


def prefer_current_env_torch_libs():
    current_dirs = [p for p in current_torch_lib_dirs() if osp.isdir(p)]
    old_entries = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    kept_entries = []
    for entry in old_entries:
        if not entry:
            continue
        normalized = osp.normpath(entry)
        if normalized in {osp.normpath(p) for p in current_dirs}:
            continue
        if '/site-packages/torch/lib' in normalized and _current_prefix() not in normalized:
            continue
        kept_entries.append(entry)
    merged = current_dirs + kept_entries
    deduped = []
    seen = set()
    for entry in merged:
        normalized = osp.normpath(entry)
        if normalized in seen:
            continue
        deduped.append(entry)
        seen.add(normalized)
    os.environ['LD_LIBRARY_PATH'] = ':'.join(deduped)


def preload_current_env_torch_libs():
    prefer_current_env_torch_libs()
    load_mode = getattr(os, 'RTLD_NOW', 0) | getattr(os, 'RTLD_GLOBAL', 0)
    lib_dirs = current_torch_lib_dirs()
    lib_names = [
        'libc10.so',
        'libtorch_cpu.so',
        'libtorch.so',
        'libtorch_python.so',
        'libtorch_cuda.so',
        'libc10_cuda.so',
        'libcudart.so.12',
    ]
    loaded = []
    seen_paths = set()
    for lib_name in lib_names:
        for lib_dir in lib_dirs:
            lib_path = osp.join(lib_dir, lib_name)
            if not osp.exists(lib_path):
                continue
            normalized = osp.normpath(lib_path)
            if normalized in seen_paths:
                continue
            ctypes.CDLL(lib_path, mode=load_mode)
            loaded.append(lib_path)
            seen_paths.add(normalized)
            break
    return loaded


def assert_cuda_torch(torch_module):
    if torch_module.version.cuda is None:
        raise RuntimeError(
            'Current environment is loading a CPU-only or mixed PyTorch build: '
            f'torch=={torch_module.__version__}, torch.version.cuda=None. '
            'Please reinstall the CUDA-enabled torch package in the relation3d '
            'environment, then rebuild pointgroup_ops / attention_rpe_ops.'
        )


def configure_spconv_runtime(torch_module):
    """Force spconv to use NVRTC kernels on Blackwell-class GPUs.

    spconv/cumm prebuilt kernels in the current environment are compiled up to
    sm_90. On RTX 5090 (sm_120), some prebuilt paths may still report
    ``no kernel image is available`` even though PTX compatibility is exposed.
    For these devices we force the runtime NVRTC path, which compiles kernels
    for the active architecture at runtime.
    """
    force_flag = os.environ.get('RELATION3D_FORCE_SPCONV_NVRTC', 'auto').lower()
    mode_name = os.environ.get('RELATION3D_SPCONV_NVRTC_MODE', 'ConstantMemory')
    try:
        if force_flag not in {'0', 'false', 'no'}:
            enable = force_flag in {'1', 'true', 'yes'}
            if not enable:
                enable = False
                if torch_module.cuda.is_available():
                    major, minor = torch_module.cuda.get_device_capability(0)
                    enable = major >= 12
                    if enable:
                        os.environ.setdefault('CUMM_CUDA_ARCH_LIST', f'{major}.{minor}')
            if enable:
                from cumm.gemm.constants import NVRTCMode
                import spconv.algo as spconv_algo
                import spconv.constants as spconv_constants

                mode = getattr(NVRTCMode, mode_name, NVRTCMode.ConstantMemory)
                spconv_constants.SPCONV_DEBUG_NVRTC_KERNELS = True
                spconv_constants.SPCONV_NVRTC_MODE = mode
                spconv_algo.SPCONV_DEBUG_NVRTC_KERNELS = True
                spconv_algo.SPCONV_NVRTC_MODE = mode
                os.environ.setdefault('SPCONV_NVRTC_FORCED', '1')
                print(
                    f'[runtime] Force spconv NVRTC mode={mode.name} '
                    f'for CUDA arch {os.environ.get("CUMM_CUDA_ARCH_LIST", "auto")}'
                )
                return True
    except Exception:
        # Best-effort runtime patching only; fall back to default spconv behavior
        # if the package internals differ across versions.
        pass
    return False
