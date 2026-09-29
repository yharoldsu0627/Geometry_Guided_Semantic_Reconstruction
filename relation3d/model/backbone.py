import functools
import gorilla
import os
import spconv.pytorch as spconv
import torch
from collections import OrderedDict
from spconv.core import ConvAlgo
from spconv.pytorch.modules import SparseModule
from torch import nn
from typing import Callable, Dict, List, Optional, Union


def _debug_cuda_sync(tag: str):
    flag = os.environ.get('RELATION3D_DEBUG_SYNC', '0').lower()
    if flag not in {'1', 'true', 'yes'}:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f'[debug-sync] {tag}', flush=True)


_SPCONV_ALGO_REPORTED = False


def get_relation3d_spconv_algo():
    """Allow forcing a safer spconv algorithm from the environment.

    RTX 5090 / sm_120 currently exercises code paths that may fail in
    MaskImplicitGemm on some spconv builds. Setting
    ``RELATION3D_SPCONV_ALGO=native`` switches all Relation3D sparse conv
    layers to the conservative native implementation without changing configs.
    """
    value = os.environ.get('RELATION3D_SPCONV_ALGO', '').strip().lower()
    if not value or value == 'auto':
        return None

    mapping = {
        'native': ConvAlgo.Native,
        'maskimplicitgemm': ConvAlgo.MaskImplicitGemm,
        'implicit': ConvAlgo.MaskImplicitGemm,
        'masksplitimplicitgemm': ConvAlgo.MaskSplitImplicitGemm,
        'splitimplicit': ConvAlgo.MaskSplitImplicitGemm,
    }
    if value not in mapping:
        raise ValueError(
            'Unsupported RELATION3D_SPCONV_ALGO='
            f'{value!r}. Expected one of: auto, native, '
            'maskimplicitgemm, masksplitimplicitgemm')

    global _SPCONV_ALGO_REPORTED
    algo = mapping[value]
    if not _SPCONV_ALGO_REPORTED:
        print(f'[runtime] Force Relation3D spconv algo={algo.name}', flush=True)
        _SPCONV_ALGO_REPORTED = True
    return algo

class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)
        
class ResidualBlock(SparseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_fn: Union[Callable, Dict] = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key: Optional[str] = None,
                 normalize_before: bool = True):
        super().__init__()
        conv_algo = get_relation3d_spconv_algo()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    algo=conv_algo,
                ))

        if isinstance(norm_fn, Dict):
            norm_caller = gorilla.nn.get_torch_layer_caller(norm_fn.pop('type'))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                    algo=conv_algo),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                    algo=conv_algo))
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                    algo=conv_algo),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                    algo=conv_algo),
                norm_fn(out_channels), nn.ReLU())

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features,
            input.indices,
            input.spatial_shape,
            input.batch_size,
            force_algo=input.force_algo,
        )

        output = self.conv_branch(input)
        _debug_cuda_sync('ResidualBlock.conv_branch')
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        _debug_cuda_sync('ResidualBlock.i_branch')
        # output.features += self.i_branch(identity).features

        return output


class UBlock(nn.Module):

    def __init__(
        self,
        nPlanes: List[int],
        norm_fn: Union[Dict, Callable] = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
        block_reps: int = 2,
        block: Union[str, Callable] = ResidualBlock,
        indice_key_id: int = 1,
        normalize_before: bool = True,
        return_blocks: bool = False,
    ):

        super().__init__()
        conv_algo = get_relation3d_spconv_algo()

        self.return_blocks = return_blocks
        self.nPlanes = nPlanes

        # process block and norm_fn caller
        if isinstance(block, str):
            area = ['residual', 'vgg', 'asym']
            assert block in area, f'block must be in {area}, but got {block}'
            if block == 'residual':
                block = ResidualBlock

        if isinstance(norm_fn, Dict):
            norm_caller = gorilla.nn.get_torch_layer_caller(norm_fn.pop('type'))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        blocks = {
            f'block{i}': block(
                nPlanes[0], nPlanes[0], norm_fn, normalize_before=normalize_before, indice_key=f'subm{indice_key_id}')
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(nPlanes[0]), nn.ReLU(),
                    spconv.SparseConv3d(
                        nPlanes[0],
                        nPlanes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}',
                        algo=conv_algo))
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        nPlanes[0],
                        nPlanes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}',
                        algo=conv_algo), norm_fn(nPlanes[1]), nn.ReLU())

            self.u = UBlock(
                nPlanes[1:],
                norm_fn,
                block_reps,
                block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks)

            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(nPlanes[1]), nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        nPlanes[1],
                        nPlanes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}',
                        algo=conv_algo))
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        nPlanes[1],
                        nPlanes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}',
                        algo=conv_algo),
                    norm_fn(nPlanes[0]), nn.ReLU())

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f'block{i}'] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key=f'subm{indice_key_id}',
                    normalize_before=normalize_before)
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input, previous_outputs: Optional[List] = None):
        output = self.blocks(input)
        _debug_cuda_sync(f'UBlock.blocks.nPlanes={self.nPlanes}')
        identity = spconv.SparseConvTensor(
            output.features,
            output.indices,
            output.spatial_shape,
            output.batch_size,
            force_algo=output.force_algo,
        )

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            _debug_cuda_sync(f'UBlock.conv.nPlanes={self.nPlanes}')
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            _debug_cuda_sync(f'UBlock.u.nPlanes={self.nPlanes}')
            output_decoder = self.deconv(output_decoder)
            _debug_cuda_sync(f'UBlock.deconv.nPlanes={self.nPlanes}')

            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))
            # output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)
            _debug_cuda_sync(f'UBlock.blocks_tail.nPlanes={self.nPlanes}')

        if self.return_blocks:
            # NOTE: to avoid the residual bug
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output
