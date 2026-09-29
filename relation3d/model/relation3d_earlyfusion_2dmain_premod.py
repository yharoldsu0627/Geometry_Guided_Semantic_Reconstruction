import gorilla
import os
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax

from relation3d.utils import cuda_cast
from .backbone import get_relation3d_spconv_algo
from .relation3d_earlyfusion import Relation3DEarlyFusion


def _debug_cuda_sync(tag: str):
    flag = os.environ.get('RELATION3D_DEBUG_SYNC', '0').lower()
    if flag not in {'1', 'true', 'yes'}:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f'[debug-sync] {tag}', flush=True)


def apply_sparse_film(x, mod_params):
    scale, shift = mod_params.chunk(2, dim=-1)
    return x.replace_feature(x.features * (1 + scale) + shift)


@gorilla.MODELS.register_module()
class Relation3DEarlyFusion2DMainPreMod(Relation3DEarlyFusion):
    """
    2D-main fusion, single modulation before input_conv:
    - main input to encoder is masked 2D voxel feature [256]
    - 3D voxel feature [9] generates FiLM parameters in 256-d space
    - modulation is applied once before 256->32 input_conv
    """

    def __init__(
        self,
        *args,
        input_mod_hidden_dim: int = 256,
        input_mod_use_layernorm: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        media = self.mlp[0].out_features
        self.input_mod_hidden_dim = int(input_mod_hidden_dim)
        self.input_mod_use_layernorm = bool(input_mod_use_layernorm)
        conv_algo = get_relation3d_spconv_algo()

        # Main encoder input becomes 2D voxel features.
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.d_2d,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
                algo=conv_algo,
            )
        )

        cond_layers = []
        if self.input_mod_use_layernorm:
            cond_layers.append(nn.LayerNorm(self.input_channel))
        cond_layers.extend([
            nn.Linear(self.input_channel, self.input_mod_hidden_dim),
            nn.GELU(),
            nn.Linear(self.input_mod_hidden_dim, self.input_mod_hidden_dim),
            nn.GELU(),
            nn.Linear(self.input_mod_hidden_dim, 2 * self.d_2d),
        ])
        self.input_modulation = nn.Sequential(*cond_layers)
        self._init_input_modulation()

    def _init_input_modulation(self):
        for module in self.input_modulation.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        last_linear = self.input_modulation[-1]
        nn.init.constant_(last_linear.weight, 0)
        nn.init.constant_(last_linear.bias, 0)

    def load_pretrain_partial(self, pretrain_path):
        ckpt = torch.load(pretrain_path, map_location='cpu')
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        model_state = self.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            model_key = key[7:] if key.startswith('module.') else key
            if (
                model_key in model_state
                and hasattr(value, 'shape')
                and model_state[model_key].shape == value.shape
            ):
                compatible[model_key] = value
            else:
                skipped.append(model_key)

        missing, unexpected = self.load_state_dict(compatible, strict=False)
        print(f'[2DMainPreMod] Loaded pretrain from {pretrain_path}')
        print(
            f'[2DMainPreMod] Compatible tensors: {len(compatible)}, '
            f'skipped: {len(skipped)}'
        )
        if missing:
            print(f'[2DMainPreMod] Missing keys ({len(missing)}): '
                  f'{missing[:5]}{"..." if len(missing) > 5 else ""}')
        if unexpected:
            print(f'[2DMainPreMod] Unexpected keys ({len(unexpected)}): '
                  f'{unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')

    def extract_feat(self, x, superpoints, p2v_map, sp_coords, coords_float, voxel_cond3d=None):
        if voxel_cond3d is not None:
            mod_params = self.input_modulation(voxel_cond3d)
            _debug_cuda_sync('2dmain_premod.input_modulation')
            x = apply_sparse_film(x, mod_params)
            _debug_cuda_sync('2dmain_premod.apply_sparse_film')

        x = self.input_conv(x)
        _debug_cuda_sync('2dmain_premod.input_conv')
        x, _ = self.unet(x)
        _debug_cuda_sync('2dmain_premod.unet')
        x = self.output_layer(x)
        _debug_cuda_sync('2dmain_premod.output_layer')
        x = x.features[p2v_map.long()]
        _debug_cuda_sync('2dmain_premod.p2v_gather')

        x_origin = x.clone()
        x = scatter_mean(x_origin, superpoints, dim=0)
        _debug_cuda_sync('2dmain_premod.scatter_mean')
        rel_fea_mean = self.pooling_linear((x[superpoints] - x_origin))
        x_mean = scatter_sum(
            scatter_softmax(rel_fea_mean, superpoints, dim=0) * x_origin,
            superpoints, dim=0)
        x, _ = scatter_max(x_origin, superpoints, dim=0)
        _debug_cuda_sync('2dmain_premod.scatter_max')
        rel_fea_max = self.pooling_linear1((x[superpoints] - x_origin))
        x_max = scatter_sum(
            scatter_softmax(rel_fea_max, superpoints, dim=0) * x_origin,
            superpoints, dim=0)
        x = self.mlp(torch.cat([x_mean, x_max], dim=-1))
        _debug_cuda_sync('2dmain_premod.final_mlp')

        return (x,
                scatter_softmax(rel_fea_mean, superpoints, dim=0),
                scatter_softmax(rel_fea_max, superpoints, dim=0))

    def _forward_stage1_view(self, voxel_coords, p2v_map, v2p_map, spatial_shape,
                             feats, superpoints, coords_float, batch_offsets,
                             feat_2d=None, insts=None, apply_mask=True, mask_indices=None):
        batch_size = len(batch_offsets) - 1
        sp_coords = scatter_mean(coords_float, superpoints, dim=0)
        _debug_cuda_sync('2dmain_premod.sp_coords')

        voxel_2d = None
        voxel_cond3d = pointgroup_ops.voxelization(feats.contiguous(), v2p_map)
        _debug_cuda_sync('2dmain_premod.voxel_cond3d')
        if feat_2d is not None:
            feat_2d_gt = feat_2d.clone()
            _debug_cuda_sync('2dmain_premod.feat_2d_clone')
            if self.feature_aux_mode == 'full':
                feat_2d_point = feat_2d[superpoints]
                if self.training:
                    mask_indices = torch.arange(
                        feat_2d.shape[0], device=feat_2d.device, dtype=torch.long)
                else:
                    mask_indices = None
            elif self.feature_aux_mode == 'object':
                if self.training and apply_mask and self.object_mask_ratio > 0:
                    if mask_indices is None:
                        feat_2d_aux, mask_indices = self._mask_object_superpoint_feats(
                            feat_2d, insts, batch_offsets)
                    else:
                        feat_2d_aux = self._apply_mask_indices(feat_2d, mask_indices)
                    feat_2d_point = feat_2d_aux[superpoints]
                else:
                    feat_2d_point = feat_2d[superpoints]
                    mask_indices = None
            elif self.feature_aux_mode == 'background':
                if self.training and apply_mask and self.background_mask_ratio > 0:
                    if mask_indices is None:
                        feat_2d_aux, mask_indices = self._mask_background_superpoint_feats(
                            feat_2d, insts, batch_offsets)
                    else:
                        feat_2d_aux = self._apply_mask_indices(feat_2d, mask_indices)
                    feat_2d_point = feat_2d_aux[superpoints]
                else:
                    feat_2d_point = feat_2d[superpoints]
                    mask_indices = None
            else:
                if self.training and apply_mask and self.mask_ratio > 0:
                    if mask_indices is None:
                        feat_2d_aux, mask_indices = self._mask_superpoint_feats(
                            feat_2d, batch_offsets)
                    else:
                        feat_2d_aux = self._apply_mask_indices(feat_2d, mask_indices)
                    feat_2d_point = feat_2d_aux[superpoints]
                else:
                    feat_2d_point = feat_2d[superpoints]
                    mask_indices = None

            voxel_2d = pointgroup_ops.voxelization(feat_2d_point.contiguous(), v2p_map)
            _debug_cuda_sync('2dmain_premod.voxel_2d')
        else:
            feat_2d_gt = None
            mask_indices = None
            voxel_2d = voxel_cond3d.new_zeros(voxel_cond3d.shape[0], self.d_2d)

        input_tensor = spconv.SparseConvTensor(
            voxel_2d,
            voxel_coords.int(),
            spatial_shape,
            batch_size,
            force_algo=get_relation3d_spconv_algo(),
        )
        _debug_cuda_sync('2dmain_premod.input_tensor')
        sp_feats, _, _ = self.extract_feat(
            input_tensor, superpoints, p2v_map, sp_coords, coords_float, voxel_cond3d=voxel_cond3d)
        _debug_cuda_sync('2dmain_premod.extract_feat_done')

        feature_aux_loss = torch.tensor(0.0, device=sp_feats.device)
        feature_aux_metric = 0.0
        feature_aux_stats = {}
        if self.training and feat_2d_gt is not None:
            if mask_indices is not None and mask_indices.numel() > 0:
                mask_weights, feature_aux_stats = self._compute_feature_aux_weights(
                    mask_indices, insts, batch_offsets, sp_coords)
                feature_aux_loss, feature_aux_metric = self._compute_mask_prediction_loss(
                    sp_feats, feat_2d_gt, mask_indices, mask_weights=mask_weights)

        out, sp_feats_update_list, _ = self.decoder(
            sp_feats, sp_coords, batch_offsets, self.epoch)
        _debug_cuda_sync('2dmain_premod.decoder')
        return {
            'out': out,
            'sp_feats_update_list': sp_feats_update_list,
            'feature_aux_loss': feature_aux_loss,
            'feature_aux_metric': feature_aux_metric,
            'feature_aux_stats': feature_aux_stats,
        }

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape,
                feats, insts, superpoints, coords_float, batch_offsets,
                sp_instance_labels, feat_2d=None, **kwargs):
        batch_size = len(batch_offsets) - 1
        voxel_cond3d = pointgroup_ops.voxelization(feats.contiguous(), v2p_map)
        if feat_2d is not None:
            feat_2d_point = feat_2d[superpoints]
            voxel_2d = pointgroup_ops.voxelization(feat_2d_point.contiguous(), v2p_map)
        else:
            voxel_2d = voxel_cond3d.new_zeros(voxel_cond3d.shape[0], self.d_2d)

        input_tensor = spconv.SparseConvTensor(
            voxel_2d,
            voxel_coords.int(),
            spatial_shape,
            batch_size,
            force_algo=get_relation3d_spconv_algo(),
        )
        sp_coords = scatter_mean(coords_float, superpoints, dim=0)
        sp_feats, _, _ = self.extract_feat(
            input_tensor, superpoints, p2v_map, sp_coords, coords_float, voxel_cond3d=voxel_cond3d)

        out, _, _ = self.decoder(
            sp_feats, sp_coords, batch_offsets, self.epoch)
        return self.predict_by_feat(scan_ids, out, superpoints, insts)
