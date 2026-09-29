# """
# Relation3D + Early Fusion + Flow Matching (三阶段)
# ===================================================
# Stage 1: GT 2D 特征 early fusion 训练 (与原始一致)
# Stage 2: 冻结 Stage 1, 训练 cond_mlp + flow matching (从 3D 生成 2D)
# Stage 3: 冻结 cond_mlp + flow matching, 小 lr 微调 backbone + decoder

# 推理: 3D 点云 → flow matching 生成 pseudo 2D → early fusion → 预测
#       不需要任何 2D 图像输入

# 放置位置: relation3d/model/relation3d_earlyfusion.py
# """

# import functools
# import gorilla
# import pointgroup_ops
# import spconv.pytorch as spconv
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax
# import numpy as np

# from relation3d.utils import cuda_cast, rle_encode
# from .backbone import ResidualBlock, UBlock, MLP
# from .loss import Criterion
# from .query_decoder import QueryDecoder


# @gorilla.MODELS.register_module()
# class Relation3DEarlyFusion(nn.Module):

#     def __init__(
#         self,
#         input_channel: int = 9,
#         blocks: int = 5,
#         block_reps: int = 2,
#         media: int = 32,
#         normalize_before=True,
#         return_blocks=True,
#         pool='mean',
#         num_class=18,
#         decoder=None,
#         criterion=None,
#         test_cfg=None,
#         norm_eval=False,
#         fix_module=[],
#         d_2d=256,
#         # ★ Flow Matching 参数 (Stage 2/3 使用, Stage 1 可不传)
#         diffusion=None,
#         ddim_steps_infer=20,
#     ):
#         super().__init__()
#         self.d_2d = d_2d
#         self.num_class = num_class
#         self.input_channel = input_channel

#         # ===================== Backbone =====================
#         self.input_conv = spconv.SparseSequential(
#             spconv.SubMConv3d(
#                 input_channel + d_2d,   # 9 + 256 = 265
#                 media,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#                 indice_key='subm1',
#             ))
#         block = ResidualBlock
#         norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
#         block_list = [media * (i + 1) for i in range(blocks)]
#         self.unet = UBlock(
#             block_list, norm_fn, block_reps, block,
#             indice_key_id=1,
#             normalize_before=normalize_before,
#             return_blocks=return_blocks)
#         self.output_layer = spconv.SparseSequential(
#             norm_fn(media), nn.ReLU(inplace=True))
#         self.pool = pool

#         # ===================== ASAM =====================
#         self.mlp = nn.Sequential(
#             nn.Linear(2 * media, media), nn.ReLU(),
#             nn.Linear(media, media))
#         self.pooling_linear = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
#         self.pooling_linear1 = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
#         self.coords_linear = MLP(3, media, norm_fn=norm_fn, num_layers=3)

#         # ===================== Decoder =====================
#         self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

#         # ===================== Criterion =====================
#         self.criterion = Criterion(**criterion, num_class=num_class)

#         # ===================== ★ Flow Matching 相关 =====================
#         self.stage = 1
#         self.ddim_steps_infer = ddim_steps_infer

#         # 条件 MLP: 丰富超点统计特征 → [M, d_2d]
#         # 输入: scatter_mean(9) + scatter_max(9) + scatter_std(9) + log(n_pts)(1) + bbox_diag(1) = 29
#         cond_input_dim = input_channel * 3 + 2
#         self.cond_mlp = nn.Sequential(
#             nn.Linear(cond_input_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, 256),
#             nn.GELU(),
#             nn.Linear(256, d_2d),
#             nn.LayerNorm(d_2d),
#         )

#         # Flow Matching 模块 (接口名保持 diffusion_module 以兼容 set_stage)
#         if diffusion is not None:
#             from .flow_matching_module import SuperpointFlowMatching
#             self.diffusion_module = SuperpointFlowMatching(
#                 d_2d=d_2d,
#                 d_3d=d_2d,       # cond_mlp 输出维度
#                 d_model=diffusion.get('d_model', 256),
#                 n_blocks=diffusion.get('n_blocks', 12),
#                 n_heads=diffusion.get('n_heads', 8),
#                 mlp_ratio=diffusion.get('mlp_ratio', 4.0),
#                 dropout=diffusion.get('dropout', 0.0),
#                 sample_steps=diffusion.get('sample_steps', 20),
#                 loss_weights=diffusion.get('loss_weights', None),
#             )
#         else:
#             self.diffusion_module = None

#         self.epoch = 0
#         self.test_cfg = test_cfg
#         self.norm_eval = norm_eval

#         for module in fix_module:
#             mod = getattr(self, module)
#             mod.eval()
#             for param in mod.parameters():
#                 param.requires_grad = False

#     # ===================== Stage 管理 =====================

#     def set_stage(self, stage):
#         """
#         Stage 1: 训全网络 (GT 2D early fusion)
#         Stage 2: 冻结 backbone/ASAM/decoder, 只训 cond_mlp + flow matching
#         Stage 3: 冻结 cond_mlp + flow matching, 小 lr 微调其余
#         """
#         self.stage = stage
#         if stage == 2:
#             # 冻结 Stage 1 所有模块
#             for name, param in self.named_parameters():
#                 if 'cond_mlp' in name or 'diffusion_module' in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False
#             n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
#             print(f'[Stage 2] Trainable params: {n_train/1e6:.2f}M '
#                   f'(cond_mlp + flow matching only)')

#         elif stage == 3:
#             # 冻结 flow matching 相关, 解冻其余
#             for name, param in self.named_parameters():
#                 if 'cond_mlp' in name or 'diffusion_module' in name:
#                     param.requires_grad = False
#                 else:
#                     param.requires_grad = True
#             n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
#             print(f'[Stage 3] Trainable params: {n_train/1e6:.2f}M '
#                   f'(backbone + decoder, flow matching frozen)')

#     def train(self, mode=True):
#         super().train(mode)
#         if mode and self.norm_eval:
#             for m in self.modules():
#                 if isinstance(m, nn.BatchNorm1d):
#                     m.eval()
#         # Stage 2: 冻结模块保持 eval (BN 不更新统计量)
#         if mode and self.stage == 2:
#             self.input_conv.eval()
#             self.unet.eval()
#             self.output_layer.eval()
#             self.mlp.eval()
#             self.pooling_linear.eval()
#             self.pooling_linear1.eval()
#             self.coords_linear.eval()
#             self.decoder.eval()
#         # Stage 3: flow matching 保持 eval
#         if mode and self.stage == 3:
#             self.cond_mlp.eval()
#             if self.diffusion_module is not None:
#                 self.diffusion_module.eval()

#     # ===================== Pretrain Loading =====================

#     def load_pretrain_partial(self, pretrain_path):
#         """
#         加载预训练权重, 对 input_conv 做部分初始化。
#         前 input_channel 通道复制, 后 d_2d 通道零初始化。
#         """
#         ckpt = torch.load(pretrain_path, map_location='cpu')
#         if 'model' in ckpt:
#             state_dict = ckpt['model']
#         elif 'state_dict' in ckpt:
#             state_dict = ckpt['state_dict']
#         else:
#             state_dict = ckpt

#         conv_key = None
#         for k in state_dict.keys():
#             if 'input_conv' in k and 'weight' in k:
#                 conv_key = k
#                 break

#         if conv_key is not None:
#             old_w = state_dict[conv_key]
#             new_w = self.state_dict()[conv_key]
#             old_in = old_w.shape[1]
#             new_in = new_w.shape[1]
#             if old_in < new_in:
#                 new_w[:, :old_in] = old_w
#                 state_dict[conv_key] = new_w
#                 print(f'[EarlyFusion] input_conv: copied {old_in}/{new_in} channels, '
#                       f'{new_in - old_in} zero-initialized')
#             elif old_in == new_in:
#                 pass
#             else:
#                 print(f'[EarlyFusion] WARNING: pretrain has more channels ({old_in} > {new_in}), skipping')
#                 del state_dict[conv_key]

#         missing, unexpected = self.load_state_dict(state_dict, strict=False)
#         if missing:
#             print(f'[EarlyFusion] Missing keys ({len(missing)}): '
#                   f'{missing[:5]}{"..." if len(missing) > 5 else ""}')
#         if unexpected:
#             print(f'[EarlyFusion] Unexpected keys ({len(unexpected)}): '
#                   f'{unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')

#     # ===================== Forward =====================

#     def forward(self, batch, mode='loss'):
#         if mode == 'loss':
#             return self.loss(**batch)
#         elif mode == 'predict':
#             return self.predict(**batch)

#     # ===================== Backbone + ASAM =====================

#     def extract_feat(self, x, superpoints, v2p_map, sp_coords, coords_float):
#         x = self.input_conv(x)
#         x, _ = self.unet(x)
#         x = self.output_layer(x)
#         x = x.features[v2p_map.long()]

#         x_origin = x.clone()
#         x = scatter_mean(x_origin, superpoints, dim=0)
#         rel_fea_mean = self.pooling_linear((x[superpoints] - x_origin))
#         x_mean = scatter_sum(
#             scatter_softmax(rel_fea_mean, superpoints, dim=0) * x_origin,
#             superpoints, dim=0)
#         x, _ = scatter_max(x_origin, superpoints, dim=0)
#         rel_fea_max = self.pooling_linear1((x[superpoints] - x_origin))
#         x_max = scatter_sum(
#             scatter_softmax(rel_fea_max, superpoints, dim=0) * x_origin,
#             superpoints, dim=0)
#         x = self.mlp(torch.cat([x_mean, x_max], dim=-1))

#         return (x,
#                 scatter_softmax(rel_fea_mean, superpoints, dim=0),
#                 scatter_softmax(rel_fea_max, superpoints, dim=0))

#     # ===================== ★ 条件计算 (丰富版) =====================

#     def _get_diffusion_cond(self, feats, superpoints):
#         """
#         从点级特征 [N, C] 提取丰富的超点级条件 [M, d_2d]
#         特征拼接: scatter_mean + scatter_max + scatter_std + log(n_pts) + bbox_diag
#         """
#         # 基础统计: mean / max / std
#         sp_mean = scatter_mean(feats, superpoints, dim=0)           # [M, C]
#         sp_max, _ = scatter_max(feats, superpoints, dim=0)          # [M, C]

#         sp_mean_sq = scatter_mean(feats ** 2, superpoints, dim=0)   # E[x^2]
#         sp_var = (sp_mean_sq - sp_mean ** 2).clamp(min=0)
#         sp_std = torch.sqrt(sp_var + 1e-8)                         # [M, C]

#         # 超点内点数 (log 归一化)
#         ones = torch.ones(feats.shape[0], 1, device=feats.device)
#         sp_count = scatter_sum(ones, superpoints, dim=0)            # [M, 1]
#         sp_count_norm = torch.log1p(sp_count)

#         # 空间范围: xyz bbox 对角线 (feats 前 3 维是 xyz)
#         sp_xyz_min, _ = scatter_max(-feats[:, :3], superpoints, dim=0)
#         sp_xyz_min = -sp_xyz_min                                    # [M, 3]
#         sp_xyz_max, _ = scatter_max(feats[:, :3], superpoints, dim=0)
#         bbox_diag = (sp_xyz_max - sp_xyz_min).norm(dim=-1, keepdim=True)  # [M, 1]

#         # 拼接: [M, C*3+2] = [M, 29]
#         cond_feat = torch.cat([sp_mean, sp_max, sp_std, sp_count_norm, bbox_diag], dim=-1)
#         return self.cond_mlp(cond_feat)

#     # ===================== ★ Early Fusion Forward (Stage 1 & 3 共用) =====================

#     def _early_fusion_forward(self, feats, feat_2d, superpoints, voxel_coords,
#                               v2p_map, p2v_map, spatial_shape, coords_float,
#                               batch_offsets, batch_size):
#         """拿到 feat_2d [M, 256] 后的标准 early fusion 流程"""
#         feat_2d_point = feat_2d[superpoints]                   # [N, 256]
#         feats_full = torch.cat([feats, feat_2d_point], dim=-1) # [N, 265]

#         voxel_feats = pointgroup_ops.voxelization(feats_full, v2p_map)
#         input_tensor = spconv.SparseConvTensor(
#             voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
#         sp_coords = scatter_mean(coords_float, superpoints, dim=0)
#         sp_feats, _, _ = self.extract_feat(
#             input_tensor, superpoints, p2v_map, sp_coords, coords_float)
#         return sp_feats, sp_coords

#     # ===================== Loss =====================

#     @cuda_cast
#     def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape,
#              feats, insts, superpoints, coords_float, batch_offsets,
#              sp_instance_labels, feat_2d=None, **kwargs):
#         batch_size = len(batch_offsets) - 1

#         # ===================== Stage 2: 只训 flow matching =====================
#         if self.stage == 2:
#             assert feat_2d is not None, 'Stage 2 needs GT feat_2d as target'
#             assert self.diffusion_module is not None, 'Stage 2 needs diffusion_module'

#             cond = self._get_diffusion_cond(feats, superpoints)  # [M, 256]
#             loss_diff, cos_sim, loss_detail = self.diffusion_module.compute_loss(
#                 cond, feat_2d, batch_offsets)

#             log_vars = {
#                 'loss': loss_diff.item(),
#                 'cos_sim': cos_sim,
#             }
#             log_vars.update(loss_detail)
#             return loss_diff, log_vars

#         # ===================== Stage 3: flow matching 生成 → early fusion =====================
#         if self.stage == 3:
#             assert self.diffusion_module is not None, 'Stage 3 needs diffusion_module'

#             with torch.no_grad():
#                 cond = self._get_diffusion_cond(feats, superpoints)
#                 feat_2d_pred = self.diffusion_module.sample(
#                     cond, batch_offsets, steps=self.ddim_steps_infer)

#             torch.cuda.empty_cache()

#             # 用 flow matching 输出做 early fusion
#             sp_feats, sp_coords = self._early_fusion_forward(
#                 feats, feat_2d_pred, superpoints, voxel_coords,
#                 v2p_map, p2v_map, spatial_shape, coords_float,
#                 batch_offsets, batch_size)

#             out, sp_feats_update_list, _ = self.decoder(
#                 sp_feats, sp_coords, batch_offsets, self.epoch)

#             loss_sim, loss_out = self._compute_clsr(
#                 sp_feats_update_list, sp_instance_labels, batch_offsets, batch_size)
#             loss, loss_dict = self.criterion(out, insts, loss_sim)
#             loss_dict.update(loss_out)
#             return loss, loss_dict

#         # ===================== Stage 1: GT 2D early fusion (原始) =====================
#         if feat_2d is not None:
#             feat_2d_point = feat_2d[superpoints]
#             feats = torch.cat([feats, feat_2d_point], dim=-1)

#         voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
#         input_tensor = spconv.SparseConvTensor(
#             voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
#         sp_coords = scatter_mean(coords_float, superpoints, dim=0)
#         sp_feats, _, _ = self.extract_feat(
#             input_tensor, superpoints, p2v_map, sp_coords, coords_float)

#         out, sp_feats_update_list, _ = self.decoder(
#             sp_feats, sp_coords, batch_offsets, self.epoch)

#         loss_sim, loss_out = self._compute_clsr(
#             sp_feats_update_list, sp_instance_labels, batch_offsets, batch_size)
#         loss, loss_dict = self.criterion(out, insts, loss_sim)
#         loss_dict.update(loss_out)
#         return loss, loss_dict

#     def _compute_clsr(self, sp_feats_update_list, sp_instance_labels, batch_offsets, batch_size):
#         """对比损失 CLSR"""
#         sp_instance_label_m_list = []
#         for i in range(batch_size):
#             sp_instance_label = sp_instance_labels[i]
#             sp_instance_label_m = torch.zeros(
#                 sp_instance_label.shape[0], sp_instance_label.shape[0]).cuda()
#             for j in torch.unique(sp_instance_label):
#                 a = torch.where(sp_instance_label == j)[0]
#                 grid_x, grid_y = torch.meshgrid(a, a, indexing='ij')
#                 sp_instance_label_m[grid_x, grid_y] = 1
#             sp_instance_label_m_list.append(sp_instance_label_m)

#         loss_sim = torch.tensor(0.0).cuda()
#         loss_out = {}
#         for layer_id in range(len(sp_feats_update_list)):
#             Sim = (F.normalize(sp_feats_update_list[layer_id])
#                    @ F.normalize(sp_feats_update_list[layer_id]).T)
#             loss_sim_i = torch.tensor(0.0).cuda()
#             for i in range(batch_size):
#                 sim_sample = Sim[batch_offsets[i]:batch_offsets[i+1],
#                                  batch_offsets[i]:batch_offsets[i+1]]
#                 loss_sim_i += F.binary_cross_entropy(
#                     torch.clamp((1 + sim_sample) / 2, 0, 1),
#                     sp_instance_label_m_list[i])
#             loss_sim_i = loss_sim_i / batch_size
#             loss_out[f'layer_{layer_id}_sim_loss'] = loss_sim_i.item()
#             loss_sim += loss_sim_i

#         return loss_sim, loss_out

#     # ===================== Predict =====================

#     @cuda_cast
#     def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape,
#                 feats, insts, superpoints, coords_float, batch_offsets,
#                 sp_instance_labels, feat_2d=None, **kwargs):
#         batch_size = len(batch_offsets) - 1

#         # ★ Stage 3 / 推理: flow matching 生成 2D 特征
#         if self.stage == 3 and self.diffusion_module is not None:
#             cond = self._get_diffusion_cond(feats, superpoints)
#             feat_2d = self.diffusion_module.sample(
#                 cond, batch_offsets, steps=self.ddim_steps_infer)
#             torch.cuda.empty_cache()

#         # Early Fusion
#         if feat_2d is not None:
#             feat_2d_point = feat_2d[superpoints]
#             feats = torch.cat([feats, feat_2d_point], dim=-1)

#         voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
#         input_tensor = spconv.SparseConvTensor(
#             voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
#         sp_coords = scatter_mean(coords_float, superpoints, dim=0)
#         sp_feats, _, _ = self.extract_feat(
#             input_tensor, superpoints, p2v_map, sp_coords, coords_float)

#         out, _, _ = self.decoder(
#             sp_feats, sp_coords, batch_offsets, self.epoch)
#         return self.predict_by_feat(scan_ids, out, superpoints, insts)

#     # ===================== 后处理 =====================

#     def predict_by_feat(self, scan_ids, out, superpoints, insts):
#         pred_labels = out['labels']
#         pred_masks = out['masks']
#         pred_scores = out['scores']

#         scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
#         nms_score = scores.max(-1)[0].squeeze()
#         proposals_pred_f = (pred_masks[0] > 0).float()
#         intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())
#         proposals_pointnum = proposals_pred_f.sum(1)
#         nms_score[proposals_pointnum == 0] = 0
#         proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(
#             1, proposals_pointnum.shape[0])
#         proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(
#             proposals_pointnum.shape[0], 1)
#         cross_ious = intersection / (
#             proposals_pn_h + proposals_pn_v - intersection + 1e-6)
#         pick_idxs = non_max_suppression(
#             cross_ious.cpu().numpy(),
#             nms_score.detach().cpu().numpy(), 0.75)

#         pred_labels = pred_labels[:, pick_idxs]
#         pred_masks[0] = pred_masks[0][pick_idxs]
#         scores = scores[pick_idxs]
#         labels = torch.arange(
#             self.num_class, device=scores.device
#         ).unsqueeze(0).repeat(pred_labels.shape[1], 1).flatten(0, 1)

#         self.test_cfg.topk_insts = min(
#             self.test_cfg.topk_insts, scores.flatten(0, 1).shape[0])
#         scores, topk_idx = scores.flatten(0, 1).topk(
#             self.test_cfg.topk_insts, sorted=False)
#         labels = labels[topk_idx]
#         labels += 1

#         topk_idx = torch.div(topk_idx, self.num_class, rounding_mode='floor')
#         mask_pred = pred_masks[0][topk_idx]
#         mask_pred_sigmoid = mask_pred.sigmoid()
#         mask_pred = (mask_pred > 0).float()
#         mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (
#             mask_pred.sum(1) + 1e-6)
#         scores = scores * mask_scores
#         mask_pred = mask_pred[:, superpoints].int()

#         score_mask = scores > self.test_cfg.score_thr
#         scores, labels, mask_pred = (
#             scores[score_mask], labels[score_mask], mask_pred[score_mask])

#         npoint_mask = mask_pred.sum(1) > self.test_cfg.npoint_thr
#         scores, labels, mask_pred = (
#             scores[npoint_mask], labels[npoint_mask], mask_pred[npoint_mask])

#         cls_pred = labels.cpu().numpy()
#         score_pred = scores.cpu().numpy()
#         mask_pred = mask_pred.cpu().numpy()

#         pred_instances = []
#         for i in range(cls_pred.shape[0]):
#             pred_instances.append({
#                 'scan_id': scan_ids[0],
#                 'label_id': cls_pred[i],
#                 'conf': round(score_pred[i], 1),
#                 'pred_mask': rle_encode(mask_pred[i]),
#             })

#         gt_instances = insts[0].gt_instances
#         return dict(
#             scan_id=scan_ids[0],
#             pred_instances=pred_instances,
#             gt_instances=gt_instances)


# def non_max_suppression(ious, scores, threshold):
#     ixs = scores.argsort()[::-1]
#     pick = []
#     while len(ixs) > 0:
#         i = ixs[0]
#         pick.append(i)
#         iou = ious[i, ixs[1:]]
#         remove_ixs = np.where((iou > threshold))[0] + 1
#         ixs = np.delete(ixs, remove_ixs)
#         ixs = np.delete(ixs, 0)
#     return np.array(pick, dtype=np.int32)



# """
# Relation3D + Early Fusion + Flow Matching (三阶段)
# ===================================================
# Stage 1: GT 2D 特征 early fusion 训练 (与原始一致)
# Stage 2: 冻结 Stage 1, 训练 cond_mlp + flow matching (从 3D 生成 2D)
# Stage 3: 冻结 cond_mlp + flow matching, 小 lr 微调 backbone + decoder

# 推理: 3D 点云 → flow matching 生成 pseudo 2D → early fusion → 预测
#       不需要任何 2D 图像输入

# 放置位置: relation3d/model/relation3d_earlyfusion.py
# """

# import functools
# import gorilla
# import pointgroup_ops
# import spconv.pytorch as spconv
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax
# import numpy as np

# from relation3d.utils import cuda_cast, rle_encode
# from .backbone import ResidualBlock, UBlock, MLP
# from .loss import Criterion
# from .query_decoder import QueryDecoder


# @gorilla.MODELS.register_module()
# class Relation3DEarlyFusion(nn.Module):

#     def __init__(
#         self,
#         input_channel: int = 9,
#         blocks: int = 5,
#         block_reps: int = 2,
#         media: int = 32,
#         normalize_before=True,
#         return_blocks=True,
#         pool='mean',
#         num_class=18,
#         decoder=None,
#         criterion=None,
#         test_cfg=None,
#         norm_eval=False,
#         fix_module=[],
#         d_2d=256,
#         # ★ Flow Matching 参数 (Stage 2/3 使用, Stage 1 可不传)
#         diffusion=None,
#         ddim_steps_infer=20,
#     ):
#         super().__init__()
#         self.d_2d = d_2d
#         self.num_class = num_class
#         self.input_channel = input_channel

#         # ===================== Backbone =====================
#         self.input_conv = spconv.SparseSequential(
#             spconv.SubMConv3d(
#                 input_channel + d_2d,   # 9 + 256 = 265
#                 media,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#                 indice_key='subm1',
#             ))
#         block = ResidualBlock
#         norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
#         block_list = [media * (i + 1) for i in range(blocks)]
#         self.unet = UBlock(
#             block_list, norm_fn, block_reps, block,
#             indice_key_id=1,
#             normalize_before=normalize_before,
#             return_blocks=return_blocks)
#         self.output_layer = spconv.SparseSequential(
#             norm_fn(media), nn.ReLU(inplace=True))
#         self.pool = pool

#         # ===================== ASAM =====================
#         self.mlp = nn.Sequential(
#             nn.Linear(2 * media, media), nn.ReLU(),
#             nn.Linear(media, media))
#         self.pooling_linear = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
#         self.pooling_linear1 = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
#         self.coords_linear = MLP(3, media, norm_fn=norm_fn, num_layers=3)

#         # ===================== Decoder =====================
#         self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

#         # ===================== Criterion =====================
#         self.criterion = Criterion(**criterion, num_class=num_class)

#         # ===================== ★ Flow Matching 相关 =====================
#         self.stage = 1
#         self.ddim_steps_infer = ddim_steps_infer

#         # 条件 MLP: scatter_mean(feats) [M, input_channel] → [M, 256]
#         self.cond_mlp = nn.Sequential(
#             nn.Linear(input_channel, 128),
#             nn.GELU(),
#             nn.Linear(128, 256),
#             nn.GELU(),
#             nn.Linear(256, d_2d),
#             nn.LayerNorm(d_2d),
#         )

#         # Flow Matching 模块 (接口名保持 diffusion_module 以兼容 set_stage)
#         if diffusion is not None:
#             from .flow_matching_module import SuperpointFlowMatching
#             self.diffusion_module = SuperpointFlowMatching(
#                 d_2d=d_2d,
#                 d_3d=d_2d,       # cond_mlp 输出维度
#                 d_model=diffusion.get('d_model', 256),
#                 n_blocks=diffusion.get('n_blocks', 12),
#                 n_heads=diffusion.get('n_heads', 8),
#                 mlp_ratio=diffusion.get('mlp_ratio', 4.0),
#                 dropout=diffusion.get('dropout', 0.0),
#                 sample_steps=diffusion.get('sample_steps', 20),
#                 loss_weights=diffusion.get('loss_weights', None),
#             )
#         else:
#             self.diffusion_module = None

#         self.epoch = 0
#         self.test_cfg = test_cfg
#         self.norm_eval = norm_eval

#         for module in fix_module:
#             mod = getattr(self, module)
#             mod.eval()
#             for param in mod.parameters():
#                 param.requires_grad = False

#     # ===================== Stage 管理 =====================

#     def set_stage(self, stage):
#         """
#         Stage 1: 训全网络 (GT 2D early fusion)
#         Stage 2: 冻结 backbone/ASAM/decoder, 只训 cond_mlp + flow matching
#         Stage 3: 冻结 cond_mlp + flow matching, 小 lr 微调其余
#         """
#         self.stage = stage
#         if stage == 2:
#             # 冻结 Stage 1 所有模块
#             for name, param in self.named_parameters():
#                 if 'cond_mlp' in name or 'diffusion_module' in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False
#             n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
#             print(f'[Stage 2] Trainable params: {n_train/1e6:.2f}M '
#                   f'(cond_mlp + flow matching only)')

#         elif stage == 3:
#             # 冻结 flow matching 相关, 解冻其余
#             for name, param in self.named_parameters():
#                 if 'cond_mlp' in name or 'diffusion_module' in name:
#                     param.requires_grad = False
#                 else:
#                     param.requires_grad = True
#             n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
#             print(f'[Stage 3] Trainable params: {n_train/1e6:.2f}M '
#                   f'(backbone + decoder, flow matching frozen)')

#     def train(self, mode=True):
#         super().train(mode)
#         if mode and self.norm_eval:
#             for m in self.modules():
#                 if isinstance(m, nn.BatchNorm1d):
#                     m.eval()
#         # Stage 2: 冻结模块保持 eval (BN 不更新统计量)
#         if mode and self.stage == 2:
#             self.input_conv.eval()
#             self.unet.eval()
#             self.output_layer.eval()
#             self.mlp.eval()
#             self.pooling_linear.eval()
#             self.pooling_linear1.eval()
#             self.coords_linear.eval()
#             self.decoder.eval()
#         # Stage 3: flow matching 保持 eval
#         if mode and self.stage == 3:
#             self.cond_mlp.eval()
#             if self.diffusion_module is not None:
#                 self.diffusion_module.eval()

#     # ===================== Pretrain Loading =====================

#     def load_pretrain_partial(self, pretrain_path):
#         """
#         加载预训练权重, 对 input_conv 做部分初始化。
#         前 input_channel 通道复制, 后 d_2d 通道零初始化。
#         """
#         ckpt = torch.load(pretrain_path, map_location='cpu')
#         if 'model' in ckpt:
#             state_dict = ckpt['model']
#         elif 'state_dict' in ckpt:
#             state_dict = ckpt['state_dict']
#         else:
#             state_dict = ckpt

#         conv_key = None
#         for k in state_dict.keys():
#             if 'input_conv' in k and 'weight' in k:
#                 conv_key = k
#                 break

#         if conv_key is not None:
#             old_w = state_dict[conv_key]
#             new_w = self.state_dict()[conv_key]
#             old_in = old_w.shape[1]
#             new_in = new_w.shape[1]
#             if old_in < new_in:
#                 new_w[:, :old_in] = old_w
#                 state_dict[conv_key] = new_w
#                 print(f'[EarlyFusion] input_conv: copied {old_in}/{new_in} channels, '
#                       f'{new_in - old_in} zero-initialized')
#             elif old_in == new_in:
#                 pass
#             else:
#                 print(f'[EarlyFusion] WARNING: pretrain has more channels ({old_in} > {new_in}), skipping')
#                 del state_dict[conv_key]

#         missing, unexpected = self.load_state_dict(state_dict, strict=False)
#         if missing:
#             print(f'[EarlyFusion] Missing keys ({len(missing)}): '
#                   f'{missing[:5]}{"..." if len(missing) > 5 else ""}')
#         if unexpected:
#             print(f'[EarlyFusion] Unexpected keys ({len(unexpected)}): '
#                   f'{unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')

#     # ===================== Forward =====================

#     def forward(self, batch, mode='loss'):
#         if mode == 'loss':
#             return self.loss(**batch)
#         elif mode == 'predict':
#             return self.predict(**batch)

#     # ===================== Backbone + ASAM =====================

#     def extract_feat(self, x, superpoints, v2p_map, sp_coords, coords_float):
#         x = self.input_conv(x)
#         x, _ = self.unet(x)
#         x = self.output_layer(x)
#         x = x.features[v2p_map.long()]

#         x_origin = x.clone()
#         x = scatter_mean(x_origin, superpoints, dim=0)
#         rel_fea_mean = self.pooling_linear((x[superpoints] - x_origin))
#         x_mean = scatter_sum(
#             scatter_softmax(rel_fea_mean, superpoints, dim=0) * x_origin,
#             superpoints, dim=0)
#         x, _ = scatter_max(x_origin, superpoints, dim=0)
#         rel_fea_max = self.pooling_linear1((x[superpoints] - x_origin))
#         x_max = scatter_sum(
#             scatter_softmax(rel_fea_max, superpoints, dim=0) * x_origin,
#             superpoints, dim=0)
#         x = self.mlp(torch.cat([x_mean, x_max], dim=-1))

#         return (x,
#                 scatter_softmax(rel_fea_mean, superpoints, dim=0),
#                 scatter_softmax(rel_fea_max, superpoints, dim=0))

#     # ===================== ★ 条件计算 (丰富版) =====================

#     def _get_diffusion_cond(self, feats, superpoints):
#         """scatter_mean(feats) [N, C] → [M, C] → cond_mlp → [M, 256]"""
#         feats_sp = scatter_mean(feats, superpoints, dim=0)   # [M, input_channel]
#         return self.cond_mlp(feats_sp)                       # [M, d_2d]

#     # ===================== ★ Early Fusion Forward (Stage 1 & 3 共用) =====================

#     def _early_fusion_forward(self, feats, feat_2d, superpoints, voxel_coords,
#                               v2p_map, p2v_map, spatial_shape, coords_float,
#                               batch_offsets, batch_size):
#         """拿到 feat_2d [M, 256] 后的标准 early fusion 流程"""
#         feat_2d_point = feat_2d[superpoints]                   # [N, 256]
#         feats_full = torch.cat([feats, feat_2d_point], dim=-1) # [N, 265]

#         voxel_feats = pointgroup_ops.voxelization(feats_full, v2p_map)
#         input_tensor = spconv.SparseConvTensor(
#             voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
#         sp_coords = scatter_mean(coords_float, superpoints, dim=0)
#         sp_feats, _, _ = self.extract_feat(
#             input_tensor, superpoints, p2v_map, sp_coords, coords_float)
#         return sp_feats, sp_coords

#     # ===================== Loss =====================

#     @cuda_cast
#     def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape,
#              feats, insts, superpoints, coords_float, batch_offsets,
#              sp_instance_labels, feat_2d=None, **kwargs):
#         batch_size = len(batch_offsets) - 1

#         # ===================== Stage 2: 只训 flow matching =====================
#         if self.stage == 2:
#             assert feat_2d is not None, 'Stage 2 needs GT feat_2d as target'
#             assert self.diffusion_module is not None, 'Stage 2 needs diffusion_module'

#             cond = self._get_diffusion_cond(feats, superpoints)  # [M, 256]
#             loss_diff, cos_sim, loss_detail = self.diffusion_module.compute_loss(
#                 cond, feat_2d, batch_offsets)

#             log_vars = {
#                 'loss': loss_diff.item(),
#                 'cos_sim': cos_sim,
#             }
#             log_vars.update(loss_detail)
#             return loss_diff, log_vars

#         # ===================== Stage 3: flow matching 生成 → early fusion =====================
#         if self.stage == 3:
#             assert self.diffusion_module is not None, 'Stage 3 needs diffusion_module'

#             with torch.no_grad():
#                 cond = self._get_diffusion_cond(feats, superpoints)
#                 feat_2d_pred = self.diffusion_module.sample(
#                     cond, batch_offsets, steps=self.ddim_steps_infer)

#             torch.cuda.empty_cache()

#             # 用 flow matching 输出做 early fusion
#             sp_feats, sp_coords = self._early_fusion_forward(
#                 feats, feat_2d_pred, superpoints, voxel_coords,
#                 v2p_map, p2v_map, spatial_shape, coords_float,
#                 batch_offsets, batch_size)

#             out, sp_feats_update_list, _ = self.decoder(
#                 sp_feats, sp_coords, batch_offsets, self.epoch)

#             loss_sim, loss_out = self._compute_clsr(
#                 sp_feats_update_list, sp_instance_labels, batch_offsets, batch_size)
#             loss, loss_dict = self.criterion(out, insts, loss_sim)
#             loss_dict.update(loss_out)
#             return loss, loss_dict

#         # ===================== Stage 1: GT 2D early fusion (原始) =====================
#         if feat_2d is not None:
#             feat_2d_point = feat_2d[superpoints]
#             feats = torch.cat([feats, feat_2d_point], dim=-1)

#         voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
#         input_tensor = spconv.SparseConvTensor(
#             voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
#         sp_coords = scatter_mean(coords_float, superpoints, dim=0)
#         sp_feats, _, _ = self.extract_feat(
#             input_tensor, superpoints, p2v_map, sp_coords, coords_float)

#         out, sp_feats_update_list, _ = self.decoder(
#             sp_feats, sp_coords, batch_offsets, self.epoch)

#         loss_sim, loss_out = self._compute_clsr(
#             sp_feats_update_list, sp_instance_labels, batch_offsets, batch_size)
#         loss, loss_dict = self.criterion(out, insts, loss_sim)
#         loss_dict.update(loss_out)
#         return loss, loss_dict

#     def _compute_clsr(self, sp_feats_update_list, sp_instance_labels, batch_offsets, batch_size):
#         """对比损失 CLSR"""
#         sp_instance_label_m_list = []
#         for i in range(batch_size):
#             sp_instance_label = sp_instance_labels[i]
#             sp_instance_label_m = torch.zeros(
#                 sp_instance_label.shape[0], sp_instance_label.shape[0]).cuda()
#             for j in torch.unique(sp_instance_label):
#                 a = torch.where(sp_instance_label == j)[0]
#                 grid_x, grid_y = torch.meshgrid(a, a, indexing='ij')
#                 sp_instance_label_m[grid_x, grid_y] = 1
#             sp_instance_label_m_list.append(sp_instance_label_m)

#         loss_sim = torch.tensor(0.0).cuda()
#         loss_out = {}
#         for layer_id in range(len(sp_feats_update_list)):
#             Sim = (F.normalize(sp_feats_update_list[layer_id])
#                    @ F.normalize(sp_feats_update_list[layer_id]).T)
#             loss_sim_i = torch.tensor(0.0).cuda()
#             for i in range(batch_size):
#                 sim_sample = Sim[batch_offsets[i]:batch_offsets[i+1],
#                                  batch_offsets[i]:batch_offsets[i+1]]
#                 loss_sim_i += F.binary_cross_entropy(
#                     torch.clamp((1 + sim_sample) / 2, 0, 1),
#                     sp_instance_label_m_list[i])
#             loss_sim_i = loss_sim_i / batch_size
#             loss_out[f'layer_{layer_id}_sim_loss'] = loss_sim_i.item()
#             loss_sim += loss_sim_i

#         return loss_sim, loss_out

#     # ===================== Predict =====================

#     @cuda_cast
#     def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape,
#                 feats, insts, superpoints, coords_float, batch_offsets,
#                 sp_instance_labels, feat_2d=None, **kwargs):
#         batch_size = len(batch_offsets) - 1

#         # ★ Stage 3 / 推理: flow matching 生成 2D 特征
#         if self.stage == 3 and self.diffusion_module is not None:
#             cond = self._get_diffusion_cond(feats, superpoints)
#             feat_2d = self.diffusion_module.sample(
#                 cond, batch_offsets, steps=self.ddim_steps_infer)
#             torch.cuda.empty_cache()

#         # Early Fusion
#         if feat_2d is not None:
#             feat_2d_point = feat_2d[superpoints]
#             feats = torch.cat([feats, feat_2d_point], dim=-1)

#         voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
#         input_tensor = spconv.SparseConvTensor(
#             voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
#         sp_coords = scatter_mean(coords_float, superpoints, dim=0)
#         sp_feats, _, _ = self.extract_feat(
#             input_tensor, superpoints, p2v_map, sp_coords, coords_float)

#         out, _, _ = self.decoder(
#             sp_feats, sp_coords, batch_offsets, self.epoch)
#         return self.predict_by_feat(scan_ids, out, superpoints, insts)

#     # ===================== 后处理 =====================

#     def predict_by_feat(self, scan_ids, out, superpoints, insts):
#         pred_labels = out['labels']
#         pred_masks = out['masks']
#         pred_scores = out['scores']

#         scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
#         nms_score = scores.max(-1)[0].squeeze()
#         proposals_pred_f = (pred_masks[0] > 0).float()
#         intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())
#         proposals_pointnum = proposals_pred_f.sum(1)
#         nms_score[proposals_pointnum == 0] = 0
#         proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(
#             1, proposals_pointnum.shape[0])
#         proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(
#             proposals_pointnum.shape[0], 1)
#         cross_ious = intersection / (
#             proposals_pn_h + proposals_pn_v - intersection + 1e-6)
#         pick_idxs = non_max_suppression(
#             cross_ious.cpu().numpy(),
#             nms_score.detach().cpu().numpy(), 0.75)

#         pred_labels = pred_labels[:, pick_idxs]
#         pred_masks[0] = pred_masks[0][pick_idxs]
#         scores = scores[pick_idxs]
#         labels = torch.arange(
#             self.num_class, device=scores.device
#         ).unsqueeze(0).repeat(pred_labels.shape[1], 1).flatten(0, 1)

#         self.test_cfg.topk_insts = min(
#             self.test_cfg.topk_insts, scores.flatten(0, 1).shape[0])
#         scores, topk_idx = scores.flatten(0, 1).topk(
#             self.test_cfg.topk_insts, sorted=False)
#         labels = labels[topk_idx]
#         labels += 1

#         topk_idx = torch.div(topk_idx, self.num_class, rounding_mode='floor')
#         mask_pred = pred_masks[0][topk_idx]
#         mask_pred_sigmoid = mask_pred.sigmoid()
#         mask_pred = (mask_pred > 0).float()
#         mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (
#             mask_pred.sum(1) + 1e-6)
#         scores = scores * mask_scores
#         mask_pred = mask_pred[:, superpoints].int()

#         score_mask = scores > self.test_cfg.score_thr
#         scores, labels, mask_pred = (
#             scores[score_mask], labels[score_mask], mask_pred[score_mask])

#         npoint_mask = mask_pred.sum(1) > self.test_cfg.npoint_thr
#         scores, labels, mask_pred = (
#             scores[npoint_mask], labels[npoint_mask], mask_pred[npoint_mask])

#         cls_pred = labels.cpu().numpy()
#         score_pred = scores.cpu().numpy()
#         mask_pred = mask_pred.cpu().numpy()

#         pred_instances = []
#         for i in range(cls_pred.shape[0]):
#             pred_instances.append({
#                 'scan_id': scan_ids[0],
#                 'label_id': cls_pred[i],
#                 'conf': round(score_pred[i], 1),
#                 'pred_mask': rle_encode(mask_pred[i]),
#             })

#         gt_instances = insts[0].gt_instances
#         return dict(
#             scan_id=scan_ids[0],
#             pred_instances=pred_instances,
#             gt_instances=gt_instances)


# def non_max_suppression(ious, scores, threshold):
#     ixs = scores.argsort()[::-1]
#     pick = []
#     while len(ixs) > 0:
#         i = ixs[0]
#         pick.append(i)
#         iou = ious[i, ixs[1:]]
#         remove_ixs = np.where((iou > threshold))[0] + 1
#         ixs = np.delete(ixs, remove_ixs)
#         ixs = np.delete(ixs, 0)
#     return np.array(pick, dtype=np.int32)





"""
Relation3D + Early Fusion (两阶段)
=================================
Stage 1: GT 2D 特征 early fusion 训练 (+ masked feature prediction)
Stage 2: 冻结 Stage 1 teacher + 纯3D encoder, 训练 diffusion:
         纯3D superpoint feat -> diffusion -> teacher superpoint feat

推理(Stage 2):
  3D 点云 -> 纯3D encoder -> diffusion -> enhanced 3D superpoint feat
           -> 冻结 decoder -> 分割预测
"""

import functools
import gorilla
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax
import numpy as np

from relation3d.utils import cuda_cast, rle_encode
from .backbone import ResidualBlock, UBlock, MLP, get_relation3d_spconv_algo
from .loss import Criterion
from .query_decoder import QueryDecoder


@gorilla.MODELS.register_module()
class Relation3DEarlyFusion(nn.Module):

    def __init__(
        self,
        input_channel: int = 9,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        num_class=18,
        decoder=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
        d_2d=256,
        # 保留旧参数仅为兼容旧配置，当前 active 版本只使用 stage1
        diffusion=None,
        ddim_steps_infer=20,
        # ★ Stage 1 辅助任务参数
        feature_aux_mode='mask',
        mask_ratio=0.3,
        mask_loss_weight=1.0,
        feature_aux_mask_policy='random',
        feature_aux_saliency_mix_ratio=0.5,
        feature_aux_saliency_warmup=0,
        feature_aux_weight_mode='none',
        feature_aux_fg_weight=2.0,
        feature_aux_boundary_weight=3.0,
        feature_aux_boundary_knn=8,
        object_mask_ratio=None,
        background_mask_ratio=None,
        object_min_points=2,
        object_logit_scale=10.0,
    ):
        super().__init__()
        self.d_2d = d_2d
        self.num_class = num_class
        self.input_channel = input_channel

        # ★ 辅助任务参数
        if feature_aux_mode not in ('mask', 'object', 'background', 'full'):
            raise ValueError(
                "feature_aux_mode must be 'mask', 'object', 'background' or 'full', "
                f'got {feature_aux_mode}')
        if feature_aux_mask_policy not in ('random', 'saliency_mix'):
            raise ValueError(
                "feature_aux_mask_policy must be 'random' or 'saliency_mix', "
                f'got {feature_aux_mask_policy}')
        if feature_aux_weight_mode not in ('none', 'fg_boundary'):
            raise ValueError(
                "feature_aux_weight_mode must be 'none' or 'fg_boundary', "
                f'got {feature_aux_weight_mode}')
        self.feature_aux_mode = feature_aux_mode
        self.mask_ratio = mask_ratio
        self.mask_loss_weight = mask_loss_weight
        self.feature_aux_mask_policy = feature_aux_mask_policy
        self.feature_aux_saliency_mix_ratio = float(feature_aux_saliency_mix_ratio)
        self.feature_aux_saliency_warmup = max(int(feature_aux_saliency_warmup), 0)
        self.feature_aux_weight_mode = feature_aux_weight_mode
        self.feature_aux_fg_weight = float(feature_aux_fg_weight)
        self.feature_aux_boundary_weight = float(feature_aux_boundary_weight)
        self.feature_aux_boundary_knn = max(int(feature_aux_boundary_knn), 0)
        if not 0.0 <= self.feature_aux_saliency_mix_ratio <= 1.0:
            raise ValueError(
                'feature_aux_saliency_mix_ratio must be in [0, 1], '
                f'got {self.feature_aux_saliency_mix_ratio}')
        if self.feature_aux_fg_weight < 1.0:
            raise ValueError(
                f'feature_aux_fg_weight must be >= 1.0, got {self.feature_aux_fg_weight}')
        if self.feature_aux_boundary_weight < 1.0:
            raise ValueError(
                'feature_aux_boundary_weight must be >= 1.0, '
                f'got {self.feature_aux_boundary_weight}')
        self.object_mask_ratio = (
            mask_ratio if object_mask_ratio is None else object_mask_ratio)
        if not 0.0 <= self.object_mask_ratio < 1.0:
            raise ValueError(
                f'object_mask_ratio must be in [0, 1), got {self.object_mask_ratio}')
        self.background_mask_ratio = (
            mask_ratio if background_mask_ratio is None else background_mask_ratio)
        if not 0.0 <= self.background_mask_ratio <= 1.0:
            raise ValueError(
                f'background_mask_ratio must be in [0, 1], got {self.background_mask_ratio}')
        self.object_min_points = max(int(object_min_points), 2)
        self.object_logit_scale = float(object_logit_scale)  # kept for config compatibility
        conv_algo = get_relation3d_spconv_algo()

        # ===================== Backbone =====================
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel + d_2d,   # 9 + 256 = 265
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
                algo=conv_algo,
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list, norm_fn, block_reps, block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks)
        self.output_layer = spconv.SparseSequential(
            norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool

        # ===================== ASAM =====================
        self.mlp = nn.Sequential(
            nn.Linear(2 * media, media), nn.ReLU(),
            nn.Linear(media, media))
        self.pooling_linear = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
        self.pooling_linear1 = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
        self.coords_linear = MLP(3, media, norm_fn=norm_fn, num_layers=3)

        # ===================== Decoder =====================
        self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

        # ===================== Criterion =====================
        self.criterion = Criterion(**criterion, num_class=num_class)

        # ===================== ★ Masked Feature Prediction Projector =====================
        # backbone 输出 superpoint feat [M, media] → 预测被 mask 掉的 2D 特征 [M, d_2d]
        self.feat_projector = nn.Sequential(
            nn.Linear(media, media * 4),
            nn.GELU(),
            nn.Linear(media * 4, media * 4),
            nn.GELU(),
            nn.Linear(media * 4, d_2d),
            nn.LayerNorm(d_2d),
        )
        # ===================== Stage =====================
        self.stage = 1
        self.ddim_steps_infer = ddim_steps_infer

        self.epoch = 0
        self.test_cfg = test_cfg
        self.norm_eval = norm_eval

        for module in fix_module:
            mod = getattr(self, module)
            mod.eval()
            for param in mod.parameters():
                param.requires_grad = False

    # ===================== Stage 管理 =====================

    def set_stage(self, stage):
        """
        Active 版本只保留 Stage 1.
        """
        if stage != 1:
            raise ValueError(
                f'Relation3DEarlyFusion now only supports stage=1, got stage={stage}')
        self.stage = 1

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    # ===================== Pretrain Loading =====================

    def load_pretrain_partial(self, pretrain_path):
        ckpt = torch.load(pretrain_path, map_location='cpu')
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        conv_key = None
        for k in state_dict.keys():
            if 'input_conv' in k and 'weight' in k:
                conv_key = k
                break

        if conv_key is not None:
            old_w = state_dict[conv_key]
            new_w = self.state_dict()[conv_key]
            # ★ spconv 权重 layout: [out, kx, ky, kz, in] — 输入通道在最后一维
            old_in = old_w.shape[-1]
            new_in = new_w.shape[-1]
            if old_in < new_in:
                new_w[..., :old_in] = old_w
                state_dict[conv_key] = new_w
                print(f'[EarlyFusion] input_conv: copied {old_in}/{new_in} channels, '
                    f'{new_in - old_in} zero-initialized')
            elif old_in == new_in:
                pass
            else:
                print(f'[EarlyFusion] WARNING: pretrain has more channels ({old_in} > {new_in}), skipping')
                del state_dict[conv_key]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[EarlyFusion] Missing keys ({len(missing)}): '
                f'{missing[:5]}{"..." if len(missing) > 5 else ""}')
        if unexpected:
            print(f'[EarlyFusion] Unexpected keys ({len(unexpected)}): '
                f'{unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')

    # ===================== Forward =====================

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    # ===================== Backbone + ASAM =====================

    def extract_feat(self, x, superpoints, v2p_map, sp_coords, coords_float):
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]

        x_origin = x.clone()
        x = scatter_mean(x_origin, superpoints, dim=0)
        rel_fea_mean = self.pooling_linear((x[superpoints] - x_origin))
        x_mean = scatter_sum(
            scatter_softmax(rel_fea_mean, superpoints, dim=0) * x_origin,
            superpoints, dim=0)
        x, _ = scatter_max(x_origin, superpoints, dim=0)
        rel_fea_max = self.pooling_linear1((x[superpoints] - x_origin))
        x_max = scatter_sum(
            scatter_softmax(rel_fea_max, superpoints, dim=0) * x_origin,
            superpoints, dim=0)
        x = self.mlp(torch.cat([x_mean, x_max], dim=-1))

        return (x,
                scatter_softmax(rel_fea_mean, superpoints, dim=0),
                scatter_softmax(rel_fea_max, superpoints, dim=0))

    # ===================== ★ Feature Auxiliary Tasks =====================

    def _apply_mask_indices(self, feat_2d, mask_indices):
        """将给定 superpoint 索引处的 2D 特征置零。"""
        feat_2d_masked = feat_2d.clone()
        if mask_indices is not None and mask_indices.numel() > 0:
            feat_2d_masked[mask_indices] = 0.0
        return feat_2d_masked

    def _sample_random_mask_indices(self, feat_2d, batch_offsets):
        """按 scene 随机采样被 mask 的 superpoint 索引。"""
        batch_size = len(batch_offsets) - 1
        mask_indices_list = []

        for i in range(batch_size):
            start = batch_offsets[i]
            end = batch_offsets[i + 1]
            n_sp = end - start
            if n_sp <= 0:
                continue

            n_mask = max(1, int(n_sp * self.mask_ratio))
            n_mask = min(n_mask, n_sp)

            perm = torch.randperm(n_sp, device=feat_2d.device)[:n_mask]
            mask_indices_list.append(perm + start)

        if mask_indices_list:
            return torch.cat(mask_indices_list, dim=0)

        return torch.empty(0, dtype=torch.long, device=feat_2d.device)

    def _sample_saliency_mix_mask_indices(self, feat_2d, saliency, batch_offsets):
        """
        saliency-guided adaptive masking:
        - 一部分 mask 位置由高 saliency superpoints 提供
        - 剩余位置继续随机采样，避免输入分布被完全改坏
        """
        stats = {
            'adaptive_ratio': 0.0,
            'selected_saliency': 0.0,
        }
        if saliency is None or saliency.numel() == 0:
            return self._sample_random_mask_indices(feat_2d, batch_offsets), stats

        batch_size = len(batch_offsets) - 1
        mask_indices_list = []
        total_mask = 0
        total_adaptive = 0
        total_selected_saliency = 0.0

        for i in range(batch_size):
            start = int(batch_offsets[i].item())
            end = int(batch_offsets[i + 1].item())
            n_sp = end - start
            if n_sp <= 0:
                continue

            n_mask = max(1, int(n_sp * self.mask_ratio))
            n_mask = min(n_mask, n_sp)

            n_adaptive = int(round(n_mask * self.feature_aux_saliency_mix_ratio))
            if self.feature_aux_saliency_mix_ratio > 0 and n_mask > 0:
                n_adaptive = max(1, n_adaptive)
            n_adaptive = min(n_adaptive, n_mask, n_sp)

            local_saliency = saliency[start:end]
            local_mask = torch.zeros(n_sp, dtype=torch.bool, device=feat_2d.device)

            if n_adaptive > 0:
                adaptive_idx = local_saliency.topk(k=n_adaptive, largest=True).indices
                local_mask[adaptive_idx] = True
            else:
                adaptive_idx = torch.empty(0, dtype=torch.long, device=feat_2d.device)

            need_more = n_mask - int(local_mask.sum().item())
            if need_more > 0:
                remain_idx = torch.nonzero(~local_mask, as_tuple=False).view(-1)
                rand_perm = torch.randperm(remain_idx.numel(), device=feat_2d.device)[:need_more]
                rand_idx = remain_idx[rand_perm]
                local_mask[rand_idx] = True

            selected_idx = torch.nonzero(local_mask, as_tuple=False).view(-1)
            mask_indices_list.append(selected_idx + start)

            total_mask += int(selected_idx.numel())
            total_adaptive += int(adaptive_idx.numel())
            if selected_idx.numel() > 0:
                total_selected_saliency += float(local_saliency[selected_idx].sum().item())

        if mask_indices_list:
            mask_indices = torch.cat(mask_indices_list, dim=0)
        else:
            mask_indices = torch.empty(0, dtype=torch.long, device=feat_2d.device)

        if total_mask > 0:
            stats['adaptive_ratio'] = total_adaptive / total_mask
            stats['selected_saliency'] = total_selected_saliency / total_mask

        return mask_indices, stats

    def _mask_superpoint_feats(self, feat_2d, batch_offsets):
        """
        随机 mask 掉一些超点的 2D 特征 (置零), 返回 mask 后的特征和 mask 索引。

        Args:
            feat_2d: [M, d_2d] 超点级 2D 特征
            batch_offsets: [B+1] batch 边界

        Returns:
            feat_2d_masked: [M, d_2d] mask 后的特征 (被选中的超点特征置零)
            mask_indices: [K] 被 mask 掉的超点全局索引
        """
        mask_indices = self._sample_random_mask_indices(feat_2d, batch_offsets)
        feat_2d_masked = self._apply_mask_indices(feat_2d, mask_indices)
        return feat_2d_masked, mask_indices

    def _compute_detection_saliency(self, voxel_coords, p2v_map, v2p_map, spatial_shape,
                                    feats, insts, superpoints, coords_float, batch_offsets,
                                    sp_instance_labels, feat_2d):
        """
        用主任务损失对 superpoint 2D 特征的梯度范数估计 token importance。
        该 saliency 仅用于 mask 策略，不直接参与参数更新。
        """
        if (not self.training or feat_2d is None or self.feature_aux_mode != 'mask' or
                self.feature_aux_mask_policy != 'saliency_mix' or self.mask_ratio <= 0 or
                self.epoch < self.feature_aux_saliency_warmup):
            return None

        batch_size = len(batch_offsets) - 1
        feat_2d_proxy = feat_2d.detach().clone().requires_grad_(True)
        proxy_view = self._forward_stage1_view(
            voxel_coords, p2v_map, v2p_map, spatial_shape, feats,
            superpoints, coords_float, batch_offsets, feat_2d=feat_2d_proxy,
            insts=insts, apply_mask=False)
        loss_sim_proxy, _ = self._compute_clsr(
            proxy_view['sp_feats_update_list'], sp_instance_labels, batch_offsets, batch_size)
        det_loss_proxy, _ = self.criterion(proxy_view['out'], insts, loss_sim_proxy)
        saliency_grad = torch.autograd.grad(
            det_loss_proxy,
            feat_2d_proxy,
            retain_graph=False,
            create_graph=False,
            allow_unused=True)[0]
        if saliency_grad is None:
            return None

        return saliency_grad.detach().float().norm(p=2, dim=-1)

    def _compute_feature_aux_weights(self, mask_indices, insts, batch_offsets, sp_coords):
        """
        根据 GT object mask 估计被 mask 超点的重要性:
        - foreground superpoint 权重更高
        - instance boundary superpoint 权重最高

        该权重仅作用在辅助重建 loss，不改变 encoder 输入分布。
        """
        stats = {
            'fg_ratio': 0.0,
            'boundary_ratio': 0.0,
        }
        if (self.feature_aux_weight_mode == 'none' or mask_indices is None or
                mask_indices.numel() == 0 or insts is None):
            return None, stats

        with torch.no_grad():
            mask_weights = torch.ones(
                mask_indices.shape[0], dtype=sp_coords.dtype, device=sp_coords.device)
            mask_is_fg = torch.zeros(
                mask_indices.shape[0], dtype=torch.bool, device=sp_coords.device)
            mask_is_boundary = torch.zeros(
                mask_indices.shape[0], dtype=torch.bool, device=sp_coords.device)

            batch_size = len(batch_offsets) - 1
            for scene_id in range(batch_size):
                start = int(batch_offsets[scene_id].item())
                end = int(batch_offsets[scene_id + 1].item())
                local_mask_sel = (mask_indices >= start) & (mask_indices < end)
                if not local_mask_sel.any():
                    continue

                n_sp = end - start
                fg_mask = torch.zeros(n_sp, dtype=torch.bool, device=sp_coords.device)
                boundary_mask = torch.zeros(n_sp, dtype=torch.bool, device=sp_coords.device)

                inst = insts[scene_id]
                if hasattr(inst, 'gt_spmasks') and inst.gt_spmasks.numel() > 0:
                    gt_spmasks = inst.gt_spmasks
                    if gt_spmasks.dim() == 1:
                        gt_spmasks = gt_spmasks.unsqueeze(0)
                    gt_spmasks = (gt_spmasks.to(sp_coords.device) > 0.5)
                    fg_mask = gt_spmasks.any(dim=0)

                    if fg_mask.any() and n_sp > 1 and self.feature_aux_boundary_knn > 0:
                        owner = torch.full(
                            (n_sp,), -1, dtype=torch.long, device=sp_coords.device)
                        owner[fg_mask] = gt_spmasks[:, fg_mask].float().argmax(dim=0)

                        scene_coords = sp_coords[start:end]
                        k = min(self.feature_aux_boundary_knn, n_sp - 1)
                        if k > 0:
                            dist = torch.cdist(scene_coords, scene_coords)
                            dist.fill_diagonal_(float('inf'))
                            knn_idx = dist.topk(k=k, largest=False).indices
                            neighbor_owner = owner[knn_idx]
                            boundary_mask = fg_mask & (
                                neighbor_owner != owner.unsqueeze(1)).any(dim=1)

                mask_idx_local = mask_indices[local_mask_sel] - start
                local_fg = fg_mask[mask_idx_local]
                local_boundary = boundary_mask[mask_idx_local]

                local_weights = torch.ones(
                    mask_idx_local.shape[0], dtype=sp_coords.dtype, device=sp_coords.device)
                local_weights[local_fg] = self.feature_aux_fg_weight
                local_weights[local_boundary] = self.feature_aux_boundary_weight

                mask_weights[local_mask_sel] = local_weights
                mask_is_fg[local_mask_sel] = local_fg
                mask_is_boundary[local_mask_sel] = local_boundary

            stats['fg_ratio'] = mask_is_fg.float().mean().item()
            stats['boundary_ratio'] = mask_is_boundary.float().mean().item()
            mask_weights = mask_weights / mask_weights.mean().clamp(min=1e-6)

        return mask_weights, stats

    def _compute_mask_prediction_loss(self, sp_feats, feat_2d_gt, mask_indices, mask_weights=None):
        """
        用 projector 从 backbone 超点特征预测被 mask 的 2D 特征, 计算重建 loss。

        Args:
            sp_feats: [M, media] backbone 输出的超点特征
            feat_2d_gt: [M, d_2d] 原始未 mask 的 GT 2D 特征
            mask_indices: [K] 被 mask 的超点索引

        Returns:
            loss_mask: scalar, 重建 loss
            cos_sim: float, mask 位置的平均 cosine similarity (用于监控)
        """
        if mask_indices.numel() == 0:
            return sp_feats.sum() * 0.0, 0.0

        # 只在被 mask 的超点上做预测 (节省计算)
        pred_feat = self.feat_projector(sp_feats[mask_indices])   # [K, d_2d]
        gt_feat = feat_2d_gt[mask_indices]                         # [K, d_2d]

        # token 级 smooth_l1，便于对 foreground / boundary 位置加权
        loss_l2 = F.smooth_l1_loss(pred_feat, gt_feat, reduction='none').mean(dim=-1)

        # token 级 cosine similarity loss
        pred_norm = F.normalize(pred_feat, dim=-1)
        gt_norm = F.normalize(gt_feat, dim=-1)
        cos_sim_per = (pred_norm * gt_norm).sum(-1)
        loss_cos = 1.0 - cos_sim_per

        per_token_loss = loss_l2 + loss_cos
        if mask_weights is not None:
            weights = mask_weights.to(per_token_loss.dtype)
            loss_mask = (per_token_loss * weights).sum() / weights.sum().clamp(min=1e-6)
        else:
            loss_mask = per_token_loss.mean()

        return loss_mask, cos_sim_per.mean().item()

    def _mask_object_superpoint_feats(self, feat_2d, insts, batch_offsets):
        """
        用 GT object-superpoint mask 在每个 object 内随机 mask 一部分 superpoint。
        任务形式仍与原始 masked reconstruction 一致，只是把 mask 位置从随机改成 object-aware。

        Returns:
            feat_2d_masked: [M, d_2d]
            mask_indices: [K] 被 mask 的超点全局索引
        """
        feat_2d_masked = feat_2d.clone()
        mask_indices_list = []
        batch_size = len(batch_offsets) - 1

        for scene_id in range(batch_size):
            start = int(batch_offsets[scene_id].item())
            inst = insts[scene_id]
            if (not hasattr(inst, 'gt_spmasks')) or inst.gt_spmasks.numel() == 0:
                continue

            gt_spmasks = inst.gt_spmasks
            if gt_spmasks.dim() == 1:
                gt_spmasks = gt_spmasks.unsqueeze(0)
            gt_spmasks = gt_spmasks.to(feat_2d.device)

            for obj_mask in gt_spmasks:
                obj_idx_local = torch.nonzero(obj_mask > 0.5, as_tuple=False).view(-1)
                if obj_idx_local.numel() < self.object_min_points:
                    continue

                n_mask = max(1, int(round(obj_idx_local.numel() * self.object_mask_ratio)))
                n_mask = min(n_mask, obj_idx_local.numel() - 1)
                if n_mask <= 0:
                    continue

                perm = torch.randperm(obj_idx_local.numel(), device=feat_2d.device)
                mask_local = obj_idx_local[perm[:n_mask]]
                mask_indices_list.append(start + mask_local)

        if mask_indices_list:
            mask_indices = torch.unique(torch.cat(mask_indices_list, dim=0))
            feat_2d_masked[mask_indices] = 0.0
        else:
            mask_indices = torch.empty(0, dtype=torch.long, device=feat_2d.device)

        return feat_2d_masked, mask_indices

    def _mask_background_superpoint_feats(self, feat_2d, insts, batch_offsets):
        """
        仅在背景 superpoint 上进行 mask，保留 object superpoints 不变。
        背景定义为：不属于任何 GT object 的 superpoint。

        Returns:
            feat_2d_masked: [M, d_2d]
            mask_indices: [K] 被 mask 的背景超点全局索引
        """
        feat_2d_masked = feat_2d.clone()
        mask_indices_list = []
        batch_size = len(batch_offsets) - 1

        for scene_id in range(batch_size):
            start = int(batch_offsets[scene_id].item())
            end = int(batch_offsets[scene_id + 1].item())
            n_sp = end - start
            if n_sp <= 0:
                continue

            bg_mask = torch.ones(n_sp, dtype=torch.bool, device=feat_2d.device)
            inst = insts[scene_id]
            if hasattr(inst, 'gt_spmasks') and inst.gt_spmasks.numel() > 0:
                gt_spmasks = inst.gt_spmasks
                if gt_spmasks.dim() == 1:
                    gt_spmasks = gt_spmasks.unsqueeze(0)
                gt_spmasks = (gt_spmasks.to(feat_2d.device) > 0.5)
                bg_mask = ~gt_spmasks.any(dim=0)

            bg_idx_local = torch.nonzero(bg_mask, as_tuple=False).view(-1)
            if bg_idx_local.numel() == 0 or self.background_mask_ratio <= 0:
                continue

            if self.background_mask_ratio >= 1.0:
                mask_local = bg_idx_local
            else:
                n_mask = max(1, int(round(bg_idx_local.numel() * self.background_mask_ratio)))
                n_mask = min(n_mask, bg_idx_local.numel())
                perm = torch.randperm(bg_idx_local.numel(), device=feat_2d.device)
                mask_local = bg_idx_local[perm[:n_mask]]

            mask_indices_list.append(start + mask_local)

        if mask_indices_list:
            mask_indices = torch.unique(torch.cat(mask_indices_list, dim=0))
            feat_2d_masked[mask_indices] = 0.0
        else:
            mask_indices = torch.empty(0, dtype=torch.long, device=feat_2d.device)

        return feat_2d_masked, mask_indices

    def _forward_stage1_view(self, voxel_coords, p2v_map, v2p_map, spatial_shape,
                             feats, superpoints, coords_float, batch_offsets,
                             feat_2d=None, insts=None, apply_mask=True, mask_indices=None):
        batch_size = len(batch_offsets) - 1
        sp_coords = scatter_mean(coords_float, superpoints, dim=0)

        if feat_2d is not None:
            feat_2d_gt = feat_2d.clone()
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
            feats_input = torch.cat([feats, feat_2d_point], dim=-1)
        else:
            feat_2d_gt = None
            mask_indices = None
            feats_input = feats

        voxel_feats = pointgroup_ops.voxelization(feats_input, v2p_map)
        input_tensor = spconv.SparseConvTensor(
            voxel_feats,
            voxel_coords.int(),
            spatial_shape,
            batch_size,
            force_algo=get_relation3d_spconv_algo(),
        )
        sp_feats, _, _ = self.extract_feat(
            input_tensor, superpoints, p2v_map, sp_coords, coords_float)

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
        return {
            'out': out,
            'sp_feats_update_list': sp_feats_update_list,
            'feature_aux_loss': feature_aux_loss,
            'feature_aux_metric': feature_aux_metric,
            'feature_aux_stats': feature_aux_stats,
        }

    # ===================== Loss =====================

    @cuda_cast
    def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape,
             feats, insts, superpoints, coords_float, batch_offsets,
             sp_instance_labels, feat_2d=None, **kwargs):
        batch_size = len(batch_offsets) - 1
        mask_indices = None
        mask_policy_stats = {}
        if self.training and feat_2d is not None and self.feature_aux_mode == 'mask' and self.mask_ratio > 0:
            if self.feature_aux_mask_policy == 'saliency_mix':
                saliency = self._compute_detection_saliency(
                    voxel_coords, p2v_map, v2p_map, spatial_shape, feats, insts,
                    superpoints, coords_float, batch_offsets, sp_instance_labels, feat_2d)
                if saliency is not None:
                    mask_indices, mask_policy_stats = self._sample_saliency_mix_mask_indices(
                        feat_2d, saliency, batch_offsets)
            if mask_indices is None:
                mask_indices = self._sample_random_mask_indices(feat_2d, batch_offsets)

        primary_view = self._forward_stage1_view(
            voxel_coords, p2v_map, v2p_map, spatial_shape, feats,
            superpoints, coords_float, batch_offsets, feat_2d=feat_2d,
            insts=insts, apply_mask=True, mask_indices=mask_indices)
        out = primary_view['out']
        feature_aux_loss = primary_view['feature_aux_loss']
        feature_aux_metric = primary_view['feature_aux_metric']
        feature_aux_stats = primary_view['feature_aux_stats']
        loss_sim, loss_out = self._compute_clsr(
            primary_view['sp_feats_update_list'], sp_instance_labels, batch_offsets, batch_size)
        loss, loss_dict = self.criterion(out, insts, loss_sim)
        loss_dict.update(loss_out)

        loss = loss + self.mask_loss_weight * feature_aux_loss
        if self.feature_aux_mode == 'full':
            loss_dict['full_pred_loss'] = feature_aux_loss.item()
            loss_dict['full_cos_sim'] = feature_aux_metric
        elif self.feature_aux_mode == 'object':
            loss_dict['object_mask_pred_loss'] = feature_aux_loss.item()
            loss_dict['object_mask_cos_sim'] = feature_aux_metric
        elif self.feature_aux_mode == 'background':
            loss_dict['background_mask_pred_loss'] = feature_aux_loss.item()
            loss_dict['background_mask_cos_sim'] = feature_aux_metric
        else:
            loss_dict['mask_pred_loss'] = feature_aux_loss.item()
            loss_dict['mask_cos_sim'] = feature_aux_metric
        if self.feature_aux_weight_mode != 'none':
            if self.feature_aux_mode == 'full':
                prefix = 'full'
            elif self.feature_aux_mode == 'object':
                prefix = 'object_mask'
            elif self.feature_aux_mode == 'background':
                prefix = 'background_mask'
            else:
                prefix = 'mask'
            loss_dict[f'{prefix}_fg_ratio'] = feature_aux_stats.get('fg_ratio', 0.0)
            loss_dict[f'{prefix}_boundary_ratio'] = feature_aux_stats.get('boundary_ratio', 0.0)
        if mask_policy_stats:
            loss_dict['mask_policy_adaptive_ratio'] = mask_policy_stats.get('adaptive_ratio', 0.0)
            loss_dict['mask_policy_selected_saliency'] = mask_policy_stats.get(
                'selected_saliency', 0.0)

        loss_dict['loss'] = loss.item()  # 更新总 loss

        return loss, loss_dict

    def _compute_clsr(self, sp_feats_update_list, sp_instance_labels, batch_offsets, batch_size):
        """对比损失 CLSR"""
        sp_instance_label_m_list = []
        for i in range(batch_size):
            sp_instance_label = sp_instance_labels[i]
            sp_instance_label_m = torch.zeros(
                sp_instance_label.shape[0], sp_instance_label.shape[0]).cuda()
            for j in torch.unique(sp_instance_label):
                a = torch.where(sp_instance_label == j)[0]
                grid_x, grid_y = torch.meshgrid(a, a, indexing='ij')
                sp_instance_label_m[grid_x, grid_y] = 1
            sp_instance_label_m_list.append(sp_instance_label_m)

        loss_sim = torch.tensor(0.0).cuda()
        loss_out = {}
        for layer_id in range(len(sp_feats_update_list)):
            Sim = (F.normalize(sp_feats_update_list[layer_id])
                   @ F.normalize(sp_feats_update_list[layer_id]).T)
            loss_sim_i = torch.tensor(0.0).cuda()
            for i in range(batch_size):
                sim_sample = Sim[batch_offsets[i]:batch_offsets[i+1],
                                 batch_offsets[i]:batch_offsets[i+1]]
                loss_sim_i += F.binary_cross_entropy(
                    torch.clamp((1 + sim_sample) / 2, 0, 1),
                    sp_instance_label_m_list[i])
            loss_sim_i = loss_sim_i / batch_size
            loss_out[f'layer_{layer_id}_sim_loss'] = loss_sim_i.item()
            loss_sim += loss_sim_i

        return loss_sim, loss_out

    # ===================== Predict =====================

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape,
                feats, insts, superpoints, coords_float, batch_offsets,
                sp_instance_labels, feat_2d=None, **kwargs):
        batch_size = len(batch_offsets) - 1

        # Early Fusion (推理时不 mask)
        if feat_2d is not None:
            feat_2d_point = feat_2d[superpoints]
            feats = torch.cat([feats, feat_2d_point], dim=-1)

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input_tensor = spconv.SparseConvTensor(
            voxel_feats,
            voxel_coords.int(),
            spatial_shape,
            batch_size,
            force_algo=get_relation3d_spconv_algo(),
        )
        sp_coords = scatter_mean(coords_float, superpoints, dim=0)
        sp_feats, _, _ = self.extract_feat(
            input_tensor, superpoints, p2v_map, sp_coords, coords_float)

        out, _, _ = self.decoder(
            sp_feats, sp_coords, batch_offsets, self.epoch)
        return self.predict_by_feat(scan_ids, out, superpoints, insts)

    # ===================== 后处理 =====================

    def predict_by_feat(self, scan_ids, out, superpoints, insts):
        pred_labels = out['labels']
        pred_masks = out['masks']
        pred_scores = out['scores']

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        nms_score = scores.max(-1)[0].squeeze()
        proposals_pred_f = (pred_masks[0] > 0).float()
        intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())
        proposals_pointnum = proposals_pred_f.sum(1)
        nms_score[proposals_pointnum == 0] = 0
        proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(
            1, proposals_pointnum.shape[0])
        proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(
            proposals_pointnum.shape[0], 1)
        cross_ious = intersection / (
            proposals_pn_h + proposals_pn_v - intersection + 1e-6)
        pick_idxs = non_max_suppression(
            cross_ious.cpu().numpy(),
            nms_score.detach().cpu().numpy(), 0.75)

        pred_labels = pred_labels[:, pick_idxs]
        pred_masks[0] = pred_masks[0][pick_idxs]
        scores = scores[pick_idxs]
        labels = torch.arange(
            self.num_class, device=scores.device
        ).unsqueeze(0).repeat(pred_labels.shape[1], 1).flatten(0, 1)

        self.test_cfg.topk_insts = min(
            self.test_cfg.topk_insts, scores.flatten(0, 1).shape[0])
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
        labels += 1

        topk_idx = torch.div(topk_idx, self.num_class, rounding_mode='floor')
        mask_pred = pred_masks[0][topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_pred = (mask_pred > 0).float()
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (
            mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        mask_pred = mask_pred[:, superpoints].int()

        score_mask = scores > self.test_cfg.score_thr
        scores, labels, mask_pred = (
            scores[score_mask], labels[score_mask], mask_pred[score_mask])

        npoint_mask = mask_pred.sum(1) > self.test_cfg.npoint_thr
        scores, labels, mask_pred = (
            scores[npoint_mask], labels[npoint_mask], mask_pred[npoint_mask])

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred_instances.append({
                'scan_id': scan_ids[0],
                'label_id': cls_pred[i],
                'conf': round(score_pred[i], 1),
                'pred_mask': rle_encode(mask_pred[i]),
            })

        gt_instances = insts[0].gt_instances
        return dict(
            scan_id=scan_ids[0],
            pred_instances=pred_instances,
            gt_instances=gt_instances)


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where((iou > threshold))[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)
