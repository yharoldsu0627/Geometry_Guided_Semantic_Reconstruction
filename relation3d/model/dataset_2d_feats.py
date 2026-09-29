# """
# Dataset Wrapper — 为 ScanNetV2 数据集添加 SegDINO3D 2D 特征
# ============================================================
# 加载 SegDINO3D 预计算的 per-point 2D 特征（4层平均 → [N, 256]）。

# 文件格式: {scan_id}.pth → list of 4 × Tensor[N, 256]
# 路径示例: /home/suyuanhao/SegDINO3D-main/data/features_2d/scannet/scene0000_00.pth

# 使用方式:
#   from relation3d.model.dataset_2d_feats import wrap_dataset_with_2d_feats
  
#   train_dataset = build_dataset(cfg.data.train, logger)
#   train_dataset = wrap_dataset_with_2d_feats(
#       train_dataset,
#       feat_dir='/home/suyuanhao/SegDINO3D-main/data/features_2d/scannet',
#       d_2d=256,
#   )
# """

# import os
# import numpy as np
# import torch
# import pointgroup_ops
# from torch.utils.data import Dataset


# class FeatureDatasetWrapper(Dataset):
#     """
#     包装原始 ScanNetV2 dataset，在每个样本后追加 feat_2d。
    
#     原始 dataset.__getitem__ 返回 8-tuple:
#       (scan_id, coord, coord_float, feat, superpoint, inst, normal, sp_instance_label)
    
#     包装后返回 9-tuple:
#       (scan_id, coord, coord_float, feat, superpoint, inst, normal, sp_instance_label, feat_2d)
    
#     同时覆盖 collate_fn 以正确处理第9个元素。
#     """

#     def __init__(self, base_dataset, feat_dir, d_2d=256):
#         self.base_dataset = base_dataset
#         self.feat_dir = feat_dir
#         self.d_2d = d_2d

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         data = self.base_dataset[idx]
#         # data = (scan_id, coord, coord_float, feat, superpoint, inst, normal, sp_instance_label)

#         scan_id = data[0]  # str, e.g. "scene0000_00"
#         n_points = data[1].shape[0]  # coord.shape[0]

#         feat_2d = self._load_feat(scan_id, n_points)

#         # 返回 9-tuple
#         return (*data, feat_2d)

#     def _load_feat(self, scan_id, n_points):
#         """
#         加载 SegDINO3D 2D 特征
        
#         返回: Tensor [N, d_2d]  per-point 2D 特征（4层平均）
#         """
#         pt_file = os.path.join(self.feat_dir, f'{scan_id}.pth')

#         if not os.path.exists(pt_file):
#             return torch.zeros(n_points, self.d_2d)

#         feat_list = torch.load(pt_file, map_location='cpu', weights_only=False)

#         if isinstance(feat_list, list):
#             # SegDINO3D 格式: list of 4 × Tensor[N, 256] → 4层平均
#             feat = torch.stack(feat_list, dim=0).mean(dim=0)  # [N, 256]
#         elif isinstance(feat_list, torch.Tensor):
#             feat = feat_list
#         else:
#             return torch.zeros(n_points, self.d_2d)

#         # 确保长度与点云一致
#         if feat.shape[0] > n_points:
#             feat = feat[:n_points]
#         elif feat.shape[0] < n_points:
#             padded = torch.zeros(n_points, self.d_2d)
#             padded[:feat.shape[0]] = feat
#             feat = padded

#         return feat

#     def collate_fn(self, batch):
#         """
#         覆盖原始 collate_fn，处理第9个元素 feat_2d。
        
#         在原始 collate_fn 返回的 dict 基础上增加 'feat_2d' 键。
#         feat_2d 需要与点云做相同的 concat（按 batch 顺序拼接）。
#         """
#         # 分离 feat_2d（第9个元素）和原始 8-tuple
#         feat_2d_list = []
#         original_batch = []
#         for item in batch:
#             *original_8, feat_2d = item
#             original_batch.append(tuple(original_8))
#             feat_2d_list.append(feat_2d)

#         # 调用原始 collate_fn 处理前 8 个元素
#         result = self.base_dataset.collate_fn(original_batch)

#         # 拼接 feat_2d: [B*N, d_2d]
#         feat_2d = torch.cat(feat_2d_list, dim=0)  # [B*N, d_2d]
#         result['feat_2d'] = feat_2d

#         return result

#     def __getattr__(self, name):
#         """代理访问 base_dataset 的属性（除了我们自己定义的）"""
#         if name in ('base_dataset', 'feat_dir', 'd_2d'):
#             raise AttributeError(name)
#         return getattr(self.base_dataset, name)


# def wrap_dataset_with_2d_feats(base_dataset, feat_dir, d_2d=256):
#     """
#     便捷函数: 包装数据集以添加 2D 特征
    
#     Args:
#         base_dataset:  原始 ScanNetV2 dataset
#         feat_dir:      SegDINO3D 特征目录
#         d_2d:          2D 特征维度 (默认 256)
    
#     Returns:
#         包装后的 dataset，collate_fn 输出的 dict 额外包含 'feat_2d' [B*N, d_2d]
#     """
#     return FeatureDatasetWrapper(base_dataset, feat_dir, d_2d)

# """
# Dataset Wrapper — 为 ScanNetV2 数据集添加预处理的 superpoint 级 2D 特征
# ====================================================================
# 加载预处理好的 per-superpoint 2D 特征 [M, 256]。

# 预处理流程 (已离线完成):
#   SegDINO3D 4层平均 → [N, 256] → scatter_mean by superpoint → [M, 256]

# 文件格式: {scan_id}.pt → Tensor[M, 256]
# 路径: dataset/scannet_v2/feat_2d_sp/

# 使用方式:
#   from relation3d.model.dataset_2d_feats import wrap_dataset_with_2d_feats
#   train_dataset = wrap_dataset_with_2d_feats(train_dataset, feat_dir, d_2d=256)
# """

# import os
# import torch
# import numpy as np
# import pointgroup_ops
# from torch.utils.data import Dataset


# class FeatureDatasetWrapper(Dataset):
#     """
#     包装原始 ScanNetV2 dataset，在每个样本后追加 superpoint 级 feat_2d。
    
#     原始 dataset.__getitem__ 返回 8-tuple:
#       (scan_id, coord, coord_float, feat, superpoint, inst, normal, sp_instance_label)
    
#     包装后返回 9-tuple:
#       (..., feat_2d)  其中 feat_2d 是 [M, 256] superpoint 级特征
    
#     collate_fn 按 superpoint batch offset 拼接 feat_2d → [B*M_total, 256]
#     """

#     def __init__(self, base_dataset, feat_dir, d_2d=256):
#         self.base_dataset = base_dataset
#         self.feat_dir = feat_dir
#         self.d_2d = d_2d

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         data = self.base_dataset[idx]
#         # data = (scan_id, coord, coord_float, feat, superpoint, inst, normal, sp_instance_label)

#         scan_id = data[0]  # str, e.g. "scene0000_00"

#         feat_2d = self._load_feat(scan_id)

#         return (*data, feat_2d)

#     def _load_feat(self, scan_id):
#         """
#         加载预处理好的 superpoint 级 2D 特征
        
#         返回: Tensor [M, d_2d]  superpoint 级 2D 特征
#         """
#         pt_file = os.path.join(self.feat_dir, f'{scan_id}.pt')

#         if os.path.exists(pt_file):
#             return torch.load(pt_file, map_location='cpu', weights_only=False)

#         # 文件不存在时返回 None，在 collate_fn 中处理
#         return None

#     def collate_fn(self, batch):
#         """
#         覆盖原始 collate_fn，处理第9个元素 feat_2d。
        
#         feat_2d 是 superpoint 级 [M_i, 256]，需要按 superpoint batch offset 拼接。
#         """
#         # 分离 feat_2d（第9个元素）和原始 8-tuple
#         feat_2d_list = []
#         original_batch = []
#         for item in batch:
#             *original_8, feat_2d = item
#             original_batch.append(tuple(original_8))
#             feat_2d_list.append(feat_2d)

#         # 调用原始 collate_fn 处理前 8 个元素
#         result = self.base_dataset.collate_fn(original_batch)

#         # 拼接 feat_2d: [B*M_total, d_2d]
#         # batch_offsets 记录了每个样本的 superpoint 偏移
#         batch_offsets = result['batch_offsets']
#         B = len(batch_offsets) - 1
#         total_sp = batch_offsets[-1].item()

#         feat_2d_cat = torch.zeros(total_sp, self.d_2d)
#         for i in range(B):
#             s, e = batch_offsets[i].item(), batch_offsets[i + 1].item()
#             M_i = e - s
#             if feat_2d_list[i] is not None:
#                 f = feat_2d_list[i]
#                 if f.shape[0] >= M_i:
#                     feat_2d_cat[s:e] = f[:M_i]
#                 else:
#                     feat_2d_cat[s:s + f.shape[0]] = f

#         result['feat_2d'] = feat_2d_cat
#         return result

#     def __getattr__(self, name):
#         """代理访问 base_dataset 的属性（除了我们自己定义的）"""
#         if name in ('base_dataset', 'feat_dir', 'd_2d'):
#             raise AttributeError(name)
#         return getattr(self.base_dataset, name)


# def wrap_dataset_with_2d_feats(base_dataset, feat_dir, d_2d=256):
#     """
#     便捷函数: 包装数据集以添加 superpoint 级 2D 特征
    
#     Args:
#         base_dataset:  原始 ScanNetV2 dataset
#         feat_dir:      预处理后的 superpoint 级特征目录 (feat_2d_sp/)
#         d_2d:          2D 特征维度 (默认 256)
    
#     Returns:
#         包装后的 dataset，collate_fn 输出的 dict 额外包含 'feat_2d' [B*M_total, d_2d]
#     """
#     return FeatureDatasetWrapper(base_dataset, feat_dir, d_2d)



"""
Dataset Wrapper — 加载预处理的超点级 2D 特征
=============================================
文件格式: {scan_id}.pt → Tensor [M, 256] (超点级，已 scatter_mean)
路径: feat_2d_sp/scene0000_00.pt

原始 dataset 返回 tuple (长度8)，本 wrapper 追加 feat_2d 到末尾 (index=8)。
配合 collate_fn_wrapper 将 feat_2d 正确 concat 后加入 batch dict。
"""

import os
import torch
from torch.utils.data import Dataset

DEBUG = False
_debug_count = 0


class FeatureDatasetWrapper(Dataset):
    """
    原始 tuple:
      (scan_id, coords, feats, normals, superpoints, insts, coords_float, sp_inst_labels)
       0        1       2      3        4            5      6              7
    
    包装后:
      (*原始, feat_2d)    feat_2d: [M, d_2d] 超点级
       0-7     8
    """

    def __init__(self, base_dataset, feat_dir, d_2d=256):
        self.base_dataset = base_dataset
        self.feat_dir = feat_dir
        self.d_2d = d_2d

        if hasattr(base_dataset, 'return_feature_indices'):
            base_dataset.return_feature_indices = True

        if hasattr(base_dataset, 'CLASSES'):
            self.CLASSES = base_dataset.CLASSES

        if DEBUG:
            pt_files = [f for f in os.listdir(feat_dir) if f.endswith('.pt')]
            pth_files = [f for f in os.listdir(feat_dir) if f.endswith('.pth')]
            print(f'[DEBUG dataset] feat_dir={feat_dir}')
            print(f'[DEBUG dataset]   .pt files: {len(pt_files)}, .pth files: {len(pth_files)}')

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        global _debug_count
        data = self.base_dataset[idx]

        feature_indices = None
        if (
            getattr(self.base_dataset, 'return_feature_indices', False)
            and isinstance(data, tuple)
            and len(data) == 9
        ):
            data, feature_indices = data[:-1], data[-1]

        scan_id = data[0]  # tuple[0] = scan_id string

        # 超点数从 sp_instance_labels (tuple[7]) 获取
        n_sp = data[7].shape[0] if hasattr(data[7], 'shape') else 0

        feat_2d = self._load_feat(scan_id, n_sp, feature_indices)

        if DEBUG and _debug_count < 3:
            print(f'[DEBUG dataset] idx={idx}, scan_id={scan_id}, '
                  f'n_sp={n_sp}, feat_2d={list(feat_2d.shape)}, '
                  f'min={feat_2d.min():.4f}, max={feat_2d.max():.4f}, '
                  f'nonzero={feat_2d.abs().sum():.2f}')
            _debug_count += 1

        return data + (feat_2d,)

    def _load_feat(self, scan_id, n_sp, feature_indices=None):
        """加载超点级 2D 特征 [M, d_2d]"""
        # 兼容两种命名:
        # 1) scene0000_00.pt / .pth
        # 2) scene0000_00_sp_2dfeats.pth  (SegDINO3D ScanNet200 导出)
        candidate_names = [
            f'{scan_id}.pt',
            f'{scan_id}.pth',
            f'{scan_id}_sp_2dfeats.pth',
        ]
        for name in candidate_names:
            path = os.path.join(self.feat_dir, name)
            if os.path.exists(path):
                feat = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(feat, torch.Tensor):
                    if feat.ndim != 2 or feat.shape[1] != self.d_2d:
                        raise ValueError(
                            f'Invalid 2D feature shape for {scan_id}: '
                            f'{tuple(feat.shape)}, expected [M, {self.d_2d}]'
                        )
                    if feature_indices is not None:
                        feature_indices = feature_indices.long()
                        if feature_indices.numel() != n_sp:
                            raise ValueError(
                                f'Superpoint index count mismatch for {scan_id}: '
                                f'{feature_indices.numel()} != {n_sp}'
                            )
                        if feature_indices.numel() and feature_indices.max().item() >= feat.shape[0]:
                            raise IndexError(
                                f'2D feature index out of range for {scan_id}: '
                                f'{feature_indices.max().item()} >= {feat.shape[0]}'
                            )
                        return feat.index_select(0, feature_indices).float()
                    feat = feat.float()
                    # 确保长度与超点数一致
                    if n_sp > 0:
                        if feat.shape[0] > n_sp:
                            feat = feat[:n_sp]
                        elif feat.shape[0] < n_sp:
                            padded = torch.zeros(n_sp, self.d_2d, dtype=torch.float32)
                            padded[:feat.shape[0]] = feat
                            feat = padded
                    return feat

        if getattr(self.base_dataset, 'strict_feature_loading', False):
            raise FileNotFoundError(
                f'No 2D superpoint feature file found for {scan_id} in {self.feat_dir}'
            )
        if DEBUG and _debug_count < 5:
            print(f'[DEBUG dataset] WARNING: not found for {scan_id}')
        return torch.zeros(max(n_sp, 1), self.d_2d, dtype=torch.float32)

    def __getattr__(self, name):
        if name in ('base_dataset', 'feat_dir', 'd_2d', 'CLASSES'):
            raise AttributeError(name)
        return getattr(self.base_dataset, name)


def collate_fn_wrapper(original_collate_fn):
    """
    分离 tuple 末尾的 feat_2d，用原始 collate 处理前8个元素，
    再把 feat_2d concat 后加入 batch dict。
    """
    def wrapped_collate(batch):
        feat_2d_list = []
        original_batch = []
        for sample in batch:
            if isinstance(sample, tuple) and len(sample) == 9:
                original_batch.append(sample[:-1])
                feat_2d_list.append(sample[-1])
            else:
                original_batch.append(sample)

        batch_dict = original_collate_fn(original_batch)

        if feat_2d_list:
            feat_2d_cat = torch.cat(feat_2d_list, dim=0)
            if isinstance(batch_dict, dict):
                batch_dict['feat_2d'] = feat_2d_cat

        return batch_dict

    return wrapped_collate


def wrap_dataset_with_2d_feats(base_dataset, feat_dir, d_2d=256):
    return FeatureDatasetWrapper(base_dataset, feat_dir, d_2d)
