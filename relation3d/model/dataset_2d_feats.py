"""
=============================================

"""

import os
import torch
from torch.utils.data import Dataset

DEBUG = False
_debug_count = 0


class FeatureDatasetWrapper(Dataset):
    """
      (scan_id, coords, feats, normals, superpoints, insts, coords_float, sp_inst_labels)
       0        1       2      3        4            5      6              7
    
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
        """Documentation."""
        # 1) scene0000_00.pt / .pth
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
