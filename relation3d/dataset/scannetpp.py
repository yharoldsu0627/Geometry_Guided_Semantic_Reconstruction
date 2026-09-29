import os.path as osp
from typing import Tuple

import numpy as np
import torch
from torch_scatter import scatter_mean

from .scannetv2 import ScanNetDataset


class ScanNetPPDataset(ScanNetDataset):
    """ScanNet++ instance benchmark dataset with global superpoint feature IDs."""

    def __init__(self, class_names_file, *args, **kwargs):
        with open(class_names_file) as handle:
            class_names = tuple(line.strip() for line in handle if line.strip())
        if len(class_names) != 84:
            raise ValueError(
                f"Expected 84 ScanNet++ instance classes in {class_names_file}, "
                f"found {len(class_names)}"
            )
        self.CLASSES = class_names
        self.NYU_ID = tuple(range(1, len(class_names) + 1))
        self.return_feature_indices = False
        self.strict_feature_loading = True
        super().__init__(*args, **kwargs)

    @staticmethod
    def _compact_instance_labels(labels: np.ndarray) -> np.ndarray:
        compact = np.full(labels.shape, -100, dtype=np.int64)
        valid = labels >= 0
        valid_ids = np.unique(labels[valid])
        if valid_ids.size:
            compact[valid] = np.searchsorted(valid_ids, labels[valid])
        return compact

    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label, normal=None):
        xyz_middle, normal = self.data_aug(xyz, True, True, True, normal)
        rgb = rgb + np.random.randn(3) * 0.1
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.0)
            xyz = self.elastic(xyz, 20, 160.0)
        xyz = xyz - xyz.min(0)
        xyz, valid_idxs = self.crop(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        instance_label = self._compact_instance_labels(instance_label[valid_idxs])
        global_superpoints = superpoint[valid_idxs]
        feature_indices, superpoint = np.unique(global_superpoints, return_inverse=True)
        if normal is not None:
            normal = normal[valid_idxs]
        return (
            xyz,
            xyz_middle,
            rgb,
            superpoint,
            semantic_label,
            instance_label,
            normal,
            feature_indices,
        )

    def transform_test(self, xyz, rgb, superpoint, semantic_label=None, instance_label=None, normal=None):
        xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz = xyz - xyz.min(0)
        feature_indices, superpoint = np.unique(superpoint, return_inverse=True)
        if instance_label is not None:
            instance_label = self._compact_instance_labels(instance_label)
        return (
            xyz,
            xyz_middle,
            rgb,
            superpoint,
            semantic_label,
            instance_label,
            normal,
            feature_indices,
        )

    def __getitem__(self, index: int) -> Tuple:
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, "")
        transformed = self.transform_train(*self.load(filename)) if self.training else self.transform_test(*self.load(filename))
        (
            xyz,
            xyz_middle,
            rgb,
            superpoint,
            semantic_label,
            instance_label,
            normal,
            feature_indices,
        ) = transformed

        coord = torch.from_numpy(np.ascontiguousarray(xyz)).long()
        coord_float = torch.from_numpy(np.ascontiguousarray(xyz_middle)).float()
        feat = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        superpoint = torch.from_numpy(np.ascontiguousarray(superpoint)).long()
        normal = (
            torch.from_numpy(np.ascontiguousarray(normal)).float()
            if normal is not None
            else None
        )
        semantic_label = (
            torch.from_numpy(np.ascontiguousarray(semantic_label)).long()
            if semantic_label is not None
            else torch.full((xyz.shape[0],), -100, dtype=torch.long)
        )
        instance_label = (
            torch.from_numpy(np.ascontiguousarray(instance_label)).long()
            if instance_label is not None
            else torch.full((xyz.shape[0],), -100, dtype=torch.long)
        )
        feature_indices = torch.from_numpy(np.ascontiguousarray(feature_indices)).long()

        inst = self.get_instance3D(
            instance_label, semantic_label, superpoint, coord_float, scan_id
        )
        sp_instance_label = scatter_mean(instance_label, superpoint)
        sample = (
            scan_id,
            coord,
            coord_float,
            feat,
            superpoint,
            inst,
            normal,
            sp_instance_label,
        )
        if self.return_feature_indices:
            return sample + (feature_indices,)
        return sample
