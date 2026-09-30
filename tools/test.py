import argparse
import hashlib
import math
import os
import os.path as osp
import sys

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
LIB_DIR = osp.join(ROOT_DIR, 'relation3d', 'lib')
for _path in (ROOT_DIR, LIB_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _runtime_env import assert_cuda_torch, configure_spconv_runtime, preload_current_env_torch_libs

preload_current_env_torch_libs()

import gorilla
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

assert_cuda_torch(torch)
configure_spconv_runtime(torch)

from relation3d.dataset import build_dataloader, build_dataset
from relation3d.evaluation import ScanNet200Eval, ScanNetEval
from relation3d.model.dataset_2d_feats import wrap_dataset_with_2d_feats, collate_fn_wrapper
from relation3d.utils import get_root_logger, save_gt_instances, save_pred_instances
from _profiling import InferenceProfiler, add_profile_args, parameter_count_million, save_profile_json


def get_args():
    parser = argparse.ArgumentParser('Relation3D Test')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', type=str, help='directory for output results')
    parser.add_argument(
        '--drop-2d-ratio',
        type=float,
        default=0.0,
        help='Test-time superpoint-level 2D feature drop ratio. Rows in feat_2d are zeroed per scene.',
    )
    parser.add_argument(
        '--drop-2d-seed',
        type=int,
        default=0,
        help='Base seed for deterministic test-time 2D feature dropout.',
    )
    parser.add_argument(
        '--noise-2d-sigma',
        type=float,
        default=0.0,
        help=(
            'Test-time relative Gaussian noise level for superpoint 2D features. '
            'Per-channel noise std is sigma * row_L2_norm / sqrt(feature_dim).'
        ),
    )
    parser.add_argument(
        '--noise-2d-seed',
        type=int,
        default=0,
        help='Base seed for deterministic scene-wise test-time 2D feature noise.',
    )
    parser.add_argument(
        '--assoc-2d-ratio',
        type=float,
        default=0.0,
        help=(
            'Test-time local 2D-to-3D association-jitter ratio. Selected superpoints receive '
            'the 2D feature of their nearest 3D superpoint centroid within the same scene.'
        ),
    )
    parser.add_argument(
        '--assoc-2d-seed',
        type=int,
        default=0,
        help='Base seed for deterministic test-time local 2D-to-3D association jitter.',
    )
    add_profile_args(parser)
    return parser.parse_args()


def build_model(cfg):
    model_cfg = dict(cfg.model)
    model_name = model_cfg.pop('name', 'Relation3D')
    if model_name == 'Relation3D':
        from relation3d.model.relation3d import Relation3D as ModelClass
    else:
        raise ValueError(f'Unsupported model.name: {model_name}')
    return ModelClass(**model_cfg).cuda()


def _stable_scene_seed(scan_id, base_seed):
    payload = f'{base_seed}:{scan_id}'.encode('utf-8')
    digest = hashlib.sha1(payload).digest()
    return int.from_bytes(digest[:8], 'little', signed=False)


def apply_test_time_2d_dropout(batch, ratio, base_seed):
    if ratio <= 0:
        return batch, 0, 0

    feat_2d = batch.get('feat_2d', None)
    batch_offsets = batch.get('batch_offsets', None)
    scan_ids = batch.get('scan_ids', None)
    if feat_2d is None or batch_offsets is None:
        return batch, 0, 0

    num_scenes = len(batch_offsets) - 1
    if scan_ids is None:
        scan_ids = [f'scene_{i}' for i in range(num_scenes)]
    if len(scan_ids) != num_scenes:
        raise ValueError(f'scan_ids length mismatch: {len(scan_ids)} vs {num_scenes}')

    feat_2d_masked = feat_2d.clone()
    total_masked = 0
    total_superpoints = 0

    for scene_idx in range(num_scenes):
        start = int(batch_offsets[scene_idx].item())
        end = int(batch_offsets[scene_idx + 1].item())
        n_sp = end - start
        if n_sp <= 0:
            continue

        total_superpoints += n_sp
        n_mask = max(1, int(n_sp * ratio))
        n_mask = min(n_mask, n_sp)

        generator = torch.Generator(device=feat_2d_masked.device)
        generator.manual_seed(_stable_scene_seed(str(scan_ids[scene_idx]), base_seed))
        perm = torch.randperm(n_sp, generator=generator, device=feat_2d_masked.device)[:n_mask]
        feat_2d_masked[perm + start] = 0.0
        total_masked += n_mask

    batch = dict(batch)
    batch['feat_2d'] = feat_2d_masked
    return batch, total_masked, total_superpoints


def apply_test_time_2d_gaussian_noise(batch, sigma, base_seed):
    if sigma <= 0:
        return batch, 0, 0.0, 0.0

    feat_2d = batch.get('feat_2d', None)
    batch_offsets = batch.get('batch_offsets', None)
    scan_ids = batch.get('scan_ids', None)
    if feat_2d is None or batch_offsets is None:
        return batch, 0, 0.0, 0.0

    num_scenes = len(batch_offsets) - 1
    if scan_ids is None:
        scan_ids = [f'scene_{i}' for i in range(num_scenes)]
    if len(scan_ids) != num_scenes:
        raise ValueError(f'scan_ids length mismatch: {len(scan_ids)} vs {num_scenes}')

    feat_2d_noisy = feat_2d.clone()
    feature_dim = feat_2d_noisy.shape[-1]
    total_perturbed = 0
    total_signal_sq = 0.0
    total_noise_sq = 0.0

    for scene_idx in range(num_scenes):
        start = int(batch_offsets[scene_idx].item())
        end = int(batch_offsets[scene_idx + 1].item())
        if end <= start:
            continue

        scene_feat = feat_2d_noisy[start:end]
        row_norm = scene_feat.norm(dim=-1, keepdim=True)
        scale = row_norm / math.sqrt(feature_dim)

        generator = torch.Generator(device=scene_feat.device)
        generator.manual_seed(_stable_scene_seed(str(scan_ids[scene_idx]), base_seed))
        noise = torch.randn(
            scene_feat.shape,
            generator=generator,
            device=scene_feat.device,
            dtype=scene_feat.dtype,
        )
        noise.mul_(scale * sigma)
        feat_2d_noisy[start:end] = scene_feat + noise

        total_perturbed += int((row_norm.squeeze(-1) > 0).sum().item())
        total_signal_sq += float(scene_feat.square().sum().item())
        total_noise_sq += float(noise.square().sum().item())

    batch = dict(batch)
    batch['feat_2d'] = feat_2d_noisy
    return batch, total_perturbed, total_signal_sq, total_noise_sq


def apply_test_time_2d_local_association_jitter(batch, ratio, base_seed):
    """Replace selected 2D priors with those of the nearest 3D superpoint."""
    if ratio <= 0:
        return batch, 0, 0

    feat_2d = batch.get('feat_2d', None)
    batch_offsets = batch.get('batch_offsets', None)
    scan_ids = batch.get('scan_ids', None)
    superpoints = batch.get('superpoints', None)
    coords_float = batch.get('coords_float', None)
    if any(value is None for value in (feat_2d, batch_offsets, superpoints, coords_float)):
        return batch, 0, 0

    num_scenes = len(batch_offsets) - 1
    if scan_ids is None:
        scan_ids = [f'scene_{i}' for i in range(num_scenes)]
    if len(scan_ids) != num_scenes:
        raise ValueError(f'scan_ids length mismatch: {len(scan_ids)} vs {num_scenes}')

    feat_2d_jittered = feat_2d.clone()
    total_reassigned = 0
    total_eligible = 0

    for scene_idx in range(num_scenes):
        start = int(batch_offsets[scene_idx].item())
        end = int(batch_offsets[scene_idx + 1].item())
        n_sp = end - start
        if n_sp <= 1:
            continue

        point_mask = (superpoints >= start) & (superpoints < end)
        if not point_mask.any():
            continue
        local_sp = superpoints[point_mask].long() - start
        local_coords = coords_float[point_mask].to(dtype=feat_2d.dtype)

        centroid_sum = torch.zeros(n_sp, 3, dtype=local_coords.dtype, device=local_coords.device)
        centroid_sum.index_add_(0, local_sp, local_coords)
        point_count = torch.bincount(local_sp, minlength=n_sp)
        valid = point_count > 0
        centroids = centroid_sum[valid] / point_count[valid].unsqueeze(-1).to(centroid_sum.dtype)
        if centroids.shape[0] <= 1:
            continue

        # Each valid superpoint can only borrow a feature from its nearest local neighbor.
        distances = torch.cdist(centroids, centroids)
        distances.fill_diagonal_(float('inf'))
        nearest_valid = distances.argmin(dim=1)
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        nearest = torch.full((n_sp,), -1, dtype=torch.long, device=feat_2d.device)
        nearest[valid_indices] = valid_indices[nearest_valid]

        scene_feat = feat_2d_jittered[start:end]
        eligible = valid & (scene_feat.norm(dim=-1) > 0) & (nearest >= 0)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(1)
        n_select = min(max(1, int(eligible_indices.numel() * ratio)), eligible_indices.numel())
        if n_select == 0:
            continue

        generator = torch.Generator(device=feat_2d.device)
        generator.manual_seed(_stable_scene_seed(str(scan_ids[scene_idx]), base_seed))
        selected = eligible_indices[
            torch.randperm(eligible_indices.numel(), generator=generator, device=feat_2d.device)[:n_select]
        ]
        scene_feat[selected] = scene_feat[nearest[selected]]
        total_reassigned += n_select
        total_eligible += int(eligible_indices.numel())

    batch = dict(batch)
    batch['feat_2d'] = feat_2d_jittered
    return batch, total_reassigned, total_eligible


def main():
    args = get_args()
    if not (0.0 <= args.drop_2d_ratio <= 1.0):
        raise ValueError(f'--drop-2d-ratio must be in [0, 1], got {args.drop_2d_ratio}')
    if args.noise_2d_sigma < 0:
        raise ValueError(f'--noise-2d-sigma must be non-negative, got {args.noise_2d_sigma}')
    if not (0.0 <= args.assoc_2d_ratio <= 1.0):
        raise ValueError(f'--assoc-2d-ratio must be in [0, 1], got {args.assoc_2d_ratio}')
    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger()
    test_data_cfg = cfg.data.get('test', cfg.data.val)
    test_loader_cfg = cfg.dataloader.get('test', cfg.dataloader.val)

    model = build_model(cfg)
    params_m = parameter_count_million(model)
    logger.info(f'Parameters: {params_m:.2f}M')
    logger.info(f'Load state dict from {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f'Checkpoint loaded. Missing={len(missing)}, Unexpected={len(unexpected)}')

    feat_2d_dir = cfg.get('feat_2d_dir', '')
    d_2d = cfg.get('d_2d', 256)
    dataset = build_dataset(test_data_cfg, logger)
    tmp_loader = build_dataloader(dataset, training=False, **test_loader_cfg)
    orig_collate = tmp_loader.collate_fn
    del tmp_loader

    if osp.isdir(feat_2d_dir):
        dataset = wrap_dataset_with_2d_feats(dataset, feat_2d_dir, d_2d=d_2d)
        collate_fn = collate_fn_wrapper(orig_collate)
        logger.info(f'2D features loaded from: {feat_2d_dir}')
    else:
        collate_fn = orig_collate
        logger.warning(f'feat_2d_dir not found: {feat_2d_dir}, running WITHOUT 2D features')

    dataloader = DataLoader(
        dataset,
        batch_size=test_loader_cfg.get('batch_size', 1),
        shuffle=False,
        num_workers=test_loader_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=test_loader_cfg.get('persistent_workers', False),
    )

    results, scan_ids, pred_insts, gt_insts = [], [], [], []
    progress_bar = tqdm(total=len(dataloader))
    profiler = InferenceProfiler(
        enabled=args.profile,
        warmup_scenes=args.profile_warmup,
        max_profile_scenes=args.profile_max_scenes,
        logger=logger,
    )
    total_masked = 0
    total_superpoints = 0
    total_perturbed = 0
    total_signal_sq = 0.0
    total_noise_sq = 0.0
    total_reassigned = 0
    total_association_eligible = 0
    if args.drop_2d_ratio > 0:
        logger.info(
            f'Apply test-time 2D feature dropout: ratio={args.drop_2d_ratio:.3f}, seed={args.drop_2d_seed}'
        )
    if args.noise_2d_sigma > 0:
        logger.info(
            'Apply test-time relative Gaussian 2D feature noise: '
            f'sigma={args.noise_2d_sigma:.3f}, seed={args.noise_2d_seed}'
        )
    if args.assoc_2d_ratio > 0:
        logger.info(
            'Apply test-time local 2D-to-3D association jitter: '
            f'ratio={args.assoc_2d_ratio:.3f}, seed={args.assoc_2d_seed}'
        )
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            batch, masked_now, total_now = apply_test_time_2d_dropout(
                batch, args.drop_2d_ratio, args.drop_2d_seed
            )
            total_masked += masked_now
            total_superpoints += total_now
            batch, perturbed_now, signal_sq_now, noise_sq_now = apply_test_time_2d_gaussian_noise(
                batch, args.noise_2d_sigma, args.noise_2d_seed
            )
            total_perturbed += perturbed_now
            total_signal_sq += signal_sq_now
            total_noise_sq += noise_sq_now
            batch, reassigned_now, eligible_now = apply_test_time_2d_local_association_jitter(
                batch, args.assoc_2d_ratio, args.assoc_2d_seed
            )
            total_reassigned += reassigned_now
            total_association_eligible += eligible_now
            start_time = profiler.before_forward()
            result = model(batch, mode='predict')
            profiler.after_forward(start_time, batch_size=1)
            results.append(result)
            progress_bar.update()
            if profiler.should_stop():
                logger.info('Profiling reached the requested number of scenes; stopping early.')
                break
    progress_bar.close()
    if args.drop_2d_ratio > 0 and total_superpoints > 0:
        logger.info(
            'Applied test-time 2D dropout to '
            f'{total_masked}/{total_superpoints} superpoints '
            f'({total_masked / total_superpoints:.4f})'
        )
    if args.noise_2d_sigma > 0 and total_signal_sq > 0:
        empirical_relative_l2 = math.sqrt(total_noise_sq / total_signal_sq)
        logger.info(
            'Applied test-time relative Gaussian 2D noise to '
            f'{total_perturbed} nonzero superpoints; '
            f'empirical relative L2={empirical_relative_l2:.4f}'
        )
    if args.assoc_2d_ratio > 0 and total_association_eligible > 0:
        logger.info(
            'Applied test-time local 2D-to-3D association jitter to '
            f'{total_reassigned}/{total_association_eligible} eligible superpoints '
            f'({total_reassigned / total_association_eligible:.4f})'
        )

    for res in results:
        scan_ids.append(res['scan_id'])
        pred_insts.append(res['pred_instances'])
        gt_insts.append(res['gt_instances'])

    if args.profile:
        profile_summary = profiler.log_summary(params_m=params_m, model_name=cfg.model.name)
        save_profile_json(args.profile_json, profile_summary)

    full_pass = len(results) == len(dataloader)
    if not full_pass and not args.skip_eval:
        logger.info('Skip evaluation because profiling stopped before the full dataset was processed.')
    elif args.skip_eval:
        logger.info('Skip evaluation by user request.')
    elif test_data_cfg.prefix != 'test':
        logger.info('Evaluate instance segmentation')
        if test_data_cfg.type == 'scannet200':
            evaluator = ScanNet200Eval(dataset.CLASSES)
        else:
            evaluator = ScanNetEval(dataset.CLASSES)
        evaluator.evaluate(pred_insts, gt_insts)

    if args.out:
        logger.info('Save results')
        nyu_id = dataset.NYU_ID
        save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
        if test_data_cfg.prefix != 'test':
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)


if __name__ == '__main__':
    main()
