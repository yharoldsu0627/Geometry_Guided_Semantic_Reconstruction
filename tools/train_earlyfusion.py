import argparse
import datetime
import os
import os.path as osp
import shutil
import sys
import time
from _runtime_env import assert_cuda_torch, configure_spconv_runtime, preload_current_env_torch_libs

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
LIB_DIR = osp.join(ROOT_DIR, 'relation3d', 'lib')
for _path in (ROOT_DIR, LIB_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

preload_current_env_torch_libs()

import gorilla
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm

assert_cuda_torch(torch)
configure_spconv_runtime(torch)

from relation3d.dataset import build_dataloader, build_dataset
from relation3d.evaluation import ScanNet200Eval, ScanNetEval
from relation3d.utils import AverageMeter, get_root_logger
from relation3d.model.dataset_2d_feats import (
    wrap_dataset_with_2d_feats,
    collate_fn_wrapper,
)


# ===================== DDP 工具函数 =====================

def setup_distributed():
    """初始化分布式环境, 兼容单卡"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        # 单卡模式
        return 0, 1, 0


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_model(model):
    """获取 DDP 包装下的原始模型"""
    if isinstance(model, DDP):
        return model.module
    return model


def reduce_tensor(tensor):
    """所有进程求平均"""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


# ===================== 主逻辑 =====================

def get_args():
    parser = argparse.ArgumentParser('Relation3D EarlyFusion Training (DDP)')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--seed', type=int, help='override train and test random seed')
    parser.add_argument('--skip_validate', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    return parser.parse_args()


def train(epoch, model, dataloader, optimizer, lr_scheduler, cfg, logger, writer, sampler):
    model.train()
    # ★ DDP: 每个 epoch 设置 sampler 的 epoch, 保证 shuffle 不同
    if sampler is not None:
        sampler.set_epoch(epoch)

    iter_time = AverageMeter()
    data_time = AverageMeter()
    meter_dict = {}
    end = time.time()
    get_model(model).epoch = epoch

    for i, batch in enumerate(dataloader, start=1):
        data_time.update(time.time() - end)

        loss, log_vars = model(batch, mode='loss')

        for k, v in log_vars.items():
            if k not in meter_dict:
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        remain_iter = len(dataloader) * (cfg.train.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = str(datetime.timedelta(seconds=int(remain_iter * iter_time.avg)))
        lr = optimizer.param_groups[0]['lr']

        if i % 10 == 0 and is_main_process():
            log_str = f'Epoch [{epoch}/{cfg.train.epochs}][{i}/{len(dataloader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, '
            log_str += f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
            for k, v in meter_dict.items():
                log_str += f', {k}: {v.val:.4f}'
            logger.info(log_str)

    lr_scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    if is_main_process():
        writer.add_scalar('train/learning_rate', lr, epoch)
        for k, v in meter_dict.items():
            writer.add_scalar(f'train/{k}', v.avg, epoch)

        save_file = osp.join(cfg.work_dir, 'lastest.pth')
        meta = dict(epoch=epoch)
        gorilla.save_checkpoint(get_model(model), save_file, optimizer, lr_scheduler, meta)


@torch.no_grad()
def eval(epoch, model, dataloader, cfg, logger, writer):
    if is_main_process():
        logger.info('Validation')

    pred_insts, gt_insts = [], []
    progress_bar = tqdm(total=len(dataloader)) if is_main_process() else None
    val_dataset = dataloader.dataset

    model.eval()
    for batch in dataloader:
        result = model(batch, mode='predict')
        pred_insts.append(result['pred_instances'])
        gt_insts.append(result['gt_instances'])
        if progress_bar is not None:
            progress_bar.update()
    if progress_bar is not None:
        progress_bar.close()

    # ★ 只在 rank 0 上做评估 (val batch_size=1, 没有 DistributedSampler 时全量跑)
    eval_res = {'all_ap': 0.0, 'all_ap_50%': 0.0, 'all_ap_25%': 0.0}
    if is_main_process():
        logger.info('Evaluate instance segmentation')
        val_data_cfg = cfg.data.get('val', {})
        if val_data_cfg.get('type') == 'scannet200':
            scannet_eval = ScanNet200Eval(val_dataset.CLASSES)
        else:
            scannet_eval = ScanNetEval(val_dataset.CLASSES)
        try:
            eval_res = scannet_eval.evaluate(pred_insts, gt_insts)
            writer.add_scalar('val/AP', eval_res['all_ap'], epoch)
            writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch)
            writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch)
            logger.info('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(
                eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
        except Exception as e:
            logger.info(str(e))

    # ★ 广播 AP 给所有进程 (用于 best model 判断)
    if dist.is_initialized():
        ap_tensor = torch.tensor([eval_res['all_ap']], device='cuda')
        dist.broadcast(ap_tensor, src=0)
        eval_res['all_ap'] = ap_tensor.item()

    return eval_res


def main():
    args = get_args()
    rank, world_size, local_rank = setup_distributed()

    cfg = gorilla.Config.fromfile(args.config)

    if args.seed is not None:
        cfg.train.seed = args.seed
        cfg.test.seed = args.seed

    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None):
        cfg.work_dir = cfg.work_dir
    else:
        cfg.work_dir = osp.join('./exps_earlyfusion_03_01',
                                osp.splitext(osp.basename(args.config))[0])

    if is_main_process():
        os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)

    # ★ 同步一下, 确保目录已创建
    if dist.is_initialized():
        dist.barrier()

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}_rank{rank}.log')
    logger = get_root_logger(log_file=log_file)

    if is_main_process():
        logger.info(f'config: {args.config}')
        logger.info(f'world_size: {world_size}, rank: {rank}, local_rank: {local_rank}')
        shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))

    writer = SummaryWriter(cfg.work_dir) if is_main_process() else None

    gorilla.set_random_seed(cfg.train.seed + rank)  # ★ 每个进程用不同种子
    if is_main_process():
        logger.info(cfg)

    # ---- Model ----
    model_cfg = dict(cfg.model)
    model_name = model_cfg.pop('name', 'Relation3DEarlyFusion')
    if model_name == 'Relation3D':
        from relation3d.model.relation3d import Relation3D as ModelClass
    elif model_name == 'Relation3DEarlyFusion':
        from relation3d.model.relation3d_earlyfusion import Relation3DEarlyFusion as ModelClass
    elif model_name == 'Relation3DEarlyFusionInputMod':
        from relation3d.model.relation3d_earlyfusion_inputmod import (
            Relation3DEarlyFusionInputMod as ModelClass,
        )
    elif model_name == 'Relation3DEarlyFusionInputModV2':
        from relation3d.model.relation3d_earlyfusion_inputmod_v2 import (
            Relation3DEarlyFusionInputModV2 as ModelClass,
        )
    elif model_name == 'Relation3DEarlyFusionInputModBottleneck':
        from relation3d.model.relation3d_earlyfusion_inputmod_bottleneck import (
            Relation3DEarlyFusionInputModBottleneck as ModelClass,
        )
    elif model_name == 'Relation3DEarlyFusionInputModStage':
        from relation3d.model.relation3d_earlyfusion_inputmod_stage import (
            Relation3DEarlyFusionInputModStage as ModelClass,
        )
    elif model_name == 'Relation3DEarlyFusion2DMainPreMod':
        from relation3d.model.relation3d_earlyfusion_2dmain_premod import (
            Relation3DEarlyFusion2DMainPreMod as ModelClass,
        )
    elif model_name == 'Relation3DEarlyFusion2DMainAdd':
        from relation3d.model.relation3d_earlyfusion_2dmain_add import (
            Relation3DEarlyFusion2DMainAdd as ModelClass,
        )
    elif model_name == 'Relation3DEarlyFusion2DMainPostMod':
        from relation3d.model.relation3d_earlyfusion_2dmain_postmod import (
            Relation3DEarlyFusion2DMainPostMod as ModelClass,
        )
    elif model_name == 'Relation3DEarlyFusionInputModShift':
        from relation3d.model.relation3d_earlyfusion_inputmod_shift import (
            Relation3DEarlyFusionInputModShift as ModelClass,
        )
    elif model_name == 'Relation3DEarlyFusion2DMainPreModShift':
        from relation3d.model.relation3d_earlyfusion_2dmain_premod_shift import (
            Relation3DEarlyFusion2DMainPreModShift as ModelClass,
        )
    else:
        raise ValueError(f'Unsupported model.name: {model_name}')

    model = ModelClass(**model_cfg).cuda()
    cfg.model_name = model_name

    if is_main_process():
        logger.info(model)
        count_parameters = gorilla.parameter_count(model)['']
        logger.info(f'Parameters: {count_parameters / 1e6:.2f}M')

    # ---- Optimizer (在 DDP 包装之前构建) ----
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    # ---- Pretrain / Resume ----
    start_epoch = 1
    if args.resume:
        if is_main_process():
            logger.info(f'Resume from {args.resume}')
        if args.eval_only:
            ckpt = torch.load(args.resume, map_location='cpu')
            if 'model' in ckpt:
                state_dict = ckpt['model']
                meta = ckpt.get('meta', {})
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                meta = ckpt.get('meta', {})
            else:
                state_dict = ckpt
                meta = {}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if is_main_process():
                logger.info(f'Loaded model-only checkpoint for eval. '
                            f'Missing: {len(missing)}, Unexpected: {len(unexpected)}')
        else:
            meta = gorilla.resume(model, args.resume, optimizer, lr_scheduler)
            if isinstance(meta, dict) and 'epoch' in meta:
                start_epoch = meta['epoch'] + 1
        if isinstance(meta, dict) and 'stage' in meta and hasattr(model, 'set_stage'):
            model.set_stage(meta['stage'])
            if is_main_process():
                logger.info(f'Set model stage from checkpoint meta: stage={meta["stage"]}')
    elif hasattr(cfg.train, 'pretrain') and cfg.train.pretrain:
        if is_main_process():
            logger.info(f'Load pretrain from {cfg.train.pretrain}')
        model.load_pretrain_partial(cfg.train.pretrain)

    # ★ DDP 包装
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # ---- Dataset: 包装 2D 特征 ----
    feat_2d_dir = cfg.get('feat_2d_dir', '')
    d_2d = cfg.get('d_2d', 256)

    train_raw = build_dataset(cfg.data.train, logger)
    val_raw = build_dataset(cfg.data.val, logger)

    # 获取原始 collate_fn
    _tmp_loader = build_dataloader(train_raw, **cfg.dataloader.train)
    orig_collate = _tmp_loader.collate_fn
    del _tmp_loader

    if osp.isdir(feat_2d_dir):
        train_dataset = wrap_dataset_with_2d_feats(train_raw, feat_2d_dir, d_2d=d_2d)
        val_dataset = wrap_dataset_with_2d_feats(val_raw, feat_2d_dir, d_2d=d_2d)
        collate_fn = collate_fn_wrapper(orig_collate)
        if is_main_process():
            logger.info(f'★ 2D features loaded from: {feat_2d_dir}')
    else:
        if is_main_process():
            logger.warning(f'★ feat_2d_dir not found: {feat_2d_dir}, running WITHOUT 2D features')
        train_dataset = train_raw
        val_dataset = val_raw
        collate_fn = orig_collate

    # ---- DataLoader ----
    train_cfg = dict(cfg.dataloader.train)

    # ★ DDP: 使用 DistributedSampler 替代 shuffle
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get('batch_size', 4),
        shuffle=(train_sampler is None),  # ★ 有 sampler 时不能 shuffle
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
        persistent_workers=train_cfg.get('persistent_workers', False))

    # ★ 验证集: 只在 rank 0 跑全量, 不做分布式切分
    val_cfg = dict(cfg.dataloader.val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_cfg.get('batch_size', 1),
        shuffle=False,
        num_workers=val_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        persistent_workers=val_cfg.get('persistent_workers', False))

    # ---- Train Loop ----
    if is_main_process():
        logger.info('Training')
    best_AP = 0.0
    best_file = None

    if args.eval_only:
        eval(0, model, val_loader, cfg, logger, writer)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train(epoch, model, train_loader, optimizer, lr_scheduler,
              cfg, logger, writer, train_sampler)

        if not args.skip_validate and (epoch % cfg.train.interval == 0):
            eval_res = eval(epoch, model, val_loader, cfg, logger, writer)

            if is_main_process() and eval_res['all_ap'] > best_AP:
                if best_file is not None and osp.exists(best_file):
                    os.remove(best_file)
                best_AP = eval_res['all_ap']
                best_file = osp.join(
                    cfg.work_dir,
                    f'epoch{epoch:03d}_AP_{eval_res["all_ap"]:.4f}'
                    f'_{eval_res["all_ap_50%"]:.4f}'
                    f'_{eval_res["all_ap_25%"]:.4f}.pth')
                meta = dict(epoch=epoch)
                gorilla.save_checkpoint(
                    get_model(model), best_file, optimizer, lr_scheduler, meta)
                shutil.copy(best_file, osp.join(cfg.work_dir, 'best.pth'))
                logger.info(f'★ New best AP: {best_AP:.4f} @ epoch {epoch}')

        if is_main_process() and writer is not None:
            writer.flush()

    if is_main_process():
        logger.info(f'Training done! Best AP: {best_AP:.4f}')

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
