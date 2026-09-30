# Geometry-Guided Semantic Reconstruction for 3D Instance Segmentation

**NeurIPS 2026**

Yuanhao Su, Shaofeng Zhang

University of Science and Technology of China, Fuzhou University

## Repository structure

```text
configs/scannet/                 Experiment configurations
data/scannetv2/                  ScanNetV2 preprocessing scripts
relation3d/dataset/              Dataset loaders and collation
relation3d/model/                Backbone, decoder, losses, and 2D feature wrapper
relation3d/evaluation/           ScanNetV2 and ScanNet200 evaluation
relation3d/lib/                  CUDA extensions used by the model
tools/train.py                   Training entry point
tools/test.py                    Evaluation entry point
```

## Environment

```bash
conda env create -f environment.yml
conda activate relation3d5090
pip install -r requirement.txt

cd lib/attention_rpe_ops && python setup.py install && cd ../..
cd relation3d/lib && python setup.py develop && cd ../..
python setup.py develop
```

The CUDA extensions require a working CUDA toolkit and `nvcc`.

## Data preparation


```text
dataset/scannet_v2/
├── train/*_inst_nostuff.pth
├── val/*_inst_nostuff.pth
├── test/*_inst_nostuff.pth
├── train/*_normals.pth
├── val/*_normals.pth
├── test/*_normals.pth
├── train/*_superpoints.pth
├── val/*_superpoints.pth
├── test/*_superpoints.pth
└── feat_2d_sp/*.pt
```

Each `*_inst_nostuff.pth` contains the processed point coordinates, colors, superpoints, and labels. The normal files are loaded automatically by the ScanNetV2 loader. Each 2D feature file contains a tensor of shape `[num_superpoints, 256]`, with the filename matching the scene ID, for example `scene0000_00.pt`.

For a new machine, download ScanNetV2 and place the raw scenes under `dataset/scannet_v2`. Copy the train, validation, and test files according to the split lists in `data/scannetv2/`. Generate the instance labels, normals, and superpoints with:

```bash
cd data/scannetv2
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```

The 2D feature extractor is outside this release. Put the projected per-superpoint features in `dataset/scannet_v2/feat_2d_sp`.

## Configuration

Before training, edit `configs/scannet/geometry_guided_scannetv2.yaml`:

```yaml
work_dir: outputs/geometry_guided_scannetv2
train:
  pretrain: checkpoints/sstnet_pretrain.pth
data:
  train:
    data_root: dataset/scannet_v2
  val:
    data_root: dataset/scannet_v2
feat_2d_dir: dataset/scannet_v2/feat_2d_sp
```


## Training

The final configuration is `configs/scannet/geometry_guided_scannetv2.yaml`.

```bash
python tools/train.py configs/scannet/geometry_guided_scannetv2.yaml
```

## Pretrained checkpoint

The ScanNetV2 checkpoint used for the reported result is:

```text
relation3d(ours)_scannetv2.pth
```

Download it from [Baidu Netdisk](https://pan.baidu.com/s/1Kas-JmRnAEVv3HD5D3Bz_w?pwd=wgvk) and use extraction password `wgvk`. After extraction, place the file at:

```text
checkpoints/relation3d(ours)_scannetv2.pth
```

The `checkpoints/` directory is local-only and is not included in this repository.

## Evaluation

```bash
python tools/test.py \
  configs/scannet/geometry_guided_scannetv2.yaml \
  'checkpoints/relation3d(ours)_scannetv2.pth' \
  --out outputs/scannetv2_val
```

The configuration uses the 18 ScanNetV2 instance classes and the same preprocessing and evaluation protocol as the reported result.
