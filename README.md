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

The repository does not include datasets, projected features, checkpoints, or experiment outputs. On the experiment machine, the prepared ScanNetV2 data is organized as:

```text
dataset/scannet_v2/
├── train/*_inst_nostuff.pth
├── val/*_inst_nostuff.pth
├── test/*_inst_nostuff.pth
├── train/*_normals.pth
├── val/*_normals.pth
├── test/*_normals.pth
└── feat_2d_sp/*.pt or *.pth
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

Change `data_root` to the directory containing the `train`, `val`, and `test` folders. Change `feat_2d_dir` to the directory containing the projected superpoint features. Set `train.pretrain` to a compatible checkpoint or set it to an empty string when no pretraining checkpoint is available. `work_dir` controls logs, TensorBoard files, and saved checkpoints.

The final method uses `feature_aux_mode: background` and `background_mask_ratio: 0.1`. Keep `d_2d: 256` unless the projected feature dimension and the model configuration are changed together. If the 2D feature directory is missing, the code runs without 2D features and does not reproduce the reported method.

## Training

The final configuration is `configs/scannet/geometry_guided_scannetv2.yaml`.

```bash
python tools/train.py configs/scannet/geometry_guided_scannetv2.yaml
```

## Evaluation

```bash
python tools/test.py \
  configs/scannet/geometry_guided_scannetv2.yaml \
  /path/to/checkpoint.pth \
  --out outputs/scannetv2_val
```

The configuration uses the 18 ScanNetV2 instance classes and the same preprocessing and evaluation protocol as the reported result.
