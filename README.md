# Geometry-Guided Semantic Reconstruction for 3D Instance Segmentation

This repository contains the reproducible implementation of our 3D instance segmentation method. The model augments a Relation3D-style 3D decoder with lifted 2D features and geometry-guided semantic reconstruction.

The release contains source code, ScanNetV2 preprocessing utilities, training and evaluation entry points, and the final experiment configuration. Datasets, DINO features, checkpoints, logs, prediction masks, and other experiment artifacts are intentionally excluded.

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

Download ScanNetV2 and place the raw scenes under `dataset/scannet_v2`. Copy the train, validation, and test files according to the split lists in `data/scannetv2/`. Generate the instance labels, normals, and superpoints with:

```bash
cd data/scannetv2
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```

The 2D feature extractor is outside this release. Put the projected per-superpoint features in `dataset/scannet_v2/feat_2d_sp` and update `feat_2d_dir` in the configuration if needed.

## Training

The final configuration is `configs/scannet/geometry_guided_scannetv2.yaml`.

```bash
python tools/train_earlyfusion.py configs/scannet/geometry_guided_scannetv2.yaml
```

## Evaluation

```bash
python tools/test_earlyfusion.py \
  configs/scannet/geometry_guided_scannetv2.yaml \
  /path/to/checkpoint.pth \
  --out outputs/scannetv2_val
```

The configuration uses the 18 ScanNetV2 instance classes and the same preprocessing and evaluation protocol as the reported result.

## Citation

```bibtex
@inproceedings{geometryguided2026,
  title     = {Geometry-Guided Semantic Reconstruction for 3D Instance Segmentation},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
