# Tutor-Student Reinforcement Learning

Official implementation of **Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection**.

This codebase is built on top of [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench/tree/main). TSRL keeps the original DeepfakeBench dataset format, preprocessing pipeline, detector registry, and training/evaluation style, and adds a reinforcement-learning tutor that dynamically assigns sample-level training weights to the student detector.

![TSRL Framework](main_arch.png)

## Overview

Deepfake detectors are usually trained with a fixed data curriculum. TSRL instead introduces a tutor-student training loop. During training, the tutor observes the student's sample-level states, including loss history, forgetting behavior, confidence, correctness, and feature representations, then outputs an action used as a loss weight. The student is optimized with these dynamic weights, while the tutor is updated through reinforcement learning.

The implementation preserves the DeepfakeBench workflow:

1. Prepare datasets with DeepfakeBench-style preprocessing.
2. Generate dataset JSON metadata and optional LMDB files.
3. Choose a detector YAML config from `training/config/detector/`.
4. Train with either standard supervised training or TSRL training.
5. Evaluate checkpoints with the original DeepfakeBench-style test script.

## What TSRL Adds to DeepfakeBench

If you already have an upstream DeepfakeBench checkout, the important TSRL additions are:

- `training/agents/`: tutor policy networks, state manager, and detector wrapper.
- `training/train_rl.py`: TSRL training entrypoint.
- `training/pretrain_tutor.py`: behavioral-cloning pretraining for the tutor.
- `training/trainer/rl_trainer.py` and `training/trainer/rltu_trainer.py`: RL tutor-student trainers.
- `training/dataset/dataset_wrapper.py`: dataset wrapper that returns sample indices for state tracking.
- TSRL detector configs such as `xception_rl.yaml`, `ucf_ppo.yaml`, `iid_ppo.yaml`, `core_ppo.yaml`, `effort_ppo.yaml`, `effort_grpo.yaml`, and related variants.
- Small integration changes in the training/config stack so the trainer can load tutor checkpoints, run warmup epochs, and use tutor-generated loss weights.

In practice, you can either use this repository directly, or start from DeepfakeBench and overlay the TSRL files above onto the DeepfakeBench root.

## Installation

Follow the DeepfakeBench environment setup first. A typical environment is:

```bash
conda create -n tsrl python=3.9 -y
conda activate tsrl

# Install PyTorch according to your CUDA version:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

The preprocessing code requires the dlib landmark model:

```text
preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat
```

Some detector configs also require pretrained detector/backbone weights. Put them under:

```text
training/pretrained/
```

and update the `pretrained` or checkpoint fields in the corresponding YAML config.

## Data Preparation

TSRL uses the same dataset organization as DeepfakeBench. First edit `preprocessing/config.yaml`:

```yaml
preprocess:
  dataset_name:
    default: FaceForensics++
  dataset_root_path:
    default: /path/to/datasets

rearrange:
  dataset_name:
    default: FaceForensics++
  dataset_root_path:
    default: /path/to/datasets
  output_file_path:
    default: ../preprocessing/dataset_json

to_lmdb:
  dataset_name:
    default: FaceForensics++
  dataset_root_path:
    default: /path/to/datasets
  output_lmdb_dir:
    default: ./datasets/lmdb
```

Then run:

```bash
cd preprocessing
python preprocess.py
python rearrange.py
cd ..
```

If your config uses LMDB, convert the extracted frames:

```bash
python preprocessing/dataset2lmdb_test.py --dataset_size 100
```

`--dataset_size` is the LMDB map size in GB. Increase it for large datasets.

By default, the training configs expect:

```text
datasets/rgb/                 # Frame-based datasets.
datasets/lmdb/                # LMDB datasets.
preprocessing/dataset_json/   # Dataset metadata JSON files.
```

You can change these paths in:

```text
training/config/train_config.yaml
training/config/test_config.yaml
training/config/detector/*.yaml
```

## Standard DeepfakeBench-Style Training

Use `training/train.py` for ordinary supervised training:

```bash
python training/train.py \
  --detector_path training/config/detector/xception.yaml \
  --train_dataset FaceForensics++ \
  --test_dataset FaceForensics++ Celeb-DF-v2
```

Distributed training follows the DeepfakeBench pattern:

```bash
torchrun --nproc_per_node=4 training/train.py \
  --ddp \
  --detector_path training/config/detector/xception.yaml \
  --train_dataset FaceForensics++ \
  --test_dataset FaceForensics++ Celeb-DF-v2
```

Useful options:

```bash
--no-save_ckpt       # Disable checkpoint saving.
--no-save_feat       # Disable feature saving.
--weighted_loss      # Enable confidence-based weighted loss in the trainer.
--ddp                # Enable distributed training.
```

## TSRL Training

Use `training/train_rl.py` for tutor-student reinforcement learning. The normal workflow is:

1. Train or load a supervised student detector.
2. Optionally pretrain the tutor from the supervised student.
3. Run TSRL training with warmup epochs.

Run TSRL without tutor pretraining:

```bash
python training/train_rl.py \
  --detector_path training/config/detector/xception_rl.yaml \
  --train_dataset FaceForensics++ \
  --test_dataset FaceForensics++ Celeb-DF-v2 \
  --use_rl_trainer \
  --rl_warmup_epochs 2
```

Run TSRL with a pretrained tutor:

```bash
python training/train_rl.py \
  --detector_path training/config/detector/xception_rl.yaml \
  --train_dataset FaceForensics++ \
  --test_dataset FaceForensics++ Celeb-DF-v2 \
  --use_rl_trainer \
  --rl_warmup_epochs 2 \
  --tutor_pretrain_path training/pretrained/pretrained_tutor_xception.pth
```

Example TSRL configs:

```text
training/config/detector/xception_rl.yaml
training/config/detector/ucf_ppo.yaml
training/config/detector/iid_ppo.yaml
training/config/detector/core_ppo.yaml
training/config/detector/effort_ppo.yaml
training/config/detector/effort_grpo.yaml
```

## Tutor Pretraining

Tutor pretraining uses a supervised student checkpoint as the expert:

```bash
python training/pretrain_tutor.py \
  --detector_path training/config/detector/xception_rl.yaml \
  --student_model_path training/pretrained/student_best.pth \
  --tutor_save_path training/pretrained/pretrained_tutor_xception.pth \
  --train_dataset FaceForensics++
```

Make sure `--student_model_path` matches the detector architecture and dataset-related settings in `--detector_path`.

## Evaluation

Evaluate a trained checkpoint:

```bash
python training/test.py \
  --detector_path training/config/detector/xception.yaml \
  --weights_path path/to/checkpoint.pth \
  --test_dataset FaceForensics++ Celeb-DF-v2
```

For TSRL-trained students, use the detector config that matches the student architecture. The tutor is only needed during training; evaluation loads the final student checkpoint.

## Running on Slurm

`train.sbatch` and `pretrain.sbatch` are examples. Before submitting, update:

- partition/account names,
- conda path and environment name,
- repository path,
- detector config,
- dataset names,
- checkpoint paths.

Submit with:

```bash
sbatch train.sbatch
sbatch pretrain.sbatch
```

## Notes for Reproducing from DeepfakeBench

When porting TSRL into a fresh DeepfakeBench checkout:

1. Install and verify DeepfakeBench first.
2. Prepare datasets with the original DeepfakeBench preprocessing format.
3. Copy TSRL's `training/agents/`, `training/train_rl.py`, `training/pretrain_tutor.py`, `training/trainer/rl_trainer.py`, `training/trainer/rltu_trainer.py`, and `training/dataset/dataset_wrapper.py`.
4. Copy the TSRL detector YAML files into `training/config/detector/`.
5. Make sure the detector registry and trainer imports include the TSRL components.
6. Run `training/train_rl.py` with `--use_rl_trainer`.

Large files are intentionally not tracked by Git, including datasets, LMDBs, logs, checkpoints, and pretrained weights.

## Citation

```bibtex
@InProceedings{Lei_2026_CVPR,
    author    = {Lei, Zhanhe and Wang, Zhongyuan and Cheng, Jikang and Huang, Baojin and Yang, Yuhong and Han, Zhen and Liang, Chao and Ye, Dengpan},
    title     = {Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {41436-41445}
}
```

## Acknowledgement

This repository is based on [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench/tree/main). We thank the DeepfakeBench authors for their open-source benchmark and codebase.
