#!/bin/bash

#SBATCH --partition=a100x4
#SBATCH --account=zywang4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00

source /project/leizhanhe/anaconda3/etc/profile.d/conda.sh

cd /project/leizhanhe/deepfake/DeepfakeBench

ls
conda init
conda activate FairnessDeepfake

conda env list

python training/train_rl.py  --detector_path ./training/config/detector/ucf.yaml --train_dataset "FaceForensics++" --test_dataset DFDC FaceForensics++ UADFV Celeb-DF-v2 --use_rl_trainer --rl_warmup_epochs 2 
