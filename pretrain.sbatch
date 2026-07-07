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

python training/pretrain_tutor.py  --detector_path training/config/detector/iid.yaml  --student_model_path training/pretrained/iid_best.pth --tutor_save_path training/pretrained/pretrained_tutor_iid.pth --train_dataset FaceForensics++