# file: training/pretrain_tutor.py
# description: CORRECTED (v3) - Hardcodes dataset count for student model 
#              initialization to match checkpoint architecture.

import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import datetime

# 导入项目中的相关模块
from detectors import DETECTOR
from dataset import *
from agents import TutorPPO, StateManager  # Assuming agents are in the python path
from logger import create_logger
from dataset.dataset_wrapper import DatasetWrapper # Import DatasetWrapper

def main():
    parser = argparse.ArgumentParser(description='Pre-train the RL Tutor model using Behavioral Cloning.')
    parser.add_argument('--detector_path', type=str, required=True, help='Path to the detector YAML config file.')
    parser.add_argument('--student_model_path', type=str, required=True, help='Path to the pre-trained supervised student model weights (.pth). This model will act as the "expert".')
    parser.add_argument('--tutor_save_path', type=str, required=True, help='Path to save the pre-trained tutor model.')
    parser.add_argument("--train_dataset", nargs="+", required=True, help="Dataset(s) to use for pre-training the tutor.")
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 2. 初始化日志 ---
    log_dir = config.get('log_dir', './logs/training/')
    pretrain_log_dir = os.path.join(log_dir, 'pretraining')
    os.makedirs(pretrain_log_dir, exist_ok=True)
    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_path = os.path.join(pretrain_log_dir, f"pretrain_tutor_{timenow}.log")
    
    logger = create_logger(log_path)
    logger.info(f"--- Starting Tutor Pre-training ---")
    logger.info(f"Log file will be saved to: {log_path}")

    # --- 3. 准备“专家”学生模型 (使用原始config) ---
    logger.info("Loading expert student model...")

    # --- ▼▼▼ 【关键修复 V3】 ▼▼▼ ---
    # 报错信息 显示 `ucf_best.pth` 的大小为 [5, 512]，
    # 这意味着它是由 len(train_dataset) = 4 (因为 4+1=5) 的配置训练的。
    # 我们必须在加载权重前，用一个相同长度的列表来初始化模型。
    original_model_dataset_count = 4 
    logger.info(f"Setting train_dataset length to {original_model_dataset_count} to match checkpoint architecture.")
    config['train_dataset'] = ['fake_dataset_entry'] * original_model_dataset_count
    # --- ▲▲▲ 修复结束 ▲▲▲ ---

    student_model_class = DETECTOR[config['model_name']]
    student_model = student_model_class(config).to(device)
    
    logger.info(f"Loading student weights from: {args.student_model_path}")
    student_model.load_state_dict(torch.load(args.student_model_path, map_location=device))
    student_model.eval()
    logger.info("Expert student model loaded successfully.")

    # 添加特征提取器的包裹
    if not hasattr(student_model, 'extract_features'):
        if hasattr(student_model, 'features'):
            student_model.extract_features = student_model.features
        else:
            raise AttributeError("The student model needs a '.features' or '.extract_features' method.")

    # --- 4. 准备导师预训练数据集 (现在覆盖config) ---
    logger.info(f"Loading dataset for tutor pre-training from: {args.train_dataset}")
    # --- ▼▼▼ 【关键】现在才覆盖为Tutor的训练集 ▼▼▼ ---
    config['train_dataset'] = args.train_dataset 
    # --- ▲▲▲ ---

    train_set_original = DeepfakeAbstractBaseDataset(config=config, mode='train')
    train_set_wrapped = DatasetWrapper(train_set_original)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set_wrapped,
        batch_size=config.get('train_batchSize', 8),
        shuffle=False, # Must be False to align with StateManager
        num_workers=int(config.get('workers', 4)),
        collate_fn=train_set_wrapped.collate_fn,
    )

    # --- 5. 初始化导师模型和状态管理器 ---
    logger.info("Initializing Tutor and State Manager...")
    dummy_input = {'image': torch.randn(1, 3, config.get('resolution', 224), config.get('resolution', 224)).to(device)}
    with torch.no_grad():
        features_output = student_model.extract_features(dummy_input)
        
    if isinstance(features_output, dict):
        features = features_output.get('forgery')
        if features is None:
            features = next(v for v in features_output.values() if isinstance(v, torch.Tensor))
    else:
        features = features_output

    if features.ndim == 4:
        features = torch.flatten(features, 1)
        
    feature_dim = features.shape[1]
    state_dim = feature_dim + 1 + 1 + 1 + 2
    logger.info(f"Detected feature dimension: {feature_dim}. Total state dim: {state_dim}")

    tutor_config = {
        'ppo_gamma': config.get('ppo_gamma', 0.99),
        'ppo_epochs': config.get('ppo_epochs', 8),
        'ppo_epsilon': config.get('ppo_epsilon', 0.2),
        'tutor_lr_actor': config.get('tutor_lr_actor', 0.0001),
        'tutor_lr_critic': config.get('tutor_lr_critic', 0.0003),
    }
    tutor_model = TutorPPO(state_dim, action_dim=1, config=tutor_config)
    tutor_model.logger = logger
    state_manager = StateManager(num_samples=len(train_set_original), config=config, device=device)

    # --- 6. 收集专家数据 (State-Action Pairs) ---
    logger.info("Collecting expert data by analyzing the student model's performance...")
    all_states = []
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Analyzing Expert"):
            indices = batch.pop('indices')
            for key, val in batch.items():
                if val is not None and key != 'video_name':
                    batch[key] = val.to(device)
            
            labels = batch['label']
            
            outputs = student_model(batch, inference=True)
            logits = outputs['cls']
            loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
            
            features_output_detached = student_model.extract_features(batch)
            if isinstance(features_output_detached, dict):
                features_detached = features_output_detached.get('forgery').detach()
            else:
                features_detached = features_output_detached.detach()
            
            if features_detached.ndim == 4:
                features_detached = torch.flatten(features_detached, 1)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            correctness = (preds == labels)

            historical_states = state_manager.get_states(indices)
            ema_loss_norm = torch.tanh(historical_states['ema_loss']).unsqueeze(1)
            forget_counts_norm = torch.tanh(historical_states['forgetting_counts'] / 10.0).unsqueeze(1)
            confidence = probs[:, 1].unsqueeze(1)
            correct_one_hot = F.one_hot(correctness.long(), num_classes=2).float()

            states_s = torch.cat([
                features_detached, ema_loss_norm, forget_counts_norm, 
                confidence, correct_one_hot
            ], dim=1)

            all_states.append(states_s.cpu())
            all_losses.extend(loss_per_sample.cpu().numpy())
            
            state_manager.update_states(indices, loss_per_sample, correctness)

    # --- 7. 定义专家策略并创建专家数据集 ---
    logger.info("Defining expert policy and creating dataset...")
    all_states_tensor = torch.cat(all_states, dim=0)
    
    avg_loss = np.mean(all_losses)
    logger.info(f"Average loss across dataset: {avg_loss:.4f}. Using this as the difficulty threshold.")
    
    expert_actions = [1.5 if loss > avg_loss else 0.75 for loss in all_losses]
    expert_actions_tensor = torch.tensor(expert_actions, dtype=torch.float32).unsqueeze(1)

    expert_dataset = TensorDataset(all_states_tensor, expert_actions_tensor)
    expert_loader = DataLoader(expert_dataset, batch_size=64, shuffle=True)

    # --- 8. 预训练导师模型的 Actor ---
    tutor_model.pretrain_actor(expert_loader, epochs=10)

    # --- 9. 保存预训练好的导师模型 ---
    tutor_model.save(args.tutor_save_path)
    logger.info(f"Pre-trained tutor model saved to {args.tutor_save_path}")
    logger.info("--- Tutor Pre-training Finished Successfully ---")

if __name__ == '__main__':
    main()