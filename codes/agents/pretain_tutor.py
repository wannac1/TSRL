# file: training/pretrain_tutor.py

import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

from detectors import DETECTOR
from dataset import *
from agents import TutorPPO, StateManager
from logger import create_logger

def main():
    parser = argparse.ArgumentParser(description='Pre-train the RL Tutor model using Behavioral Cloning.')
    parser.add_argument('--detector_path', type=str, required=True, help='Path to the detector YAML config file.')
    parser.add_argument('--student_model_path', type=str, required=True, help='Path to the pre-trained supervised student model weights (.pth). This model will act as the "expert".')
    parser.add_argument('--tutor_save_path', type=str, required=True, help='Path to save the pre-trained tutor model.')
    parser.add_argument("--train_dataset", nargs="+", required=True)
    args = parser.parse_args()

    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    config['train_dataset'] = args.train_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = create_logger()
    logger.info("--- Starting Tutor Pre-training ---")

    logger.info("Loading dataset and expert student model...")
    train_set = DeepfakeAbstractBaseDataset(config=config, mode='train')
    train_set_wrapped = DatasetWrapper(train_set)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set_wrapped,
        batch_size=config.get('train_batchSize', 8),
        shuffle=False, 
        num_workers=int(config.get('workers', 4)),
        collate_fn=train_set_wrapped.collate_fn,
    )

    student_model_class = DETECTOR[config['model_name']]
    student_model = student_model_class(config).to(device)
    student_model.load_state_dict(torch.load(args.student_model_path, map_location=device))
    student_model.eval()

    logger.info("Initializing Tutor and State Manager...")
    dummy_input = {'image': torch.randn(1, 3, config.get('resolution', 224), config.get('resolution', 224)).to(device)}
    if hasattr(student_model, 'extract_features'):
        features = student_model.extract_features(dummy_input)
    else: 
        features = student_model.features(dummy_input)
    if features.ndim == 4:
        features = torch.flatten(features, 1)
    feature_dim = features.shape[1]
    state_dim = feature_dim + 1 + 1 + 1 + 2

    tutor_model = TutorPPO(state_dim, action_dim=1, config=config)
    tutor_model.logger = logger
    state_manager = StateManager(num_samples=len(train_set), config=config, device=device)

    logger.info("Collecting expert data by analyzing the student model's performance...")
    all_states = []
    all_expert_actions = []
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Analyzing Expert"):
            indices = batch.pop('indices')
            for key, val in batch.items():
                if val is not None and key != 'video_name':
                    batch[key] = val.to(device)
            
            labels = batch['label']
            
            outputs = student_model(batch)
            logits = outputs['cls']
            loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
            
            features_detached = student_model.extract_features(batch).detach()
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

    logger.info("Defining expert policy and creating dataset...")
    all_states_tensor = torch.cat(all_states, dim=0)
    
    avg_loss = np.mean(all_losses)
    logger.info(f"Average loss across dataset: {avg_loss:.4f}. Using this as the difficulty threshold.")
    
    expert_actions = [1.5 if loss > avg_loss else 0.75 for loss in all_losses]
    expert_actions_tensor = torch.tensor(expert_actions, dtype=torch.float32).unsqueeze(1)

    expert_dataset = TensorDataset(all_states_tensor, expert_actions_tensor)
    expert_loader = DataLoader(expert_dataset, batch_size=64, shuffle=True)

    tutor_model.pretrain_actor(expert_loader, epochs=10) 

    tutor_model.save(args.tutor_save_path)
    logger.info(f"Pre-trained tutor model saved to {args.tutor_save_path}")
    logger.info("--- Tutor Pre-training Finished Successfully ---")

if __name__ == '__main__':
    main()