# file: training/agents/state_manager.py
# description: Manages the dynamic state of each sample throughout the training process.

import torch
import os

class StateManager:

    def __init__(self, num_samples, config, device):
        self.num_samples = num_samples
        self.alpha = config.get('alpha_ema', 0.1)
        self.device = device
        
        self.ema_loss = torch.full((num_samples,), 0.693, device=device)

        self.forgetting_counts = torch.zeros(num_samples, dtype=torch.int32, device=self.device)
        

        self.last_epoch_correct = torch.ones(num_samples, dtype=torch.bool, device=self.device)

        print(f"StateManager initialized for {num_samples} samples on device {device}.")

    def get_states(self, indices):

        indices = indices.to(self.device)
        return {
            'ema_loss': self.ema_loss[indices],
            'forgetting_counts': self.forgetting_counts[indices].float() 
        }

    def update_states(self, indices, current_losses, current_correctness):

        indices = indices.to(self.device)
        current_losses = current_losses.detach().to(self.device)
        current_correctness = current_correctness.to(self.device)
        
        old_ema = self.ema_loss[indices]
        new_ema = self.alpha * current_losses + (1 - self.alpha) * old_ema
        self.ema_loss[indices] = new_ema
        
        last_correct_batch = self.last_epoch_correct[indices]
        forgotten_mask = (~last_correct_batch) & current_correctness
        
        self.forgetting_counts[indices] += forgotten_mask.int()
        

        self.last_epoch_correct[indices] = current_correctness

    def save_states(self, checkpoint_dir, epoch):

        state_path = os.path.join(checkpoint_dir, f'state_manager_epoch_{epoch}.pth')
        torch.save({
            'ema_loss': self.ema_loss,
            'forgetting_counts': self.forgetting_counts,
            'last_epoch_correct': self.last_epoch_correct
        }, state_path)
        print(f"StateManager states saved to {state_path}")

    def load_states(self, checkpoint_path):

        if not os.path.exists(checkpoint_path):
            print(f"Warning: StateManager checkpoint not found at {checkpoint_path}. Starting with fresh states.")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.ema_loss = checkpoint['ema_loss'].to(self.device)
        self.forgetting_counts = checkpoint['forgetting_counts'].to(self.device)
        self.last_epoch_correct = checkpoint['last_epoch_correct'].to(self.device)
        print(f"StateManager states loaded from {checkpoint_path}")