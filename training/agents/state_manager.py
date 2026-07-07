# file: training/agents/state_manager.py
# description: Manages the dynamic state of each sample throughout the training process.

import torch
import os

class StateManager:
    """
    状态管理器 (State Manager):
    负责在多个训练周期(epoch)中，持久化地跟踪和管理数据集中每个样本的动态难度信息。
    这是RL导师智能体做出决策所需历史信息的来源。
    """
    def __init__(self, num_samples, config, device):
        """
        初始化状态管理器。
        
        Args:
            num_samples (int): 训练集中的样本总数。
            config (dict): 包含配置参数的字典，如alpha_ema。
            device (torch.device): 计算设备 (e.g., 'cuda:0')。
        """
        self.num_samples = num_samples
        self.alpha = config.get('alpha_ema', 0.1)
        self.device = device
        
        # 使用张量存储所有样本的状态，以实现高效的批量读写
        # 1. 指数移动平均损失 (EMA Loss)
        self.ema_loss = torch.full((num_samples,), 0.693, device=device)
        # 2. 遗忘次数 (Forgetting Counts)
        # 根据您的定义：如果上一轮错误，这一轮正确，则计数增加
        self.forgetting_counts = torch.zeros(num_samples, dtype=torch.int32, device=self.device)
        
        # 3. 上一轮的预测正确性 (用于计算遗忘事件)
        # 初始化为True，这样第一轮预测错误的样本不会被错误地计为“遗忘”
        self.last_epoch_correct = torch.ones(num_samples, dtype=torch.bool, device=self.device)

        print(f"StateManager initialized for {num_samples} samples on device {device}.")

    def get_states(self, indices):
        """
        根据一批样本的索引，获取它们的历史状态。
        
        Args:
            indices (torch.Tensor): 当前批次样本的索引张量。
            
        Returns:
            dict: 包含这批样本历史状态的字典。
        """
        indices = indices.to(self.device)
        return {
            'ema_loss': self.ema_loss[indices],
            'forgetting_counts': self.forgetting_counts[indices].float() # 转换为浮点数以便输入网络
        }

    def update_states(self, indices, current_losses, current_correctness):
        """
        根据当前批次的训练结果，更新对应样本的状态。
        
        Args:
            indices (torch.Tensor): 当前批次样本的索引张量。
            current_losses (torch.Tensor): 当前批次样本各自的损失值。
            current_correctness (torch.Tensor): 当前批次样本是否被正确分类的布尔张量。
        """
        indices = indices.to(self.device)
        current_losses = current_losses.detach().to(self.device)
        current_correctness = current_correctness.to(self.device)
        
        # --- 更新EMA损失 ---
        old_ema = self.ema_loss[indices]
        new_ema = self.alpha * current_losses + (1 - self.alpha) * old_ema
        self.ema_loss[indices] = new_ema
        
        # --- 更新遗忘次数 ---
        # 找出那些“上一轮预测错误”且“这一轮预测正确”的样本
        last_correct_batch = self.last_epoch_correct[indices]
        forgotten_mask = (~last_correct_batch) & current_correctness
        
        # 对发生“遗忘”的样本，计数器加一
        self.forgetting_counts[indices] += forgotten_mask.int()
        
        # --- 为下一轮更新做准备 ---
        # 记录当前批次的预测结果
        self.last_epoch_correct[indices] = current_correctness

    def save_states(self, checkpoint_dir, epoch):
        """
        将所有样本的状态保存到文件，以便于恢复训练。
        """
        state_path = os.path.join(checkpoint_dir, f'state_manager_epoch_{epoch}.pth')
        torch.save({
            'ema_loss': self.ema_loss,
            'forgetting_counts': self.forgetting_counts,
            'last_epoch_correct': self.last_epoch_correct
        }, state_path)
        print(f"StateManager states saved to {state_path}")

    def load_states(self, checkpoint_path):
        """
        从文件加载状态。
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: StateManager checkpoint not found at {checkpoint_path}. Starting with fresh states.")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.ema_loss = checkpoint['ema_loss'].to(self.device)
        self.forgetting_counts = checkpoint['forgetting_counts'].to(self.device)
        self.last_epoch_correct = checkpoint['last_epoch_correct'].to(self.device)
        print(f"StateManager states loaded from {checkpoint_path}")