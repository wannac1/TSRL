# file: training/agents/tutor_ppo.py
# description: PPO agent for the Tutor model.
# MODIFIED: Changed torch.stack to torch.cat and removed unnecessary Tensor conversion.

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import numpy as np
import pandas as pd
from copy import deepcopy

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()  # Use Sigmoid to ensure output is in [0, 1]
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        # Using a fixed covariance for exploration stability
        cov_matrix = torch.diag(torch.full((action_mean.size(-1),), 0.1)).to(action_mean.device)
        # Repeat for batch size
        cov_matrix = cov_matrix.unsqueeze(0).repeat(action_mean.size(0), 1, 1)
        dist = MultivariateNormal(action_mean, covariance_matrix=cov_matrix)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = torch.full((action_mean.shape[-1],), 0.01).to(action_mean.device)
        cov_mat = torch.diag(action_var).unsqueeze(0).repeat(action_mean.shape[0], 1, 1)
        
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class TutorPPO:
    def __init__(self, state_dim, action_dim, config):
        self.gamma = config.get('ppo_gamma', 0.99)
        self.epochs = config.get('ppo_epochs', 8)
        self.eps_clip = config.get('ppo_epsilon', 0.2)
        
        # --- ▼▼▼ 【【【核心修复 1：添加 Mini-batch 大小】】】 ▼▼▼
        # (设置一个合理的 mini-batch 大小，例如 512 或 1024，以防止 OOM)
        # (您可以稍后在 .yaml 配置文件中覆盖 'ppo_mini_batch_size')
        self.mini_batch_size = config.get('ppo_mini_batch_size', 512)
        # --- ▲▲▲ 【【【修改结束】】】 ▲▲▲

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim, 0.1).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': config.get('tutor_lr_actor', 0.0001)},
            {'params': self.policy.critic.parameters(), 'lr': config.get('tutor_lr_critic', 0.0003)}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, 0.1).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def to(self, device):
        """
        【【【核心修复】】】
        添加一个 .to() 方法，以响应来自 rl_trainer.py 的设备移动请求。
        
        此方法将 Agent 内部的所有 nn.Module 组件（策略网络）
        移动到指定的设备上。
        """
        
        # 1. 更新 Agent 内部的设备记录
        # (这会覆盖 __init__ 中的硬编码)
        self.device = device
        
        # 2. 将真正的 PyTorch 网络移动到指定设备
        self.policy.to(self.device)
        self.policy_old.to(self.device)
        
        self.MseLoss.to(self.device) # 同样移动损失函数

        print(f"TutorPPO: Internal networks (policy, policy_old) successfully moved to {self.device}")
        
        # 3. 返回 self 以支持链式调用
        return self

# (确保这个方法被正确缩进，与 __init__ 和 select_action 处于同一级别)
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = state.to(self.device)
            action, action_logprob, _ = self.policy_old.act(state_tensor)
        
        # --- ▼▼▼ 【【【核心修复：将张量移至 CPU 缓解显存压力】】】 ▼▼▼
        # (不要在 buffer 列表中保留 GPU 张量)
        self.buffer.states.append(state_tensor.cpu())
        self.buffer.actions.append(action.cpu())
        self.buffer.logprobs.append(action_logprob.cpu())
        # --- ▲▲▲ 【【【修改结束】】】 ▲▲▲

        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy().flatten()
    
    # (在 tutor_ppo.py 中)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # --- ▼▼▼ 【【【核心修复 2：在 CPU 上处理数据】】】 ▼▼▼
        
        # 1. 在 CPU 上准备所有数据 (不使用 .to(self.device))
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_actions = torch.cat(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.cat(self.buffer.logprobs, dim=0).detach()
        
        buffer_size = old_states.shape[0]

        # 2. PPO 训练循环 (保持不变)
        for _ in range(self.epochs):
            
            # 3. 创建一个随机索引列表，用于打乱数据
            indices = np.arange(buffer_size)
            np.random.shuffle(indices)

            # 4. 【【【新增：Mini-batch 循环】】】
            #    遍历整个缓冲区 (56.10 GiB)，但一次只处理 mini_batch_size
            for start in range(0, buffer_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                # 5. 从 CPU 切片数据，*然后*将这个小批次 (mini-batch) 发送到 GPU
                mb_states = old_states[mb_indices].to(self.device)
                mb_actions = old_actions[mb_indices].to(self.device)
                mb_logprobs = old_logprobs[mb_indices].to(self.device)
                mb_rewards = rewards[mb_indices].to(self.device)

                # 6. 在这个小批次上执行 PPO 更新
                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                state_values = torch.squeeze(state_values)
                
                ratios = torch.exp(logprobs - mb_logprobs.detach())

                surr1 = ratios * mb_rewards
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * mb_rewards

                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, mb_rewards) - 0.01 * dist_entropy
                
                # 7. 反向传播这个小批次的损失
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        # --- ▲▲▲ 【【【修改结束】】】 ▲▲▲
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    def pretrain_actor(self, expert_data_loader, epochs):
        """
        使用专家数据对 Actor 网络进行监督学习（行为克隆）。
        """
        self.logger.info("Starting Actor pre-training with expert data...")
        # 我们只训练 Actor，所以为其创建一个单独的优化器
        actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=0.0001)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for states, expert_actions in expert_data_loader:
                states = states.to(self.device)
                expert_actions = expert_actions.to(self.device)

                # 获取 Actor 的输出
                predicted_actions = self.policy.actor(states)

                # 计算损失
                loss = loss_fn(predicted_actions, expert_actions)

                # 更新 Actor
                actor_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(expert_data_loader)
            self.logger.info(f"Actor Pre-training Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")
        
        # 预训练结束后，同步 policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.logger.info("Actor pre-training finished. Old policy synchronized.")

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))