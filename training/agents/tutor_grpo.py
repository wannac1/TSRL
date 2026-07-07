# file: training/agents/tutor_grpo.py
# description: GRPO agent for the Tutor model, using a Graph Attention Network (GAT).
# MODIFIED: Added an overridden 'pretrain_actor' method to fix dimension mismatch during pre-training.

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

# 导入 torch_geometric 相关组件
from torch_geometric.nn import GATConv, knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

# 从原有的tutor_ppo.py中导入RolloutBuffer和TutorPPO作为基类
from .tutor_ppo import RolloutBuffer, TutorPPO

class ActorCriticGNN(nn.Module):
    """
    使用图注意力网络 (GAT) 的 Actor-Critic 模型。
    """
    def __init__(self, state_dim, action_dim, feature_dim, action_std_init):
        super(ActorCriticGNN, self).__init__()

        self.feature_dim = feature_dim  # 用于构建图的特征维度
        
        # GAT Layer
        self.gat_conv1 = GATConv(state_dim, 64, heads=4, dropout=0.1)
        self.gat_conv2 = GATConv(64 * 4, 128, heads=1, concat=False, dropout=0.1)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def _create_graph_from_batch(self, states):
        """
        从批次状态中构建k-NN图。我们只使用视觉特征来计算近邻。
        """
        # 假设前 self.feature_dim 是视觉特征
        visual_features = states[:, :self.feature_dim]
        # 为批次中的所有节点构建一个k=5的近邻图
        edge_index = knn_graph(visual_features, k=5, batch=None, loop=True)
        return edge_index

    def forward(self, states):
        # 1. 构建图
        edge_index = self._create_graph_from_batch(states)
        x = states
        
        # 2. GNN 前向传播
        x = torch.relu(self.gat_conv1(x, edge_index))
        x = self.gat_conv2(x, edge_index) # 输出维度 [num_nodes, 128]
        
        # 3. 传入 Actor 和 Critic
        action_mean = self.actor(x)
        state_val = self.critic(x)
        
        return action_mean, state_val, edge_index

    def act(self, state):
        action_mean, state_val, _ = self.forward(state)
        
        # 和PPO一样，使用多元正态分布进行采样
        cov_matrix = torch.diag(torch.full((action_mean.size(-1),), 0.1)).to(action_mean.device)
        cov_matrix = cov_matrix.unsqueeze(0).repeat(action_mean.size(0), 1, 1)
        dist = MultivariateNormal(action_mean, covariance_matrix=cov_matrix)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_mean, state_values, _ = self.forward(state)
        
        action_var = torch.full((action_mean.shape[-1],), 0.01).to(action_mean.device)
        cov_mat = torch.diag(action_var).unsqueeze(0).repeat(action_mean.shape[0], 1, 1)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy


class TutorGRPO(TutorPPO):
    """
    GRPO Tutor Agent
    """
    def __init__(self, state_dim, action_dim, feature_dim, config):
        # 使用父类的构造函数，但策略网络会在这里被重写
        super().__init__(state_dim, action_dim, config)
        
        # 重写 policy 和 policy_old 为GNN版本
        self.policy = ActorCriticGNN(state_dim, action_dim, feature_dim, 0.1).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.gat_conv1.parameters(), 'lr': config.get('tutor_lr_critic', 0.0003)},
            {'params': self.policy.gat_conv2.parameters(), 'lr': config.get('tutor_lr_critic', 0.0003)},
            {'params': self.policy.actor.parameters(), 'lr': config.get('tutor_lr_actor', 0.0001)},
            {'params': self.policy.critic.parameters(), 'lr': config.get('tutor_lr_critic', 0.0003)}
        ])

        self.policy_old = ActorCriticGNN(state_dim, action_dim, feature_dim, 0.1).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    # --- 【关键修复】为GRPO重写pretrain_actor方法 ---
    def pretrain_actor(self, expert_data_loader, epochs):
        """
        为 GRPO 重写的 Actor 预训练方法。
        这个版本会正确地调用完整的 GNN 前向传播，以避免维度不匹配的错误。
        """
        self.logger.info("Starting GRPO Actor pre-training with expert data...")
        # 优化器需要包含GNN层和Actor层，因为它们都是Actor决策路径的一部分
        actor_params = list(self.policy.gat_conv1.parameters()) + \
                       list(self.policy.gat_conv2.parameters()) + \
                       list(self.policy.actor.parameters())
        actor_optimizer = torch.optim.Adam(actor_params, lr=0.0001)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for states, expert_actions in expert_data_loader:
                states = states.to(self.device)
                expert_actions = expert_actions.to(self.device)

                # 正确的调用方式：通过完整的 forward 方法来获取 Actor 的输出
                # self.policy.forward 返回 (action_mean, state_val, edge_index)
                predicted_actions, _, _ = self.policy.forward(states)

                # 计算损失
                loss = loss_fn(predicted_actions, expert_actions)

                # 更新 Actor
                actor_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(expert_data_loader)
            self.logger.info(f"GRPO Actor Pre-training Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")
        
        # 预训练结束后，同步 policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.logger.info("GRPO Actor pre-training finished. Old policy synchronized.")