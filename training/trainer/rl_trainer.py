# file: trainer/rl_trainer.py
# description: 最终优化版的RL_Trainer，应用了所有关键修复和改进。

import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import os

# 导入您原有的Trainer作为基类，以及新创建的RL组件
from .trainer_tu import Trainer as SupervisedTrainer
from training.agents import TutorPPO, StateManager, TutorGRPO
from training.utils import AverageMeter
import inspect
class RL_Trainer(SupervisedTrainer):
    """
    强化学习训练协调器 (RL_Trainer) - 最终优化版
    
    继承自监督学习Trainer，并实现“导师-学生”强化学习框架。
    此版本包含以下关键改进：
    1.  【修复】实现了真正的监督学习Warmup，在warmup阶段不使用导师权重。
    2.  【优化】设计了更智能的、基于“状态变化”的奖励函数，以提供更精确的指导信号。
    3.  【集成】增加了加载预训练导师模型的逻辑，以实现行为克隆。
    """
    def __init__(self, config, model, optimizer, scheduler, logger, train_dataset, **kwargs):
        super().__init__(config, model, optimizer, scheduler, logger, **kwargs)
        self.student_model = self.model
        self.student_optimizer = self.optimizer
        self.current_epoch = 0 

        self.logger.info("Initializing RL components for Tutor-Student training (Optimized Version)...")
        self.device = next(self.student_model.parameters()).device #

        # --- ▼▼▼ 【【【核心修复 3：检查模型签名】】】 ▼▼▼
        # (检查 'student_model' (例如 ucf_detector) 是否接受 'calc_reconstruction' 参数)
        self.logger.info("Checking model's forward() signature...")
        try:
            forward_params = inspect.signature(self.student_model.forward).parameters
            if 'calc_reconstruction' in forward_params:
                self.model_accepts_calc_reconstruction = True
                self.logger.info("Model accepts 'calc_reconstruction' argument. Will pass 'False' to prevent OOM.")
            else:
                self.model_accepts_calc_reconstruction = False
                self.logger.info("Model does not accept 'calc_reconstruction'. Standard forward() will be used.")
        except Exception as e:
            self.logger.warning(f"Could not inspect model signature: {e}. Assuming standard forward().")
            self.model_accepts_calc_reconstruction = False
        # --- ▲▲▲ 【【【修改结束】】】 ▲▲▲


        # (修复 'AttributeError: dataset' 的 Bug)
        underlying_dataset = train_dataset.original_dataset #
        
        self.num_fake_samples = len(underlying_dataset) 
        
        if not hasattr(underlying_dataset, 'real_imglist'):
            self.logger.info(f"Initializing StateManager for standard dataset. Size: {self.num_fake_samples}")
            total_samples = self.num_fake_samples
            self.is_pair_dataset = False
        else:
            self.logger.info("Initializing StateManager for pairDataset.")
            self.is_pair_dataset = True
            num_real_samples = len(underlying_dataset.real_imglist) 
            total_samples = self.num_fake_samples + num_real_samples
            self.logger.info(f"StateManager size: {self.num_fake_samples} (fakes) + {num_real_samples} (reals) = {total_samples}")

        self.state_manager = StateManager(
            num_samples=total_samples,
            config=config,
            device=self.device
        )
        
        # (其余 __init__ 代码保持不变)
        if hasattr(self.student_model, 'features'):
            self.logger.info("Detector has a '.features()' method. It will be used directly.")
            self.student_model.extract_features = self.student_model.features
        elif not hasattr(self.student_model, 'extract_features'):
            self.logger.info("Detector does not have a native '.features()' method. Creating a wrapper.")
            feature_layer_name = self.config.get('feature_layer_name', 'avgpool')
            self.student_model = self._wrap_model_for_feature_extraction(self.student_model, feature_layer_name)
        
        feature_dim = self._get_feature_dim(config)
        state_dim = feature_dim + 1 + 1 + 1 + 2
        rl_agent_type = config.get('rl_agent', 'ppo')
        if rl_agent_type.lower() == 'grpo':
            self.logger.info(f"Initializing TutorGRPO with feature_dim={feature_dim} for graph construction.")
            self.tutor_model = TutorGRPO(state_dim, action_dim=1, feature_dim=feature_dim, config=config)
        else:
            self.logger.info("Initializing TutorPPO.")
            self.tutor_model = TutorPPO(state_dim, action_dim=1, config=config)

        try:
            self.tutor_model.to(self.device) 
            self.logger.info(f"Tutor model successfully moved to {self.device}.")
        except Exception as e:
            self.logger.error(f"Failed to move tutor model to device: {e}")
            raise e

        self.logger.info(f"RL components initialized. State vector dimension: {state_dim}")

        if self.config.get('tutor_pretrain_path'):
            tutor_path = self.config['tutor_pretrain_path']
            if os.path.exists(tutor_path):
                self.logger.info(f"Loading pre-trained tutor model from: {tutor_path}")
                try:
                    self.tutor_model.load(tutor_path) 
                    self.logger.info("Successfully loaded pre-trained tutor model.")
                except Exception as e:
                    self.logger.error(f"Failed to load pre-trained tutor model: {e}")
            else:
                self.logger.warning(f"Tutor pre-train path specified but file not found: {tutor_path}. Starting with a random tutor.")
        else:
            self.logger.info("No pre-trained tutor model specified. Starting with a random tutor.")

    def _wrap_model_for_feature_extraction(self, model, feature_layer_name):
        from torch.nn import Sequential
        if hasattr(model, 'is_wrapped_for_rl'): return model

        feature_extractor = None
        if hasattr(model, 'model') and feature_layer_name == 'avgpool':
             if hasattr(model.model, 'avgpool'):
                modules = list(model.model.children())
                avgpool_index = -1
                for i, module in enumerate(modules):
                    if 'AvgPool' in str(module):
                        avgpool_index = i
                        break
                if avgpool_index != -1:
                    feature_extractor = Sequential(*modules[:avgpool_index+1])

        if feature_extractor is None:
            raise NotImplementedError(f"Could not automatically create a feature extractor for model type {type(model.model)}.")

        def extract_features(data_dict):
            return feature_extractor(data_dict['image'])
        
        model.extract_features = extract_features
        model.is_wrapped_for_rl = True
        return model

    # --- 【修复 1 (来自上一轮)】修改此方法以处理字典返回值 ---
    def _get_feature_dim(self, config):
        dummy_input = {'image': torch.randn(1, 3, config.get('resolution', 224), config.get('resolution', 224)).to(self.device)}
        with torch.no_grad():
            self.student_model.eval()
            features_out = self.student_model.extract_features(dummy_input) #

            # --- 【修复】处理UCF模型返回dict的情况 ---
            if isinstance(features_out, dict):
                if 'forgery' in features_out:
                    features = features_out['forgery'] #
                else:
                    # 作为备选，取字典中的第一个值
                    self.logger.warning("Could not find 'forgery' key in features dict. Using first available value.")
                    features = next(iter(features_out.values()))
            else:
                features = features_out
            # --- 修复结束 ---

            if features.ndim == 4:
                features = torch.flatten(features, 1)
        return features.shape[1]

    # --- 【修复 1 (来自上一轮)】修改此方法以处理字典返回值 ---
    def train_step(self, data_dict, all_indices):
    
        labels = data_dict['label']

        with torch.no_grad():
            self.student_model.eval()
            
            # --- ▼▼▼ 【【【核心修复 4：条件化调用】】】 ▼▼▼
            if self.model_accepts_calc_reconstruction:
                initial_outputs = self.student_model(data_dict, calc_reconstruction=False)
            else:
                initial_outputs = self.student_model(data_dict) #
            # --- ▲▲▲ 【【【修改结束】】】 ▲▲▲

            initial_logits = initial_outputs['cls']
            initial_probs = F.softmax(initial_logits, dim=1)
            initial_preds = torch.argmax(initial_probs, dim=1)
            initial_correctness = (initial_preds == labels)

            # (其余代码保持不变...)
            features_out_detached = self.student_model.extract_features(data_dict) 
            if isinstance(features_out_detached, dict):
                features_detached = features_out_detached['forgery'].detach() if 'forgery' in features_out_detached else next(iter(features_out_detached.values())).detach()
            else:
                features_detached = features_out_detached.detach()
            
            if features_detached.ndim == 4:
                features_detached = torch.flatten(features_detached, 1)

            historical_states = self.state_manager.get_states(all_indices) 
            ema_loss_norm = torch.tanh(historical_states['ema_loss']).unsqueeze(1) 
            forget_counts_norm = torch.tanh(historical_states['forgetting_counts'] / 10.0).unsqueeze(1) 
            confidence = initial_probs[:, 1].unsqueeze(1) 
            correct_one_hot = F.one_hot(initial_correctness.long(), num_classes=2).float() 

        states_s = torch.cat([ 
            features_detached, ema_loss_norm, forget_counts_norm, 
            confidence, correct_one_hot
        ], dim=1)

        if self.current_epoch < self.config.get('rl_warmup_epochs', 2):
            weights_w = torch.ones(states_s.shape[0], 1, device=self.device).float()
        else:
            weights_w_np, _ = self.tutor_model.select_action(states_s) 
            weights_w = torch.from_numpy(weights_w_np).to(self.device).float()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device) #

        self.student_model.train()
        
        # --- ▼▼▼ 【【【核心修复 4：条件化调用】】】 ▼▼▼
        if self.model_accepts_calc_reconstruction:
            student_outputs_train = self.student_model(data_dict, calc_reconstruction=False)
        else:
            student_outputs_train = self.student_model(data_dict) #
        # --- ▲▲▲ 【【【修改结束】】】 ▲▲▲
        
        logits_train = student_outputs_train['cls']
        loss_per_sample_train = F.cross_entropy(logits_train, labels, reduction='none')
        
        weighted_loss = (loss_per_sample_train * weights_w.squeeze()).mean()
        
        self.student_optimizer.zero_grad()
        weighted_loss.backward() #
        self.student_optimizer.step()
        
        losses_to_return = {'overall': weighted_loss.item()}
        predictions_to_return = student_outputs_train

        with torch.no_grad():
            self.student_model.eval()
            
            # --- ▼▼▼ 【【【核心修复 4：条件化调用】】】 ▼▼▼
            if self.model_accepts_calc_reconstruction:
                updated_outputs = self.student_model(data_dict, calc_reconstruction=False)
            else:
                updated_outputs = self.student_model(data_dict) #
            # --- ▲▲▲ 【【【修改结束】】】 ▲▲▲
            
            if self.current_epoch >= self.config.get('rl_warmup_epochs', 2):
                rewards_r = self._calculate_rewards(initial_outputs, updated_outputs, labels)
                self.tutor_model.buffer.rewards.extend(rewards_r.cpu().numpy())
                self.tutor_model.buffer.is_terminals.extend([True] * len(rewards_r))
            
            current_loss_per_sample = F.cross_entropy(initial_logits, labels, reduction='none')
            self.state_manager.update_states(all_indices, current_loss_per_sample.detach(), initial_correctness.detach()) 

        return losses_to_return, predictions_to_return

    # --- 【关键改进 2】全新设计的、基于“状态变化”的奖励函数 ---
    def _calculate_rewards(self, initial_outputs, updated_outputs, labels):
        initial_probs = F.softmax(initial_outputs['cls'], dim=1).detach()
        initial_preds = torch.argmax(initial_probs, dim=1)
        initial_correctness = (initial_preds == labels)
        initial_confidence = initial_probs.gather(1, labels.unsqueeze(1)).squeeze()

        updated_probs = F.softmax(updated_outputs['cls'], dim=1).detach()
        updated_preds = torch.argmax(updated_probs, dim=1)
        updated_correctness = (updated_preds == labels)
        updated_confidence = updated_probs.gather(1, labels.unsqueeze(1)).squeeze()

        rewards = torch.zeros_like(labels, dtype=torch.float32)

        for i in range(len(labels)):
            reward = 0.0
            if not initial_correctness[i] and updated_correctness[i]:
                reward = 1.0
            elif initial_correctness[i] and not updated_correctness[i]:
                reward = -1.0
            elif initial_correctness[i] and updated_correctness[i]:
                reward = 0.5 * (updated_confidence[i] - initial_confidence[i])
            else:
                reward = -0.5 * (updated_confidence[i] - initial_confidence[i])
            rewards[i] = reward
        return rewards

    def train_epoch(self, epoch, train_data_loader, test_data_loaders=None):
        self.current_epoch = epoch
        self.logger.info(f"===> RL Training Epoch[{epoch}] | Warmup: {'Yes' if epoch < self.config.get('rl_warmup_epochs', 2) else 'No'}")

        times_per_epoch = self.config.get('test_times_per_epoch', 2)
        if epoch < self.config.get('rl_warmup_epochs', 2):
            times_per_epoch = self.config.get('warmup_test_times_per_epoch', 1)
        
        test_step = len(train_data_loader) // times_per_epoch if times_per_epoch > 0 else len(train_data_loader)
        step_cnt = epoch * len(train_data_loader)

        train_recorder_loss = defaultdict(AverageMeter)
        train_recorder_metric = defaultdict(AverageMeter)

        for iteration, batch_dict in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            self.setTrain()
            
            # --- ▼▼▼ 【新修复 2】检查是否为空批次 (collate_fn返回了{}) ---
            if not batch_dict:
                self.logger.warning(f"Epoch[{epoch}], Iter[{iteration+1}] Skipping empty batch (all samples failed to load).")
                continue
            # --- 修复结束 ---

            sample_indices = batch_dict.pop('indices', None)
            if sample_indices is None:
                # 如果 batch_dict 不是空的，但 'indices' 是 None，说明 collate_fn 逻辑出错了
                # (但基于我们上面的检查，这里应该总是安全的)
                if not batch_dict: # 再次检查以防万一
                    self.logger.warning(f"Epoch[{epoch}], Iter[{iteration+1}] Skipping bad/empty batch.")
                    continue
                else:
                    self.logger.error(f"Epoch[{epoch}], Iter[{iteration+1}] DataLoader yielded batch without 'indices'. Batch keys: {batch_dict.keys()}")
                    raise ValueError("DataLoader must yield 'indices'. Please use DatasetWrapper.")
            
            for key, val in batch_dict.items():
                if val is not None and key != 'video_name':
                    batch_dict[key] = val.to(self.device)
            
            losses, predictions_dict = self.train_step(batch_dict, sample_indices)
            
            batch_metrics = self.student_model.get_train_metrics(batch_dict, predictions_dict)
            if losses is None and predictions_dict is None:
                continue  # 跳过这个损坏的批次
            for name, value in losses.items():
                train_recorder_loss[name].update(value)
            for name, value in batch_metrics.items():
                if 'Confusion Matrix' not in name:
                    train_recorder_metric[name].update(value)

            if (iteration + 1) % self.config.get('rec_iter', 300) == 0 and self.config['local_rank'] == 0:
                loss_str = f"Epoch: {epoch}, Iter: {iteration+1}    "
                for k, v in train_recorder_loss.items():
                    avg_loss = v.average()
                    loss_str += f"training-loss, {k}: {avg_loss:.4f}    "
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', avg_loss, global_step=step_cnt + iteration)
                self.logger.info(loss_str)
                
                metric_str = f"Epoch: {epoch}, Iter: {iteration+1}    "
                for k, v in train_recorder_metric.items():
                    avg_metric = v.average()
                    metric_str += f"training-metric, {k}: {avg_metric:.4f}    "
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', avg_metric, global_step=step_cnt + iteration)
                self.logger.info(metric_str)

                for recorder in train_recorder_loss.values(): recorder.clear()
                for recorder in train_recorder_metric.values(): recorder.clear()

            current_step = step_cnt + iteration
            if (current_step + 1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp'] or (self.config['ddp'] and torch.distributed.get_rank() == 0)):
                    self.logger.info("===> Test start!")
                    self.model = self.student_model 
                    self.test_epoch(epoch, iteration, test_data_loaders, current_step)
        
        if epoch >= self.config.get('rl_warmup_epochs', 2):
            self.logger.info(f"===> Updating Tutor model at the end of Epoch[{epoch}]...")
            self.tutor_model.update()
        
        if self.config['local_rank'] == 0:
            self.state_manager.save_states(self.log_dir, epoch)
        
        return self.best_metrics_all_time