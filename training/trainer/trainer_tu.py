# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics
import torch.nn.functional as F
from training.agents import StateManager

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from .rl_trainer import RL_Trainer
except ImportError:
    RL_Trainer = None # 稍后用于检查
class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None, 
        train_dataset=None,
        **kwargs # --- 新增 kwargs ---
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.hard_sample_log = []
        self.speed_up()  # move model to GPU
        if not hasattr(self, 'state_manager'):
            if train_dataset is None:
                raise ValueError("Trainer requires 'train_dataset' to be passed for StateManager initialization.")
            self.logger.info("Base Trainer: Initializing StateManager for tracking...")
            self.state_manager = StateManager(
                num_samples=len(train_dataset),
                config=config,
                device=device
            )
        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']],find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:

            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()


            return losses,predictions


    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        if epoch>=1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1


        #times_per_epoch=4

        test_step = len(train_data_loader) // times_per_epoch    # test 10 times per epoch
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.original_dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            data_dict_1 = data_dict.copy()
            sample_indices = data_dict_1.pop('indices', None)
            if sample_indices is None:
                raise ValueError("DataLoader did not yield 'indices'. Please ensure DatasetWrapper is used for train_dataset.")
            self.setTrain()
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda()

            losses,predictions=self.train_step(data_dict)

            # update learning rate

            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)
            is_rl_trainer = RL_Trainer is not None and isinstance(self, RL_Trainer)
            if not is_rl_trainer:
                try:
                    with torch.no_grad():
                        # 我们需要单样本损失和正确性
                        # (假设 'predictions' 包含 'cls' (logits), 'data_dict' 包含 'label')
                        logits = predictions['cls']
                        labels = data_dict['label']
                        
                        # 使用与 RL_Trainer 一致的损失计算 (CrossEntropy)
                        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
                        
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        correctness = (preds == labels)
                        
                        # 更新状态
                        self.state_manager.update_states(sample_indices, loss_per_sample.detach(), correctness.detach())
                except Exception as e:
                    self.logger.warning(f"Warning: Failed to update StateManager in base Trainer. Error: {e}")
            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['local_rank']==0:
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)



                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            if (step_cnt+1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp'] ):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                else:
                    test_best_metric = None

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1
        if self.config['local_rank'] == 0: # 仅在主进程
            ema_losses = self.state_manager.ema_loss # 获取所有样本的EMA loss
            difficulty_threshold = self.config.get('difficulty_threshold', 0.7)
            
            hard_sample_count = (ema_losses > difficulty_threshold).sum().item()
            total_samples = len(ema_losses)
            
            self.logger.info(f"Epoch {epoch} - Hard Samples (EMA Loss > {difficulty_threshold}): {hard_sample_count} / {total_samples}")
            
            # 记录到日志
            self.hard_sample_log.append({
                'epoch': epoch, 
                'hard_samples': hard_sample_count, 
                'total_samples': total_samples
            })
        return test_best_metric

    def get_respect_acc(self,prob,label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real,acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            if isinstance(v, torch.Tensor):
                try:
                    if v.ndim == 0:
                        v = v.item()
                    else:
                        v = v.squeeze().item()
                except Exception as e:
                    print(f"[Error] Cannot convert metric {k} to scalar: {v}, error: {e}")
                    continue
            else:
                try:
                    v = float(v)
                except Exception as e:
                    print(f"[Error] Metric {k} is not a number: {v}, error: {e}")
                    continue
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)
    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
    def extract_training_set_state(self, extraction_loader):
        """
        在训练结束后，遍历一次训练集，提取所有特征、标签和对应的EMA Loss。
        这个方法必须在 Rank 0 上运行，并使用一个*非DDP*、*非Shuffle*的数据加载器。
        """
        self.logger.info("Extracting final features and state from the entire training set...")
        self.setEval() # 设置为评估模式

        all_features = []
        all_labels = []
        all_indices = []

        # 检查模型是否有 'extract_features'，以防万一
        # (在我们的实现中，我们依赖 inference() 返回 'feat')
        if not hasattr(self.model, 'extract_features') and self.config['local_rank'] == 0:
             # RL_Trainer 会添加 extract_features, Base Trainer 不会
             # 但 Base Trainer 的 inference() 会返回 'feat'
             pass # 这是正常的

        for data_dict in tqdm(extraction_loader, desc="Extracting features"):
            # 1. 准备数据
            sample_indices = data_dict.pop('indices', None)
            if sample_indices is None:
                self.logger.error("DataLoader did not yield 'indices' during final extraction.")
                continue 

            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda()

            # 2. 提取特征
            predictions = self.inference(data_dict)
            
            if 'feat' not in predictions:
                self.logger.error("Model's 'inference' output does not contain 'feat' key.")
                break # 停止提取
            
            features = predictions['feat']
            labels = data_dict['label']
            
            # 3. 收集数据
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_indices.append(sample_indices.cpu().numpy())

        if not all_features:
            self.logger.error("Feature extraction failed. No features were collected.")
            return None, None, None

        # 4. 合并所有批次
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)
        
        # 5. 根据索引从 StateManager 获取 EMA Loss
        self.logger.info("Aligning features with final EMA losses...")
        
        # 获取 StateManager 中的 EMA loss (这个已经是 0-N 排好序的)
        ema_losses_tensor = self.state_manager.ema_loss.cpu().numpy()
        
        # 我们需要重新排序提取到的 features 和 labels，使其与 StateManager 对齐
        # all_indices 包含了 [10, 5, 100, ...] 这样的乱序索引
        # all_features[i] 对应 all_indices[i]
        
        # 创建一个排序索引，它会告诉我们如何排列 all_indices 才能得到 [0, 1, 2, ...]
        sort_idx = np.argsort(all_indices)
        
        # 使用这个排序索引来排列 features 和 labels
        all_features_sorted = all_features[sort_idx]
        all_labels_sorted = all_labels[sort_idx]
        
        # 现在:
        # all_features_sorted[0] 对应 索引为 0 的样本
        # all_labels_sorted[0] 对应 索引为 0 的样本
        # ema_losses_tensor[0] 对应 索引为 0 的样本
        # 它们现在是对齐的。
        
        self.logger.info("Extraction complete.")
        return all_features_sorted, all_labels_sorted, ema_losses_tensor
