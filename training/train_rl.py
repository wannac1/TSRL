# file: training/train_rl.py
# description: 最终版RL训练脚本，集成了预训练导师加载、Warmup等功能。

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

# --- MODIFICATION START: Import new RL components ---
from trainer.trainer import Trainer # This is your original Trainer
from trainer.rl_trainer import RL_Trainer # This is our new RL_Trainer
from dataset.dataset_wrapper import DatasetWrapper # Our new DatasetWrapper
# --- MODIFICATION END ---

from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')

# --- MODIFICATION START: Add RL trainer switch, WARMUP, and PRETRAIN arguments ---
parser.add_argument('--use_rl_trainer', action='store_true', default=False, help='Enable the RL-based Tutor-Student trainer.')
parser.add_argument('--rl_warmup_epochs', type=int, default=2, help='Number of supervised warmup epochs before starting RL.')
parser.add_argument('--tutor_pretrain_path', type=str, default=None, help='Path to a pre-trained tutor model to load before starting RL training.')
# --- MODIFICATION END ---

args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config):
    # This function is modified to return both the dataloader and the dataset instance
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set_original = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set_original = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set_original = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda':
            train_set_original = LSDADataset(config, mode='train')
        else:
            raise NotImplementedError
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set_original = pairDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set_original = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set_original = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set_original = LRLDataset(config, mode='train')
    else:
        train_set_original = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='train',
                )

    # Wrap the dataset with DatasetWrapper to include sample indices for RL training
    train_set = DatasetWrapper(train_set_original)
    
    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler,
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                )

    # Return both loader and dataset for RL_Trainer
    return train_data_loader, train_set


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        config = config.copy()
        config['test_dataset'] = test_name
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
        else:
            test_set = LRLDataset(
                config=config,
                mode='test',
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last = (test_name=='DeepFakeDetection'),
            )
        return test_data_loader
    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(),
            optim.SGD,
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    scheduler_name = config['lr_scheduler']
    
    # 检查 scheduler_name 是否为 Python 的 None 对象，或者字符串 'None'
    if scheduler_name is None or scheduler_name == 'None':
        return None
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))
    return scheduler


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False

    # --- MODIFICATION START: Add RL trainer args to config ---
    config['use_rl_trainer'] = args.use_rl_trainer
    config['rl_warmup_epochs'] = args.rl_warmup_epochs
    config['tutor_pretrain_path'] = args.tutor_pretrain_path
    # --- MODIFICATION END ---

    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json'

    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    rl_suffix = "_RL" if config['use_rl_trainer'] else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + rl_suffix + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp

    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))

    # prepare data
    train_data_loader, train_dataset = prepare_training_data(config)
    test_data_loaders = prepare_testing_data(config)

    # prepare model, optimizer, scheduler, metric
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)

    # --- MODIFICATION START: Instantiate the correct trainer based on config ---
    if config['use_rl_trainer']:
        logger.info("Initializing RL_Trainer for Tutor-Student training.")
        trainer = RL_Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            train_dataset=train_dataset,
            metric_scoring=metric_scoring,
            time_now=timenow
        )
    else:
        logger.info("Initializing standard supervised Trainer.")
        trainer = Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            metric_scoring=metric_scoring,
            time_now=timenow
        )
    # --- MODIFICATION END ---

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        if hasattr(trainer.model, 'epoch'):
             trainer.model.epoch = epoch

        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
        
        if scheduler is not None:
            scheduler.step()

    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric)))

    if 'svdd' in config['model_name']:
        if isinstance(trainer, RL_Trainer):
             trainer.student_model.update_R(epoch)
        else:
             model.update_R(epoch)

    if hasattr(trainer, 'writers'):
        for writer in trainer.writers.values():
            writer.close()

if __name__ == '__main__':
    main()