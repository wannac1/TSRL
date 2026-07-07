# file: dataset/dataset_wrapper.py
# description: 修正后的 DatasetWrapper，它能正确委托 collate_fn。

import torch
from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    """
    这个 Wrapper 包裹一个现有的 Dataset，主要目的是
    在 __getitem__ 中添加样本索引，并在 collate_fn 中
    将其作为 'indices' 键打包到批次中，以供 RL_Trainer 使用。
    
    它通过调用原始数据集的 collate_fn 来实现这一点，
    从而使其与 DeepfakeAbstractBaseDataset 和 pairDataset 
    （以及任何其他具有 .collate_fn 的自定义数据集）兼容。
    """
    def __init__(self, original_dataset):
        """
        Args:
            original_dataset (Dataset): 一个__getitem__返回字典, 
                                        且拥有 .collate_fn 方法的数据集实例。
        """
        super().__init__()
        self.original_dataset = original_dataset
        
        # 关键：获取原始数据集的 collate_fn
        if not hasattr(self.original_dataset, 'collate_fn'):
            raise ValueError(f"Wrapped dataset {type(original_dataset)} must have a 'collate_fn' method.")
        # 保存原始的 collate_fn 供后续调用
        self.original_collate_fn = self.original_dataset.collate_fn

    def __len__(self):
        """
        长度直接代理给原始数据集。
        """
        return len(self.original_dataset)

    def __getitem__(self, index):
        """
        获取数据时，调用原始数据集获取字典，然后在字典中添加'index'键。
        """
        # 1. 从原始数据集获取字典
        item_dict = self.original_dataset[index]
        
        if not isinstance(item_dict, dict):
            raise TypeError(f"Wrapped dataset {type(self.original_dataset)} __getitem__ must return a dict.")
            
        # 2. 在字典中添加索引
        item_dict['index'] = index
        
        return item_dict

    def collate_fn(self, batch):
        """
        自定义的数据打包函数 (核心修复点)。
        """
        
        # 1. 过滤掉数据集中可能返回的 None 项 (例如，加载失败的样本)
        valid_batch = [d for d in batch if d is not None]
        if not valid_batch:
            # 如果整个批次都是 None，返回一个空字典。
            # rl_trainer.py 中的检查会处理这个空字典。
            return {} 

        # 2. 提取所有 'index'，并从字典中移除它们
        indices = [d.pop('index') for d in valid_batch]
        
        # 3. 'valid_batch' 现在是一个干净的列表，只包含原始数据
        
        # 4. 【关键】调用原始的 collate_fn
        #    - 如果是 pairDataset，则调用 pairDataset.collate_fn
        #    - 如果是 abstract_dataset，则调用 abstract_dataset.collate_fn
        try:
            collated_data = self.original_collate_fn(valid_batch)
        except Exception as e:
            if not valid_batch:
                 return {} 
            print(f"\nERROR: The original collate_fn ({self.original_collate_fn.__module__}) failed.")
            print(f"Error: {e}\n")
            raise e

        # 5. 将 'indices' 键添加到 collate_fn 返回的字典中
        #    (pairDataset 和 abstract_dataset 的 collate_fn 都会返回一个包含'label'的字典)
        collated_data['indices'] = torch.tensor(indices, dtype=torch.long)
        
        return collated_data