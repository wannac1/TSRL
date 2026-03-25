import torch
from torch.utils.data import Dataset

class DatasetWrapper(Dataset):

    def __init__(self, original_dataset):
        super().__init__()
        self.original_dataset = original_dataset
        
        if not hasattr(self.original_dataset, 'collate_fn'):
            raise ValueError(f"Wrapped dataset {type(original_dataset)} must have a 'collate_fn' method.")
        self.original_collate_fn = self.original_dataset.collate_fn

    def __len__(self):

        return len(self.original_dataset)

    def __getitem__(self, index):

        item_dict = self.original_dataset[index]
        
        if not isinstance(item_dict, dict):
            raise TypeError(f"Wrapped dataset {type(self.original_dataset)} __getitem__ must return a dict.")
            
        item_dict['index'] = index
        
        return item_dict

    def collate_fn(self, batch):
        
        valid_batch = [d for d in batch if d is not None]
        if not valid_batch:
            return {} 

        indices = [d.pop('index') for d in valid_batch]
        

        try:
            collated_data = self.original_collate_fn(valid_batch)
        except Exception as e:
            if not valid_batch:
                 return {} 
            print(f"\nERROR: The original collate_fn ({self.original_collate_fn.__module__}) failed.")
            print(f"Error: {e}\n")
            raise e

        collated_data['indices'] = torch.tensor(indices, dtype=torch.long)
        
        return collated_data