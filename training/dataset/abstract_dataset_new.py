# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.
import lmdb
import sys

import torchvision
from torch.utils.data import Subset

sys.path.append('.')
from scipy.ndimage import binary_erosion, binary_dilation
import os
import math
import yaml
import glob
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A
from dataset.utils.attribution_mask import *
from dataset.albu import IsotropicResize

FFpp_pool=['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT']#
DiffFace_pool=['SDv15','SDv21','DiffSwap']
save_index=0
def save_tuple(img_tuple,labels,idx,type2):
    imgs=img_tuple*0.5+0.5
    pil_img=torchvision.transforms.ToPILImage()(imgs[0])

    pil_img.save(f"{type2}_296_img/{idx}_{labels[0]}.png")

def shift_image(image, dx, dy,**kwargs):
    image = np.roll(image, dy, axis=0)
    image = np.roll(image, dx, axis=1)
    if dy > 0:
        image[:dy, :] = 0
    elif dy < 0:
        image[dy:, :] = 0
    if dx > 0:
        image[:, :dx] = 0
    elif dx < 0:
        image[:, dx:] = 0
    return image


def get_shift_transform(dx, dy,p=1):
    return A.Lambda(image=lambda image, **kwargs: shift_image(image, dx, dy, **kwargs), p=p)

def shift_image_v2(image, dx, dy,**kwargs):
    if dy > 0:
        image=image[dy:-dy, :]
    if dx > 0:
        image=image[:, dx:-dx]
    return image


def get_shift_transform_v2(dx, dy,p=1):
    return A.Lambda(image=lambda image, **kwargs: shift_image_v2(image, dx, dy, **kwargs), p=p)

def all_in_pool(inputs,pool):
    for each in inputs:
        if each not in pool:
            return False
    return True

class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """
        
        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.data_manner = config['data_manner']

        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        self.output_device="cpu"
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        self.env={}
        # if mode == 'train' and len(config['train_dataset'])>1 and self.data_manner == 'lmdb' and (not np.all(np.isin(config['train_dataset'], FFpp_pool))):
        #     raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
        suffix="lmdb"
        if config['compression'] == 'c40':
            suffix='c40_'+suffix
        if config['sample_size'] == 296:
            suffix='296_'+suffix
        if config['sample_size'] == 384:
            suffix='384_c23_'+suffix
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            self.dataset_list = dataset_list
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                if self.data_manner == 'lmdb':

                    if one_data in FFpp_pool:
                        if one_data=='DeepFakeDetection':
                            self.DFD=True
                        else:
                            self.DFD=False
                        dataset_name = 'FaceForensics++'
                    elif one_data in DiffFace_pool:
                        dataset_name = 'DiffusionFace'
                    else:
                        dataset_name = one_data
                    lmdb_path = os.path.join(config['lmdb_dir'], f"{dataset_name}_{suffix}")
                    self.env[dataset_name] = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                if one_data not in DiffFace_pool:
                    tmp_image, tmp_label = self.collect_img_and_label_for_one_dataset(one_data)
                else:
                    tmp_image, tmp_label = self.load_diffFace_json(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
        elif mode == 'test':
            if self.data_manner == 'lmdb':
                if config['test_dataset'] in FFpp_pool:
                    if config['test_dataset'] == 'DeepFakeDetection':
                        self.DFD = True
                    else:
                        self.DFD = False
                    dataset_name = 'FaceForensics++'
                elif config['test_dataset'] in DiffFace_pool:
                    dataset_name = 'DiffusionFace'
                else:
                    self.DFD = False
                    dataset_name = config['test_dataset']
                if dataset_name=='FaceForensics++' and (config['compression']=='c40' or config['sample_size']==384) and not self.DFD:
                    lmdb_path = os.path.join(config['lmdb_dir'], f"{dataset_name}_{suffix}")
                else:

                    lmdb_path = os.path.join(config['lmdb_dir'], f"{dataset_name}_lmdb")
                self.env[dataset_name] = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
            one_data = config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            if config['test_dataset'] not in DiffFace_pool:
                image_list, label_list = self.collect_img_and_label_for_one_dataset(one_data)
            else:
                image_list, label_list = self.load_diffFace_json(one_data)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list
        self.dataset_name=dataset_name
        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        if mode == 'train':
            self.transform = self.init_training_data_aug_method()
            if 'norm_class' in config:
                self.init_subset()
        elif (self.mode == 'test') and ('test_aug' in self.config):
            self.transform = self.init_testing_data_aug_method()

    def init_subset(self):

        train_idx_normal = np.argwhere(np.isin(np.array(self.label_list.copy()), self.config['norm_class'])).flatten().tolist()
        self.sub_set = Subset(self, train_idx_normal)
        self.sub_set.data_dict = {
            'image': list(np.array(self.image_list)[train_idx_normal]),
            'label': list(np.array(self.label_list)[train_idx_normal]),
        }

    def subsets(self):
        ssets={}
        train_idx_normal = np.argwhere(np.isin(np.array(self.label_list.copy()), 0)).flatten().tolist()
        ssets['real'] = Subset(self, train_idx_normal)
        ssets['real'].data_dict = {
            'image': list(np.array(self.image_list)[train_idx_normal]),
            'label': list(np.array(self.label_list)[train_idx_normal]),
        }
        train_idx_normal = np.argwhere(np.isin(np.array(self.label_list.copy()), 0)).flatten().tolist()
        ssets['fake'] = Subset(self, train_idx_normal)
        ssets['fake'].data_dict = {
            'image': list(np.array(self.image_list)[train_idx_normal]),
            'label': list(np.array(self.label_list)[train_idx_normal]),
        }

        return ssets

    def init_testing_data_aug_method(self):
        if self.config['test_aug']['use_train_aug']:
            return self.init_training_data_aug_method()
        transform=A.Compose([
            A.RandomGridShuffle(grid=(self.config['test_aug']['GridShuffle']['grid_g'], self.config['test_aug']['GridShuffle']['grid_g']),
                                p=self.config['test_aug']['GridShuffle']['p']),
            A.CropAndPad(px=self.config['test_aug']['crop']['crop_px'], keep_size=True,p=self.config['test_aug']['crop']['p']),
            A.CropAndPad(px=self.config['test_aug']['pad']['pad_px'], keep_size=True, p=self.config['test_aug']['pad']['p']),
            get_shift_transform(dx=self.config['test_aug']['shift']['shift_dx'], dy=self.config['test_aug']['shift']['shift_dy'],
                                p=self.config['test_aug']['shift']['p']),
            A.MedianBlur(blur_limit=self.config['test_aug']['blur']['blur_limit'], p=self.config['test_aug']['blur']['p']),
            A.RandomBrightnessContrast(brightness_limit=self.config['test_aug']['brightness']['brightness_limit'],
                                       contrast_limit=0,p=self.config['test_aug']['brightness']['p']),
            A.ImageCompression(quality_lower=self.config['test_aug']['quality']['quality_lower'],
                               quality_upper=self.config['test_aug']['quality']['quality_upper'], p=self.config['test_aug']['quality']['p']),

            A.GridDropout(ratio=self.config['test_aug']['GridDropout']['ratio'],p=self.config['test_aug']['GridDropout']['p']),
            A.GaussNoise(var_limit=(20.0,20.0), mean=0, per_channel=True, always_apply=False, p=self.config['test_aug']['GaussNoise']['p']),

        ])
        return transform

    def init_training_data_aug_method(self):
        trans = A.Compose([
            # A.GridDropout(ratio=0.5, holes_number_x = 2, holes_number_y = 2,
            #               p=self.config['data_aug']['gridmask_prob']),
            # A.RandomGridShuffle(grid=(self.config['GridShuffle']['grid_g'],self.config['GridShuffle']['grid_g']),p=self.config['GridShuffle']['p']),
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            # get_shift_transform_v2(dx=self.config['Shift']['shift_dx'],
            #                     dy=self.config['Shift']['shift_dy'],
            #                     p=self.config['Shift']['p']),
            # A.Resize(height=256, width=256),
            # A.OneOf([
            #     IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            #     IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            #     IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            # ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def load_diffFace_json(self,dataset_name):
        path = os.path.join(self.config['dataset_json_folder'], dataset_name+ '.json')
        with open(path, 'r') as f:
            dataset_info = json.load(f)
        fake_paths= dataset_info[dataset_name][self.mode]
        real_paths= dataset_info['real'][self.mode]
        file_path = fake_paths+real_paths
        labels=[1]*len(fake_paths)+[0]*len(real_paths)
        return file_path, labels




    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.
        
        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []
        cp = None
        # Try to get the dataset information from the JSON file
        try:
            suffix='_296' if self.config['sample_size']==296 and self.mode=='train' and self.config['compression']=='c23' else ''
            if self.config['sample_size']==384 and self.config['compression']=='c23' and dataset_name=='FaceForensics++' and not self.DFD:
                suffix='_384_c23'

            if self.config['compression']=='c40' and dataset_name=='FaceForensics++' and not self.DFD:
                cp = 'c40'
                path = os.path.join(self.config['dataset_json_folder'][:-2]+"v5", dataset_name+suffix + '.json')
            else:
                path=os.path.join(self.config['dataset_json_folder'], dataset_name + suffix + '.json')
            with open(path, 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        # If JSON file exists, do the following data collection
        # FIXME: ugly, need to be modified here.

        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'
        elif dataset_name == 'FF-DF_c40':
            dataset_name = 'FF-DF'
            cp = 'c40'
        elif dataset_name == 'FF-F2F_c40':
            dataset_name = 'FF-F2F'
            cp = 'c40'
        elif dataset_name == 'FF-FS_c40':
            dataset_name = 'FF-FS'
            cp = 'c40'
        elif dataset_name == 'FF-NT_c40':
            dataset_name = 'FF-NT'
            cp = 'c40'
        # Get the information for the current dataset
        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                sub_dataset_info = sub_dataset_info['c23']
            elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                sub_dataset_info = sub_dataset_info['c40']
            # Iterate over the videos in the dataset
            for video_name, video_info in sub_dataset_info.items():
                # Get the label and frame paths for the current video
                if video_info['label'] not in self.config['label_dict']:
                    raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                label = self.config['label_dict'][video_info['label']]
                # if label == 0 and dataset_name == 'FaceShifter':
                #     continue
                frame_paths = video_info['frames']

                # Select self.frame_num frames evenly distributed throughout the video
                total_frames = len(frame_paths)
                if self.frame_num < total_frames:
                    step = total_frames // self.frame_num
                    selected_frames = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]
                    # Append the label and frame paths to the lists according the number of frames
                    label_list.extend([label]*len(selected_frames))
                    frame_path_list.extend(selected_frames)
                else:
                    label_list.extend([label]*total_frames)
                    frame_path_list.extend(frame_paths)
            
        # Shuffle the label and frame path lists in the same order
        if self.mode == 'train':
            shuffled = list(zip(label_list, frame_path_list))
            random.shuffle(shuffled)
            label_list, frame_path_list = zip(*shuffled)
        
        return frame_path_list, label_list

     
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['sample_size'] # if self.mode == "train" else self.config['resolution']
        if self.data_manner == 'img':
            file_path=os.path.join(self.config['dataset_root_dir'],file_path)
            assert os.path.exists(file_path), f"{file_path} does not exist"
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        elif self.data_manner == 'lmdb':
            dataset_name = file_path.split('\\')[0]
            with self.env[dataset_name].begin(write=False) as txn:
                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if self.data_manner == 'img':
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            dataset_name = file_path.split('\\')[0]
            with self.env[dataset_name].begin(write=False) as txn:
                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    mask = np.zeros((size, size,3))
                else:
                    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
                    mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if self.data_manner == 'img':
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        elif self.data_manner == 'lmdb':
            dataset_name = file_path.split('\\')[0]
            with self.env[dataset_name].begin(write=False) as txn:
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
        landmark = np.clip(landmark,0,self.config['resolution']-1)
        return np.float32(landmark)

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)


    def re_crop(self,img,landmark=None,mask=None,ratio=None):
        # Create a dictionary of arguments
        kwargs = {'image': img}
        if self.config['sample_size']==256:
            random_max = 0.11 #1.3*(1-0.11*2)=1
        else:
            random_max = 0.16 #1.5*(1-0.16*2)=1
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask
        if 'crop_p' in self.config:
            p=self.config['crop_p']
        else:
            p=1
        if ratio is None:
            ratio = random.uniform(0, random_max)
        abs_margin = -int(img.shape[0]*ratio)
        if p < random.random():
            abs_margin = 0 if self.config['sample_size']==256 else -20

        if (self.config['sample_size']==296 and self.config['use_crop'] == False) or (self.mode=='test' and self.config['compression']=='c40'):
            abs_margin = -20

        crop_transform_256 = A.Compose([
            A.CropAndPad(px=abs_margin, keep_size=True,
                         p=1),
        ])
        crop_transform_296 = A.Compose([
            A.CropAndPad(px=abs_margin, keep_size=False,
                         p=1),
            A.Resize(256, 256)
        ])

        crop_transform = crop_transform_256 if self.config['sample_size'] == 256 else crop_transform_296
        transformed = crop_transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        return augmented_img, augmented_landmark, augmented_mask


    def data_aug(self, img, landmark=None, mask=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """
        
        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None and mask.max()>0:
            kwargs['mask'] = mask.squeeze(2)

        # Apply data augmentation
        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')

        augmented_mask = transformed.get('mask',mask)

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            if len(augmented_mask.shape)!=4:
                augmented_mask=np.expand_dims(augmented_mask, axis=2)
            augmented_landmark = np.array(augmented_landmark)

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_path = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        # Get the mask and landmark paths
        mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
        landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark
        
        # Load the image
        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}, path {image_path}: {e}")
            return self.__getitem__(0)
        image = np.array(image)  # Convert to numpy array for data augmentation
        
        # Load mask and landmark (if needed)
        if self.config['with_mask']:
            mask = self.load_mask(mask_path)
        else:
            mask = None
        if self.config['with_landmark']:
            landmarks = self.load_landmark(landmark_path)
            landmarks=np.clip(landmarks, 0, self.config['resolution']-1)
        else:
            landmarks = None

        #Do cropping
        if self.mode == 'train':
            if (self.config .get('use_crop',False) and self.config['sample_size']==256) or self.config['sample_size']==296:
                crop_ratio = self.config['crop_ratio']
                crop_ratio = None if crop_ratio=='r' else crop_ratio
                image_trans, landmarks_trans, mask_trans = self.re_crop(image, landmarks, mask,ratio=crop_ratio)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)
        else:
            if (self.config['compression'] == 'c40' and self.dataset_name == 'FaceForensics++' and not self.DFD):
                image_trans, landmarks_trans, mask_trans = self.re_crop(image, landmarks, mask, ratio=None)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)
        # Do Data Augmentation
        if self.mode=='train' and self.config['use_data_augmentation']:
            image_trans, landmarks_trans, mask_trans = self.data_aug(image_trans, landmarks_trans, mask_trans)
        elif (self.mode=='test') and ('test_aug' in self.config):
            if self.config['test_aug']['use_aug']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image_trans, landmarks_trans, mask_trans)

        if 'remove_attribute' in self.config['data_aug'] and self.config['data_aug']['remove_attribute']:
            line1=0
            line2=0
            p1=random.random()
            p2=random.random()
            if self.config['data_aug']['remove_nose_prob']>p1:
                line1 = remove_nose(image_trans, landmarks_trans)
            if self.config['data_aug']['remove_eyes_prob']>p2:
                line2 = remove_eyes(image_trans,landmarks_trans)
            if isinstance(line1, np.ndarray) and isinstance(line2, np.ndarray):
                line = line1 + line2
            elif isinstance(line1, np.ndarray):
                line = line1
            elif isinstance(line2, np.ndarray):
                line = line2
            else:
                line=0
            if type(line) is not int:
                #here are removing instead of replacing
                image_trans=image_trans*line[:, :, np.newaxis]
            # center = (int(np.mean(x)), int(np.mean(y)))  # 计算质心
            # mix_img = Image.fromarray(np.uint8(
            #     line[:, :, np.newaxis] * (fake_image * 0.8 + real_image * 0.2) + (1 - line)[:, :, np.newaxis] * real_image))

        if no_norm:
            if mask_trans is None:
                return image_trans, label
            else:
                return image_trans, label, mask_trans
        # To tensor and normalize
        image_trans = self.normalize(self.to_tensor(image_trans))
        if self.config['with_landmark']:
            landmarks_trans = torch.from_numpy(landmarks)
        if self.config['with_mask']:
            mask_trans = torch.from_numpy(mask_trans)
        if self.output_device=='cuda':
            image_trans=image_trans.cuda()
        return image_trans, label, landmarks_trans, mask_trans



    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks = zip(*batch)
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        
        # Special case for landmarks and masks if they are None
        if landmarks[0] is not None:
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if masks[0] is not None:
            masks = torch.stack(masks, dim=0)
        else:
            masks = None
        # global save_index
        # save_index += 1
        # save_tuple(images,labels, save_index, 'original')
        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'

        return len(self.image_list)


if __name__ == "__main__":
    # with open('/home/zhiyuanyan/disfin/deepfake_benchmark/training/config/detector/xception.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # train_set = DeepfakeAbstractBaseDataset(
    #             config = config,
    #             mode = 'train',
    #         )
    # train_data_loader = \
    #     torch.utils.data.DataLoader(
    #         dataset=train_set,
    #         batch_size=config['train_batchSize'],
    #         shuffle=True,
    #         num_workers=int(config['workers']),
    #         collate_fn=train_set.collate_fn,
    #     )
    # from tqdm import tqdm
    # for iteration, batch in enumerate(tqdm(train_data_loader)):
    #     # print(iteration)
    #     ...
    #     # if iteration > 10:
    #     #     break
    def trans_PIL_img(input,trans):
        input=np.array(input)
        return Image.fromarray(trans(image=input)['image'])




    img_test_path = r"H:\code\DeepfakeBench\datasets_v2\FaceForensics++_296\original_sequences\youtube\c23\frames\000\000.png"
    original_image = Image.open(img_test_path)
    img_v2 = r'H:\code\DeepfakeBench\datasets_v2\FaceForensics++\original_sequences\youtube\c23\frames\000\000.png'
    original_image_v2 = Image.open(img_v2)
    tran_v0 = A.Compose([
        get_shift_transform_v2(dx=50,
                               dy=0,
                               p=1,)
    ])
    img1 = trans_PIL_img(original_image, tran_v0)
    img1.show()
    transform_v1 = A.Compose([
        A.CropAndPad(px=10,keep_size=True)
    ])

    transform_v2 = A.Compose([
        A.CropAndPad(px=-10,keep_size=True)
    ])

    transform_v3 = A.Compose([
        get_shift_transform(dx=100, dy=50),
    ])
    transform_v4 = A.Compose([
        A.GaussianBlur(p=1),
    ])
    trans_PIL_img(original_image, transform_v4).show()
    original_image.show()
    img1=trans_PIL_img(original_image,transform_v1)
    img2=trans_PIL_img(original_image,transform_v2)
    img3=trans_PIL_img(original_image,transform_v3)

    img1.show()
    img2.show()
    img3.show()

