import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from PIL import Image
import numpy as np



# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class Dataset_flowers_list(Dataset):
    def __init__(self, path_to_dataset_train, transform, dict_id, training_data, del_labels):
        self.data_dir = path_to_dataset_train
        self.transform = transform
        data = [os.path.join(path_to_dataset_train, x) for x in training_data]

        if len(del_labels) > 0:
            list_indx = []
            # for i, (i1, i2) in enumerate(dict_id.values()):
            #     if i2 in del_labels:
            #         list_indx.append(i)
            for i, row in enumerate(data):
                basename = os.path.basename(row)
                if dict_id[basename][1] in del_labels:
                    list_indx.append(i)
            list_indx.sort(reverse=True)
            for indx in list_indx:
                data.pop(indx)
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]
        image = Image.open(file)
        image = image.resize((224, 224))

        if len(image.size) == 2:
            image = image.convert('RGB')
        # np_img = np.array(image)
        if self.transform:
            image = self.transform(image)
        return image, idx


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        filter_label=None,
        limit=-1,
        dict_id=None,
        training_data=None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.filter_label = filter_label
        self.limit = limit
        self.dict_id = dict_id
        self.training_data = training_data

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              # transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])

        self.train_dataset = Dataset_flowers_list(self.data_dir, transform=train_transforms,
                                             dict_id=self.dict_id, training_data=self.training_data,
                                             del_labels=self.filter_label, )
        self.val_dataset = Dataset_flowers_list(self.data_dir, transform=val_transforms,
                                           dict_id=self.dict_id, training_data=self.training_data,
                                           del_labels=self.filter_label, )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            # num_workers=self.num_workers,
            shuffle=True,
            # pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            # num_workers=self.num_workers,
            shuffle=False,
            # pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            # num_workers=self.num_workers,
            shuffle=True,
            # pin_memory=self.pin_memory,
        )
     