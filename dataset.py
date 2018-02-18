""" Heavily influenced by torchvision's MNIST implementation
"""
import os
import numpy as np
import torch.utils.data as data
import skimage.io as io
import torchvision.transforms as transforms
import augmenter
from glob import glob
from torch.utils.data.sampler import SubsetRandomSampler 

class NucleiDataset(data.Dataset):

    """Class to load official data provided by the organizer
    """
    def __init__(self, root, with_masks=True, augmenter=None, transform=None, mask_transform=None):
        self.root = os.path.expanduser(root)
        self.with_masks = with_masks
        self.augment = augmenter
        self.transform = transform
        self.mask_transform = mask_transform

        self._load_data()


    def __getitem__(self, index):
        path = glob(os.path.join(self.paths[index], 'images', '*'))

        img = io.imread(path[0])
        if img.shape[2] > 3:
            img = img[:,:,:3]

        if self.with_masks:
            path = glob(os.path.join(self.paths[index], 'masks', '*'))
            masks = np.moveaxis(io.imread_collection(path).concatenate(),0,-1)

            if self.augment is not None:
                img, masks = self.augment(img, masks)

            if self.transform is not None:
                img = self.transform(img)

            if self.mask_transform is not None:
                masks = self.mask_transform(masks)

            return img, masks

        if self.augment is not None:
            img = self.augment(img)

        if self.transform is not None:
            img = self.transform(img)

        return img


    def __len__(self):
        return len(self.paths)


    def _load_data(self):
        path = os.path.join(self.root, '*')
        self.paths = glob(path)
        if not self.paths:
            raise RuntimeError("{} is empty.".format(self.root))


    def __repr__(self):
        size = self.__len__()
        fmt_str = '{} Dataset with {} datapoints located at {}'.format(self.__class__.__name__, size, self.root)
        return fmt_str


def get_train_loaders(data_dir,
        validation_dir=None,
        validation_ratio=0.1,
        batch_size=1,
        augment=True,
        random_seed=123,
        shuffle=True,
        num_workers=4,
        pin_memory=False):
    """ Utility to setup data loaders
    """
    if validation_dir is None:
        validation_dir = data_dir
    else:
        validation_ratio = 0.

    general_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    augmenters = augmenter.Compose([
        augmenter.Sometimes(augmenter.iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
        augmenter.iaa.Fliplr(0.5),
        augmenter.iaa.Flipud(0.5)
    ])

    train_dataset = NucleiDataset(
            data_dir,
            with_masks = True,
            transform = general_transforms,
            mask_transform = general_transforms
    )

    validation_dataset = NucleiDataset(
            validation_dir,
            with_masks = True,
            transform = general_transforms,
            mask_transform = general_transforms
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(validation_ratio * num_train)

    if split != 0 and shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split] if split != 0 else list(range(len(validation_dataset)))
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(valid_idx)

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    validation_loader = data.DataLoader(
        validation_dataset, batch_size=batch_size, sampler=validation_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, validation_loader


def get_test_loader(data_dir,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    data_loader = data.DataLoader(
            NucleiDataset(
                data_dir,with_masks=False,transform=transform
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
    )

    return data_loader
