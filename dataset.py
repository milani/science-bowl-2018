""" Heavily influenced by torchvision's MNIST implementation
"""
import os
import collections
import functools
import numpy as np
import torch
import torch.utils.data as data
import skimage.io as io
import torchvision.transforms as transforms
import augmenter
import config
from glob import glob
from torch._six import string_classes
from torch.utils.data.sampler import SubsetRandomSampler 
from torch.utils.data.dataloader import default_collate

class NucleiDataset(data.Dataset):
    """Class to load official data provided by the organizer
    """
    def __init__(self, root, with_masks=True, with_names=True, augmenter=None, transform=None, mask_transform=None):
        self.root = os.path.expanduser(root)
        self.with_masks = with_masks
        self.with_names = True
        self.augment = augmenter
        self.transform = transform
        self.mask_transform = mask_transform

        self._load_data()


    def __getitem__(self, index):
        path = self.paths[index]
        name = os.path.basename(os.path.normpath(path))

        img = self._load_image(path)

        if self.with_masks:
            masks = self._load_masks(path)

            if self.augment is not None:
                img, masks = self.augment(img, masks)

            if self.transform is not None:
                # copy is required to avoid negative strides
                # which is not supported by torch.Tensor
                img = self.transform(img.copy())

            if self.mask_transform is not None:
                masks = self.mask_transform(masks.copy())

            if self.with_names:
                return img, masks, name
            return img, masks

        if self.augment is not None:
            img = self.augment(img)

        if self.transform is not None:
            img = self.transform(img.copy())

        if self.with_names:
            return img, name
        return img


    def __len__(self):
        return len(self.paths)


    def _load_image(self, path):
        path = glob(os.path.join(path, 'images', '*'))[0]
        img = io.imread(path)
        if img.shape[2] > 3:
            img = img[:,:,:3]
        return img


    def _load_masks(self, path):
        path = glob(os.path.join(path, 'masks', '*'))
        masks = np.moveaxis(
                io.imread_collection(path).concatenate(),
                0,-1
        )
        return masks


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
        min_size=256,
        max_size=800,
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
#        augmenter.Sometimes(augmenter.iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
        #augmenter.iaa.OneOf([
        #    CropAndPad(),
        #    ResizeAndPad()
        #]),
        augmenter.FixedResizeAndPad(config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, interpolation='linear'),
        augmenter.iaa.Fliplr(0.5),
        augmenter.iaa.Flipud(0.5)
    ])

    train_dataset = NucleiDataset(
            data_dir,
            with_masks = True,
            transform = general_transforms,
            mask_transform = general_transforms,
            augmenter = augmenters if augment else None
    )

    validation_dataset = NucleiDataset(
            validation_dir,
            with_masks = True,
            transform = general_transforms,
            mask_transform = general_transforms,
            augmenter = augmenters if augment else None
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
        collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory
    )

    validation_loader = data.DataLoader(
        validation_dataset, batch_size=batch_size, sampler=validation_sampler,
        collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, validation_loader


def get_test_loader(data_dir,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False):

    augmenters = augmenter.Compose([
        augmenter.FixedResizeAndPad(config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, interpolation='linear'),
    ])

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    data_loader = data.DataLoader(
            NucleiDataset(
                data_dir,with_masks=False,transform=transform, augmenter=augmenters
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
    )

    return data_loader


def _collate(batch):
    # Attempts to resize all tensors to a fixed size
    # and calls default_collate.
    # Assumes height and width of tensors are the same
    # and they only differ in number of channels.
    if torch.is_tensor(batch[0]):
        batch = list(batch)
        h, w = batch[0].shape[1:]
        channels = np.array([t.shape[0] for t in batch])
        max_channel = channels.max()
        new_channels = max_channel - channels
        for i in np.where(new_channels > 0)[0]:
            c = int(new_channels[i])
            size = torch.Size((c,h,w))
            batch[i] = torch.cat((batch[i], torch.zeros(size)),0)

        return default_collate(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [_collate(samples) for samples in transposed]
    else:
        return default_collate(batch)
