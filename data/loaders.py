import collections
import numpy as np
import torch
import torchvision.transforms as transforms
import data.augments as augments
from data.dataset import NucleiDataset
from data.utils import ToTensor
from imgaug.augmenters import Fliplr, Flipud
from torch._six import string_classes
from torch.utils.data.sampler import SubsetRandomSampler 
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

def train_validation_loaders(data_dir, validation_dir=None,
        validation_ratio=0.1, batch_size=1,
        min_size=256, max_size=320,
        augment=True, random_seed=123,
        shuffle=True, num_workers=4, pin_memory=False):

    if validation_dir is None:
        validation_dir = data_dir
    else:
        validation_ratio = 0.

    general_transforms = transforms.Compose([
        ToTensor()
    ])

    augmenters = augments.Compose([
        augments.FixedResizeAndPad(min_size, max_size, interpolation='linear'),
        Fliplr(0.5),
        Flipud(0.5)
    ])

    train_dataset = NucleiDataset(
            data_dir,
            with_masks = True,
            transform = general_transforms,
            augmenter = augmenters if augment else None
    )

    validation_dataset = NucleiDataset(
            validation_dir,
            with_masks = True,
            transform = general_transforms,
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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory
    )

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, sampler=validation_sampler,
        collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, validation_loader


def test_loader(data_dir, min_size=256, max_size=320, batch_size=1, shuffle=False,
        num_workers=4,pin_memory=False):

    augmenters = augments.Compose([
        augments.FixedResizeAndPad(min_size, max_size, interpolation='linear'),
    ])

    transform = transforms.Compose([
        ToTensor()
    ])

    data_loader = DataLoader(
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
    # Assumes height (and width if ndim ==3) of tensors are the same
    # and they only differ in number of channels.
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        batch = list(batch)
        dims = []
        if len(batch[0].shape) == 3:
            h, w = batch[0].shape[1:]
            dims.extend([h,w])
        elif len(batch[0].shape) == 2:
            w = batch[0].shape[1]
            dims.append(w)
        channels = np.array([t.shape[0] for t in batch])
        max_channel = channels.max()
        new_channels = max_channel - channels
        for i in np.where(new_channels > 0)[0]:
            c = int(new_channels[i])
            size = torch.Size([c] + dims)
            batch[i] = torch.cat((batch[i], torch.zeros(1).expand(c,*dims)),0)

        return default_collate(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [_collate(samples) for samples in transposed]
    else:
        return default_collate(batch)

__all__ = ['train_validation_loaders', 'test_loader']
