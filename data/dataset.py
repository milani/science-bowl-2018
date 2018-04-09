import os
import numpy as np
import skimage.io as io
from glob import glob
from torch.utils.data import Dataset

class NucleiDataset(Dataset):
    """Class to load official data provided by the organizer
    """
    def __init__(self, root, with_masks=True, augmenter=None, transform=None):
        self.root = os.path.expanduser(root)
        self.with_masks = with_masks
        self.augment = augmenter
        self.transform = transform

        self._load_data()


    def __getitem__(self, index):
        path = self.paths[index]
        name = os.path.basename(os.path.normpath(path))

        img = self._load_image(path)

        if self.with_masks:
            masks = self._load_masks(path)
            classes = self._generate_classes(masks)

            if self.augment is not None:
                img, masks, classes = self.augment(img, masks, classes)

            boxes = self._generate_boxes(masks)

            if self.transform is not None:
                # copy is required to avoid negative strides
                # which is not supported by torch.Tensor
                img = self.transform(img.copy())
                masks = self.transform(masks.copy())
                boxes = self.transform(boxes.copy())
                classes = self.transform(classes.copy())

            return img, masks, boxes, classes, name

        if self.augment is not None:
            img = self.augment(img)

        if self.transform is not None:
            img = self.transform(img.copy())

        return img, name


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


    def _generate_classes(self, masks):
        # TODO: support multi-class
        num_masks = masks.shape[-1]
        return np.array([1]*num_masks)


    def _generate_boxes(self, masks):
        # TODO: support multi-class
        num_masks = masks.shape[-1]
        h, w = masks.shape[:2]
        boxes = np.zeros((num_masks,4),dtype=np.int32)

        for i in range(num_masks):
            m = masks[:,:,i]
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # To fully contain the mask
                x1 -= 1 if x1 > 0 else 0
                y1 -= 1 if y1 > 0 else 0
                x2 += 1 if x2 < w else 0
                y2 += 1 if y2 < h else 0
            else:
                raise RuntimeError("invalid mask encountered")

            boxes[i] = np.array([x1, y1, x2, y2])

        return boxes


    def _load_data(self):
        path = os.path.join(self.root, '*')
        self.paths = glob(path)
        if not self.paths:
            raise RuntimeError("{} is empty.".format(self.root))


    def __repr__(self):
        size = self.__len__()
        fmt_str = '{} Dataset with {} datapoints located at {}'.format(self.__class__.__name__, size, self.root)
        return fmt_str

