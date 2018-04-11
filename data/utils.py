import torch
import torchvision.transforms.functional as F

class ToTensor(object):
    """Converts a ``numpy.ndarray`` to tensor.

    If ndim == 3, it is considered an image and this transform works as its torchvision's counterpart
    If ndim < 2, it simply converts the numpy to torch.Tensor.
    """
    def __call__(self, array):
        """
        Args:
            array: numpy.ndarray of ndim < 3

        Returns:
            Tensor: converted numpy.ndarray
        """
        if array.ndim == 3:
            return self.to_tensor(array)
        else:
            return torch.from_numpy(array).float()

    def to_tensor(self, img):
        img = torch.from_numpy(img.transpose((2, 0, 1)))

        if isinstance(img, torch.ByteTensor) and float(img.max()) > 1:
            return img.float().div(255)

        return img.float()
