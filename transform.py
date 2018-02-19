import torchvision.transforms.functional as F
import torch

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
            return F.to_tensor(array)
        else:
            return torch.from_numpy(array).float()
