from imgaug.augmenters import Sometimes
from .compose import Compose
from .fixed_resize_and_pad import FixedResizeAndPad

def Sometimes(aug):
    return Sometimes(0.5, aug)


__all__ = ['Compose', 'FixedResizeAndPad', 'Sometimes']
