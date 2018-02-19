from imgaug.augmenters import Scale, Pad
from imgaug.augmenters.meta import Augmenter
from imgaug.parameters import Deterministic
from math import floor
import six.moves as sm

class FixedResizeAndPad(Augmenter):
    """Resizes the image so that the smaller dimension == min_dim
    and larger dimension should not exceed max_dim, otherwise scales
    the image so that it is contained in the final size.

    min_dim: int(default=256)
            The smaller dimension of the image will be scaled to min_dim

    max_dim: int(default=640)
            The dimensions of the image should not exceed this size. It also defines
            the final size of the image (max_dim x max_dim)

    name: string, optional(default=None)
            see `imgaug.Augmenter.__init__()`

    deterministic : bool, optional(default=False)
            see `imgaug.Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
            see `imgaug.Augmenter.__init__()`

    pad_mode: ia.ALL or string or list of strings or StochasticParameter, optional(default="constant")
            see `imgaug.Augmenters.CropAndPad`

    pad_cval : float or int or tuple of two ints/floats or list of ints/floats or StochasticParameter, optional(default=0)
            see `imgaug.Augmenters.CropAndPad`
    """
    def __init__(
            self, min_dim=256, max_dim=640,
            name=None, deterministic=False, random_state=None,
            pad_mode='constant', pad_cval=0, interpolation='linear'
    ):
        super(FixedResizeAndPad,self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.min_dim = min_dim
        self.max_dim = max_dim
        # size will be changed later in augmentation
        self.scale = Scale(size=0.2, name=name, deterministic=deterministic,
                random_state=random_state, interpolation=interpolation)
        # px will be changed later in augmentation
        self.pad = Pad(
                px=(0,0,0,0),
                pad_mode=pad_mode,
                pad_cval=pad_cval,
                name=name,
                deterministic=deterministic,
                random_state=random_state,
                keep_size=False
        )


    def get_parameters(self):
        return [self.min_dim, self.max_dim]


    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        for i in sm.xrange(nb_images):
            image = images[i]
            h, w = image.shape[:2]
            scale, top_pad, right_pad, bottom_pad, left_pad = self._calculate_scale_pad(h, w)

            self.scale.size = Deterministic(scale)
            self.pad.top = Deterministic(top_pad)
            self.pad.right = Deterministic(right_pad)
            self.pad.bottom = Deterministic(bottom_pad)
            self.pad.left = Deterministic(left_pad)

            scaled = self.scale._augment_images([image], random_state, parents, hooks)
            padded = self.pad._augment_images(scaled, random_state, parents, hooks)
            result += padded

        return result


    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        for i in sm.xrange(nb_images):
            keypoint = keypoints_on_images[i]
            h, w = keypoint.shape[:2]
            scale, top_pad, right_pad, bottom_pad, left_pad = self._calculate_scale_pad(h, w)

            self.scale.size = Deterministic(scale)
            self.pad.top = Deterministic(top_pad)
            self.pad.right = Deterministic(right_pad)
            self.pad.bottom = Deterministic(bottom_pad)
            self.pad.left = Deterministic(left_pad)

            scaled = self.scale._augment_keypoints([keypoint], random_state, parents, hooks)
            padded = self.pad._augment_keypoints(scaled, random_state, parents, hooks)
            result += padded

        return result


    def _calculate_scale_pad(self, h, w):
        min_dim = self.min_dim
        max_dim = self.max_dim
        image_max = max(h, w)
        scale = max(1.0, min_dim / min(h, w))

        if floor(image_max * scale) > max_dim:
            scale = max_dim / image_max

        new_h, new_w = floor(scale * h), floor(scale * w)

        top_pad = (max_dim - new_h) // 2
        bottom_pad = max_dim - new_h - top_pad
        left_pad = (max_dim - new_w) // 2
        right_pad = max_dim - new_w - left_pad

        return scale, top_pad, right_pad, bottom_pad, left_pad

