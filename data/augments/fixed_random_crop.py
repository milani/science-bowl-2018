import math
import imgaug as ia
import six.moves as sm
from imgaug.augmenters.meta import Augmenter
from imgaug.augmenters import Scale, CropAndPad
from imgaug.parameters import Deterministic, DiscreteUniform

def do_assert(condition, message="Assertion failed."):
    if not condition:
        raise AssertionError(str(message))

class FixedRandomCrop(Augmenter):
    """Crops a fixed size box from an image. If the image is smaller
    than the required size, it is scaled so that the smaller dimension is equal
    to the required size.
    """
    def __init__(self, size=256, interpolation='linear', name=None, deterministic=False, random_state=None):
        super(FixedRandomCrop,self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        def handle(val):
            if ia.is_single_integer(val):
                do_assert(val > 0)
                return (val, val)
            elif ia.is_single_float(val):
                do_assert(val > 0)
                return (val, val)
            elif isinstance(val, tuple):
                do_assert(len(val) == 2)
                do_assert(val[0] > 0 and val[1] > 0)
                return val
            else:
                raise Exception("Expected size to be int, float or tuple, got %s." % type(size))

        self.size = handle(size)
        # px will be changed later during augmentation
        self.crop = CropAndPad(px=(-1,-1,-1,-1), keep_size=False, name=name, deterministic=deterministic, random_state=random_state)
        # size will be changed later during augmentation
        self.scale = Scale(size={"width":self.size[0], "height":self.size[1]}, name=name, deterministic=deterministic,
                random_state=random_state, interpolation=interpolation)


    def get_parameters(self):
        return [self.size]


    def _augment_images(self, images, random_state, parents, hooks):
        size = self.size
        result = []
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i]
            image = self._scale_if_necessary(image, random_state, parents, hooks)
            h, w = image.shape[:2]

            top_px, left_px, bottom_px, right_px = self._crop_points(size[0], size[1], w, h, seeds[i])

            self.crop.top = Deterministic(top_px)
            self.crop.left = Deterministic(left_px)
            self.crop.bottom = Deterministic(bottom_px)
            self.crop.right = Deterministic(right_px)
            cropped = self.crop._augment_images([image], random_state, parents, hooks)
            result += cropped

        return result


    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        size = self.size
        result = []
        nb_images = len(keypoints_on_images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            keypoint = keypoints_on_images[i]
            keypoint = self._scale_if_necessary(keypoint, random_state, parents, hooks)
            h, w = keypoint.shape[:2]

            top_px, left_px, bottom_px, right_px = self._crop_points(size[0], size[1], w, h, seeds[i])

            self.crop.top = Deterministic(top_px)
            self.crop.left = Deterministic(left_px)
            self.crop.bottom = Deterministic(bottom_px)
            self.crop.right = Deterministic(right_px)
            cropped = self.crop._augment_keypoints([keypoint], random_state, parents, hooks)
            result += cropped

        return result


    def _crop_points(self, box_w, box_h, img_w, img_h, seed):
        center_x, center_y = self._draw_sample_point(box_w, box_h, img_w, img_h, seed)
        top_crop = -(center_y - box_h / 2)
        bottom_crop = -(img_h - (center_y + box_h / 2))
        left_crop = -(center_x - box_w / 2)
        right_crop = -(img_w - (center_x + box_w / 2))

        return int(top_crop), int(left_crop), int(bottom_crop), int(right_crop)


    def _draw_sample_point(self, box_w, box_h, img_w, img_h, seed):
        x1 = math.ceil(box_w / 2)
        y1 = math.ceil(box_h / 2)
        x2 = math.floor(img_w - x1)
        y2 = math.floor(img_h - y1)

        random_state = ia.new_random_state(seed)
        x = DiscreteUniform(x1, x2)
        y = DiscreteUniform(y1, y2)

        center_x = x.draw_sample(random_state=random_state)
        center_y = y.draw_sample(random_state=random_state)

        return center_x, center_y


    def _scale_if_necessary(self, obj, random_state, parents, hooks):
        h, w = obj.shape[:2]
        box_w, box_h = self.size
        ratio = max(box_w/w, box_h/h)

        if ratio <= 1:
            return obj

        self.scale.size = (Deterministic(math.ceil(h*ratio)), Deterministic(math.ceil(w*ratio)))

        if isinstance(obj, ia.Keypoint):
            obj = self.scale._augment_keypoints([obj], random_state, parents, hooks)
        else:
            obj = self.scale._augment_images([obj], random_state, parents, hooks)
        return obj[0]
