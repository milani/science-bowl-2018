import numpy as np
from imgaug import BoundingBox, BoundingBoxesOnImage
from imgaug.augmenters import Sequential

class Compose(object):
    """Composes several augmenters together.
    Unlike original pytorch implementation, it accepts a second argument (of mask type).

    Args:
        augmenters: list of imgaug.augmenters to compose.

    """
    def __init__(self, augmenters):
        assert type(augmenters) == list, 'augmenters should be of type `list`'
        self.augmenters = Sequential(augmenters)


    def __call__(self, img, mask=None, boxes=None):
        if mask is not None:
            aug_det = self.augmenters.to_deterministic()
            bboxes = BoundingBoxesOnImage([BoundingBox(*box) for box in boxes], img.shape)
            new_bboxes = aug_det.augment_bounding_boxes([bboxes])[0]
            boxes = [[box.x1, box.y1, box.x2, box.y2] for box in new_bboxes.bounding_boxes]
            return aug_det.augment_image(img), aug_det.augment_image(mask), np.array(boxes)

        return self.augmenters.augment_image(img)

