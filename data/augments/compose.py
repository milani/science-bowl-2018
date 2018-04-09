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


    def __call__(self, img, masks=None, classes=None, boxes=None):
        if masks is not None:
            returns = self.augment(img, masks, classes, boxes)
            while returns[1].shape[-1] == 0:
                returns = self.augment(img, masks, classes, boxes)
            return returns
        return self.augmenters.augment_image(img)


    def augment(self, img, masks=None, classes=None, boxes=None):
        returns = []
        aug_det = self.augmenters.to_deterministic()

        # augment image
        input_size = img.shape
        img = aug_det.augment_image(img)
        returns.append(img)

        # augment masks
        new_masks = aug_det.augment_image(masks)
        null_masks = new_masks.sum(axis=0).sum(axis=0) == 0
        new_masks = new_masks[:,:,~null_masks]
        returns.append(new_masks)

        # if removed any mask, remove corresponding class
        if classes is not None:
            classes = classes[~null_masks]
            returns.append(classes)

        if boxes is not None:
            # augment boxes
            bboxes = BoundingBoxesOnImage([BoundingBox(*box) for box in boxes], input_size)
            new_bboxes = aug_det.augment_bounding_boxes([bboxes])[0]
            new_bboxes = new_bboxes.remove_out_of_image().cut_out_of_image()
            boxes = [[box.x1, box.y1, box.x2, box.y2] for box in new_bboxes.bounding_boxes]
            boxes = np.array(boxes)
            returns.append(boxes)
        return tuple(returns)

