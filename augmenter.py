from imgaug import augmenters as iaa

class Compose(object):
    """Composes several augmenters together.
    Unlike original pytorch implementation, it accepts a second argument (of mask type).

    Args:
        augmenters: list of imgaug.augmenters to compose.

    """
    def __init__(self, augmenters):
        assert type(augmenters) == list, 'augmenters should be of type `list`'
        self.augmenters = iaa.Sequential(augmenters)


    def __call__(self, img, mask=None):
        if mask is not None:
            aug_det = self.augmenters.to_deterministic()
            return aug_det(img), aug_det(mask)

        return self.augmenters.augment_images(img)

def Sometimes(aug):
    return iaa.Sometimes(0.5, aug)


