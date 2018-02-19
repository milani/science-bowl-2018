"""Configurations

IMAGE_MIN_DIM: The shortest dimension of the image will be scaled to match this number, keeping aspect ratio.
IMAGE_MAX_DIM: The final image size won't exceed this number. Depending on the augmentation, if the image is
            not fully contained, it is either cropped or rescaled to fit.
"""
IMAGE_MIN_DIM=256
IMAGE_MAX_DIM=500

