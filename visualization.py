"""Visualization helpers. Heavily influenced by `matterport/Mask_RCNN`
"""
import random
import torch
import numpy as np
import colorsys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
plt.ion()

def display_images(images, cols=4, size=14, channel_first=True):
    """
    Display images ([b x c x h x w]) in a grid
    """
    if torch.is_tensor(images) and channel_first:
        images = np.transpose(images,(0,2,3,1))
    rows = len(images) // cols + 1
    plt.figure(figsize=(size, size * rows // cols))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.imshow(image)

    plt.show()


def display_image_masks(image, masks, title="",
                      figsize=(16, 16),  ax=None, channel_first=True):
    """
    Displays image and its masks. Supports channel first types for both
    image and masks.
    image: either [c, w, h] or [w, h, c]
    masks: [num, w, h]
    """
    if isinstance(image,torch.Tensor):
        image = (255*image.numpy()).astype(np.uint8)
    if isinstance(masks,torch.Tensor):
        masks = (255*masks.numpy()).astype(np.uint8)

    if channel_first:
        image = np.transpose(image,(1,2,0))
    # Number of instances
    N = masks.shape[0]
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.copy()
    for i in range(N):
        color = colors[i]

        # Mask
        mask = masks[i, :, :]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    print(masked_image.shape)
    ax.imshow(masked_image)
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


