import numpy as np
import scipy.misc, scipy.ndimage
import torch
from skimage.transform import rotate, warp, AffineTransform
import logging
log = logging.getLogger()

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

def mold_meta(meta):
    """ flatten dict values """
    out = []
    for x in meta.values():
        out.extend(x)
    return torch.tensor(out)

def unmold_meta(meta):
    meta = list(meta.cpu().numpy())
    return dict(window=meta[:4])

def mold_image(image, config):
    """ Prepares RGB image with 0-255 values for input to model
    """
    if config.COMPAT:
        image = torch.tensor(image, dtype=torch.double)
        image = image - torch.tensor(config.MEAN_PIXEL, dtype=torch.double)
    else:
        image = image - config.MEAN_PIXEL
        image = torch.tensor(image)
    # channel first
    image = image.permute(2, 0, 1).float()
    return image

def unmold_image(image, config):
    """ reverses mold_image """
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image + config.MEAN_PIXEL).astype(np.uint8)
    return image

def resize_image(image, config):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    min_dim = min(config.IMAGE_SHAPE)
    max_dim = max(config.IMAGE_SHAPE)
    padding = config.IMAGE_PADDING

    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


############################## mask ########################################


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask

def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

######### image and mask

def augment(img, masks=None):
    """ augment image and masks """

    # use same seed for image and masks for identical transforms
    seed = np.random.randint(1e6)

    img = augment_image(img, seed)
    if masks is not None:
        for i in range(masks.shape[2]):
            masks[:, :, i] = augment_image(masks[:, :, i], seed)
        # remove empty masks
        masks = masks[:, :, np.any(masks!=0, axis=(0,1))]
    return img, masks

def augment_image(img, vflip=.5, hflip=.5, angle=360, shear=.3, seed=np.random.randint(1e6)):
    """ apply random transformations to an image
    vflip/hflip: probabilities
    angle/shear: maximums
    seed: set to apply same transforms on multiple images
    """
    np.random.seed(seed)

    vflip = np.random.random() > vflip
    hflip = np.random.random() > hflip
    angle = np.random.random() * angle
    shear = np.random.random() * shear

    if vflip:
        img = np.flip(img, 0)
    if hflip:
        img = np.flip(img, 1)
    img = rotate(img, angle)
    img = warp(img, inverse_map=AffineTransform(shear=shear))
    img = (img * 255).astype(np.uint8)
    # log.info(f"hflip={hflip}, vflip={vflip}, angle={angle:.0f}, shear={shear:.2f}")

    return img

def unmold_detections(boxes, class_ids, scores, masks, image_shape, image_meta):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)]
    mrcnn_mask: [N, height, width, num_classes]
    image_shape: [height, width, depth] Original size of the image before resizing
    window: [y1, x1, y2, x2] Box in the image where the real image is
            excluding the padding.

    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    """
    # strip padding
    ix = class_ids.ne(0).nonzero()[:, 0].unique()
    boxes, class_ids, scores, masks = [var[ix] for var in [boxes, class_ids, scores, masks]]

    masks = masks.permute(0, 2, 3, 1)

    # select relevant class_id
    masks = masks[range(len(masks)), :, :, class_ids.long()]
    window = torch.tensor(unmold_meta(image_meta)["window"]).float()

    # Compute scale and shift to translate coordinates to image domain.
    h_scale = image_shape[0] / (window[2] - window[0])
    w_scale = image_shape[1] / (window[3] - window[1])
    scale = torch.tensor(min(h_scale, w_scale))
    shift = window[:2]  # y, x
    scales = torch.tensor([scale, scale, scale, scale])
    shifts = torch.tensor([shift[0], shift[1], shift[0], shift[1]])
    boxes = (boxes - shifts) * scales

############### convert to numpy before using skimage and for output ########################################
    boxes = boxes.long().cpu().numpy()
    class_ids = class_ids.cpu().numpy()
    scores = scores.cpu().numpy()
    masks = masks.cpu().numpy()
    image_shape = image_shape.cpu().numpy()

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(len(class_ids)):
        # Convert neural network mask to full size mask
        full_mask = unmold_mask(masks[i], boxes[i], image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty((0,) + masks.shape[1:3])

    return boxes, class_ids, scores, full_masks