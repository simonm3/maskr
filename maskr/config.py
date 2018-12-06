"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import numpy as np
import os
from os.path import join
import torch
from maskr.datagen.anchors import generate_pyramid_anchors
import logging
log = logging.getLogger()


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

##### datagen ##################################################################

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Images are resized to >= min and <=max. if cant do both then max is enforced
    IMAGE_SHAPE = [1024, 1024]

    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = [123.7, 116.8, 103.9]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    WORKERS = os.cpu_count()
    BATCH_SIZE = 1
    AUGMENT = False
    SHUFFLE = True

####### training ##################################################################

    # names of weight files
    IMAGENET_MODEL_WEIGHTS = "resnet50_imagenet.pth"
    COCO_MODEL_WEIGHTS = "mask_rcnn_coco.pth"

    # NUMBER OF GPUs to use. For CPU use 0
    GPU_COUNT = torch.cuda.device_count()

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

#### calculated

    ANCHORS = None
    BACKBONE_SHAPES = None
    DEVICE = "cpu"
    WEIGHTS = None

######### backbone ################################################################

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

########## RPN ################################################################

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Bounding box refinement standard deviation for RPN
    RPN_BBOX_STD_DEV = [0.1, 0.1, 0.2, 0.2]

##### roialign ###########################################################################

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

#### detection #########################################################################

    # for final detections
    BBOX_STD_DEV = [0.1, 0.1, 0.2, 0.2]

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

#### development and debugging ##############################################################

    # stay compatible with original for comparison
    # NOTE GPU convolutions do not produce consistent results on same input.
    COMPAT = False

    # if false then run rpn only
    HEAD = True

############################################################################################

    def __init__(self):
        """Set values of computed attributes."""
        if self.GPU_COUNT > 0:
            self.DEVICE = "cuda"
            torch.backends.cudnn.benchmark = True
        else:
            self.DEVICE = "cpu"
            torch.backends.cudnn.benchmark = False

        # default weights is pretrained coco
        self.WEIGHTS = os.path.abspath(join(os.path.dirname(__file__),
                                       os.pardir, "data/models",
                                       self.COCO_MODEL_WEIGHTS))

        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

        # Generate Anchors here as used by dataset and model
        self.ANCHORS = generate_pyramid_anchors(self.RPN_ANCHOR_SCALES,
                                                self.RPN_ANCHOR_RATIOS,
                                                self.BACKBONE_SHAPES,
                                                self.BACKBONE_STRIDES,
                                                self.RPN_ANCHOR_STRIDE)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")