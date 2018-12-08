import os
import re

import torch
import torch.nn as nn
import torch.utils.data
from torch import from_numpy
import skimage
import numpy as np

from maskr.utils import image_utils
from maskr.utils.batch import batch_slice, unpack

from maskr.datagen.head_targets import build_head_targets
from maskr.filters.proposals import proposals
from maskr.filters.detections import detections
from maskr.filters.roialign import roialign

from .rpn import RPN
from .resnet import ResNet
from .resnetFPN import FPN
from .head import Classifier, Mask

import logging
log = logging.getLogger()


class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super().__init__()
        self.config = config

        # must be on same device as model
        self.anchors = from_numpy(config.ANCHORS)

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # backbone
        resnet = ResNet("resnet101", stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()

        # feature pyramid
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        # head Classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # head Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

    def forward(self, *inputs):

        targets = len(inputs)>2

        # split into two paths for training and prediction
        if targets:
            # training/validation mode. inputs from dataloader/dataset
            # tgt_rpn_match and tgt_rpn_bbox not used but passed through because.....
            # fastai loss is calculated in callback to store results but this has single param output of this function.
            # fastai loss_func is not able to store intermediate results
            images, image_metas, \
            tgt_rpn_match, tgt_rpn_bbox, \
            gt_class_ids, gt_boxes, gt_masks = inputs

            # remove any images without any gt_class_ids
            # todo is there a cleaner way to reject an image? unclean to zero pad then remove like this
            keep = [i for i in range(len(gt_class_ids)) if gt_class_ids[i].gt(0).any()]
            images = images[keep]
            image_metas = image_metas[keep]
            tgt_rpn_match = tgt_rpn_match[keep]
            tgt_rpn_bbox = tgt_rpn_bbox[keep]
            gt_class_ids = gt_class_ids[keep]
            gt_boxes = gt_boxes[keep, :]
            gt_masks = gt_masks[keep,:]

            # fastai databunch/devicedataloader only support tensors
            # zero padding was added to allow tensors to be stacked.
            # after unpack the batch dimension is a list with one tensor per item
            gt_class_ids, gt_boxes, gt_masks = unpack([gt_class_ids, gt_boxes, gt_masks])
        else:
            images, image_metas = inputs

        config = self.config

        # Feature extraction
        feature_maps = self.fpn(images)

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in feature_maps:
            layer_outputs.append(self.rpn(p))

        # last feature map not used for classifier/mask head
        feature_maps = feature_maps[:-1]

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        ############################################################################
        # from here onwards each function uses batch_slice
        # splits the batch, iterates over items, returns list of results
        # batch dimension is the list
        ############################################################################

        # Generate proposals [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        proposal_count = config.POST_NMS_ROIS_TRAINING if targets else config.POST_NMS_ROIS_INFERENCE
        rois = proposals(rpn_class, rpn_bbox, proposal_count,
                         self.anchors.to(config.DEVICE),
                         config=config)

        if not config.HEAD:
            return dict(out=[tgt_rpn_match, tgt_rpn_bbox, \
                             rpn_class_logits, rpn_bbox, 0,0,0,0,0,0])

        if targets:
            # Filter proposals to get target proportion of positive rois; and get targets
            with torch.no_grad():
                rois, target_class_ids, target_deltas, target_mask = \
                    build_head_targets(rois, gt_class_ids, gt_boxes, gt_masks, config)

            # class head
            x = roialign(rois, *feature_maps, config.POOL_SIZE, config.IMAGE_SHAPE)
            mrcnn_class_logits, mrcnn_probs, mrcnn_deltas = batch_slice(1)(self.classifier)(x)

            # mask head
            x = roialign(rois, *feature_maps, config.MASK_POOL_SIZE, config.IMAGE_SHAPE)
            mrcnn_mask = batch_slice(1)(self.mask)(x)

            return dict(out=[tgt_rpn_match, tgt_rpn_bbox, \
                             rpn_class_logits, rpn_bbox, \
                             target_class_ids, target_deltas, target_mask, \
                             mrcnn_class_logits, mrcnn_deltas, mrcnn_mask])
        else:
            # class head
            x = roialign(rois, *feature_maps, config.POOL_SIZE, config.IMAGE_SHAPE)
            mrcnn_class_logits, mrcnn_probs, mrcnn_deltas = batch_slice()(self.classifier)(x)

            # detections filter speeds inference and improves accuracy (see maskrcnn paper)
            #### putting this after mask head is much worse!!!
            # boxes are image domain for output. rois are as above but filtered i.e. suitable for mask head.
            boxes, class_ids, scores, rois = batch_slice(4)\
                            (detections)(rois, mrcnn_probs, mrcnn_deltas, image_metas, config)

            # mask head
            x = roialign(rois, *feature_maps, config.MASK_POOL_SIZE, config.IMAGE_SHAPE)
            masks = batch_slice()(self.mask)(x)

            return boxes, class_ids, scores, masks

    def predict(self, images):
        """ predict list of images without targets, bypassing dataset """
        if not isinstance(images, list):
            images = [images]

        # setup model
        self.eval()
        if self.config.DEVICE=="cuda":
            self.cuda()
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            self.cpu()
            torch.set_default_tensor_type(torch.FloatTensor)

        # dataset
        molded_images = []
        image_shapes = []
        image_metas = []
        for image in images:
            # If grayscale or rgba then convert to RGB for consistency (load_image)
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image) * 255
            elif image.shape[-1] == 4:
                image = skimage.color.rgba2rgb(image) * 255
            image = image.astype(np.uint8)

            # image and shape (__get_item__
            image_shapes.append(torch.tensor(image.shape))
            image, window, scale, padding = image_utils.resize_image(image, self.config)
            image = image_utils.mold_image(image, self.config)
            molded_images.append(image)

            # image meta
            image_meta = image_utils.mold_meta(dict(window=window))
            image_metas.append(image_meta)

        # dataloader
        molded_images = torch.stack(molded_images)
        image_metas = torch.stack(image_metas)
        image_shapes = torch.stack(image_shapes)

         # predict
        with torch.no_grad():
            boxes, class_ids, scores, masks = self(molded_images, image_metas)

        # prepare outputs
        results = []
        for i in range(len(images)):
            detections1 = [var[i] for var in [boxes, class_ids, scores, masks, image_shapes, image_metas]]
            results.append(image_utils.unmold_detections(*detections1))
        return results

    def initialize_weights(self):
        """Initialize model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")