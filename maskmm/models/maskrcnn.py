import datetime
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from maskmm.utils import image_utils

from maskmm.datagen.head_targets import build_head_targets
from maskmm.filters.proposals import proposals
from maskmm.filters.detections import filter_detections
from maskmm.filters.roialign import roialign

from .rpn import RPN
from .resnet import ResNet
from .resnetFPN import FPN
from .head import Classifier, Mask

from maskmm.tracker import save

import logging
log = logging.getLogger()


class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super().__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the head (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()

        # Top-down Layers
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        # FPN Classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # FPN Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

    def forward(self, *inputs):
        targets = len(inputs)>2

        if targets:
            # training/validation mode. inputs from dataloader/dataset
            # tgt_rpn_match and tgt_rpn_bbox not used but passed through because.....
            # fastai loss is calculated in callback to store results but this has single param output of this function.
            # fastai loss_func is not able to store intermediate results
            images, image_metas,\
            tgt_rpn_match, tgt_rpn_bbox, \
            gt_class_ids, gt_boxes, gt_masks = inputs
        else:
            images, image_metas = inputs

        config = self.config

        # Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(images)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if self.training \
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposals([rpn_class, rpn_bbox],
                             proposal_count=proposal_count,
                             config=config)

        if not config.HEAD:
            return dict(out=[tgt_rpn_match, tgt_rpn_bbox, \
                             rpn_class_logits, rpn_bbox, 0,0,0,0,0,0])

        if targets:
            with torch.no_grad():
                # Subsample proposals and generate target outputs for training
                # Note inputs and outputs are zero padded.
                rois, target_class_ids, target_deltas, target_mask = \
                    build_head_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

        if len(rois) == 0:
            mrcnn_class_logits = torch.empty(0)
            mrcnn_probs = torch.empty(0).int()
            mrcnn_bbox = torch.empty(0)
            mrcnn_mask = torch.empty(0)
        else:
            # roialign, merge batch dimension and run classifier head
            x = roialign([rois] + mrcnn_feature_maps, config.POOL_SIZE, config.IMAGE_SHAPE)
            mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = self.classifier(x)

            # roialign, merge batch dimension and run mask head
            x = roialign([rois] + mrcnn_feature_maps, config.MASK_POOL_SIZE, config.IMAGE_SHAPE)
            mrcnn_mask = self.mask(x)

        if targets:
            return dict(out=[tgt_rpn_match, tgt_rpn_bbox,\
                             rpn_class_logits, rpn_bbox,\
                             target_class_ids, target_deltas, target_mask,\
                             mrcnn_class_logits, mrcnn_bbox, mrcnn_mask])
        else:
            return mrcnn_probs, mrcnn_bbox, mrcnn_mask

    def predict(self, image):
        """ predict list of images without targets, bypassing dataset """

        # todo extend to multiple images
        # mold image
        image_shape = image.shape
        image, window, scale, padding = image_utils.resize_image(image, self.config)
        image = image_utils.mold_image(image, self.config)

        # predict
        class_probs, boxes, masks = self(image)
        class_probs = class_probs.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()

        # unmold boxes
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            class_probs = np.delete(class_probs, exclude_ix, axis=0)
            boxes = np.delete(boxes, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_probs.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = image_utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
            full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty((0,) + masks.shape[1:3])

        class_ids = np.argmax(class_probs, axis=1)

        return class_ids, class_probs, boxes, full_masks

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

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir,
                                    f"{self.config.NAME.lower()}{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir,
                                            "mask_rcnn_{}_*epoch*.pth".format(
                                                self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

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

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def detect(self, images):
        """Runs the detection pipeline (on list of images rather than a dataset)

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        config = self.config
        self.eval()
        with torch.no_grad():
            # Mold inputs to format expected by the neural network
            molded_images, image_metas, windows = image_utils.mold_inputs(images, config)

            # Run object detection
            detections, mrcnn_mask = self([molded_images, image_metas], mode="detection")

            # Convert to numpy
            detections = detections.cpu().numpy()
            mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).cpu().numpy()

            # Process detections
            results = []
            for i, image in enumerate(images):
                final_rois, final_class_ids, final_scores, final_masks = \
                    image_utils.unmold_detections(detections[i], mrcnn_mask[i],
                                                  image.shape, windows[i])
                results.append({
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                })
            return results
