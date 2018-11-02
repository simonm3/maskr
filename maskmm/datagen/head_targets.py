import torch
from maskmm.lib.roialign.roi_align.crop_and_resize import CropAndResizeFunction
from maskmm.utils import box_utils, utils
import numpy as np
from maskmm.tracker import save, saveall
from maskmm.utils.utils import batch_slice

import logging
log = logging.getLogger()

@batch_slice
def build_head_targets(inputs, config):
    """ Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """
    proposals, gt_class_ids, gt_boxes, gt_masks = inputs

    # Normalize coordinates
    h, w = config.IMAGE_SHAPE[:2]
    scale = torch.tensor([h, w, h, w]).float()
    gt_boxes = gt_boxes / scale

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    if len(torch.nonzero(gt_class_ids < 0)):
        crowd_ix = torch.nonzero(gt_class_ids < 0)[:, 0]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
        crowd_boxes = gt_boxes[crowd_ix.data, :]
        crowd_masks = gt_masks[crowd_ix.data, :, :]
        gt_class_ids = gt_class_ids[non_crowd_ix.data]
        gt_boxes = gt_boxes[non_crowd_ix.data, :]
        gt_masks = gt_masks[non_crowd_ix.data, :]

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = box_utils.compute_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = torch.tensor(len(proposals) * [True], dtype=torch.uint8)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = box_utils.compute_overlaps(proposals, gt_boxes)

    # Determine postive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs.
    if len(torch.nonzero(positive_roi_bool)):
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(len(positive_indices))
        rand_idx = rand_idx[:positive_count]
        positive_indices = positive_indices[rand_idx]
        positive_count = len(positive_indices)
        positive_rois = proposals[positive_indices.data, :]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data, :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data, :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        # Compute bbox refinement for positive ROI
        deltas = box_utils.box_refinement(positive_rois.data, roi_gt_boxes.data)
        std_dev = torch.tensor(config.BBOX_STD_DEV).float()
        deltas /= std_dev

        # Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data, :, :]

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = torch.arange(len(roi_masks)).int()
        masks = CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0) \
            (roi_masks.unsqueeze(1), boxes, box_ids)
        masks = masks.squeeze(1)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks)
    else:
        positive_count = 0

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if len(torch.nonzero(negative_roi_bool)) and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(len(negative_indices))
        rand_idx = rand_idx[:negative_count]
        negative_indices = negative_indices[rand_idx]
        negative_count = len(negative_indices)
        negative_rois = proposals[negative_indices.data, :]
    else:
        negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = torch.zeros(negative_count).float()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = torch.zeros(negative_count, 4)
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1])
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = torch.zeros(negative_count)
        roi_gt_class_ids = zeros
        zeros = torch.zeros(negative_count, 4).int()
        deltas = zeros
        zeros = torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1])
        masks = zeros
    else:
        rois = torch.empty(0)
        roi_gt_class_ids = torch.empty(0)
        deltas = torch.empty(0)
        masks = torch.empty(0)

    rois = utils.pad(rois, config.TRAIN_ROIS_PER_IMAGE)
    roi_gt_class_ids = utils.pad(roi_gt_class_ids, config.TRAIN_ROIS_PER_IMAGE)
    deltas = utils.pad(deltas, config.TRAIN_ROIS_PER_IMAGE)
    masks = utils.pad(masks, config.TRAIN_ROIS_PER_IMAGE)

    return rois, roi_gt_class_ids, deltas, masks
