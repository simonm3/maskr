from maskmm.utils import box_utils
import torch
from torch import tensor
import numpy as np
import logging
log = logging.getLogger()
from maskmm.tracker import save, saveall, get_type

@saveall
def build_rpn_targets(anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # strip the zeros (else allocates an anchor to them)
    ids = gt_class_ids.nonzero().squeeze(-1)
    gt_class_ids = gt_class_ids[ids]
    gt_boxes = gt_boxes[ids]

    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = torch.zeros([anchors.shape[0]]).int()
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = torch.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = gt_class_ids.lt(0).nonzero()
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = gt_class_ids.gt(0).nonzero()
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = box_utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1, dtype=torch.uint8)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = torch.ones([anchors.shape[0]], dtype=torch.uint8)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = box_utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    if config.COMPAT:
        anchor_iou_max = tensor(np.max(overlaps.cpu().numpy(), axis=1))
        anchor_iou_argmax = tensor(np.argmax(overlaps.cpu().numpy(), axis=1)).long()
    else:
        anchor_iou_max, anchor_iou_argmax = torch.max(overlaps, dim=1)

    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1

    # 2. Set an anchor for each GT box (regardless of IoU value).
    if config.COMPAT:
        gt_iou_argmax = tensor(np.argmax(overlaps.cpu().numpy(), axis=0)).long()
    else:
        gt_iou_argmax = torch.argmax(overlaps, dim=0)
    rpn_match[gt_iou_argmax] = 1

    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = rpn_match.eq(1).nonzero().squeeze(-1)
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        if config.COMPAT:
            ids = tensor(np.random.choice(ids.cpu().numpy(), extra, replace=False)).long()
        else:
            ids = ids[torch.randperm(len(ids))][:extra]
        rpn_match[ids] = 0

    # Same for negative proposals
    ids = rpn_match.eq(-1).nonzero().squeeze(-1)
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - rpn_match.eq(1).sum())
    if extra > 0:
        # Rest the extra ones to neutral
        if config.COMPAT:
            ids = tensor(np.random.choice(ids.cpu().numpy(), extra.item(), replace=False)).long()
        else:
            ids = ids[torch.randperm(len(ids))][:extra]
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = rpn_match.eq(1).nonzero().squeeze(-1)

    # boxes
    boxes = box_utils.box_refinement(anchors[ids], gt_boxes[anchor_iou_argmax[ids]])
    rpn_bbox[:len(boxes)] = boxes

    # Normalize
    rpn_bbox /= torch.tensor(config.RPN_BBOX_STD_DEV, dtype=torch.float)

    return rpn_match, rpn_bbox
