from maskmm.utils import box_utils
import torch
import numpy as np
from maskmm.mytools import *

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
    device = config.DEVICE

    #torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # todo move earlier
    anchors = torch.Tensor(anchors).to(device)
    gt_class_ids = torch.Tensor(gt_class_ids).to(device)
    gt_boxes = torch.Tensor(gt_boxes).to(device)

    # todo move padding to end??
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = torch.zeros([anchors.shape[0]], dtype=torch.int32, device=device)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = torch.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4), device=device)

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
        crowd_overlaps = box_utils.torch_compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1, dtype=torch.uint8, device=device)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = torch.ones([anchors.shape[0]], dtype=torch.uint8, device=device)

    # Compute overlaps [num_anchors, num_gt_boxes]
    save(anchors, "anchors_pre")
    save(gt_boxes, "gt_boxes_pre")
    overlaps = box_utils.torch_compute_overlaps(anchors, gt_boxes)
    save(overlaps, "overlaps")

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

    anchor_iou_max, anchor_iou_argmax = torch.max(overlaps, dim=1)
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    save(rpn_match, "test1")

    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = torch.argmax(overlaps, dim=0)
    rpn_match[gt_iou_argmax] = 1
    save(rpn_match, "test2")

    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1
    save(rpn_match, "test3")

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = rpn_match.eq(1).nonzero().squeeze(-1)
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        if config.NPRANDOM:
            ids = np.random.choice(ids, extra, replace=False)
        else:
            ids = ids[torch.randperm(len(ids))][:extra]
        rpn_match[ids] = 0

    # Same for negative proposals
    ids = rpn_match.eq(-1).nonzero().squeeze(-1)
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - rpn_match.eq(1).sum())
    if extra > 0:
        # Rest the extra ones to neutral
        if config.NPRANDOM:
            ids = np.random.choice(ids, extra.item(), replace=False)
        else:
            ids = ids[torch.randperm(len(ids))][:extra]
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = rpn_match.eq(1).nonzero().squeeze(-1)

    data = box_utils.box_refinement(anchors[ids], gt_boxes[anchor_iou_argmax[ids]])

    # todo is padding needed? included to enable match with original
    rpn_bbox = torch.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4), dtype=torch.float, device=device)
    rpn_bbox[:len(data)] = data

    # Normalize
    rpn_bbox /= torch.tensor(config.RPN_BBOX_STD_DEV, dtype=torch.float, device=device)

    return rpn_match, rpn_bbox
