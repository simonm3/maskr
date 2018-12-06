import torch
import numpy as np
from maskr.utils import box_utils
from maskr.lib.nms.nms_wrapper import nms
from maskr.utils.batch import batch_slice

from maskr.test.baseline import saveall
import logging
log = logging.getLogger()

@saveall
@batch_slice(2)
def proposals(rpn_class, rpn_bbox, proposal_count, anchors, config):
    """Receives anchor scores and selects a subset to pass as proposals
       to the second stage. Filtering is done based on anchor scores and
       non-max suppression to remove overlaps. It also applies bounding
       box refinement to anchors.

       Inputs:
           rpn_class: [batch, anchors, (bg prob, fg prob)]
           rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

       Returns:
           Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
       """
    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    rpn_class = rpn_class[:, 1]

    # standardise
    std_dev = torch.tensor(config.RPN_BBOX_STD_DEV).float().reshape([1,4])
    rpn_bbox = rpn_bbox * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, len(anchors))
    rpn_class, order = rpn_class.sort(descending=True)
    order = order[:pre_nms_limit]
    rpn_class = rpn_class[:pre_nms_limit]
    rpn_bbox = rpn_bbox[order, :]
    anchors = anchors[order, :]

    # Apply deltas to anchors to get refined anchors.
    boxes = box_utils.apply_box_deltas(anchors, rpn_bbox)

    # Clip to image boundaries
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = box_utils.clip_to_window(window, boxes)

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    keep = nms(torch.cat((boxes, rpn_class.unsqueeze(1)), dim=1), config.RPN_NMS_THRESHOLD)
    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    norm = torch.tensor([height, width, height, width]).float()
    rpn_rois = boxes / norm

    return rpn_rois
