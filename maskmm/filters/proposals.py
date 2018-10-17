import torch
import numpy as np
from maskmm.utils import box_utils
from maskmm.lib.nms.nms_wrapper import nms
import logging
log = logging.getLogger()

def proposals(inputs, proposal_count, nms_threshold, anchors, config):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """
    # todo fudge to get working
    anchors = torch.Tensor(anchors).cuda()

    with torch.no_grad():
        # Currently only supports batchsize 1
        inputs[0] = inputs[0].squeeze(0)
        inputs[1] = inputs[1].squeeze(0)

        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, 1]

        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float()
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas = deltas * std_dev

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(6000, len(anchors))
        scores, order = scores.sort(descending=True)
        order = order[:pre_nms_limit]
        scores = scores[:pre_nms_limit]
        deltas = deltas[order.data, :] # TODO: Support batch size > 1 ff.
        anchors = anchors[order.data, :]

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = box_utils.apply_box_deltas(anchors, deltas)

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        height, width = config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        boxes = box_utils.clip_to_window(window, boxes)

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
        keep = keep[:proposal_count]
        boxes = boxes[keep, :]

        # Normalize dimensions to range of 0 to 1.
        norm = torch.from_numpy(np.array([height, width, height, width])).float()
        if config.GPU_COUNT:
            norm = norm.cuda()
        normalized_boxes = boxes / norm

        # Add back batch dimension
        normalized_boxes = normalized_boxes.unsqueeze(0)

        return normalized_boxes