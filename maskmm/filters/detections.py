
import torch
from maskmm.utils import box_utils, image_utils
from maskmm.utils.batch import batch_slice
from maskmm.lib.nms.nms_wrapper import nms
import numpy as np

import logging
log = logging.getLogger()

### utils #########################

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = torch.tensor([True], dtype=torch.uint8)
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool]


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1])]


###############################################

@batch_slice(4, 3)
def get_detections(rois, probs, deltas, image_meta, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    window = image_utils.unmold_meta(image_meta)["window"]

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    class_scores = probs[idx, class_ids]
    deltas_specific = deltas[idx, class_ids]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = torch.tensor(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float()
    refined_rois = box_utils.apply_box_deltas(rois, deltas_specific * std_dev)

    unscaled_rois = torch.tensor(refined_rois)

    # Convert coordinates from 0-1 to image scale
    height, width = config.IMAGE_SHAPE[:2]
    scale = torch.tensor(np.array([height, width, height, width])).float()
    refined_rois *= scale

    # Clip boxes to image window
    refined_rois = box_utils.clip_to_window(window, refined_rois)

    # Round and cast to int since we're dealing with pixels now
    refined_rois = torch.round(refined_rois)

    # Filter out boxes with zero area or background
    areas = (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])
    keep_bool = areas.ne(0) & class_ids.ne(0)

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:,0]

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]

    for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
        # Pick detections of this class
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]

        # Sort
        ix_rois = pre_nms_rois[ixs]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order,:]

        class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), config.DETECTION_NMS_THRESHOLD)

        # Map indicies
        class_keep = keep[ixs[order[class_keep]]]

        if i==0:
            nms_keep = class_keep
        else:
            nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)

    # Keep top detections
    top_ids = class_scores[keep].sort(descending=True)[1][:config.DETECTION_MAX_INSTANCES]
    keep = keep[top_ids]

    # apply filter
    boxes = refined_rois[keep]
    class_ids = class_ids[keep]
    scores = class_scores[keep]

    return boxes, class_ids, scores, unscaled_rois[keep]
