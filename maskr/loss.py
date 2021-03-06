import torch
import torch.nn.functional as F
from maskr.test.baseline import saveall
from maskr.utils.batch import batch_slice, pad
import logging
log = logging.getLogger()

@saveall
@batch_slice(2)
def rpn_class(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    if len(anchor_class)==0:
        return None

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = rpn_match.ne(0).nonzero()[:,0]

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices]
    anchor_class = anchor_class[indices]

    loss = F.cross_entropy(rpn_class_logits, anchor_class)
    return loss

@saveall
@batch_slice(3)
def rpn_bbox(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
    rpn_match: [batch, anchors, 1].
                Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = rpn_match.eq(1).nonzero()

    if len(indices)==0:
        return None

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices[:, 0]]

    # Trim target bounding box deltas to the same length as rpn_bbox
    target_bbox = target_bbox[:len(rpn_bbox)]

    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss

@saveall
@batch_slice(2)
def mrcnn_class(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois].
    pred_class_logits: [batch, num_rois, num_classes]
    """
    # add back the background class_ids
    target_class_ids = pad(target_class_ids, len(pred_class_logits))

    if len(target_class_ids)==0:
        return None

    loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    return loss

@saveall
@batch_slice(3)
def mrcnn_bbox(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = target_class_ids[positive_roi_ix].long()
    indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

    if len(indices)==0:
        return None

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = target_bbox[indices[:, 0], :]
    pred_bbox = pred_bbox[indices[:, 0], indices[:, 1], :]

    loss = F.smooth_l1_loss(pred_bbox, target_bbox)

    return loss

@saveall
@batch_slice(3)
def mrcnn_mask(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_masks: [batch, proposals, height, width, num_classes]
    """
    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
    positive_class_ids = target_class_ids[positive_ix].long()

    indices = torch.stack((positive_ix, positive_class_ids), dim=1)

    if len(indices)==0:
        return None

    # Gather the masks (predicted and true) that contribute to loss
    y_true = target_masks[indices[:, 0], :, :]
    y_pred = pred_masks[indices[:, 0], indices[:, 1], :, :]

    loss = F.binary_cross_entropy(y_pred, y_true)
    return loss
