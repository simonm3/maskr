import torch
import torch.nn.functional as F
from maskr.test.baseline import saveall
from maskr.utils.batch import unbatch
import logging
log = logging.getLogger()

@saveall
def rpn_class(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = rpn_match.ne(0).nonzero()

    # Pick rows that contribute to the loss and filter out the rest.
    # note this flattens the batch dimension
    rpn_class_logits = rpn_class_logits[indices[:,0],indices[:,1]]
    anchor_class = anchor_class[indices[:,0],indices[:,1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)
    return loss

@saveall
def rpn_bbox(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    targets = []
    rpns = []
    # process each item per batch separately as need to trim the target box to right size
    for target_bbox, rpn_match, rpn_bbox in zip(target_bbox, rpn_match, rpn_bbox):
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        indices = rpn_match.eq(1).nonzero()

        # Pick bbox deltas that contribute to the loss
        rpn_bbox = rpn_bbox[indices[:, 0]]

        # Trim target bounding box deltas to the same length as rpn_bbox
        # todo if this is needed then also needed in head?
        if len(target_bbox) < len(rpn_bbox):
            log.warning("more rpn_targets than rpns")
            target_bbox = target_bbox[:len(rpn_bbox)]

        targets.append(target_bbox)
        rpns.append(rpn_bbox)

    # flatten the batch dimension
    target_bbox = torch.cat(targets)
    rpn_bbox = torch.cat(rpns)

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss

@saveall
def mrcnn_class(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the numpy.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    # remove batch dimension
    target_class_ids, pred_class_logits = unbatch([target_class_ids, pred_class_logits])

    # remove zero padding rois. note include background rois with class_id=0
    ix = pred_class_logits.ne(0).nonzero()[:, 0].unique()
    target_class_ids = target_class_ids[ix]
    pred_class_logits = pred_class_logits[ix]

    if len(target_class_ids)==0:
        return torch.tensor([0], requires_grad=False).float()

    loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    return loss

@saveall
def mrcnn_bbox(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # remove batch dimension
    target_bbox, target_class_ids, pred_bbox = unbatch([target_bbox, target_class_ids, pred_bbox])

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = target_class_ids[positive_roi_ix].long()
    indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

    if len(indices)==0:
        log.warning("no positive rois")
        return torch.tensor([0], requires_grad=False).float()

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = target_bbox[indices[:, 0], :]
    pred_bbox = pred_bbox[indices[:, 0], indices[:, 1], :]

    loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    return loss

@saveall
def mrcnn_mask(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill numpy.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # remove batch dimension
    target_masks, target_class_ids, pred_masks = unbatch([target_masks, target_class_ids, pred_masks])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
    positive_class_ids = target_class_ids[positive_ix].long()

    indices = torch.stack((positive_ix, positive_class_ids), dim=1)

    if len(indices)==0:
        log.warning("no positive rois for mask")
        return torch.tensor([0], requires_grad=False).float()

    # Gather the masks (predicted and true) that contribute to loss
    y_true = target_masks[indices[:, 0], :, :]
    y_pred = pred_masks[indices[:, 0], indices[:, 1], :, :]

    loss = F.binary_cross_entropy(y_pred, y_true)
    return loss
