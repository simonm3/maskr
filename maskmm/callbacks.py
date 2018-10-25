from fastai import *
from maskmm import loss
import logging
log = logging.getLogger()

class Multiloss(Callback):
    """ handle multiple loss functions """

    def __init__(self, learner):
        self.learner = learner
        learner.losses = []

    def on_loss_begin(self, **kwargs):
        """ calculate losses, save and return sum """
        tgt_rpn_match, tgt_rpn_bbox,\
        rpn_class_logits, rpn_bbox,\
        target_class_ids, target_deltas, target_mask,\
        mrcnn_class_logits, mrcnn_bbox, mrcnn_mask = kwargs["last_output"]

        # calculate
        rpn_class_loss = loss.rpn_class(tgt_rpn_match, rpn_class_logits)
        rpn_bbox_loss = loss.rpn_bbox(tgt_rpn_bbox, tgt_rpn_match, rpn_bbox)
        mrcnn_class_loss = loss.mrcnn_class(target_class_ids, mrcnn_class_logits)
        mrcnn_bbox_loss = loss.mrcnn_bbox(target_deltas, target_class_ids, mrcnn_bbox)
        mrcnn_mask_loss = loss.mrcnn_mask(target_mask, target_class_ids, mrcnn_mask)

        log.info(target_class_ids)
        log.info(mrcnn_class_logits)

        # save
        losses = [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
        log.info(losses)
        self.learner.losses.append(losses)

        return sum(losses).squeeze()