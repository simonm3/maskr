from fastai import *
from maskmm import loss

class Multiloss(LearnerCallback):
    """ handle multiple loss functions """

    def __init__(self, learner):
        self.learner = learner
        learner.losses = []

    def on_loss_begin(self, x, y):
        """ calculate losses, save and return sum """

        rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, \
                    target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = x
        rpn_match, rpn_bbox = y

        # calculate
        rpn_class_loss = loss.rpn_class(rpn_match, rpn_class_logits)
        rpn_bbox_loss = loss.rpn_bbox(rpn_bbox, rpn_match, rpn_pred_bbox)
        mrcnn_class_loss = loss.mrcnn_class(target_class_ids, mrcnn_class_logits)
        mrcnn_bbox_loss = loss.mrcnn_bbox(target_deltas, target_class_ids, mrcnn_bbox)
        mrcnn_mask_loss = loss.mrcnn_mask(target_mask, target_class_ids, mrcnn_mask)

        # save
        losses = [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
        self.learner.losses.append(losses)

        return sum(losses)