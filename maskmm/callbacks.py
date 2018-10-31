from fastai import *
from maskmm import loss
from maskmm.tracker import save
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
        mrcnn_class_logits, mrcnn_bbox, mrcnn_mask = kwargs["last_output"]["out"]

        # calculate
        rpn_class_loss = loss.rpn_class(tgt_rpn_match, rpn_class_logits)
        rpn_bbox_loss = loss.rpn_bbox(tgt_rpn_bbox, tgt_rpn_match, rpn_bbox)
        mrcnn_class_loss = loss.mrcnn_class(target_class_ids, mrcnn_class_logits)
        mrcnn_bbox_loss = loss.mrcnn_bbox(target_deltas, target_class_ids, mrcnn_bbox)
        mrcnn_mask_loss = loss.mrcnn_mask(target_mask, target_class_ids, mrcnn_mask)

        # save
        losses = [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
        total = sum(losses).squeeze()
        losses = [total] + losses
        #log.info(losses)
        self.learner.losses.append(losses)

        return total

###### save checkpoint objects ##############################################################

class PostBackwardSave(LearnerCallback):
    def on_backward_end(self, **kwargs:Any):
        for name, param in self.learn.model.named_parameters():
            if param.requires_grad:
                save(param, "post_back" + name)
        for name, param in self.learn.model.named_parameters():
            if param.requires_grad:
                save(param.grad, "grad"+name)

class PostStepSave(LearnerCallback):
    def on_step_end(self, **kwargs:Any):
        for name, param in self.learn.model.named_parameters():
            if param.requires_grad:
                save(param, "post_step"+name)

class PreBatch(LearnerCallback):
    def on_batch_begin(self, xb:Tensor, yb:Tensor, train:bool=True):
        images, image_metas, tgt_rpn_match, tgt_rpn_bbox, gt_class_ids, gt_boxes, gt_masks = xb
        save(images, "images")
        save(gt_class_ids, "gt_class_ids")
        save(gt_boxes, "gt_boxes")
        ### UNSQUEEZE FOR COMPARISON WITH MASKMM0 HAS STRANGE LAST DIMENSION THAT IS NEVER USED?
        save(tgt_rpn_match.unsqueeze(-1), "rpn_match")
        save(tgt_rpn_bbox, "rpn_bbox")