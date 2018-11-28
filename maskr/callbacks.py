from fastai import *
from maskr import loss
from maskr.baseline import save
import logging
log = logging.getLogger()

class Multiloss(LearnerCallback):
    """ calculate multiple loss functions, sum, and save results """

    def on_train_begin(self, **kwargs:Any):
        self.losses = []

    def on_loss_begin(self, **kwargs):
        config = self.learn.model.config

        # get inputs
        tgt_rpn_match, tgt_rpn_bbox,\
        rpn_class_logits, rpn_bbox,\
        target_class_ids, target_deltas, target_mask,\
        mrcnn_class_logits, mrcnn_bbox, mrcnn_mask = kwargs["last_output"]["out"]

        # rpn loss
        rpn_class_loss = loss.rpn_class(tgt_rpn_match, rpn_class_logits)
        rpn_bbox_loss = loss.rpn_bbox(tgt_rpn_bbox, tgt_rpn_match, rpn_bbox)
        losses = [rpn_class_loss, rpn_bbox_loss]

        # head loss
        if config.HEAD:
            mrcnn_class_loss = loss.mrcnn_class(target_class_ids, mrcnn_class_logits)
            mrcnn_bbox_loss = loss.mrcnn_bbox(target_deltas, target_class_ids, mrcnn_bbox)
            mrcnn_mask_loss = loss.mrcnn_mask(target_mask, target_class_ids, mrcnn_mask)
            losses.extend([mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss])

        # output losses
        total = sum(losses).squeeze()
        losses = [total] + losses
        log.info([f"{x}={loss.item():0.4f}" for x, loss in zip(["tot", "rc", "rb", "c", "b", "m"],losses)])
        self.losses.append(losses)

        return total

class Cuda(LearnerCallback):
    """ sets default tensors during training/validation to config.DEVICE
    sets to cpu for train/valid dataloader as cuda does not work with multiprocessing workers>0)
    """
    def on_train_begin(self, **kwargs:Any):
        # use cpu for train dataloader
        torch.set_default_tensor_type(torch.FloatTensor)

    def on_batch_begin(self, **kwargs:Any):
        # use cuda after dataloader initialised
        if self.learn.model.config.DEVICE=="cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def on_batch_end(self, **kwargs:Any):
        # use cpu for valid dataloader
        torch.set_default_tensor_type(torch.FloatTensor)

###### debugging ##############################################################

class TrainSave(LearnerCallback):
    """ save data during weight update """
    def on_batch_begin(self, **kwargs):
        xb = kwargs["last_input"]
        images, image_metas, tgt_rpn_match, tgt_rpn_bbox, gt_class_ids, gt_boxes, gt_masks = xb
        save(images, "images")
        save(gt_class_ids, "gt_class_ids")
        save(gt_boxes, "gt_boxes")
        ### UNSQUEEZE FOR COMPARISON WITH MASKMM0 HAS STRANGE LAST DIMENSION THAT IS NEVER USED?
        save(tgt_rpn_match.unsqueeze(-1), "rpn_match")
        save(tgt_rpn_bbox, "rpn_bbox")

    def on_backward_end(self, **kwargs:Any):
        """ save weights and gradients before step """
        for name, param in self.learn.model.named_parameters():
            if param.requires_grad:
                save(param, "back_" + name)
        for name, param in self.learn.model.named_parameters():
            if param.requires_grad:
                save(param.grad, "grad_"+name)

    def on_step_end(self, **kwargs:Any):
        """ save weights after step """
        for name, param in self.learn.model.named_parameters():
            if param.requires_grad:
                save(param, "step_"+name)

class StrictBnFreeze(LearnerCallback):
    """ set all batchnorm to eval as original maskr
    fastai Bnfreeze only does this if next layer has requires_grad=False
    """
    def on_epoch_begin(self, **kwargs:Any):
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.learn.model.apply(set_bn_eval)