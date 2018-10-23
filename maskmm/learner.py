import torch
import torch.optim as optim
import torch.utils.data

from maskmm.utils import visualize
import maskmm.loss as loss
import logging
log = logging.getLogger()
from maskmm.mytools import *

class Learner:
    """ training/validation loop encapsulating model, datasets, optimizer """

    def __init__(self, model, train_dataset, val_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.loss_history = []
        self.val_loss_history = []

    def train(self, learning_rate, epochs, layers):
        """
            learning_rate: The learning rate to train with
            epochs: Number of training epochs. Note that previous training epochs
                    are considered to be done alreay, so this actually determines
                    the epochs to train in total rather than in this particaular
                    call.
            layers: Allows selecting wich layers to train. It can be:
                - A regular expression to match layer names to train
                - One of these predefined values:
                  heads: The RPN, classifier and mask heads of the network
                  all: All the layers
                  3+: Train Resnet stage 3 and up
                  4+: Train Resnet stage 4 and up
                  5+: Train Resnet stage 5 and up
        """
        model = self.model
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # From a specific Resnet stage and up
            "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Train
        log1("\nStarting at epoch {}. LR={}\n".format(model.epoch + 1, learning_rate))
        log1("Checkpoint Path: {}".format(model.checkpoint_path))
        model.set_trainable(layers)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in model.named_parameters() if
                            param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
        model.optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': model.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=model.config.LEARNING_MOMENTUM)

        for epoch in range(model.epoch + 1, epochs + 1):
            log1(f"Epoch {epoch}/{epochs}.")

            # Training
            model.train()
            model.optimizer.zero_grad()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            #model.apply(set_bn_eval)

            losses = self.run_epoch(train_generator, model.config.STEPS_PER_EPOCH,
                                    mode="training")

            self.loss_history.append(losses)

            # Validation
            model.eval()
            #model.apply(set_bn_eval)
            with torch.no_grad():
                losses = self.run_epoch(val_generator, model.config.VALIDATION_STEPS,
                                        mode="validation")
                self.val_loss_history.append(losses)

            # finish epoch
            visualize.plot_loss(self.loss_history, self.val_loss_history, save=True, log_dir=model.log_dir)
            torch.save(model.state_dict(), model.checkpoint_path.format(epoch))

        model.epoch = epochs

    def run_epoch(self, datagenerator, steps, mode):
        model = self.model
        device = model.config.DEVICE

        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        step = 0

        for inputs in datagenerator:
            batch_count += 1

            # get data
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = inputs
            image_metas = image_metas.cpu().numpy()

            save(images, "images")
            save(gt_class_ids, "gt_class_ids")
            save(gt_boxes, "gt_boxes")
            ### UNSQUEEZE FOR COMPARISON WITH MASKMM0 HAS STRANGE LAST DIMENSION THAT IS NEVER USED?
            save(rpn_match.unsqueeze(-1), "rpn_match")
            save(rpn_bbox, "rpn_bbox")

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, \
            target_mask, mrcnn_mask = model([images, image_metas, gt_class_ids, gt_boxes, gt_masks],
                                            mode=mode)

            # Compute losses
            rpn_class_loss = loss.rpn_class(rpn_match, rpn_class_logits)
            rpn_bbox_loss = loss.rpn_bbox(rpn_bbox, rpn_match, rpn_pred_bbox)
            mrcnn_class_loss = loss.mrcnn_class(target_class_ids, mrcnn_class_logits)
            mrcnn_bbox_loss = loss.mrcnn_bbox(target_deltas, target_class_ids, mrcnn_bbox)
            mrcnn_mask_loss = loss.mrcnn_mask(target_mask, target_class_ids, mrcnn_mask)
            totloss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

            if mode=="training":
                # Backpropagation
                totloss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if (batch_count % model.config.BATCH_SIZE) == 0:
                    model.optimizer.step()
                    model.optimizer.zero_grad()
                    batch_count = 0

            # Progress
            printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                             suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}".format(
                                 totloss.item(), rpn_class_loss.item(), rpn_bbox_loss.item(),
                                 mrcnn_class_loss.item(), mrcnn_bbox_loss.item(),
                                 mrcnn_mask_loss.item()), length=10)

            # Statistics
            loss_sum += totloss.item() / steps
            loss_rpn_class_sum += rpn_class_loss.item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.item() / steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.item() / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, \
               loss_mrcnn_mask_sum


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\n')
    # Print New Line on Complete
    if iteration == total:
        print()


def log1(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)
