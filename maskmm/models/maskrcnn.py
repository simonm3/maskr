
import datetime
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from maskmm.utils import image_utils, visualize

from maskmm.datagen.head_targets import build_head_targets
from maskmm.datagen.anchors import generate_pyramid_anchors
from maskmm.datagen.rpn_targets import build_rpn_targets

from .rpn import RPN
from filters.proposals import proposals
from .resnet import ResNet
from .resnetFPN import FPN
from .head import Classifier, Mask
from filters.detections import filter_detections_batch

from maskmm.loss.head_loss import compute_losses

class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture.
        """

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)

        # Generate Anchors
        with torch.no_grad():
            self.anchors = torch.from_numpy(generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                    config.RPN_ANCHOR_RATIOS,
                                                                    config.BACKBONE_SHAPES,
                                                                    config.BACKBONE_STRIDES,
                                                                    config.RPN_ANCHOR_STRIDE)).float()
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        # FPN Classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # FPN Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def detect(self, images):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = image_utils.mold_inputs(images, self.config)

        # Convert images to torch tensor
        with torch.no_grad():
            molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

        # To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        # Run object detection
        detections, mrcnn_mask = self.predict([molded_images, image_metas], mode='inference')

        # Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                image_utils.unmold_detections(detections[i], mrcnn_mask[i],
                                         image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def predict(self, input, mode):
        molded_images = input[0]
        image_metas = input[1]

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        # Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposals([rpn_class, rpn_bbox],
                             proposal_count=proposal_count,
                             nms_threshold=self.config.RPN_NMS_THRESHOLD,
                             anchors=self.anchors,
                             config=self.config)

        if mode == 'inference':
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)

            with torch.no_grad():
                # Detections
                # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
                detections = filter_detections_batch(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)

                # Convert boxes to normalized coordinates
                # TODO: let DetectionLayer return normalized coordinates to avoid
                #       unnecessary conversions
                h, w = self.config.IMAGE_SHAPE[:2]
                scale = torch.from_numpy(np.array([h, w, h, w])).float()
                if self.config.GPU_COUNT:
                    scale = scale.cuda()
                detection_boxes = detections[:, :4] / scale

                # Add back batch dimension
                detection_boxes = detection_boxes.unsqueeze(0)

            # Create masks for detections
            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)

            # Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)

            return [detections, mrcnn_mask]

        elif mode == 'training':

            with torch.no_grad():
                gt_class_ids = input[2]
                gt_boxes = input[3]
                gt_masks = input[4]

                # Normalize coordinates
                h, w = self.config.IMAGE_SHAPE[:2]
                scale = torch.from_numpy(np.array([h, w, h, w])).float()
                if self.config.GPU_COUNT:
                    scale = scale.cuda()
                gt_boxes = gt_boxes / scale

                # Generate detection targets
                # Subsamples proposals and generates target outputs for training
                # Note that proposal class IDs, gt_boxes, and gt_masks are zero
                # padded. Equally, returned rois and targets are zero padded.
                rois, target_class_ids, target_deltas, target_mask = \
                    build_head_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            if not rois.size():
                mrcnn_class_logits = torch.FloatTensor()
                mrcnn_class = torch.IntTensor()
                mrcnn_bbox = torch.FloatTensor()
                mrcnn_mask = torch.FloatTensor()
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)

                # Create masks for detections
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois)

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask]

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """

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
        train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch+1, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch+1, epochs+1):
            log("Epoch {}/{}.".format(epoch,epochs))

            # Training
            loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask = self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)

            # Validation
            val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

            # Statistics
            self.loss_history.append([loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask])
            self.val_loss_history.append([val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask])
            visualize.plot_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            # Save model
            torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs



    def train_epoch(self, datagenerator, optimizer, steps):
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        step = 0

        optimizer.zero_grad()

        for inputs in datagenerator:
            batch_count += 1

            images, image_metas, gt_class_ids, gt_boxes = inputs
            image_metas = image_metas.numpy()
            rpn_match, rpn_bbox = build_rpn_targets(self.anchors, gt_class_ids, gt_boxes, self.config)

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0

            # Progress
            printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                             suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}".format(
                                 loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                                 mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                                 mrcnn_mask_loss.data.cpu()[0]), length=10)

            # Statistics
            loss_sum += loss.data.cpu()[0]/steps
            loss_rpn_class_sum += rpn_class_loss.data.cpu()[0]/steps
            loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0]/steps
            loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0]/steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0]/steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0]/steps

            # Break after 'steps' steps
            if step==steps-1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum

    def valid_epoch(self, datagenerator, steps):

        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0

        for inputs in datagenerator:
            images, image_metas, gt_class_ids, gt_boxes = inputs
            image_metas = image_metas.numpy()
            rpn_match, rpn_bbox = build_rpn_targets(self.anchors, gt_class_ids, gt_boxes, self.config)

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            if not target_class_ids.size():
                continue

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

            # Progress
            printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                             suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}".format(
                                 loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                                 mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                                 mrcnn_mask_loss.data.cpu()[0]), length=10)

            # Statistics
            loss_sum += loss.data.cpu()[0]/steps
            loss_rpn_class_sum += rpn_class_loss.data.cpu()[0]/steps
            loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0]/steps
            loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0]/steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0]/steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0]/steps

            # Break after 'steps' steps
            if step==steps-1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
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
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
    # Print New Line on Complete
    if iteration == total:
        print()

def log(text, array=None):
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