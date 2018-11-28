import pytest
from .main import t
import torch
from maskr.config import Config
config = Config()

import model
from maskr.datagen.rpn_targets import build_rpn_targets
from maskr.datagen.head_targets import build_head_targets

""" 
anchors is relatively simple and unchanged
dataset is largely tested by rpn_targets and head_targets
"""

@pytest.mark.parametrize("index", range(4))
def test_build_rpn_targets(t, index):
    def postLoad():
        image_shape, anchors, gt_class_ids, gt_boxes, config = t.inputs
        anchors = torch.tensor(anchors).double()
        gt_class_ids = torch.tensor(gt_class_ids).double()
        gt_boxes = torch.tensor(gt_boxes).double()
        config.COMPAT=True
        t.inputs = anchors, gt_class_ids, gt_boxes, config
    t.postLoad = postLoad
    t.run(model.build_rpn_targets, build_rpn_targets)

@pytest.mark.parametrize("index", range(4))
def test_build_head_targets(t, index):
    def postLoad():
        rpn_rois, gt_class_ids, gt_boxes, gt_masks, config = t.inputs
        h, w = config.IMAGE_SHAPE[:2]
        scale = torch.tensor([h, w, h, w]).float()
        gt_boxes = gt_boxes * scale
        t.inputs = rpn_rois, gt_class_ids.float(), gt_boxes, gt_masks, config
    t.postLoad = postLoad
    t.run(model.detection_target_layer, build_head_targets)
