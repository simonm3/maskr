import pytest
import model
import torch
from maskmm.filters.proposals import proposals
from maskmm.filters.roialign import roialign
from maskmm.filters.detections import get_detections

import logging
log = logging.getLogger()

@pytest.mark.parametrize("index", range(4))
def test_proposals(t, index):
    def postLoad():
        inputs, proposal_count, nms_threshold, anchors, config = t.inputs
        rpn_class, rpn_bbox = inputs
        t.inputs = rpn_class, rpn_bbox, proposal_count, config
    t.postLoad = postLoad
    t.run(model.proposal_layer, proposals)

@pytest.mark.parametrize("index", range(4))
def test_roialign(t, index):
    def postLoad():
        inputs, pool_size, image_shape = t.inputs
        rois,p2,p3,p4,p5 = inputs
        rois = rois.unsqueeze(0)
        t.inputs = rois,p2,p3,p4,p5,pool_size, image_shape
    t.postLoad = postLoad
    t.run(model.pyramid_roi_align, roialign)

def test_detections():
    # detections baseline created from demo.py using single image
    base = Baseline("maskmm0_p")
    t = Test(base)
    def postLoad():
        rois, probs, deltas, image_meta, config = t.inputs
        image_meta = torch.tensor(image_meta)
        t.inputs = rois, probs, deltas, image_meta, config
        t.baseline.results = t.baseline.results[:, :4], t.baseline.results[:, 4], t.baseline.results[:, 5]
    t.postLoad = postLoad
    t.run(model.refine_detections, get_detections)