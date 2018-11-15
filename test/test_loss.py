import pytest
from .main import t

import model
from maskmm import loss

@pytest.mark.parametrize("index", range(4))
def test_rpn_class(t, index):
    t.run(model.compute_rpn_class_loss, loss.rpn_class)

@pytest.mark.parametrize("index", range(4))
def test_rpn_bbox(t, index):
    t.run(model.compute_rpn_bbox_loss, loss.rpn_bbox)

@pytest.mark.parametrize("index", range(4))
def test_mrcnn_class(t, index):
    t.run(model.compute_mrcnn_class_loss, loss.mrcnn_class)

@pytest.mark.parametrize("index", range(4))
def test_mrcnn_bbox(t, index):
    t.run(model.compute_mrcnn_bbox_loss, loss.mrcnn_bbox)

@pytest.mark.parametrize("index", range(4))
def test_mrcnn_mask(t, index):
    t.run(model.compute_mrcnn_mask_loss, loss.mrcnn_mask)