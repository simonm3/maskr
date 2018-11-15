import pytest
from .main import t
import torch
from maskmm.config import Config
config = Config()

import model
from maskmm.datagen.rpn_targets import build_rpn_targets
from maskmm.samples.nuke.dataset import Dataset
from maskmm.samples.nuke import nuke

@pytest.mark.parametrize("index", range(4))
def test_build_rpn_targets(t, index):
    def postLoad():
        t.inputs.pop(0)
        t.inputs[0] = torch.tensor(t.inputs[0]).double()
        t.inputs[1] = torch.tensor(t.inputs[1]).double()
        t.inputs[2] = torch.tensor(t.inputs[2]).double()
        t.inputs[3].COMPAT=True
    t.postLoad = postLoad
    t.run(model.build_rpn_targets, build_rpn_targets)

"""@pytest.mark.parametrize("index", range(4))
def test_getitem(t, index):
    train_ds, val_ds = nuke.get_data()

    def postLoad():
        t.inputs.pop(0)
    t.postLoad = postLoad

    def postRun():
        t.results = t.results[0]
        t.results.pop(1)
        t.baseline.results.pop(1)
    t.postRun = postRun

    t.run(model.Dataset.__getitem__, train_ds.__getitem__)
"""