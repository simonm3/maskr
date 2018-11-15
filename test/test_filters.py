import pytest
import model
from maskmm.filters.proposals import proposals
from maskmm.filters.roialign import roialign
from .main import t

@pytest.mark.parametrize("index", range(4))
def test_proposals(t, index):
    def postLoad():
        t.inputs.pop(3)
        t.inputs.pop(2)
    t.postLoad = postLoad
    t.run(model.proposal_layer, proposals)

@pytest.mark.parametrize("index", range(4))
def test_roialign(t, index):
    def postLoad():
        t.inputs[0] = t.inputs[0].squeeze()
    t.postLoad = postLoad
    t.run(model.pyramid_roi_align, roialign)
