import pytest
from maskr.baseline import Test, Baseline

@pytest.fixture()
def t():
    """ generate Test object for each function
    postLoad and postRun can be used to manipulate inputs, baseline.results, results
    """
    base = Baseline("maskmm0")
    return Test(base)