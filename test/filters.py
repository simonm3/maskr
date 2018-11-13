from maskmm.tracker import *
from maskmm.filters import proposals
from maskmm.loss import rpn_class, rpn_bbox
import inspect

t = Tracker("maskmm", basename="maskmm0")

def run(f, name0=None, params0_map=None, n=4):
    """ generic test of function

    f: function to be tested
    name0: name of baseline function. default is same name.
    params0_map: map of params to baseline params with different names

    function names are assumed to be unique
    repository stores params function.param_0; and return function.0_0
    number after the _ increments each time item is saved i.e. for each test run
    """

    # todo save and load use pickle only, save config. save random state OR reset
    # baseline = saveall, clear, save
    # test = load, match, mse. match functions
    # adhoc = add saves; baseline_func using current data; test_func

    # current
    name = f.__name__
    params = inspect.getfullargspec(f).args

    # baseline
    if name0 is None:
        name0 = name
    params0 = params.copy()
    params0.update(params0.map)

    # for each run
    for n in range(n):
        # run
        inputs = [load0(f"{name0}.{param}_{n}") for param in params0]
        res = f(*inputs)
        res0 = [load0(f"{name}.r{ix}_{n}") for ix in range(len(res))]

        # check results
        for r0, r in zip(res0, res):
            diff = match(r0, r)
            assert diff < t.tolerance

def test_proposals():
    run(proposals)

def test_rpn_class_loss():
    run(rpn_class)

def test_rpn_bbox_loss():
    run(rpn_bbox)