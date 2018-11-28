""" tools for debugging and testing """

import os
from os.path import join
from functools import wraps
import shutil
import pickle
import inspect
from glob import glob

import tensorflow as tf
import torch
import numpy as np
from maskr.utils.batch import listify, unlistify
import numbers
import random
import logging
log = logging.getLogger()

# baseline
ROOT = join(os.path.expanduser("~"), "tracking")

class Baseline:
    """ repository for storing baseline results

    example:
        in main
            from baseline import baseline
            baseline.start("savepath")
        in module
            from baseline import *
            exposes save, saveall, load
    """
    # do nothing until start called
    enabled = False

    def __init__(self, name=None):
        """ name: subfolder to load results"""
        if name:
            self.path = join(ROOT, name)

    def start(self, name, clear=True):
        """ enable saving
        name: subfolder to save/load results
        """
        log.warning(f"baseline {name} started. This is ignored for any modules already imported!!!")
        self.path = join(ROOT, name)
        self.enabled = True
        if clear:
            self.clear()

    def clear(self):
        """ clear saved files """
        shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path, exist_ok=True)

    def save(self, obj, filename):
        """ save object
        suffix _n ensures unique
        """
        if not self.enabled:
            return

        # get unique filepath and filename
        index = 0
        while True:
            filepath = join(self.path, filename+"_"+str(index))
            if os.path.isfile(filepath):
                index = index + 1
                continue
            break

        # save object
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            try:
                pickle.dump(obj, f)
            except Exception as e:
                log.exception(e)
                log.warning(f"save failed for {filename} {type(obj)}")

    def load(self, filename):
        """ load object """
        filepath = join(self.path, filename)
        return pickle.load(open(filepath, "rb"))

    def saveall(base, f):
        """ function decorator. save parameters, random state and return values

        can also be used for checkpoints e.g. this would create check1.a, check1.b
            @saveall
            def check1(a, b):
                pass

        todo tensorflow bug requires "self" as parameter to call function
        HACKAROUND WRAPPER BELOW USES SELF AND THIS FUNCTION USES BASE AS FIRST PARAMETER
        """
        if not base.enabled:
            return f
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            func_name = f.__name__

            # todo tensorflow bug workaround
            args = list(args)
            args.insert(0, self)
            del self

            # save parameters
            params = dict(zip(f.__code__.co_varnames, args))
            params.update(kwargs)

            for param, v in params.items():
                base.save(v, f"{func_name}.{param}")

            # save random state
            randomstate = [random.getstate(), np.random.get_state(), torch.random.get_rng_state()]
            try:
                randomstate.append(torch.cuda.get_rng_state())
                torch.backends.cudnn.deterministic = True
            except:
                pass
            save(randomstate, f"{func_name}.randomstates")

            # run
            out = f(*args, **kwargs)

            # save return values
            out = listify(out)
            for i, ret in enumerate(out):
                base.save(ret, f"results/{func_name}.{i}")

            # revert from list to single return
            out = unlistify(out)
            return out

        return wrapper

    def load_state(self, func, index):
        """ load random state """
        state = self.load(f"{func.__name__}.randomstates_{index}")
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.random.set_rng_state((state[2]))
        if len(state)>=4:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state[3])
                torch.backends.cudnn.deterministic = True
            else:
                log.warning("original run used GPU which is not available")

    def load_params(self, func, index):
        """ return params for run n """
        inputs = inspect.getfullargspec(func).args
        params = []
        for iname in inputs:
            params.append(self.load(f"{func.__name__}.{iname}_{index}"))
        return params

    def load_results(self, func, index):
        " return results for run n "
        results= sorted(glob(f"{self.path}/results/{func.__name__}.*_{index}"))
        return unlistify([self.load(r) for r in results])

class Test:
    """ base class to test a function against a baseline
    """
    def __init__(self, baseline, tolerance=1e-8):
        self.baseline = baseline
        self.tolerance = tolerance

        # other class variables to be set
        self.inputs = None
        self.baseline.results = None
        self.results = None
        self.func = None

        # each run with different data increments index
        self.index = 0

    def run(self, basefunc, func, tolerance=None):
        """ test function against baseline """
        if tolerance:
            self.tolerance = tolerance

        # load baseline
        b = self.baseline
        b.load_state(basefunc, self.index)
        self.inputs = b.load_params(basefunc, self.index)
        self.baseline.results = b.load_results(basefunc, self.index)
        self.func = func
        self.postLoad()

        # execute
        self.results = self.func(*self.inputs)
        self.postRun()

        # match
        for a, b in zip(listify(self.results), listify(self.baseline.results)):
            diff = match(a,b)
            assert diff <= self.tolerance

        self.index += 1

    def postLoad(self):
        """ override to manipulate inputs and baseline.results """
        pass

    def postRun(self):
        """ override to manipulate results """
        pass

################# testing ###########################################################

def match(a, b):
    """ return difference score. 999 if not comparable.
    """
    try:
        if isinstance(a, numbers.Number):
            diff = abs(a-b)
        elif isinstance(a, str):
            diff = a==b
        else:
            diff = mse(a, b)
    except (TypeError, RuntimeError) as e:
        log.info(f"cannot match {ftype(a)}, {ftype(b)}")
        log.exception(e)
        diff = 999
    return diff

def mse(a, b):
    """ return mean squared error of two tensors or arrays """
    a = numpy(a)
    b = numpy(b)
    return ((a - b) ** 2).sum()

def numpy(x):
    """ convert to numpy double """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, tf.Tensor):
        x  = x.numpy()
    return x.astype(np.float64)

########### debugging #######################################################################

def ftype(obj):
    """ return full type of object
    e.g. type, dtype, shape
    """
    if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
        types = " ".join([str(x) for x in [type(obj), obj.dtype, list(obj.shape)]])
        if isinstance(obj, torch.Tensor):
            types = " ".join([obj.device.type, types])
    else:
        types = str(type(obj))

    for s in ["class ", "'", "<", ">", "(", ")", "torch.", "numpy." ]:
        types = types.replace(s, "")
    return types

def rngreset(seed=0):
    """ resets all random number generators """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def rngnext():
    """ return next random numbers to be generated without changing states """
    out = []
    # random
    state = random.getstate()
    out.append(f"r={random.random():0.4f}")
    random.setstate(state)

    # numpy
    state = np.random.get_state()
    out.append(f"n={np.random.random():0.4f}")
    np.random.set_state(state)

    # torch
    state = torch.random.get_rng_state()
    out.append(f"t={torch.rand(1)[0]:0.4f}")
    torch.random.set_rng_state(state)

    # cuda
    if torch.cuda.is_available():
        state = torch.cuda.get_rng_state()
        # note there is no function for generating a random in cuda but this may work?
        out.append(f"c={state.float().std()%1:0.4f} {torch.backends.cudnn.deterministic}")

    return out


################################################################################################

baseline = Baseline()
load = baseline.load
save = baseline.save
saveall = baseline.saveall