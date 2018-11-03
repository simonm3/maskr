""" tools for debugging """

import torch
import numpy as np
import os
from os.path import join
import inspect
from functools import wraps
import shutil
import sys
import random
import pickle
import logging

log = logging.getLogger()

def mse(a, b):
    """ return mean squared error of two tensors or arrays """
    return ((a - b) ** 2).sum()

def get_type(obj):
    """ return detailed type of object
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

class Tracker:
    """ enables comparison of results between base and live package

    main in package1
        from maskmm.tracker import t
        t.init("package1", ...)

    module
        from maskmm.tracker import save, saveall
        @saveall to save parameters and return values
        save(object, "name") to manually save an object

    for benchmarking package2
        setup main and modules as for package1
        t.init("package2", baseline="package1", ...)
        every file will automatically be compared to base

        todo mapping functions/files with different names
    """
    def __init__(self,
                 name,
                 enabled=True,
                 exclude_funcs=None,
                 exclude_vars=None,
                 basename=None,
                 tolerance=1e-8,
                 log_save=False,
                 log_type=False):
        """ name: subfolder to save/load results
            enabled: enable/disable saving of files. set to False for normal running.
            exclude_funcs: turn off save for named functions
            basename: name of base to benchmark against. None when creating a base.
            tolerance: mse<tolerance is considered zero
            log_save=logs every file saved
            log_type=logs type differences even of content same.
        """
        trackpath = join(os.path.expanduser("~"), "tracking")

        # set paths for saving
        self.path = join(trackpath, name)
        self.basepath = None
        if basename:
            self.basepath = join(trackpath, basename)

        self.enabled = enabled
        self.exclude_funcs = exclude_funcs or []
        self.exclude_vars = exclude_vars or []
        self.tolerance = tolerance
        self.log_save = log_save
        self.log_type = log_type

    def clear(self):
        """ clear last run by deleting saved files """
        shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path, exist_ok=True)

    def save(self, obj, filename):
        """ save a file and compare to baseline
        """
        if not self.enabled or filename.startswith(self.exclude_vars):
            return

        # get unique filepath and filename
        index = 0
        while True:
            filepath = join(self.path, filename+"_"+str(index))
            if os.path.isfile(filepath):
                index = index + 1
                continue
            break
        filename = os.path.basename(filepath)

        # log file source
        if self.log_save:
            frame = inspect.stack()[1]
            log.info(f"saving {filename} from {os.path.basename(frame.filename)}/{frame.lineno}")

        # save file
        with open(filepath, "wb") as f:
            if isinstance(obj, torch.Tensor):
                torch.save(obj, f)
            elif isinstance(obj, np.ndarray):
                np.save(f, obj)
            else:
                try:
                    pickle.dump(obj, f)
                except:
                    raise Exception(f"save does not support {type(obj)}")

        # match against baseline
        if self.basepath:
            m = self.match(filename)

    def load(self, filename, trackpath=None):
        """ load object from file """
        trackpath = trackpath or self.path
        filepath = join(trackpath, filename)
        try:
            return torch.load(filepath)
        except:
            try:
                return np.load(filepath)
            except:
                return pickle.load(open(filepath, "rb"))

    def load0(self, filename):
        """ load object from basepath """
        return self.load(filename, self.basepath)

    def saveall(self, f):
        """ function decorator that saves params and return values to files in self.trackpath

        @track
        def fname(a, b):
            c = 123
            d = 456
            return c, d
        run0: params=[fname_a, fname_b] return=[fname0_r0, fname0_r1]
        run1: params=[fname_a_1, fname_b_1] return=[fname1_r0, fname1_r1_1]
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            func_name = f.__name__
            if not self.enabled or (func_name in self.exclude_funcs):
                return f(*args, **kwargs)

            # save inputs
            if self.log_save:
                log.info(f"*** {func_name} inputs ***********************************")
            params = dict(zip(f.__code__.co_varnames, args))
            params.update(kwargs)
            for param, v in params.items():
                if param=="self":
                    continue
                self.save(v, f"{func_name}.{param}")

            # execute
            out = f(*args, **kwargs)

            # save outputs
            if self.log_save:
                log.info("*** outputs")
            # convert single returns to list
            if not isinstance(out, (list, tuple)):
                out = [out]
            for i, ret in enumerate(out):
                self.save(ret, f"{func_name}.r{i}")
            if self.log_save:
                log.info(f"*** {func_name} ends *************************************")
            # revert from list to single return
            if len(out)==1:
                out = out[0]
            return out

        return wrapper

    def match(self, a, b=None):
        """ return mean squared error or -1 if type mismatch
        a, b can be objects or filenames
        if a is filename them b is same name in base folder
        if type mismatch then attempt to convert
        log errors
        """
        if b is None:
            b = a

        # load data
        if isinstance(a, str):
            filename = a
            a = self.load(a)
        if isinstance(b, str):
            try:
                b = self.load(b, self.basepath)
            except FileNotFoundError:
                log.warning(f"no base found for {filename}")
                return 0

        # compare types. if necessary then convert
        af = get_type(a)
        bf = get_type(b)
        typemess = ""
        if af!=bf:
            typemess = f" * {af} * {bf}"
            # convert types for comparison
            if isinstance(a, np.ndarray):
                a = torch.tensor(a)
            if isinstance(b, np.ndarray):
                b = torch.tensor(b)
            a = a.cpu().double()
            b = b.cpu().double()

        # compare content
        try:
            diff = mse(a, b)
            if diff > self.tolerance:
                log.warning(f"content unequal {filename}={diff:.0e}{typemess}")
            elif self.log_type and (af!=bf):
                log.warning(f"types unequal {filename}{typemess}")
            else:
                log.info(f"matched {filename}")
        except (TypeError, RuntimeError):
            log.warning(f"not comparable {filename}{typemess}")
            diff = -1
        return diff

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

tracker = Tracker("unnamed")
load = tracker.load
save = tracker.save
saveall = tracker.saveall
match = tracker.match