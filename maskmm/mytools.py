""" tools for debugging """

import torch
import random
import numpy as np
import os
from os.path import join
from inspect import currentframe, getframeinfo
import logging
log = logging.getLogger()

###########################################################
##### tools to compare tensors/arrays from two apps
###########################################################

SAVEPATH = "/home/ubuntu/saved"

# two packages whose results need to be compared
APP = "maskmm"
APP0 = "maskmm0"
os.makedirs(SAVEPATH, exist_ok=True)

# mean squared error considered zero if less than TOL. allows small differences seen with cuda/gpu
TOL = 1e-8

def save(arr, filename):
    """ save array
    files saved from app0 have 0 appended to filename to allow same save code to be cut/pasted between apps..
    """
    c = getframeinfo(currentframe().f_back)
    if c.filename.find(f"/{APP0}/") >= 0:
        filename = filename + "0"
    # logs saves so you can check if same file written more than once
    log.info(f"saving {filename} at {os.path.basename(c.filename)}/{c.lineno}")

    with open(join(SAVEPATH, filename), "wb") as f:
        if isinstance(arr, torch.Tensor):
            torch.save(arr, f)
        elif isinstance(arr, np.ndarray):
            np.save(f, arr)

def load(filename):
    """ load array """
    filename = filename
    c = getframeinfo(currentframe().f_back)
    # log.info(f"loading {filename} at {os.path.basename(c.filename)}/{c.lineno}")

    filename = join(SAVEPATH, filename)
    try:
        return torch.load(filename).cpu()
    except:
        return np.load(filename)

def mse(a, b):
    """ return mean squared error of two matrices
     a,b are nd.arrays or torch.tensors """
    return ((a-b)**2).sum()

def match(file1, file2=None, tol=TOL):
    """ return true if mse < tol
    file1, file2 are filenames. file contains ndarray or tensor
    if b==None then uses a+"0"
    """
    if file2 is None:
        file2 = file1 + "0"
    a = load(file1)
    b = load(file2)
    if a.shape != b.shape:
        log.info(f"different shapes a={a.shape} b={b.shape}")
        return False
    return bool(mse(a,b) <= tol)

##################################################################################
## diagnose random seeds
##################################################################################

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