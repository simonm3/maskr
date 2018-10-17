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
##### tools to compare two output tensors or numpy arrays
###########################################################

SAVEPATH = "/home/ubuntu/saved"
os.makedirs(SAVEPATH, exist_ok=True)

def save(arr, filename):
    """ save array """
    c = getframeinfo(currentframe().f_back)
    if c.filename.find("/maskmm0/") >= 0:
        filename = filename + "0"
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
    log.info(f"loading {filename} at {os.path.basename(c.filename)}/{c.lineno}")

    filename = join(SAVEPATH, filename)
    try:
        return torch.load(filename).cpu()
    except:
        return np.load(filename)

def match(filename):
    """ return true if filename=filename0 """
    a = load(filename)
    b = load(filename+"0")
    if isinstance(a, np.ndarray):
        try:
            return np.equal(a, b).all()
        except:
            log.info((a.shape, b.shape))
            return False
    return torch.equal(a, b)

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
        out.append(f"c={state.float().std()%1:0.4f}")
        #torch.cuda.set_rng_state(state)
        out.append(f"c={str(torch.backends.cudnn.deterministic)}")

    return out