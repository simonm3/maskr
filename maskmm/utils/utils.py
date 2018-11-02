import torch
from functools import wraps
import logging
log = logging.getLogger()
from maskmm.tracker import get_type

def pad(x, length, dim=0):
    """ pad tensor with zeros to allow batch to be stacked """
    x = x.transpose(0, dim)
    zeropadded = torch.zeros((length, *x.shape[1:]), dtype=x.dtype)
    zeropadded[:len(x)] = x
    zeropadded = zeropadded.transpose(0, dim)
    return zeropadded

def batch_slice(f):
    """ decorator that converts function to handle batches of input tensors
    inputs are in format [[batch, a], [batch, b], ...]
    function is called on each a, b to return outputs c, d
    outputs are stacked in shape [[batch, c], [batch, d]]
    """
    @wraps(f)
    def wrapper(inputs, *args, **kwargs):
        out = []
        for i in range(len(inputs[0])):
            inputs_item = [x[i] for x in inputs]
            out_item = f(inputs_item, *args, **kwargs)
            out.append(out_item)

        # convert list of instance outputs to list of stacked output objects
        # [[c1, d1], [c2, d2]] => [[c1, c2], [d1, d2]]
        out = list(zip(*out))
        # [[c1, c2], [d1, d2]] => [[batch, c], [batch, d]]
        out = [torch.stack(x) for x in out]

        # avoid need to index a single return object
        if len(out)==1:
            return out[0]
        return out
    return wrapper