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

def batch_slice(aggfunc=None):
    """ decorator that converts function to handle batches of input tensors

    inputs are in format [[batch, a], [batch, b], ...]
    function is called on each a, b to return outputs c, d
    outputs are aggregated in shape [[batch, c], [batch, d]]
    if len(output)==1 then returns output
    aggfunc aggregates the output and ignores empty items
    aggfunc could be torch.stack, torch.cat or lambda x:x (for list). default is stack.
    """
    aggfunc = aggfunc or torch.stack

    def batch_slice_inner(f):
        @wraps(f)
        def wrapper(inputs, *args, **kwargs):
            out = []
            for i in range(len(inputs[0])):
                inputs_item = [x[i] for x in inputs]
                out_item = f(inputs_item, *args, **kwargs)
                if not isinstance(out_item, list):
                    out_item = [out_item]
                out.append(out_item)

            # convert list of instance outputs to list of output objects
            # [[c1, d1], [c2, d2]] => [[c1, c2], [d1, d2]]
            out = list(zip(*out))
            # [[c1, c2], [d1, d2]] => [[batch, c], [batch, d]]
            if aggfunc is torch.cat:
                # cat fails if len(x)==0
                out = [aggfunc(x) for x in out if len(x) > 0]
            else:
                out = [aggfunc(x) for x in out]

            # for single return variable then return it rather than a list
            if len(out)==1:
                return out[0]
            return out
        return wrapper
    return batch_slice_inner