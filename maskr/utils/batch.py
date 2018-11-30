import torch
import numpy as  np
from functools import wraps
import logging
log = logging.getLogger()

def unbatch(vars):
    """ combine first 2 dimensions of each var """
    vars = listify(vars)
    return unlistify([x.reshape(-1, *x.shape[2:]) for x in vars])

def listify(x):
    """ allow single item to be treated same as list. simplifies loops and list comprehensions """
    if isinstance(x, tuple):
        return list(x)
    if not isinstance(x, list):
        return [x]
    return x

def unlistify(x):
    """ remove list wrapper from single item. similar to function returning list or item """
    return x[0] if len(x)==1 else x

def pad(x, tgt_shape):
    """ return zeropadded x with target shape
    any dimensions of shape less than x.shape are ignored
    integer shape extends dim=0 and pads rest with zeros
    """
    # empty return zeros
    if len(x)==0:
        return torch.zeros(tgt_shape, dtype=x.dtype)

    # pad dim0 only
    if isinstance(tgt_shape, int):
        tgt_shape = [tgt_shape, *x.shape[1:]]

    padding = [max(0, t-x) for t,x in zip(tgt_shape, x.shape)]
    # format for torch pad to pad at bottom
    padding = reversed([(0, p) for p in padding])
    padding = np.concatenate(list(padding)).tolist()
    if (np.array(padding)==0).all():
        return x
    return torch.nn.functional.pad(x, padding)

def pack(variables):
    """ add zero padding and stack each variable

    variables is a list where each variable is a list of tensors.
    return is a list with one tensor per variable
    e.g. three tensors [[5, 4], [2, 4], [1, 4]] would pack to a tensor [3, 5, 4]
    """
    stacked = []
    for v in variables:
        # get maximum dimensions
        dims = max([len(x.shape) for x in v])
        maxshape = []
        for dim in range(dims):
            maxdim  = max([0 if dim>=len(item.shape) else item.shape[dim] for item in v])
            maxshape.append(maxdim)

        # resize items to maximum and stack
        padded = [pad(item, maxshape) for item in v]
        stacked.append(torch.stack(padded))
    return stacked

def unpack(variables):
    """ remove first dimension and zero padding from each variable

      variables is a list with one tensor per variable
      a tensor [3, 5, 4] may be unpacked to [[5, 4], [2, 4], [1, 4]]
    """
    all_unpacked = []
    for v in variables:
        unpacked = [item[item.ne(0).nonzero()[:, 0].unique()] for item in v]
        all_unpacked.append(unpacked)
    return all_unpacked

def batch_slice(slice=1, packed=None):
    """ converts a function to process batches
    slice=number of params to slice
    packed=number of zero-padded params. None assumes all.

    Split batch dimension and remove padding
    Process each item
    Stack results by item. Pad with zeros if needed.

    Benefits
    ========
    * Simplify code by not vectorizing the batch dimension, leaving optimization until later.
    * Enable inputs of [batch, N, x1, x2, x3] on pytorch modules expecting [batch, x1, x2, x3]

    Inputs and outputs to target function
    =====================================
    * example c, d = func(inputs, config, z=4)
    * inputs [[batch, a], [batch, b]]
    * return [[batch, c], [batch, d], ...] or single variable [batch, c]
    * inputs and outputs are zeropadded and stacked
    """
    def batch_slice_inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            inputs, constants = args[:slice], args[slice:]

            inputs = listify(inputs)
            try:
                pass#log.info(f.__name__)
            except:
                pass#log.info("unknown")

            # unpack up to packed. default is all sliced vars
            n = packed or slice
            inputs[:n] = unpack(inputs[:n])

            # convert from variables/items to items/variables
            # e.g. inputs [[batch, a], [batch, b]] ==> [[a1, b1], [a2, b2]]
            items = list(zip(*inputs))

            # process each item. listify forces return to be list for zip.
            results = [listify(f(*item, *constants, **kwargs)) for item in items]

            # convert results from items/variables to variables/items
            # e.g. function returns c,d ==> [[c1, d1], [c2, d2]] => [[c1, c2], [d1, d2]]
            results = list(zip(*results))

            results = pack(results)

            results = unlistify(results)

            return results
        return wrapper
    return batch_slice_inner