import torch
from functools import wraps
import logging
log = logging.getLogger()
import numpy as  np

listify = lambda x: [x] if not isinstance(x, (list, tuple)) else x
unlistify = lambda x: x[0] if len(x)==1 else x

def pad(x, shape):
    """ pad tensor to target shape
    if any dimensions of shape are less than x.shape then these are ignored
    """
    if isinstance(shape, int):
        shape = [shape, *x.shape[1:]]
    if len(x.shape)!=len(shape):
        raise Exception("x and shape must have same number of dimensions")
    padding = [max(0, t-x) for t,x in zip(shape, x.shape)]
    padding = reversed([(0, p) for p in padding])
    padding = np.concatenate(list(padding)).tolist()
    return torch.nn.functional.pad(x, padding)

def pack(inputs):
    """ add zero padding and stack
    e.g. class_logits from three items [[12,2], [44,2], [5, 2]] => [3,44,2]
    """
    inputs = listify(inputs)

    maxlength = max([np.array(x.shape[0]) for x in inputs])
    padded = [pad(x, [maxlength, *x.shape[1:]]) for x in inputs]
    stacked = [torch.stack(x) for x in padded]

    stacked = unlistify(stacked)
    return stacked

def unpack(inputs):
    """ unstack and remove zero padding
    inputs is list of tensors each is padded/stacked e.g. [[batch, a], [batch, b]
    return is list of list of tensors without the padding e.g. [[a1, a2], [b1, b2]
    """
    inputs = listify(inputs)

    # unpad based on index from first variable
    x = inputs[0]
    dim = 0 if len(x.shape)==1 else 1
    ix = x.ne(0).any(dim=dim).nonzero()[:, 0].unique()
    unpacked = [i[ix] for i in inputs]

    unpacked = unlistify(unpacked)
    return unpacked

def batch_slice(in_pad=None):
    """ converts a function to process batches
    Split the into items and strip padding
    Process each item
    Aggregate return variables into batches using zeropad/stack

    Benefits
    ========
    * Simplify code by not vectorizing the batch dimension, leaving optimization until later.
    * Enable inputs of [batch, N, x1, x2, x3] on pytorch modules expecting [batch, x1, x2, x3]

    Inputs and outputs
    ==================
    * input tensors are in format [[batch, a], [batch, b], ...]
    * function is called on each a, b to return outputs c, d
    * output tensors are in format [[batch, c], [batch, d], ...] or for single output [batch, c]

    Parameters
    ==========
    in_pad=True. strips input padding from all variables. int or list restricts to indexed variables.
    """
    in_pad = in_pad or True

    def batch_slice_inner(f):

        @wraps(f)
        def wrapper(inputs, *args, **kwargs):
            inputs = listify(inputs)

            in_pad2 = range(len(inputs)) if in_pad == True else in_pad
            inputs = [unpack(input) if i in in_pad2 else input for i, input in inputs]

            # convert from variables/items to items/variables
            # e.g. inputs [[batch, a], [batch, b]] ==> [[a1, b1], [a2, b2]]
            items = list(zip(*inputs))

            # process each item
            results = [f(item, *args, **kwargs) for item in items]

            # convert results from items/variables to variables/items
            # e.g. function returns c,d ==> [[c1, d1], [c2, d2]] => [[c1, c2], [d1, d2]]
            results = list(zip(*results))
            results = pack(results)

            return results
        return wrapper
    return batch_slice_inner