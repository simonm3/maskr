import torch
from functools import wraps
import logging
log = logging.getLogger()
import numpy as  np
from maskmm.tracker import get_type

def pad(x, shape):
    """ pad tensor to target shape
    if shape is int then pad dim=0
    else if any dimensions of shape are less than x.shape then these are ignored
    """
    if isinstance(shape, int):
        shape = (shape, *x.shape[1:])
    if len(x.shape)!=len(shape):
        raise Exception("x and shape must have same number of dimensions")
    padding = [max(0, t-x) for t,x in zip(shape, x.shape)]
    padding = reversed([(0, p) for p in padding])
    padding = np.concatenate(list(padding)).tolist()
    return torch.nn.functional.pad(x, padding)

def pad_all(inputs):
    """ pads each item in x to same shape so they can be stacked """
    # get max_shape on every dimension
    shapes = [np.array(x.shape) for x in inputs]
    max_shape = np.stack(shapes).max(axis=0).tolist()

    # reshape each to max_shape
    padded = [pad(x, max_shape) for x in inputs]

    return padded

def unpad(inputs):
    """ removes zero padding from list of inputs
    uses first input to determine index
    """
    inputs = [inputs] if not isinstance(inputs, (list, tuple)) else inputs

    ix = get_data_index(inputs[0])
    inputs = [i[ix] for i in inputs]

    if len(inputs)==1:
        inputs = inputs[0]
    return inputs

def get_data_index(x):
    """ return dim=0 index of data items without zero padding
    """
    dim = 0 if len(x.shape)==1 else 1
    ix = x.ne(0).any(dim=dim).nonzero()[:, 0].unique()
    return ix

def batch_slice(aggfunc=None, unpadding=True):
    """ converts a function to process each item in a batch; and stack the output

    Simplifies code by not vectorizing the batch dimension, leaving optimization until later.
    Enables one to run inputs of [batch, N, x1, x2, x3] on modules expecting [batch, x1, x2, x3]

    input tensors are in format [[batch, a], [batch, b], ...]
    function is called on each a, b to return outputs c, d
    output tensors are in format [[batch, c], [batch, d], ...] or for single output [batch, c]

    aggfunc default is torch.stack. This automatically zero pads each output.
    alternatives are torch.cat or lambda x:x (which outputs lists). Both of these lose the batch dimension.
    strip=True strips zeros from all inputs
    strip=[1,2,3] strips from only inputs 1,2,3
    strip=0 strips only input 0
    """
    aggfunc = aggfunc or torch.stack

    def batch_slice_inner(f):

        @wraps(f)
        def wrapper(inputs, *args, **kwargs):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            # convert from variables/items to items/variables
            # e.g. inputs [[batch, a], [batch, b]] ==> [[a1, b1], [a2, b2]]
            inputs = list(zip(*inputs))

            # process each item
            results = []
            for item in inputs:
                # strip zero padding
                item = [item] if not isinstance(item, (list, tuple)) else item
                strip = range(len(item)) if unpadding==True else unpadding
                strip = [strip] if not isinstance(strip, (list,tuple)) else strip
                item = [unpad(var) if i in strip else var for i, var in enumerate(item)]
                item = item[0] if len(item)==1 else item

                r = f(item, *args, **kwargs)

                # temporarily make a list to allow list comprehensions
                r = [r] if not isinstance(r, (list, tuple)) else r

                results.append(r)

            # convert returns from items/variables to variables/items
            # e.g. function returns c,d ==> [[c1, d1], [c2, d2]] => [[c1, c2], [d1, d2]]
            results = list(zip(*results))

            # aggregate outputs
            if aggfunc is torch.stack:
                results = [aggfunc(pad_all(output)) for output in results]
            else:
                results = [aggfunc(x) for x in results if len(x) > 0]

            # if only one return variable then return it rather than a list
            results = results[0] if len(results)==1 else results

            return results
        return wrapper
    return batch_slice_inner