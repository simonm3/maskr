import torch
from functools import wraps
import logging
log = logging.getLogger()
import numpy as  np

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

def pad_length(inputs,):
    """ adds zero padding to tensors so they can be stacked
    e.g. class_logits from three items [[12,2], [44,2], [5, 2]] => [[44, 2], [44, 2], [44, 2]]
    """
    maxlength = max([np.array(x.shape[0]) for x in inputs])
    padded = [pad(x, [maxlength, *x.shape[1:]]) for x in inputs]

    return padded

def unpad_length(inputs):
    """ removes zero padding from tensors
    inputs is list of tensors each is padded/stacked e.g. [[batch, a], [batch, b]
    return is list of list of tensors without the padding e.g. [[a1, a2], [b1, b2]
    """
    # single tensor could still need unpadding
    inputs = [inputs] if not isinstance(inputs, (list, tuple)) else inputs

    # get index from first variable
    x = inputs[0]
    dim = 0 if len(x.shape)==1 else 1
    ix = x.ne(0).any(dim=dim).nonzero()[:, 0].unique()

    # strip zeros
    inputs = [i[ix] for i in inputs]
    inputs = inputs[0] if len(inputs)==1 else inputs
    return inputs

def batch_slice(aggfunc=None, in_pad=None):
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
    aggfunc=torch.stack. aggregate function for return variables. alternatives are torch.cat or lambda x:x (list)
    in_pad=True. strips input padding from all variables. int or list restricts to indexed variables.
    """
    aggfunc = aggfunc or torch.stack
    in_pad = in_pad or True

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
                strip = range(len(item)) if in_pad == True else in_pad
                strip = [strip] if not isinstance(strip, (list,tuple)) else strip
                item = [unpad_length(var) if i in strip else var for i, var in enumerate(item)]
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
                results = [aggfunc(pad_length(output)) for output in results]
            else:
                results = [aggfunc(x) for x in results if len(x) > 0]

            # if only one return variable then return it rather than a list
            results = results[0] if len(results)==1 else results

            return results
        return wrapper
    return batch_slice_inner