import torch
from functools import wraps

def batch_slice(f):
    """ converts function to handle batches of input
    inputs are in shape [batch, a], [batch, b]
    function is called on each a, b to generate c, d
    outputs are stacked in shape [batch, c], [batch, d]
    """
    @wraps(f)
    def wrapper(inputs, *args, **kwargs):
        out = []
        for i in range(len(inputs[0])):
            inputs = [x[i] for x in inputs]
            out.append(f(inputs, *args, **kwargs))
        out = [torch.stack(x) for x in out]
        return out

    return wrapper