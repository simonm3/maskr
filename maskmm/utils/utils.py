import torch
from functools import wraps
import logging
log = logging.getLogger()

def batch_slice(f):
    """ converts function to handle batches of input
    inputs are in shape [[batch, a], [batch, b], ...]
    function is called on each a, b to generate c, d
    outputs are stacked in shape [[batch, c], [batch, d]]
    both inputs and outputs are lists EVEN IF ONLY ONE INPUT OR OUTPUT
    """
    @wraps(f)
    def wrapper(inputs, *args, **kwargs):
        out = []
        for i in range(len(inputs[0])):
            inputs = [x[i] for x in inputs]
            out.append(f(inputs, *args, **kwargs))
        # convert [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        out = list(zip(*out))
        # convert to [[batch, a], [batch, b], [batch, c]]
        out = [torch.stack(x) for x in out]

        # avoid need to index a single return object
        if len(out)==1:
            return out[0]
        return out
    return wrapper