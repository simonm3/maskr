"""
configures notebooks with common extensions and imports

Usage:
     at top of notebook:
        from ipstartup import *
"""
import os
import sys

################## analysis ################################
import scipy as sp
import pandas as pd
import numpy as np

################# visualisation #############################
from pprint import pprint
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from IPython.display import display as d

################# logging #############################
from os.path import join, expanduser
import yaml
import logging
from logging.config import dictConfig
home = expanduser("~")
log = logging.getLogger()
try:
    dictConfig(yaml.load(open(join(home, "logging.yaml"))))
except:
    logging.basicConfig(level=logging.INFO)
    log.warning("~/logging.yaml not found. using basic logging")

if log.getEffectiveLevel() > logging.DEBUG:
    import warnings
    warnings.filterwarnings("ignore")
log.info("")

################# notebook extensions #############################
try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    get_ipython().magic('matplotlib inline')
    # show start time, elapsed time; alert on finish; %%s suppress execution
    get_ipython().magic('load_ext cellevents')
except:
    pass