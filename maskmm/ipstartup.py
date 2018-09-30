"""
setup jupyter for datagen analysis.
can put in spyder preferences to load when ipython console starts. 
"""
import os
import sys
from os.path import join, expanduser
import yaml
import logging
from logging.config import dictConfig
dictConfig(yaml.load(open(join(expanduser("~"),"logging.yaml"))))
log = logging.getLogger(__name__)
log.info("running ipstartup")

# configure ipython
try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')      # autoreload all modules
    get_ipython().magic('matplotlib inline')  # show charts inline
    get_ipython().magic('load_ext cellevents')  # show time and alert
except:
    pass

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
from IPython.core.display import HTML

# my stuff
try:
    from analysis.explore import xlo
except:
    pass
try:
    from analysis.plot import Gridplot
    gridplot = Gridplot()
except:
    pass


def wide():
    """ makes notebook fill screen width """
    d(HTML("<style>.container { width:100% !important; }</style>"))


def flog(text):
    """ for finding logging problems """
    with open("c:/log1.txt", "a") as f:
        f.write(str(text))
