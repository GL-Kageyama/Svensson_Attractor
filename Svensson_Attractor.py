#=================================================================================
#-------------------------      Svensson Attractor     ---------------------------
#=================================================================================

#-----------------     X = d * sin(a * X) - sin(b * Y)     -----------------------
#-----------------     Y = c * cos(a * X) + cos(b * Y)     -----------------------

#              This program must be run in a Jupyter notebook.
#---------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib as mpl
import pandas as pd
import sys
import datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import Greys9, inferno, viridis
from datashader.utils import export_image
from functools import partial
from numba import jit
import numba
from colorcet import palette

#---------------------------------------------------------------------------------

background = "white"
img_map = partial(export_image, export_path="svensson_maps", background=background)

n = 10000000

#---------------------------------------------------------------------------------

@jit
def trajectory(fn, a, b, c, d, x0=0, y0=0, n=n):

    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0

    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(a, b, c, d, x[i], y[i])

    return pd.DataFrame(dict(x=x, y=y))

@jit
def svensson(a, b, c, d, x, y):

    return d*np.sin(a*x) - np.sin(b*y),   c*np.cos(a*x) + np.cos(b*y)

#---------------------------------------------------------------------------------

cmaps =  [palette[p][::-1] for p in ['bgy', 'bmw', 'bgyw', 'bmy', 'fire', 'gray', 'kbc', 'kgy']]
cmaps += [inferno[::-1], viridis[::-1]]
cvs = ds.Canvas(plot_width = 500, plot_height = 500)
ds.transfer_functions.Image.border=0

#---------------------------------------------------------------------------------

# Parameter  :               a=xxx,   b=xxx,   c=xxx,   d=xxx,
df = trajectory(svensson,    1.40,    1.56,    1.40,   -6.56,   0,   0)
#df = trajectory(svensson,   -0.91,   -1.29,   -1.97,   -1.56,   0,   0)
#df = trajectory(svensson,     1.5,    -1.8,     1.6,     0.9,   0,   0)
#df = trajectory(svensson,   -1.78,    1.29,   -0.09,   -1.18,   0,   0)

# Try to put a value in xxx.
#df = trajectory(svensson,    xxx,    xxx,    xxx,    xxx,   0,   0)

agg = cvs.points(df, 'x', 'y')
img = tf.shade(agg, cmap = cmaps[6], how='linear', span = [0, n/60000])
img_map(img,"attractor")

