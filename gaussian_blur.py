import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import pickle

import constants as c
import binning

# Gaussian filter parameters
sigma=2.0                  # standard deviation for Gaussian kernel
truncate=4.0               # truncate filter at this many sigmas

xmesh = pickle.load(open('xmesh.p', 'rb'))
ymesh = pickle.load(open('ymesh.p', 'rb'))
hmesh = pickle.load(open('hmesh.p', 'rb'))

# mask SR
mask_indices = binning.binInSR(xmesh,ymesh)
x = np.ma.array(xmesh, mask=mask_indices)
y = np.ma.array(ymesh, mask=mask_indices)
h = np.ma.array(hmesh, mask=mask_indices)

fig, axes = plt.subplots(1, 2, True, True)
im = axes[0].pcolormesh(xmesh,ymesh,hmesh)
fig.colorbar(im, ax=axes[0])

# apply filter
V=h.copy()
V[mask_indices] = np.inf
VV=sp.ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)

# replace SR 
VVarr = np.array(VV)
#VVarr[mask_indices] = 0

im = axes[1].pcolormesh(xmesh,ymesh,VVarr)
fig.colorbar(im, ax=axes[1])

plt.show()