#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:54:31 2022

@author: pierre
"""
## Libraries
import sys
import os
import numpy as np
from scipy import signal
from skimage.measure import label, regionprops
import brydson_sampling
import matplotlib.pyplot as plt
import imageio
from skimage.morphology import erosion, disk, dilation
from skimage.measure import label, regionprops

## Parameters
N = 256
radius = 20
sigma1 = 3
sigma2 = 1
sigma = 1
ratio = 0.3

##
I = np.arange(-radius,radius)
X,Y = np.meshgrid(I,I)

sigma_max = np.max([sigma1,sigma2,sigma])

II = np.arange(-5*sigma_max,5*sigma_max)
XX,YY = np.meshgrid(II,II)

g1 = np.exp(-(XX**2+YY**2)/(2*sigma1**2))
g2 = np.exp(-(XX**2+YY**2)/(2*sigma2**2))
g = np.exp(-(XX**2+YY**2)/(2*sigma**2))
g1 = g1/np.sum(g1)
g2 = g2/np.sum(g2)
g = g/np.sum(g)

## The indicator of a cell is an idiotic disk
cell = np.sqrt((X**2+Y**2))<=radius/2

## Centers of the cells are drawn at random
pts = brydson_sampling.Bridson_sampling(width=N, height=N, radius=radius+2)
pts = pts.astype(int)
npts = len(pts)
perm = np.random.permutation(npts)
pts = pts[perm,:]

cx1 = pts[0:int(ratio*npts),0]
cy1 = pts[0:int(ratio*npts),1]
cx2 = pts[int(ratio*npts)+1:,0]
cy2 = pts[int(ratio*npts)+1:,1]

## Masks
u1 = np.zeros((N,N))
u1[cx1,cy1] = 1
u1 = signal.convolve2d(u1, cell, mode='same')
plt.figure(1)
plt.imshow(u1)


u2 = np.zeros((N,N))
u2[cx2,cy2] = 1
u2 = signal.convolve2d(u2, cell, mode='same')

eroded = 1 - erosion(u2, disk(9))

holes = u2 * eroded

labs = label(holes)
reg = regionprops(labs)

# enlever les objets au bords
for r in reg:
       if 255 in r.coords or 0 in r.coords:
              holes[r.coords[:, 0], r.coords[:, 1]] = 0

labs = label(u1)
reg = regionprops(labs)

# enlever les objets au bords
for r in reg:
       if 255 in r.coords or 0 in r.coords:
              u1[r.coords[:, 0], r.coords[:, 1]] = 0

plt.figure(2)
plt.imshow(u2)

plt.figure(3)
plt.imshow(holes)
plt.show()

from skimage.io import imsave

imsave("/mnt/86e98852-2345-4dcb-ae92-58406694998c/Documents/Codes/napari_svetlana/images/hole_image.png", u1 + holes)
imsave("/mnt/86e98852-2345-4dcb-ae92-58406694998c/Documents/Codes/napari_svetlana/images/mask_hole_image.png",
       label(u1 + holes))
