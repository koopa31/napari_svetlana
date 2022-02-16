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

plt.ion()

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
plt.imshow(u1);plt.show()

u2 = np.zeros((N,N))
u2[cx2,cy2] = 1
u2 = signal.convolve2d(u2, cell, mode='same')
plt.imshow(u2);plt.show()

## Textures
text1 = np.random.randn(N,N)
text2 = np.random.randn(N,N)
text1 = signal.convolve2d(text1,g1, mode='same')
text2 = signal.convolve2d(text2,g2, mode='same')
# text1 = np.real(np.fft.ifft2(np.fft.fft2(text1)*np.fft.fft2(g1)))
# text2 = np.real(np.fft.ifft2(np.fft.fft2(text2)*np.fft.fft2(g2)))
text1 = (text1 - np.min(text1))/np.max(text1)
text2 = (text2 - np.min(text2))/np.max(text2)

u = u1*text1 + u2*text2
u = signal.convolve2d(u, g, mode='same')

plt.imshow(u);plt.show()

## mask 
mask = u1 + u2
labels = label(mask,connectivity=1)
plt.imshow(labels)
plt.show()

imageio.imsave("label.png", labels.astype('uint16'))
imageio.imsave("mask1.png", u1)
imageio.imsave("mask2.png", u2)
imageio.imsave("image.png", u)
imageio.imsave("mask.png", u1 + u2)

## Saving patches
perm = np.random.permutation(npts)
pts = pts[perm,:]
ps = 20
u=np.uint8(255*u/np.max(u))
for i in range(20):
  if (pts[i,0]-ps>0) & (pts[i,0]+ps<N) & (pts[i,1]-ps>0) & (pts[i,1]+ps<N):
    imageio.imsave("annotate_%i.png"%i, u[pts[i,0]-ps:pts[i,0]+ps,pts[i,1]-ps:pts[i,1]+ps])

ucol = np.zeros((N,N,3))
ucol[:,:,0] = 255*u1
ucol[:,:,1] = 255*u2

imageio.imsave("prediction.png",ucol)


