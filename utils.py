#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:28:35 2025

@author: pweiss
"""

import numpy as np 

# Rescaling the images in 0,1
def rescale_01(y_t):
    y_t -= y_t.amin(dim = (2,3),keepdim=True)
    y_t /= y_t.amax(dim = (2,3),keepdim=True)
    return y_t

# scaling to UINT8 for saving as PNG files
def rescale_uint8(x):
    return np.uint8(np.clip(255*x,0,255))

def SNR(xref,x):
    return -10*np.log10(np.sum((xref-x)**2)/np.sum(xref**2))

def MSE(xref,x):
    return np.sum((xref-x)**2)/np.sum(xref**2)