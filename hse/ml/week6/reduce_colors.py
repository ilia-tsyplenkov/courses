#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TODO: add module docs."""

import math
import skimage
import skimage.io
import numpy
from sklearn.cluster import KMeans


def get_PNSR(X, Y):
    """Calculate PNSR for RGB."""
    if X.shape != Y.shape:
        raise TypeError("Shape doesn't match: %s %s" % (X.shape, Y.shape))

    mse = 1.0 / (len(X) * 3) * sum([squared_diff(X[i], Y[i])
                                    for i in range(len(X))])
    max_i = max(max(Y), max(X))
    psnr = 20 * math.log(max_i, 10) - 10 * math.log(mse, 10)
    return psnr


def squared_diff(x, y):
    """Calculate squared difference between 2 points."""
    return sum([math.pow(x[i] - y[i], 2) for i in range(len(x))])

image = skimage.io.imread('parrots.jpg')
img_data = skimage.img_as_float(image)
m, n, x = img_data.shape
X = numpy.reshape(img_data, (m * n, x))
