#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TODO: add module docs."""

import math
import skimage
import skimage.io
import numpy as np
import pandas
from sklearn.cluster import KMeans


def get_PNSR(X, Y):
    """Calculate PNSR for RGB."""
    mse = np.mean((X - Y) ** 2)
    max_i = max(X.max(), Y.max())
    psnr = 20 * math.log(max_i, 10) - 10 * math.log(mse, 10)
    return psnr


def get_clusters(cluster_n, X, *dargs):
    """Split X to clusters using KMeans."""

    """Returns arrays with mean/median
    colors by clusters."""
    X_c = X.copy()
    kmeans = KMeans(n_clusters=cluster_n, random_state=241)
    kmeans.fit(X_c)
    X_c['target'] = kmeans.predict(X_c)

    medians = X_c.groupby('target').median().values
    X_med = [medians[i] for i in X_c['target'].values]

    means = X_c.groupby('target').mean().values
    X_mean = [means[i] for i in X_c['target'].values]

    X_med = np.reshape(X_med, dargs)
    X_mean = np.reshape(X_mean, dargs)

    return X_mean, X_med


if __name__ == "__main__":
    image = skimage.io.imread('parrots.jpg')
    image = skimage.img_as_float(image)
    m, n, x = image.shape
    X = np.reshape(image, (m * n, x))
    X = pandas.DataFrame(X)

    for cls_n in range(1, 21):
        X_mean, X_med = get_clusters(cls_n, X, m, n, x)
        mean_PNSR = get_PNSR(image, X_mean)
        med_PNSR = get_PNSR(image, X_med)
        if mean_PNSR > 20 or med_PNSR > 20:
            with open('ans.txt', 'w') as f:
                f.write(str(cls_n))
            break
