#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Find best 'p' parameter for Minkowski metric."""

import sklearn.datasets
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing
import numpy

if __name__ == '__main__':
    # loading data
    boston_obj = sklearn.datasets.load_boston()

    # split data to features and target variable"
    X = boston_obj.data
    y = boston_obj.target

    # provide feature scaling
    X = sklearn.preprocessing.scale(X, copy=False)

    # create cross-validator
    kf = sklearn.model_selection.KFold(n_splits=5,
                                       shuffle=True,
                                       random_state=42)

    # create range for 'p'
    p_range = numpy.linspace(start=1, stop=10, num=200)
    res = []
    for p in p_range:

        # create KNN classifier with specified 'p'
        neigh_reg = sklearn.neighbors.KNeighborsRegressor(weights='distance',
                                                          p=p)

        # get cross-validation quality
        cval_score = sklearn.model_selection.cross_val_score(
                neigh_reg,
                X,
                y=y,
                scoring='neg_mean_squared_error',
                cv=kf)
        res.append(cval_score.mean())

    # find best quality
    optimal_quality = max(res)
    # find best 'p'
    best_p = p_range[res.index(optimal_quality)]
    # print it to file
    with open('metric_q1', 'w') as f:
        f.write(str(best_p))
