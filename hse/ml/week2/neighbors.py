#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Find optimal nearest neighbors number."""

import pandas
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing


def best_knn_number(X, y, scaling=False, max_neighbor=50, min_neighbor=1):
    """Get best knn number and quality for it."""
    if scaling:
        X = sklearn.preprocessing.scale(X, copy=False)
    kf = sklearn.model_selection.KFold(n_splits=5,
                                       shuffle=True,
                                       random_state=241)
    quality = []
    for neigh_n in range(min_neighbor, max_neighbor + 1):
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neigh_n)
        q = sklearn.model_selection.cross_val_score(clf,
                                                    X,
                                                    y=y,
                                                    scoring='accuracy',
                                                    cv=kf)
        quality.append(q.mean())

    b_q = max(quality)
    b_neigh_n = quality.index(b_q) + 1
    return b_neigh_n, round(b_q, 2)


if __name__ == '__main__':

    data = pandas.read_csv('wine.data')
    data.columns = range(1, 15)
    y = data[1]
    X = data[range(2, 15)]

    kf = sklearn.model_selection.KFold(n_splits=5,
                                       shuffle=True,
                                       random_state=241)
    neigh_n, best_quality = best_knn_number(X, y)
    with open('neigh_q1', 'w') as f:
        f.write(str(neigh_n))
    with open('neigh_q2', 'w') as f:
        f.write(str(best_quality))

    neigh_n, best_quality = best_knn_number(X, y, scaling=True)
    with open('neigh_q3', 'w') as f:
        f.write(str(neigh_n))
    with open('neigh_q4', 'w') as f:
        f.write(str(best_quality))
