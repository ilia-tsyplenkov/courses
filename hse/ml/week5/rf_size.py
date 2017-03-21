#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Find best tree number for random forest tree."""

import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def scorer(estimator, X, y):
    """Wrapper for cross_val_score."""
    return r2_score(y, estimator.predict(X))

if __name__ == '__main__':

    data = pandas.read_csv('abalone.csv', index_col=None)
    col = data.columns
    data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M'
                                  else (-1 if x == 'F' else 0))
    X = data[col[:-1]]
    y = data[col[-1]]
    X = X.as_matrix()
    y = y.as_matrix()
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for tree_n in range(1, 51):
        clf = RandomForestRegressor(n_estimators=tree_n, random_state=1)
        quality = cross_val_score(clf, X, y, scoring=scorer, cv=kf)
        mean_q = np.mean(quality)
        if mean_q > 0.52:
            with open('rf_size_q1', 'w') as f:
                f.write(str(tree_n))
            break
