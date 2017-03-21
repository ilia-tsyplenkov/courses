#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Get quality difference for Perceptron.

Difference calculate between quality for
scaled and standard features.
"""

import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def calculate_quality(X_train, y_train, X_test, y_test):
    """Calculate perseptron quality."""
    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


train_data = pandas.read_csv('perceptron-train.csv',
                             index_col=None,
                             header=None)
test_data = pandas.read_csv('perceptron-test.csv',
                            index_col=None,
                            header=None)

y_train = train_data[0]
X_train = train_data[range(1, 3)]
y_test = test_data[0]
X_test = test_data[range(1, 3)]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

res = calculate_quality(X_train, y_train, X_test, y_test)
scaled_res = calculate_quality(X_train_scaled, y_train, X_test_scaled, y_test)

diff = round(scaled_res - res, 3)

with open('norm_attr_q1', 'w') as f:
    f.write(str(diff))
