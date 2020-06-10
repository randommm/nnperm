#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.    If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
from nnperm import NNPTest
import itertools
from generate_data import generate_data
from db_structure import Result, db
import os
from sstudy import do_simulation_study

if 'estimator' in os.environ:
    estimator = [os.environ['estimator']]
else:
    estimator = ["ann", "rf", "linear"]

to_sample = dict(
    distribution = range(6),
    method = ["permutation", "shuffle_once", "cpi"],
    db_size = [1_000, 10_000],
    betat = [0, 0.01, 0.1, 0.6],
    retrain_permutations = [True, False],
    complexity = [1],
    estimator = estimator,
)

def sample_filter(
    distribution,
    method,
    db_size,
    betat,
    retrain_permutations,
    complexity,
    estimator,
    ):
    if retrain_permutations and method == "cpi":
        return False
    if not retrain_permutations and method == "remove":
        return False
    if distribution in [4,5] and (betat == 0.01 or db_size == 10000):
        return False
    if estimator == 'linear':
        return 1000
    return True

def func(
    distribution,
    method,
    db_size,
    betat,
    retrain_permutations,
    complexity,
    estimator,
    ):
    hidden_size = 100
    n_train = db_size
    x_train, y_train = generate_data(n_train, betat, distribution)

    if distribution == 1:
        feature_to_test = 3
    elif distribution == 0 or distribution == 2 or distribution == 3:
        feature_to_test = 1
    elif distribution == 4 or 5:
        feature_to_test = 0

    if estimator == "ann":
        nn_obj = NNPTest(
        verbose=1,
        es=True,
        hidden_size=hidden_size,
        num_layers=complexity * 5,
        y_train = y_train,
        x_train = np.delete(x_train, feature_to_test, 1),
        x_to_permutate = x_train[:, feature_to_test],
        retrain_permutations = retrain_permutations,
        estimator = estimator,
        method = method,
        )
    elif estimator == "rf":
        nn_obj = NNPTest(
        y_train = y_train,
        x_train = np.delete(x_train, feature_to_test, 1),
        x_to_permutate = x_train[:, feature_to_test],
        retrain_permutations = retrain_permutations,
        estimator = "rf",
        method = method,
        n_estimators = complexity * 300,
        )
    elif estimator == "linear":
        nn_obj = NNPTest(
        y_train = y_train,
        x_train = np.delete(x_train, feature_to_test, 1),
        x_to_permutate = x_train[:, feature_to_test],
        retrain_permutations = retrain_permutations,
        estimator = "linear",
        method = method,
        )

    return dict(
        pvalue=nn_obj.pvalue, elapsed_time=nn_obj.elapsed_time,
    )

do_simulation_study(to_sample, func, db, Result, max_count=200,
    sample_filter=sample_filter)
