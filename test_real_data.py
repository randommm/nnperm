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
import pandas as pd
from nnperm import NNPTest
import itertools
import os
from scipy import stats
from db_structure_real_data import ResultRealData, db
from sstudy import do_simulation_study

# Prepare training dataset
# Data from https://github.com/vgeorge/pnad-2015/tree/master/dados
df = pd.read_csv("dbs/diamonds.csv")
df = df.iloc[:, 1:]
color = pd.factorize(df['color'])[0]
clarity = pd.factorize(df['clarity'])[0]

dummies = ['cut', 'color', 'clarity']
for column in dummies:
    new_df = pd.get_dummies(df[column], dummy_na=False,
                            drop_first=True, prefix=column)
    df = pd.concat([df, new_df], axis=1)
    df = df.drop(column, 1)

y_train = np.array(df[["price"]])
x_train = df.drop("price", 1)

rvs = stats.norm.rvs(size=df.shape[0], random_state=10)
x_train = x_train.assign(new_var_1=(clarity + rvs))
x_train = x_train.assign(new_var_2=(clarity + color + rvs))

columns = x_train.columns
x_train = np.array(x_train)

assert(x_train.shape[1]==25)

to_remove = []
to_add = []
for c in dummies:
    dcols = [x[:len(c) + 1] == c + "_" for x in columns]
    dcols, = np.where(dcols)
    dcols = list(dcols)
    to_remove.extend(dcols)
    to_add.append(tuple(dcols))
features_tested = range(len(columns))
features_tested = list(np.delete(features_tested, to_remove))
features_tested.extend(to_add)

# Code to obtain RF importance measures:
# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor(300).fit(x_train, y_train)
# res = [np.sum(rf.feature_importances_[np.array(x)])
#     for x in feature_tested]
# print(["{0:.2f}".format(x) for x in res])

# Permutations to run
to_sample = dict(
    estimator = ['ann', 'rf', 'linear'],
    method = ["permutation", "shuffle_once", "cpi"],
    retrain_permutations = [True, False],
    feature_tested = features_tested,
    include_extra = range(3),
)

if 'estimator' in os.environ:
    to_sample['estimator'] = [os.environ['estimator']]


def sample_filter(
    estimator,
    method,
    retrain_permutations,
    feature_tested,
    include_extra,
    ):
    if retrain_permutations and method == "cpi":
        return False
    if not retrain_permutations and method == "remove":
        return False
    if include_extra == 0 and feature_tested in [23, 24]:
        return False
    elif include_extra == 1 and feature_tested == 24:
        return False
    elif include_extra == 2 and feature_tested == 23:
        return False
    return True

def func(
    estimator,
    method,
    retrain_permutations,
    feature_tested,
    include_extra,
    ):

    feature_testedd = feature_tested
    if include_extra == 0:
        x_train_n = np.delete(x_train, [23, 24], 1)
    elif include_extra == 1:
        x_train_n = np.delete(x_train, [24], 1)
    elif include_extra == 2:
        x_train_n = np.delete(x_train, [23], 1)
        if feature_tested == 24:
            feature_testedd = 23 # since feature 23 was deleted

    hidden_size = 100

    if estimator == "ann":
        nn_obj = NNPTest(
        verbose=2,
        es=True,
        hidden_size=hidden_size,
        num_layers=5,
        y_train = y_train,
        x_train = np.delete(x_train_n, feature_testedd, 1),
        x_to_permutate = x_train_n[:, feature_testedd],
        retrain_permutations = retrain_permutations,
        estimator = estimator,
        method = method,
        )
    elif estimator == "rf":
        nn_obj = NNPTest(
        y_train = y_train,
        x_train = np.delete(x_train_n, feature_testedd, 1),
        x_to_permutate = x_train_n[:, feature_testedd],
        retrain_permutations = retrain_permutations,
        estimator = "rf",
        method = method,
        n_estimators = 300,
        )
    elif estimator == "linear":
        nn_obj = NNPTest(
        y_train = y_train,
        x_train = np.delete(x_train_n, feature_testedd, 1),
        x_to_permutate = x_train_n[:, feature_testedd],
        retrain_permutations = retrain_permutations,
        estimator = "linear",
        method = method,
        )

    return dict(
        pvalue=nn_obj.pvalue, elapsed_time=nn_obj.elapsed_time,
    )

do_simulation_study(to_sample, func, db, ResultRealData, max_count=1,
    sample_filter=sample_filter)
