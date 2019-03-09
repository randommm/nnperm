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
from peewee import *
import os
from sstudy_storage import do_simulation_study

# Prepare storage dataset
db = SqliteDatabase('real_data.sqlite3')
class Result(Model):
    estimator = TextField()
    method = TextField()
    retrain_permutations = IntegerField()
    feature_tested = IntegerField()
    pvalue = DoubleField()
    elapsed_time = DoubleField()

    class Meta:
        database = db
Result.create_table()

# Prepare training dataset
# Data from https://github.com/vgeorge/pnad-2015/tree/master/dados
df = pd.read_csv("sgemm_product.csv")
mean_col = df.iloc[:, -1] + df.iloc[:, -2]
mean_col += df.iloc[:, -3] + df.iloc[:, -4]
mean_col /= 4
df = pd.concat([df.iloc[:, :14], mean_col], axis=1)
columns = list(df.columns)
columns[-1] = "run (ms)"
df.columns = columns
y_train = np.array(df)[:, -1:]
x_train = np.array(df)[:, :-1]

# Permutations to run
to_sample = dict(
    estimator = ['ann', 'rf', 'linear'],
    method = ["permutation", "shuffle_once"],
    retrain_permutations = [True, False],
    feature_tested = range(15),
)

def func(estimator,
    method,
    retrain_permutations,
    feature_tested,):

    hidden_size = 100

    if estimator == "ann":
        nn_obj = NNPTest(
        verbose=1,
        es=True,
        hidden_size=hidden_size,
        num_layers=5,
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
        n_estimators = 300,
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

do_simulation_study(to_sample, func, db, Result, max_count=1)
