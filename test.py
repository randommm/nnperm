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

import torch
import torch.nn.functional as F

import numpy as np
import scipy.stats as stats

from nnperm import NNPredict, NNPTest
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import time
import hashlib
import itertools
import pickle
from sklearn.externals import joblib
import os

from generate_data import generate_data
from db_structure import Result

for [distribution, db_size, nh_istrue, nhlayers, hl_nnodes] in (
     itertools.product([1], [10_000, 100_000], [True, False]
        ,[10], [100, 1000])):
    while True:
        query = Result.select().where(
            Result.distribution==distribution, Result.db_size==db_size,
            Result.nh_istrue==nh_istrue, Result.nhlayers==nhlayers,
            Result.hl_nnodes==hl_nnodes)
        if query.count() >= 200:
            pv_avg = np.mean([res.pvalue for res in query])
            print("P-values average:", pv_avg)
            break

        n_train = db_size
        x_train, y_train = generate_data(n_train)

        feature_to_test = 3 if nh_istrue else 4

        nn_obj = NNPTest(
        verbose=2,
        es=True,
        hl_nnodes=hl_nnodes,
        nhlayers=nhlayers,
        y_train = y_train,
        x_train = x_train[:, -feature_to_test],
        x_to_permutate = x_train[:, feature_to_test],
        )

        print("Quantile:", nn_obj.quantile)
        print("Pvalue:", nn_obj.pvalue)

        Result.create(
            distribution=distribution, db_size=db_size,
            nh_istrue=nh_istrue, nhlayers=nhlayers,
            hl_nnodes=hl_nnodes, quantile=nn_obj.quantile,
            pvalue=nn_obj.pvalue, elapsed_time=nn_obj.elapsed_time
        )
