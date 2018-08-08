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
from db_structure import Result

db_size_sample = [1_000, 10_000]
betat_sample = [0, 0.01, 0.05, 0.1, 0.3, 0.6]
method_sample = ["permutation", "remove", "shuffle_once"]
distribution_sample = range(3)
retrain_permutations_sample = [True, False]
full_sample = set(itertools.product(db_size_sample, betat_sample,
    retrain_permutations_sample, distribution_sample, method_sample))

nhlayers = 10
hl_nnodes = 100
estimator = "rf"

while full_sample:
    sample = np.random.choice(len(full_sample))
    sample = list(full_sample)[sample]
    (db_size, betat, retrain_permutations, distribution,
        method) = sample

    if not retrain_permutations and method == "remove":
        full_sample.discard(sample)
        continue

    query = Result.select().where(
        Result.distribution==distribution, Result.db_size==db_size,
        Result.betat==betat, Result.nhlayers==nhlayers,
        Result.estimator==estimator, Result.method==method,
        Result.hl_nnodes==hl_nnodes, Result.estimator==estimator,
        Result.retrain_permutations==retrain_permutations)
    if query.count() >= 200:

        pv_avg = np.mean([res.pvalue for res in query])
        print(
            "distribution:", distribution, "\n",
            "betat:", betat, "\n",
            "db_size:", db_size, "\n",
            "retrain_permutations:", retrain_permutations, "\n",
            "nhlayers:", nhlayers, "\n",
            "hl_nnodes:", hl_nnodes,
        )
        print("P-values average:", pv_avg, flush=True)

        full_sample.discard(sample)
        continue

    n_train = db_size
    x_train, y_train = generate_data(n_train, betat, distribution)

    if distribution == 1:
        feature_to_test = 3
    elif distribution == 0 or distribution == 2:
        feature_to_test = 1

    if estimator == "ann":
        nn_obj = NNPTest(
        verbose=1,
        es=True,
        hl_nnodes=hl_nnodes,
        nhlayers=nhlayers,
        y_train = y_train,
        x_train = np.delete(x_train, feature_to_test, 1),
        x_to_permutate = x_train[:, feature_to_test],
        retrain_permutations = retrain_permutations,
        estimator = estimator,
        method = method,
        )
    else:
        nn_obj = NNPTest(
        y_train = y_train,
        x_train = np.delete(x_train, feature_to_test, 1),
        x_to_permutate = x_train[:, feature_to_test],
        retrain_permutations = retrain_permutations,
        estimator = estimator,
        method = method,
        n_estimators = 300,
        )

    print("Pvalue:", nn_obj.pvalue)

    Result.create(
        distribution=distribution, db_size=db_size,
        betat=betat, nhlayers=nhlayers,
        hl_nnodes=hl_nnodes, estimator = estimator, method = method,
        pvalue=nn_obj.pvalue, elapsed_time=nn_obj.elapsed_time,
        retrain_permutations=retrain_permutations
    )
