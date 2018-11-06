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

estimator = os.environ['estimator'] if 'estimator' in os.environ else ''
while estimator == "":
    print("Set estimator")
    estimator = input("")

db_size_sample = [1_000, 10_000]
betat_sample = [0, 0.01, 0.1, 0.6]
method_sample = ["permutation", "remove", "shuffle_once"]

if estimator != "linear":
    complexity_sample = [1, 2]
else:
    complexity_sample = [1]

distribution_sample = range(5)
retrain_permutations_sample = [True, False]
full_sample = set(itertools.product(db_size_sample, betat_sample,
    retrain_permutations_sample, distribution_sample, method_sample,
    complexity_sample))

hidden_size = 100


while full_sample:
    sample = np.random.choice(len(full_sample))
    sample = list(full_sample)[sample]
    (db_size, betat, retrain_permutations, distribution,
        method, complexity) = sample

    if not retrain_permutations and method == "remove":
        full_sample.discard(sample)
        continue

    query = Result.select().where(
        Result.distribution==distribution, Result.db_size==db_size,
        Result.betat==betat, Result.complexity==complexity,
        Result.estimator==estimator, Result.method==method,
        Result.retrain_permutations==retrain_permutations,)
    if query.count() >= 200:

        pv_avg = np.mean([res.pvalue for res in query])
        print(
            "Final results for: \n",
            "distribution:", distribution, "\n",
            "betat:", betat, "\n",
            "db_size:", db_size, "\n",
            "retrain_permutations:", retrain_permutations, "\n",
            "method:", method, "\n",
            "complexity:", complexity, "\n",
        )
        print("P-values average:", pv_avg, flush=True)

        full_sample.discard(sample)
        print(len(full_sample), "combinations left")
        continue

    db.close()

    n_train = db_size
    x_train, y_train = generate_data(n_train, betat, distribution)

    if distribution == 1:
        feature_to_test = 3
    elif distribution == 0 or distribution == 2 or distribution == 3:
        feature_to_test = 1
    elif distribution == 4:
        feature_to_test = 0

    if estimator == "ann":
        nn_obj = NNPTest(
        verbose=1,
        es=True,
        hidden_size=hidden_size,
        num_layers=complexity * 10,
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

    print("Pvalue:", nn_obj.pvalue)

    Result.create(
        distribution=distribution, db_size=db_size,
        betat=betat, complexity=complexity,
        estimator = estimator, method = method,
        pvalue=nn_obj.pvalue, elapsed_time=nn_obj.elapsed_time,
        retrain_permutations=retrain_permutations
    )

    print("Result stored in the database", flush=True)
