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
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

db_size_sample = {1_000, 10_000}
betat_sample = {0, 0.01, 0.05, 0.1, 0.3, 0.6}
retrain_permutations_sample = {True, False}
full_sample = set(itertools.product(db_size_sample, betat_sample,
    retrain_permutations_sample))

distribution = 0
nhlayers = 10
hl_nnodes = 100

while full_sample:

    sample = np.random.choice(len(full_sample))
    sample = list(full_sample)[sample]
    db_size, betat, retrain_permutations = sample

    query = Result.select().where(
        Result.distribution==distribution, Result.db_size==db_size,
        Result.betat==betat, Result.nhlayers==nhlayers,
        Result.hl_nnodes==hl_nnodes,
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

    feature_to_test = 3

    nn_obj = NNPTest(
    verbose=2,
    es=True,
    hl_nnodes=hl_nnodes,
    nhlayers=nhlayers,
    y_train = y_train,
    x_train = x_train[:, -feature_to_test],
    x_to_permutate = x_train[:, feature_to_test],
    retrain_permutations = retrain_permutations,
    )

    print("Pvalue:", nn_obj.pvalue)

    Result.create(
        distribution=distribution, db_size=db_size,
        betat=betat, nhlayers=nhlayers,
        hl_nnodes=hl_nnodes,
        pvalue=nn_obj.pvalue, elapsed_time=nn_obj.elapsed_time,
        retrain_permutations=retrain_permutations
    )

cls = ["-", ":", "-.", "-", "--", "-."]
clw = [1.0, 2.0, 2.0, 2.0, 2.5, 1.0]
ax = plt.figure(figsize=[8.4, 5.8]).add_subplot(111)
df = pd.DataFrame(list(Result.select().dicts()))
ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000))
i=0
for db_size in np.sort(list(db_size_sample))[:1]:
    for betat in np.sort(list(betat_sample)):
        label = "betat = " + str(betat)
        label += " and "
        label += str(db_size) + " instances"

        idx1 = df['betat'] == betat
        idx2 = df['db_size'] == db_size
        idx3 = df['retrain_permutations'] == False
        idxs = np.logical_and(idx1, idx2)
        idxs = np.logical_and(idxs, idx3)
        pvals = np.sort(df[idxs]['pvalue'])
        ax.plot(pvals, np.linspace(0, 1, len(pvals)), label=label,
            linestyle=cls[i], lw=clw[i])
        i += 1

legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

with PdfPages("plots/cdf.pdf") as ps:
    ps.savefig(ax.get_figure(), bbox_inches='tight')
