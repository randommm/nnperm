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
from scipy import stats
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from db_structure import Result

def ecdf(x, ax, *args, **kwargs):
    xc = np.concatenate(([0], np.sort(x), [1]))
    y = np.linspace(0, 1, len(x) + 1)
    yc = np.concatenate(([0], y))
    ax.step(xc, yc, *args, **kwargs)

cls = ["-", ":", "-.", "--"]
clw = [1.0, 2.0, 1.5, 3.0, 0.5, 4.0]
clws = list(itertools.product(clw, cls))

df = pd.DataFrame(list(Result.select().dicts()))

#for db_size in np.sort(db_size_sample):

def plotcdfs(distribution, method, retrain_permutations, db_size,
        estimator, dfpvalues, i, ax):
    label = str(method)
    if retrain_permutations and method != "remove":
        label += " retrain"


    idx1 = df['betat'] == 0.0
    idx2 = df['db_size'] == db_size
    idx3 = (df['retrain_permutations']
        == retrain_permutations)
    idx4 = df['method'] == method
    idx5 = df['estimator'] == estimator
    idx6 = df['distribution'] == distribution
    idxs = np.logical_and(idx1, idx2)
    idxs = np.logical_and(idxs, idx3)
    idxs = np.logical_and(idxs, idx4)
    idxs = np.logical_and(idxs, idx5)
    idxs = np.logical_and(idxs, idx6)
    pvals = np.sort(df[idxs]['pvalue'])

    test_unif = stats.kstest(pvals, 'uniform')
    test_unif = test_unif.pvalue
    vals = [
        method, retrain_permutations, db_size,
        estimator, distribution, np.round(test_unif, 2)
    ]
    dfpvalues.loc["new"] = vals
    dfpvalues.index = range(dfpvalues.shape[0])

    if db_size == 1000 or estimator == "rf":
        return

    ecdf(pvals, ax, label=label, linestyle=clws[i[0]][1],
         lw=clws[i[0]][0])
    i[0] += 1

dfpvalues = [
    "method", "retrain_permutations", "db_size", "estimator",
    "distribution", "pvalue"
    ]
dfpvalues = pd.DataFrame(columns=dfpvalues)

method_sample = ["permutation", "remove", "shuffle_once"]
fig = plt.figure(figsize=[8.4, 5.8])

for distribution in range(3):
    ax = fig.add_subplot(1, 3, distribution + 1)
    ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000))
    i = [0]
    for method in np.sort(method_sample):
        for retrain_permutations in [True, False]:
            for db_size in [1_000, 10_000]:
                for estimator in ["ann", "rf"]:
                    if retrain_permutations or not method == "remove":
                        plotcdfs(distribution, method,
                            retrain_permutations, db_size,
                            estimator, dfpvalues, i, ax)
    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

filename = "plots/"
filename += "null.pdf"
with PdfPages(filename) as ps:
    ps.savefig(fig, bbox_inches='tight')
plt.close(fig)

print(dfpvalues.to_latex())
