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

cls = [":", "-.", "--", '-']
clw = [1.0, 2.0, 1.5, 3.0, 0.5, 4.0]
clws = list(itertools.product(clw, cls))

df = pd.DataFrame(list(Result.select().where(Result.betat==0).dicts()))

#for db_size in np.sort(db_size_sample):

def plotcdfs(distribution, method, retrain_permutations, db_size,
        estimator, dfpvalues, i, ax):
    label = str(method)
    if retrain_permutations and method != "remove":
        label += " retrain"


    idx1 = df['betat'] == 0.0
    idx2 = df['db_size'] == db_size
    idx3 = df['retrain_permutations'] == retrain_permutations
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
    test_unif = np.round(test_unif, 2)
    vals = [
        method, retrain_permutations, db_size,
        estimator, distribution, test_unif
    ]
    if test_unif > 0:
        dfpvalues.loc["new"] = vals
        dfpvalues.index = range(dfpvalues.shape[0])

    if db_size == 1000:
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
fig = plt.figure(figsize=[11.4, 16.9])
axarr = fig.subplots(4, 3)
fig.subplots_adjust(wspace=0.17, hspace=0.17)

for distribution in range(4):
    for est_ind, estimator in enumerate(["ann", "rf", "linear"]):
        ax = axarr[distribution, est_ind]
        ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000))
        i = [0]
        for method in np.sort(method_sample):
            for retrain_permutations in [True, False]:
                for db_size in [1_000, 10_000]:
                    if retrain_permutations or not method == "remove":
                        plotcdfs(distribution, method,
                            retrain_permutations, db_size,
                            estimator, dfpvalues, i, ax)
        ax.legend(loc="auto", frameon=False,
                  ncol=1, borderaxespad=0.2)
        ax.set_ylim(0, 1.6)
        #ax.set_xlim(-0.1, 1.1)
        ax.set_title("Distribution " + str(distribution)
            + " (" + str(estimator).upper() + ")")
plt.setp([a.get_xticklabels() for a in axarr[1:3, :].reshape(-1)],
    visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1:2].reshape(-1)],
    visible=False)

filename = "plots/"
filename += "null.pdf"
with PdfPages(filename) as ps:
    ps.savefig(fig, bbox_inches='tight')
plt.close(fig)

print(dfpvalues.to_latex())
