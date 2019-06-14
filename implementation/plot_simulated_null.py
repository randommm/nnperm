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

def ecdf_plot(x, ax, *args, **kwargs):
    xc = np.concatenate(([0], np.sort(x), [1]))
    y = np.linspace(0, 1, len(x) + 1)
    yc = np.concatenate(([0], y))
    ax.step(xc, yc, *args, **kwargs)

cls = ["-.", ":", '-', "--"]
clw = [2.0, 1.0, 3.0, 1.5, 0.5, 4.0]
clws = list(itertools.product(clw, cls))
colors = ['red', 'black', 'green', 'blue', 'yellow']

df = pd.DataFrame(list(Result.select().where(Result.complexity==1, Result.betat==0).dicts()))

#for db_size in np.sort(db_size_sample):

def plotcdfs(distribution, method, retrain_permutations, db_size,
        estimator, dfpvalues, i, ax):
    label = str(method)
    if label == 'permutation':
        label = 'COINP'
    if label == 'shuffle_once':
        label = 'CPI'
    if (not retrain_permutations) and method != "remove":
        label = "Approximate " + label

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

    if estimator == 'ann' and distribution == 0:
        ecdf_plot(pvals, ax, label=label, linestyle=clws[i[0]][1],
             lw=clws[i[0]][0], color=colors[i[0]])
    else:
        ecdf_plot(pvals, ax, linestyle=clws[i[0]][1],
             lw=clws[i[0]][0], color=colors[i[0]])
    i[0] += 1

dfpvalues = [
    "method", "retrain_permutations", "db_size", "estimator",
    "distribution", "pvalue"
    ]
dfpvalues = pd.DataFrame(columns=dfpvalues)

method_sample = ["permutation", "remove", "shuffle_once"]
method_sample = ["permutation", "shuffle_once"]

for db_size in [1_000, 10_000]:
    fig = plt.figure(figsize=[11.4, 16.9])
    axarr = fig.subplots(4, 3)
    fig.subplots_adjust(wspace=0.25, hspace=0.3)
    for distribution in range(4):
        for est_ind, estimator in enumerate(["ann", "rf", "linear"]):
            ax = axarr[distribution, est_ind]
            ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000))
            i = [0]
            for method in np.sort(method_sample):
                for retrain_permutations in [True, False]:
                    if retrain_permutations or not method == "remove":
                        plotcdfs(distribution, method,
                            retrain_permutations, db_size,
                            estimator, dfpvalues, i, ax)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('p-value')
            ax.set_ylabel('Cumulative probability')
            #ax.set_xlim(-0.1, 1.1)
            ax.set_title("Distribution " + str(distribution+1)
                + " (" + str(estimator).upper() + ")")
    #plt.setp([a.get_xticklabels() for a in axarr[1:3, :].reshape(-1)],
    #    visible=False)
    #plt.setp([a.get_yticklabels() for a in axarr[:, 1:2].reshape(-1)],
    #    visible=False)

    fig.legend(loc='upper center', borderaxespad=5.1, ncol=4
       , fancybox=True, shadow=True, columnspacing=6.5)

    filename = "plots/"
    filename += "null_db_size_of_"
    filename += str(db_size)
    filename += ".pdf"
    with PdfPages(filename) as ps:
        ps.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    print(dfpvalues.to_latex())
