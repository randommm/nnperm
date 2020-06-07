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
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from db_structure import Result

def ecdf_plot(x, ax, *args, **kwargs):
    xc = np.concatenate(([0], np.sort(x), [1]))
    y = np.linspace(0, 1, len(x) + 1)
    yc = np.concatenate(([0], y))
    ax.step(xc, yc, *args, **kwargs)

cls = ["-", ":"]
clw = [2.0, 1.5, 1.0, 3.0, 0.5, 4.0]
clws = list(itertools.product(clw, cls))

df = pd.DataFrame(list(Result
    .select()
    .where(Result.complexity==1, Result.distribution in [4, 5])
    .dicts()
))

#for db_size in np.sort(db_size_sample):

def plotcdfs(db_size, betat, distribution):
    fig, axes = plt.subplots(2, 2, figsize=[8.4, 5.8])
    axes = axes.flatten()
    for idx, estimator in enumerate(["ann", "rf", "linear"]):
        ax = axes[idx]
        method_sample = [
            ["COINP", "permutation", True],
            ["CPI", "cpi", False],
            ["SCPI", "shuffle_once", True],
        ]
        betat_sample = [0, betat]

        ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000))
        ax.set_title(estimator)

        i = 0
        for method in method_sample:
            for betat in np.sort(betat_sample):
                label = "$\\beta_s$ = " + str(betat)
                label += " and "
                #label += str(db_size) + " instances"
                label += method[0]

                idx1 = df['betat'] == betat
                idx2 = df['db_size'] == db_size
                idx3 = df['retrain_permutations'] == method[2]
                idx4 = df['method'] == method[1]
                idx5 = df['estimator'] == estimator
                idx6 = df['distribution'] == distribution
                idxs = np.logical_and(idx1, idx2)
                idxs = np.logical_and(idxs, idx3)
                idxs = np.logical_and(idxs, idx4)
                idxs = np.logical_and(idxs, idx5)
                idxs = np.logical_and(idxs, idx6)
                pvals = np.sort(df[idxs]['pvalue'])

                ecdf_plot(pvals, ax, label=label,
                    linestyle=clws[i][1], lw=clws[i][0])
                i += 1

    legend = ax.legend(bbox_to_anchor=(1.34, 0.16), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

    axes[3].get_xaxis().set_visible(False)
    axes[3].get_yaxis().set_visible(False)
    axes[3].set_frame_on(False)
    fig.subplots_adjust(hspace = 0.3)

    filename = "plots/"
    filename += "dist_" + str(distribution+1)
    filename += "_and_db_size_" + str(db_size)
    filename += "_and_betat_" + str(betat)[-1]
    filename += ".pdf"
    with PdfPages(filename) as ps:
        ps.savefig(fig, bbox_inches='tight')
    plt.close(fig)

for db_size in [1_000]:
    for betat in [0.1, 0.6]:
        for distribution in [4, 5]:
            plotcdfs(db_size, betat, distribution)
