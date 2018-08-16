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

cls = ["-", ":", "-.", "--"]
clw = [1.0, 2.0, 1.5, 3.0, 0.5, 4.0]
clws = list(itertools.product(clw, cls))

df = pd.DataFrame(list(Result.select().dicts()))

#for db_size in np.sort(db_size_sample):

def plotcdfs(distribution, retrain_permutations, db_size, estimator):

    method_sample = ["permutation", "remove", "shuffle_once"]
    betat_sample = [0, 0.05, 0.3, 0.6]
    if not retrain_permutations:
        method_sample.remove("remove")

    ax = plt.figure(figsize=[8.4, 5.8]).add_subplot(111)
    ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000))

    i = 0
    for method in np.sort(method_sample):
        for betat in np.sort(betat_sample):
            label = "betat = " + str(betat)
            label += " and "
            #label += str(db_size) + " instances"
            label += method

            idx1 = df['betat'] == betat
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

            # Uncomment to plot two-tailed tests
            #for j in range(len(pvals)):
            #    pvals[j] = 2 * pvals[j] if pvals[j] <= 0.5 else 2 * (1 - pvals[j])
            #pvals = np.sort(pvals)

            ax.step(pvals, np.linspace(0, 1, len(pvals), False), label=label,
                linestyle=clws[i][1], lw=clws[i][0])
            i += 1

    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    filename = "plots/"
    filename += "estimator_" + str(estimator)
    filename += "_and_distribution_" + str(distribution)
    filename += "_and_retrain_permutations_" + str(retrain_permutations)
    filename += "_and_db_size_" + str(db_size)
    filename += ".pdf"
    with PdfPages(filename) as ps:
        ps.savefig(ax.get_figure(), bbox_inches='tight')
    plt.close(ax.get_figure())

for distribution in range(3):
    for retrain_permutations in [True, False]:
        for db_size in [1_000, 10_000]:
            for estimator in ["ann", "rf"]:
                plotcdfs(distribution, retrain_permutations, db_size,
                    estimator)
