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
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from db_structure import Result

df = pd.DataFrame(list(Result
    .select()
    .where(
        Result.complexity==1,
        Result.method=="permutation",
        Result.retrain_permutations==True,
        Result.estimator=="ann"
    )
    .dicts()
))

def ecdf(x, ax, *args, **kwargs):
    xc = np.concatenate(([0], np.sort(x), [1]))
    y = np.linspace(0, 1, len(x) + 1)
    yc = np.concatenate(([0], y))
    ax.step(xc, yc, *args, **kwargs)

cls = ["-.", ":", '-', "--", "-.", ":", '-', "--"]
clw = [2.0, 1.0, 3.0, 1.5, 0.5, 4.0, 2.0, 1.0, 3.0, 1.5, 0.5, 4.0]
clws = list(itertools.product(clw, cls))
colors = ['red', 'black', 'green', 'blue', 'yellow']

def plotcdfs(distribution, db_size, ax, idx):
    betat_sample = [0, 0.01, 0.1, 0.6]

    for betat in np.sort(betat_sample):
        label = str(betat)
        idx1 = df['db_size'] == db_size
        idx2 = df['distribution'] == distribution
        idx3 = df['betat'] == betat
        idxs = np.logical_and(idx1, idx2)
        idxs = np.logical_and(idxs, idx3)
        pvals = np.sort(df[idxs]['pvalue'])

        test_unif = stats.kstest(pvals, 'uniform')
        test_unif = test_unif.pvalue
        test_unif = np.round(test_unif, 2)

        if distribution == 0:
            ecdf(pvals, ax, label=label)
        else:
            ecdf(pvals, ax)

fig = plt.figure(figsize=[11.4, 16.9])
axarr = fig.subplots(4, 2)
fig.subplots_adjust(wspace=0.25, hspace=0.3)

idx = 0
for est_ind, db_size in enumerate([1_000, 10_000]):
    for distribution in range(4):
        ax = axarr[distribution, est_ind]
        ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000))
        plotcdfs(distribution, db_size, ax, idx)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('p-value')
        ax.set_ylabel('Cumulative probability')
        #ax.set_xlim(-0.1, 1.1)
        ax.set_title("Distribution " + str(distribution+1)
            + " with " + str(db_size).upper() + " instances")
        idx += 1

#plt.setp([a.get_xticklabels() for a in axarr[1:3, :].reshape(-1)],
#    visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 1:2].reshape(-1)],
#    visible=False)

fig.legend(loc='upper center', borderaxespad=1.8, ncol=2,
    fancybox=True, shadow=True, columnspacing=6.5,
    title="$\\beta_\\kappa$")

filename = "plots/coinp_only.pdf"
with PdfPages(filename) as ps:
    ps.savefig(fig, bbox_inches='tight')
plt.close(fig)

