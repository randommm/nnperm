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
from db_structure_real_data import ResultRealData, db
import pickle
import collections

cls = ["-", ":", "-.", "--"]
clw = [1.0, 2.0, 1.5, 3.0, 0.5, 4.0]
clws = list(itertools.product(clw, cls))

for include_extra in range(3):
    df = pd.DataFrame(list(ResultRealData
        .select()
        .where(ResultRealData.include_extra==include_extra)
        .dicts()
    ))

    def renamer(x):
        label, retrain_permutations = x
        if label == 'permutation':
            label = 'COINP'
        if label == 'shuffle_once':
            label = 'SCPI'
        if label == 'cpi':
            label = 'CPI'
        if (not retrain_permutations) and label not in ["remove",'CPI']:
            label = "Approx " + label
        return label

    nc = df[['method', 'retrain_permutations']].apply(renamer, axis=1)
    df[['method']] = nc

    df['feature_tested'] = df['feature_tested'].apply(pickle.loads)
    ext = lambda x: (x[0]
        if isinstance(x, collections.abc.Sequence)
        else x)
    df['feature_tested'] = df['feature_tested'].apply(ext)

    df['estimator_method'] = df[['estimator', 'method']].apply(
        lambda x: (x[0], x[1]),
        axis=1
        )

    datap = dict()
    for estimator_method in np.sort(np.unique(df['estimator_method'])):
        valsp = []
        for feature_tested in np.unique(df['feature_tested']):
            try:
                valp = df[df['estimator_method'] == estimator_method]
                valp = valp[valp['feature_tested'] == feature_tested]
                valp = valp.pvalue.iloc[[0]].item()
            except (ValueError, IndexError):
                valp = np.nan

            valsp.append("{0:.2f}".format(valp))

        datap[estimator_method] = valsp


    dfp = pd.DataFrame.from_dict(datap)
    #dfp.index = [str(x) for x in range(1, 10)]
    columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
    columns.extend(['cut', 'color', 'clarity'])
    if include_extra == 1:
        columns.append('c+n')
    if include_extra == 2:
        columns.append('c+c+n')
    dfp = dfp.transpose()
    dfp.columns = columns

    print(dfp)
    print(dfp.to_latex(multirow=True))
