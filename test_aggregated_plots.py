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
from plotnine import *
from db_structure import Result

df = pd.DataFrame(list(Result.select().where(
    (Result.betat == 0.01) | (Result.betat == 0.1) |
    (Result.betat == 0.6)
).dicts()))

def plotcdfs(df, distribution, power=0.05):
    idx1 = df['distribution'] == distribution
    idx2 = df['betat'] > 0
    idxs = np.logical_and(idx1, idx2)
    df = df[idxs]

    plot = ggplot()
    for db_size in [1000, 10000]:
        dfs = df[df['db_size'] == db_size]

        ccolor = '#555555'
        dodge_text = position_dodge(width=0.9)

        #aggregate
        dfs = dfs.groupby(["method", "estimator", 'betat', 'retrain_permutations'],
            as_index=False)["pvalue"].apply(
            lambda x: sum(x<0.05)/len(x)).reset_index()
        dfs = dfs.rename(columns = {0:'pvalue'})
        dfs['pvalue'] = np.round(dfs['pvalue'] * 100, 1)
        pvalue_max = np.max(dfs['pvalue'])

        #new column
        dfs['retrain_permutations'] = np.array(dfs['retrain_permutations'], dtype="str")
        to_append = map('\n'.join, zip(dfs["method"], dfs["retrain_permutations"]))
        dfs['retrain_and_method'] = list(to_append)


        dfs['betat'] = np.array(dfs['betat'], dtype="str")
        dfs['estimator'] = dfs["estimator"].apply(lambda x: x.upper())
        to_append = map(' and '.join, zip(dfs["betat"], dfs["estimator"]))
        dfs['betat_and_estimator'] = list(to_append)


        if db_size == 1000:
            plot += geom_col(dfs,
            aes(x='retrain_and_method', y='pvalue', fill='betat_and_estimator'),
            show_legend=True, position = "dodge",
            )

            plot += guides(fill=guide_legend(title="betat and \n estimator \n"))
        else:
            plot += geom_col(dfs,
            aes(x='retrain_and_method', y='pvalue', fill='betat_and_estimator'),
            show_legend=False, position = "dodge", alpha=0.0, color="#110011"
            )
            plot += scale_color_discrete(l=.4)

        #plot += geom_text(dfs,
        #             aes(label='pvalue', y='pvalue', x='retrain_and_method'),
        #             position=dodge_text, angle=45,
        #             size=8, va='bottom', format_string='{}%')

    plot += theme(panel_background=element_rect(fill='white'),               # new
             #axis_title_y=element_blank(),
             axis_line_x=element_line(color='black'),
             #axis_line_y=element_blank(),
             #axis_text_y=element_blank(),
             axis_text_x=element_text(color=ccolor, rotation=90),
             #axis_ticks_major_y=element_blank(),
             #axis_ticks_major_x=element_blank(),
             panel_grid=element_blank(),
             panel_border=element_blank(),
             )

    plot += ggtitle("Distribution " + str(distribution))
    plot += ylab("Test power")
    plot += xlab("Method and retrain")
    plot += lims(y=(0, np.max(dfs['pvalue'])+2))

    return plot

for distribution in range(3):
    for retrain_permutations in [True, False]:
        filename = "plots/"
        filename += "aggregated"
        filename += "_distribution" + str(distribution)
        filename += ".pdf"
        plotcdfs(df.copy(), distribution).save(filename)
