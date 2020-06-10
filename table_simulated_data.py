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
from db_structure import Result, db
import os

pd.set_option('display.max_rows', 1000)

if 'estimator' in os.environ:
    estimator = [os.environ['estimator']]
else:
    estimator = ["ann", "rf", "linear"]

df = pd.DataFrame(list(Result
    .select()
    .where(
        Result.estimator == estimator,
        Result.method != 'remove',
        (Result.distribution not in [4,5]) | (Result.betat != 0.01),
        (Result.distribution not in [4,5]) | (Result.db_size == 1000),
    )
    .dicts()
))
del df['id']
assert all(df['complexity']==1)
del df['complexity']

to_group = ['distribution', 'db_size', 'betat',
    'estimator', 'method', 'retrain_permutations']

def mpse(data):
    if all([x == '-' for x in data]):
        return '-'
    mean = data.mean()
    std_error = np.std(data) / np.sqrt(len(data))
    return "{0:.3f} ({1:.3f})".format(mean, std_error)

gdf = df.groupby(to_group).agg(mpse)

count = df.groupby(to_group).count().iloc[:,-1]
gdf['nsim'] = count

print(gdf)
