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
#along with this program. If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
import itertools
from collections import OrderedDict

def do_simulation_study(to_sample, func, db, Result, max_count=1):
    to_sample = OrderedDict(to_sample) # ensure order won't be messed up
    full_sample = set(itertools.product(*to_sample.values()))
    keys = list(to_sample)

    while full_sample:
        sample = np.random.choice(len(full_sample))
        sample = list(full_sample)[sample]
        dsample = dict(zip(keys, sample))


        # check count of rows in db
        query = Result.select().where(
            *[getattr(Result, x[0]) == x[1] for x in dsample.items()]
            )
        print(len(full_sample) * max_count, "combinations left")
        if query.count() >= max_count:
            full_sample.discard(sample)
            continue
        db.close()

        func_res = func(**dsample)
        print("Results:")
        print(func_res)

        Result.create(**dsample, **func_res)
        print("Result stored in the database", flush=True)
