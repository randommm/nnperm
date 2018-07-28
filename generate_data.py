#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 2 or 3 the License.
#
#Obs.: note that the other files are licensed under GNU GPL 3. This
#file is licensed under GNU GPL 2 or 3 for compatibility with flexcode
#license only.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You can get a copy of the GNU General Public License version 2 at
#<http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats

def generate_data(n_gen, betat, distribution):
    if distribution == 0:
        x_dim = 5
    elif distribution == 1:
        x_dim = 50

    beta = stats.norm.rvs(size=x_dim, scale=0.4, random_state=(x_dim-5))
    beta0 = -.3
    sigma = 1.1

    beta[3] = betat

    def func(x):
        x_transf = x.copy()
        for i in range(0, x_dim, 5):
            x_transf[i] = np.abs(x[i]) ** 1.3
            x_transf[i+1] = np.cos(x[i+1])
            x_transf[i+2] = np.log(np.abs(x[i]*x[i+2]))
            x_transf[i+3] = np.log(np.abs(x[i+3]))
            x_transf[i+4] = np.sqrt(np.abs(x[i+4]))
        return np.dot(beta, x_transf)

    x_gen = stats.skewnorm.rvs(scale=0.1, size=n_gen*x_dim, a=2)
    x_gen = x_gen.reshape((n_gen, x_dim))
    mu_gen = np.apply_along_axis(func, 1, x_gen)

    y_gen = stats.skewnorm.rvs(loc=beta0, scale=sigma, size=n_gen, a=4)
    y_gen = mu_gen + y_gen

    return x_gen, y_gen
