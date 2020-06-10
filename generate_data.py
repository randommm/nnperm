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
    if distribution <= 1:
        return generate_data_01(n_gen, betat, distribution)
    elif distribution == 2:
        return generate_data_2(n_gen, betat)
    elif distribution == 3:
        return generate_data_3(n_gen, betat)
    elif distribution == 4:
        return generate_data_4(n_gen, betat)
    elif distribution == 5:
        return generate_data_5(n_gen, betat)

def generate_data_2(n_gen, betat):
    beta = [3, np.nan]
    beta[1] = betat

    cov_matrix = [[1, 0.9], [0.9, 1]]
    x_gen = stats.multivariate_normal.rvs(cov=cov_matrix, size=n_gen)

    mu_gen = np.dot(x_gen, beta)
    y_gen = stats.norm.rvs(scale=0.5, size=n_gen)
    y_gen = mu_gen + y_gen

    assert(y_gen.shape == (n_gen,))
    assert(x_gen.shape == (n_gen, 2))

    return x_gen, y_gen

def generate_data_3(n_gen, betat):
    beta = [3, np.nan]
    beta[1] = betat

    x_gen = stats.beta.rvs(1, 1, size=n_gen * 2).reshape((n_gen, 2))
    x_gen += stats.norm.rvs(-0.5, size=n_gen).reshape((n_gen, 1))

    mu_gen = np.dot(x_gen, beta)
    y_gen = stats.beta.rvs(2, 2)
    y_gen = mu_gen + y_gen

    assert(y_gen.shape == (n_gen,))
    assert(x_gen.shape == (n_gen, 2))

    return x_gen, y_gen

def generate_data_4(n_gen, betat):
    beta = [np.nan, 1]
    beta[0] = betat

    x_gen_1 = stats.norm.rvs(size=n_gen)
    x_gen_2 = x_gen_1 ** 2
    x_gen = np.column_stack((x_gen_1, x_gen_2))

    mu_gen = np.dot(x_gen, beta)
    y_gen = stats.norm.rvs(size=n_gen)
    y_gen = mu_gen + y_gen

    assert(y_gen.shape == (n_gen,))
    assert(x_gen.shape == (n_gen, 2))

    return x_gen, y_gen

def generate_data_5(n_gen, betat):
    x_gen_1 = stats.norm.rvs(size=[n_gen, 1])
    x_gen_2 = stats.norm.rvs(size=[n_gen, 1])
    x_gen = np.column_stack((x_gen_1, x_gen_2))

    mu_gen = x_gen_1 * x_gen_2 * betat + x_gen_2**2
    y_gen = stats.norm.rvs(size=[n_gen, 1])
    y_gen = mu_gen + y_gen

    assert(y_gen.shape == (n_gen, 1))
    y_gen = y_gen[:, 0]

    assert(y_gen.shape == (n_gen, ))
    assert(x_gen.shape == (n_gen, 2))

    return x_gen, y_gen

def generate_data_01(n_gen, betat, distribution):
    x_dim = 5

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

    y_gen = mu_gen + y_gen

    assert(y_gen.shape == (n_gen,))

    if distribution == 0:
        x_gen = x_gen[:, [2, 3]]
        assert(x_gen.shape == (n_gen, 2))
    else:
        assert(x_gen.shape == (n_gen, 5))

    return x_gen, y_gen
