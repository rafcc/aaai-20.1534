#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

import click
import numpy as np
import pandas as pd
from grouplasso import GroupLassoRegressor
from numba import f8, i2, jit
from sklearn.preprocessing import MinMaxScaler


def add_intercept(X):
    """
    add intercept to X
    """
    return np.c_[X, np.ones(len(X))]


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


@jit(f8[:](f8[:], f8[:], i2[:]))
def _proximal_operator(coef, thresholds, group_ids):
    """
    Proximal operator.
    """
    result = np.zeros_like(coef).astype(np.float64)
    for i in range(group_ids[-1] + 1):
        group_idx = group_ids == i
        group_coef = coef[group_idx]
        group_size = np.sum(group_idx)
        group_norm = np.linalg.norm(group_coef, ord=2) / np.sqrt(group_size)
        multiplier = 0 if group_norm == 0 else max(0, 1 - thresholds[i] / group_norm)
        result[group_idx] = multiplier * group_coef

    return result


@jit(f8(f8[:], f8[:], i2[:]))
def _group_lasso_penalty(alphas, coef, group_ids):
    penalty = 0.0
    for i in range(group_ids[-1] + 1):
        group_idx = group_ids == i
        group_size = np.sum(group_idx)
        penalty += (
            alphas[i] * np.linalg.norm(coef[group_idx], ord=2) / np.sqrt(group_size)
        )

    return penalty


class MoGroupLassoRegressor(GroupLassoRegressor):
    def fit(self, X, y):
        if not isinstance(self.group_ids, np.ndarray):
            raise TypeError("group_ids must be numpy.array.")

        if self.group_ids.dtype != np.int:
            raise TypeError("type of group_id must be int.")

        if np.any(self.alpha <= 0):
            raise ValueError("alpha must be greater than zero.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        self._losses.clear()

        alpha = self.alpha
        group_ids = self.group_ids.astype(np.int16)
        n_samples = len(X)
        X = add_intercept(X)
        n_features = X.shape[1]
        if self.initial_weights is not None:
            if self.initial_weights.shape != (n_features,):
                raise ValueError("initial_weights must have shape (n_features, ).")
            w = self.initial_weights.copy()
        else:
            w = np.zeros(n_features)
        thresh = self.eta * alpha
        itr = 0
        while itr < self.max_iter:
            w_old = w.copy()
            pred = X @ w
            if self.verbose and itr % self.verbose_interval == 0:
                penalty = _group_lasso_penalty(alpha, w[:-1], group_ids)
                loss = mean_squared_error(y, pred) + penalty
                self._losses.append(loss)
                print("training loss:", loss)

            diff = 1 / n_samples * X.T @ (pred - y)
            out = w - self.eta * diff

            w[:-1] = _proximal_operator(out[:-1], thresh, group_ids)
            w[-1] = out[-1]

            if np.linalg.norm(w_old - w, 2) / self.eta < self.tol:
                if self.verbose:
                    print("Converged. itr={}".format(itr))
                break

            itr += 1

        if itr >= self.max_iter:
            warnings.warn("Failed to converge. Increase the " "number of iterations.")
        self.coef_ = w[:-1]
        self.intercept_ = w[-1]
        self.n_iter_ = itr


class Problem(object):
    def __init__(self, group_ids, X, y, rho, eps, seed):
        self.group_ids = group_ids
        self.X = X
        self.y = y
        self.rho = rho
        self.eps = eps
        self.seed = seed

    def __call__(self, w):
        """w |--> (x^*, f(x^*))"""
        alpha = weight_to_alpha(w, self.rho, self.eps)
        X = self.X
        y = self.y
        regr = MoGroupLassoRegressor(
            group_ids=self.group_ids, alpha=alpha, random_state=self.seed, verbose=False
        )
        regr.fit(X, y)
        f = np.array([err(regr, X, y)] + [reg(regr, i) for i in range(self.groups())])
        return regr.coef_, f

    def groups(self):
        """Return the number of groups"""
        return self.group_ids[-1] + 1

    def objectives(self):
        """Return the number of objectives"""
        return self.groups() + 1


def weight_to_alpha(w, rho, eps):
    """Convert weight to alpha"""
    v = w + eps
    alpha = v[1:] / v[0]
    return alpha * rho


def err(regr, X, y):
    """f1(x) = ||Ax - y||^2 / N"""
    y_ = regr.predict(X)
    return np.linalg.norm(y_ - y, ord=2) ** 2 / len(X)


def reg(regr, i):
    """f2(x) = ||x||^2 / #x"""
    grp = regr.group_ids == i
    return np.linalg.norm(regr.coef_[grp], ord=2) ** 2 / sum(grp)


def uniform_on_simplex(dim):
    """Generate a point uniform randomly on a (dim-1)-simplex"""
    z = np.sort(np.random.uniform(0.0, 1.0, dim - 1))
    return np.append(z, 1.0) - np.append(0.0, z)


def runif_simplex(n, dim, face):
    """Sample `n` points on a face of a simplex of dimension `dim-1`."""
    t = np.zeros(shape=(n, dim))
    for i in range(n):
        t[i][face] = uniform_on_simplex(len(face))
    return t


def validate_subproblem(ctx, param, value):
    """Check ordering and duplication of indices"""
    i_ = 0
    for i in value:
        if i <= i_:
            raise click.BadParameter(
                "{} cannot follow {} (They must be increasing)".format(i, i_)
            )
        i_ = i
    return value


@click.command(help="Sampling Pareto set/front of multi-objective group lasso")
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.argument(
    "groups",
    type=click.Path(exists=True, dir_okay=False),
    callback=lambda ctx, param, f: np.loadtxt(f, delimiter=",", skiprows=1, dtype=int),
)
@click.argument(
    "subproblem",
    type=click.IntRange(1),
    nargs=-1,
    required=True,
    callback=validate_subproblem,
)
@click.option(
    "-n",
    "--size",
    type=click.IntRange(1),
    required=True,
    help="Sample size to generate (value >= 1)",
)
@click.option(
    "-r",
    "--rho",
    type=click.FloatRange(0.0),
    required=True,
    help="Regularization coefficient (value >= 0)",
)
@click.option(
    "-e",
    "--eps",
    type=click.FloatRange(0.0),
    required=True,
    help="Epsilon to avoid numerical instability (value >= 0)",
)
@click.option(
    "-s",
    "--seed",
    type=click.IntRange(0, 2 ** 32 - 1),
    required=True,
    help="Random seed (0 <= value <= 2**32-1)",
)
def main(data, groups, subproblem, size, rho, eps, seed):
    click.echo(
        "data = {}\n"
        "groups = {}\n"
        "subproblem = {}\n"
        "size = {}\n"
        "rho = {}\n"
        "eps = {}\n"
        "seed = {}\n".format(data, groups, subproblem, size, rho, eps, seed)
    )
    np.random.seed(seed)

    yX = np.loadtxt(data, delimiter=",", skiprows=1)
    mm = MinMaxScaler()
    yX = mm.fit_transform(yX)
    y = yX[:, 0]  # The 1st column is an output variable
    X = yX[:, 1:]  # The remaining columns are input variables

    xf_star = Problem(groups, X, y, rho, eps, seed)
    weights = runif_simplex(
        n=size, dim=xf_star.objectives(), face=[i - 1 for i in subproblem]
    )

    click.echo("Generating Pareto set/front samples...")
    np.set_printoptions(formatter={"float": "{:5.3f}".format})

    def isf(w):
        if w is None:
            return "done."
        alpha = weight_to_alpha(w, rho, eps)
        return "w={}, alpha={}]".format(w, alpha)

    with click.progressbar(weights, show_pos=True, item_show_func=isf) as bar:
        results = [xf_star(w) for w in bar]
    decisions = [x for x, _ in results]
    objectives = [f for _, f in results]
    s = "_".join(str(i) for i in subproblem)
    fn = (
        "{},n_{},r_{:1.0e},e_{:1.0e},s_{}".format(data, size, rho, eps, seed)
        + ",{}_"
        + s
    )
    np.savetxt(fn.format("w"), weights)
    np.savetxt(fn.format("x"), decisions)
    np.savetxt(fn.format("f"), objectives)


if __name__ == "__main__":
    main()
