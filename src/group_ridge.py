#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

import click
import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
from numba import f8, i2, jit
from sklearn.base import BaseEstimator, RegressorMixin


def add_intercept(X):
    """
    add intercept to X
    """
    return np.c_[X, np.ones(len(X))]


def mean_squared_error(alpha, y_true, y_pred):
    return alpha * np.mean((y_true - y_pred) ** 2)


@jit(f8(f8[:], f8[:], i2[:]))
def _group_lasso_penalty(alphas, coef, group_ids):
    penalty = 0.0
    for i in range(group_ids[-1] + 1):
        group_idx = group_ids == i
        group_size = np.sum(group_idx)
        penalty += (
            alphas[i + 1] * np.linalg.norm(coef[group_idx], ord=2) ** 2 / group_size
        )

    return penalty


@jit(f8[:](f8[:], f8[:, :], f8[:], f8[:], i2[:]))
def _grad(alphas, X, y, w, group_ids):
    """
    Gradient.
    """
    n_samples = len(X)
    err = X.T @ (X @ w - y)
    result = alphas[0] / n_samples * err
    for i in range(group_ids[-1] + 1):
        group_idx = group_ids == i
        group_coef = w[group_idx]
        group_size = np.sum(group_idx)
        result[group_idx] += alphas[i + 1] / group_size * group_coef

    return result


class MoGroupLassoRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        group_ids,
        random_state=None,
        alpha=1e-3,
        eta=1e-1,
        tol=1e-7,
        max_iter=1000,
        initial_weights=None,
        verbose=True,
        verbose_interval=1,
    ):
        self.group_ids = group_ids
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self.alpha = alpha
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        self.initial_weights = initial_weights
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self._losses = []

    def fit(self, X, y):
        if not isinstance(self.group_ids, np.ndarray):
            raise TypeError("group_ids must be numpy.array.")

        if self.group_ids.dtype != np.int:
            raise TypeError("type of group_id must be int.")

        if np.any(self.alpha < 0):
            raise ValueError("alpha must be non-negative.")

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
            # w = np.random.normal(size=n_features)
            w = np.zeros(n_features)
        itr = 0
        while itr < self.max_iter:
            w_old = w.copy()
            w -= self.eta * _grad(alpha, X, y, w, group_ids)

            pred = X @ w
            if self.verbose and itr % self.verbose_interval == 0:
                mse = mean_squared_error(alpha[0], y, pred)
                penalty = _group_lasso_penalty(alpha, w[:-1], group_ids)
                self._losses.append(mse + penalty)
                print("training loss: {} = {} + {}".format(mse + penalty, mse, penalty))

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

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class Problem(object):
    def __init__(
        self, group_ids, X, y, rho, eps, mse_eps, seed, eta, tol, max_iter, verbose
    ):
        self.group_ids = group_ids
        self.X = X
        self.y = y
        self.rho = rho
        self.eps = eps
        self.mse_eps = mse_eps
        self.seed = seed
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def __call__(self, w):
        """w |--> (x^*, f(x^*))"""
        alpha = weight_to_alpha(w, self.rho, self.eps, self.mse_eps)
        X = self.X
        y = self.y
        regr = MoGroupLassoRegressor(
            group_ids=self.group_ids,
            random_state=self.seed,
            alpha=alpha,
            eta=self.eta,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
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


def weight_to_alpha(w, rho, eps, mse_eps):
    """Convert weight to alpha"""
    alpha = w.copy()
    alpha[0] += mse_eps
    alpha[1:] += eps
    alpha[1:] *= rho
    return alpha


def err(regr, X, y):
    """f1(x) = ||Ax - y||^2 / N"""
    y_ = regr.predict(X)
    return np.linalg.norm(y_ - y, ord=2) ** 2 / len(X)


def reg(regr, i):
    """f2(x) = ||x||^2 / #x"""
    grp = regr.group_ids == i
    return np.linalg.norm(regr.coef_[grp], ord=2) ** 2 / np.sum(grp)


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
    help="Epsilon added to REG terms (value >= 0)",
)
@click.option(
    "-m",
    "--mse-eps",
    type=click.FloatRange(0.0),
    default=0,
    help="Epsilon added to MSE term (value >= 0)",
)
@click.option(
    "-s",
    "--seed",
    type=click.IntRange(0, 2 ** 32 - 1),
    required=True,
    help="Random seed (0 <= value <= 2**32-1)",
)
@click.option(
    "-z",
    "--normalize",
    type=click.Choice(["standard", "minmax", "none"]),
    default="standard",
    help="Normalizer",
)
@click.option(
    "-l",
    "--eta",
    type=click.FloatRange(0.0),
    default=1,
    help="Learning rate (value >= 0)",
)
@click.option(
    "-t",
    "--tol",
    type=click.FloatRange(0.0),
    default=1e-7,
    help="Tolerance (value >= 0)",
)
@click.option(
    "-i",
    "--iteration",
    type=click.IntRange(1),
    default=1000,
    help="Max iteration of grad descent",
)
@click.option("-v", "--verbose", type=bool, default=False, help="Verbosity")
def main(
    data,
    groups,
    subproblem,
    size,
    rho,
    eps,
    mse_eps,
    seed,
    normalize,
    eta,
    tol,
    iteration,
    verbose,
):
    click.echo(
        "data = {}\n"
        "groups = {}\n"
        "subproblem = {}\n"
        "size = {}\n"
        "rho = {}\n"
        "reg_eps = {}\n"
        "mse_eps = {}\n"
        "seed = {}\n"
        "normalize = {}\n"
        "eta = {}\n"
        "tol = {}\n"
        "iteration = {}\n".format(
            data,
            groups,
            subproblem,
            size,
            rho,
            eps,
            mse_eps,
            seed,
            normalize,
            eta,
            tol,
            iteration,
        )
    )
    np.random.seed(seed)

    normalizers = {"standard": sp.scale, "minmax": sp.minmax_scale, "none": lambda x: x}
    normalizer = normalizers[normalize]
    yX = np.loadtxt(data, delimiter=",", skiprows=1)
    yX = normalizer(yX)
    y = yX[:, 0]  # The 1st column is an output variable
    X = yX[:, 1:]  # The remaining columns are input variables

    xf_star = Problem(
        groups, X, y, rho, eps, mse_eps, seed, eta, tol, iteration, verbose
    )
    weights = runif_simplex(
        n=size, dim=xf_star.objectives(), face=[i - 1 for i in subproblem]
    )

    click.echo("Generating Pareto set/front samples...")
    np.set_printoptions(formatter={"float": "{:5.3f}".format})

    def isf(w):
        if w is None:
            return "done."
        alpha = weight_to_alpha(w, rho, eps, mse_eps)
        return "w={}, alpha={}]".format(w, alpha)

    with click.progressbar(weights, show_pos=True, item_show_func=isf) as bar:
        results = [xf_star(w) for w in bar]
    decisions = [x for x, _ in results]
    objectives = [f for _, f in results]
    s = "_".join(str(i) for i in subproblem)
    fn = (
        "{},n_{},r_{:1.0e},e_{:1.0e},m_{:1.0e},s_{},l_{:1.0e},t_{:1.0e},i_{}".format(
            data, size, rho, eps, mse_eps, seed, eta, tol, iteration
        )
        + ",{}_"
        + s
    )
    np.savetxt(fn.format("w"), weights)
    np.savetxt(fn.format("x"), decisions)
    np.savetxt(fn.format("f"), objectives)


if __name__ == "__main__":
    main()
