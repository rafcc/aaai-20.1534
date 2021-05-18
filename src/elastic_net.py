#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import numpy as np
from sklearn.linear_model import ElasticNet


def weight_to_alpha_rho(w, eps):
    """Convert (w1,w2,w3) to (alpha,rho)"""
    v = w + eps
    alpha = (v[1] + v[2]) / v[0]
    l1_ratio = v[1] / (v[1] + v[2])
    return alpha, l1_ratio


def f1(regr, X, y):
    """f1(x) = ||Ax - y||^2 / N"""
    y_ = regr.predict(X)
    return np.linalg.norm(y_ - y, ord=2) ** 2 / len(X)


def f2(regr):
    """f2(x) = |x|"""
    return np.linalg.norm(regr.coef_, ord=1)


def f3(regr):
    """f3(x) = ||x||^2"""
    return np.linalg.norm(regr.coef_, ord=2) ** 2


def xf_star(w, X, y, eps, seed):
    """w |--> (x^*, f(x^*))"""
    alpha, l1_ratio = weight_to_alpha_rho(w, eps)
    regr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed)
    regr.fit(X, y)
    f = np.array([f1(regr, X, y), f2(regr), f3(regr)])
    return regr.coef_, f


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


@click.command(help="Sampling Pareto set/front of 3-objective Elastic net")
@click.argument("data", type=click.Path(exists=True, dir_okay=False))
@click.argument(
    "subproblem",
    type=click.IntRange(1, 3),
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
def main(data, subproblem, size, eps, seed):
    click.echo(
        "data = {}\n"
        "subproblem = {}\n"
        "size = {}\n"
        "eps = {}\n"
        "seed = {}\n".format(data, subproblem, size, eps, seed)
    )
    np.random.seed(seed)

    yX = np.loadtxt(data, delimiter=",", skiprows=1)
    y = yX[:, 0]  # The 1st column is an output variable
    X = yX[:, 1:]  # The remaining columns are input variables

    weights = runif_simplex(n=size, dim=3, face=[i - 1 for i in subproblem])

    click.echo("Generating Pareto set/front samples...")
    np.set_printoptions(formatter={"float": "{:5.3f}".format})

    def isf(w):
        if w is None:
            return "done."
        alpha, rho = weight_to_alpha_rho(w, eps)
        return "w={}, [alpha rho]=[{:9.3f} {:5.3f}]".format(w, alpha, rho)

    with click.progressbar(weights, show_pos=True, item_show_func=isf) as bar:
        results = [xf_star(w, X, y, eps, seed) for w in bar]
    decisions = [x for x, _ in results]
    objectives = [f for _, f in results]
    s = "_".join(str(i) for i in subproblem)
    np.savetxt(data + ",w_" + s, weights)
    np.savetxt(data + ",x_" + s, decisions)
    np.savetxt(data + ",f_" + s, objectives)


if __name__ == "__main__":
    main()
