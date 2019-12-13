#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import numpy as np


class Med(object):
    '''MED problem.
    >>> med = Med(dim=3, obj=3, convexity=2)
    >>> med([1, 0, 0])
    array([0., 1., 1.])

    >>> med = Med(dim=4, obj=3, convexity=2)
    >>> med([0, 1, 0, 0])
    array([1., 0., 1.])

    >>> med = Med(dim=4, obj=4, convexity=2)
    >>> med([0, 0, 1, 0])
    array([1., 1., 0., 1.])
    '''

    def __init__(self, dim, obj, convexity):
        self.e = np.eye(obj, dim)
        self.A = np.eye(dim)
        self.convexity = convexity

    def __call__(self, x):
        '''f_i(x) = ||A_i x - e_i||^2'''
        return np.array([self.f(i, x) for i in range(self.obj())])

    def f(self, i, x):
        return np.linalg.norm((self.A.dot(x) - self.e[i]) / np.sqrt(2), ord=2)**self.convexity

    def obj(self):
        return self.e.shape[0]

    def dim(self):
        return self.e.shape[1]


def x_star(w, f):
    '''w |--> (x^*, f(x^*))'''
    x = np.zeros(f.dim())
    x[:f.obj()] = w
    return x


def uniform_on_simplex(dim):
    '''Generate a point uniform randomly on a (dim-1)-simplex'''
    z = np.sort(np.random.uniform(0.0, 1.0, dim - 1))
    return np.append(z, 1.0) - np.append(0.0, z)


def runif_simplex(n, dim, face):
    '''Sample `n` points on a face of a simplex of dimension `dim-1`.'''
    t = np.zeros(shape=(n, dim))
    for i in range(n):
        t[i][face] = uniform_on_simplex(len(face))
    return t


def validate_subproblem(ctx, param, value):
    '''Check ordering and duplication of indices'''
    i_ = 0
    for i in value:
        if i <= i_:
            raise click.BadParameter(
                '{} cannot follow {} (They must be increasing)'.format(i, i_))
        i_ = i
    return value


@click.command(help='Sampling Pareto set/front of MED')
@click.argument('subproblem',
                type=click.IntRange(1), nargs=-1, required=True,
                callback=validate_subproblem)
@click.option('-d', '--dim',
              type=click.IntRange(1), required=True,
              help='Number of decision variables (value >= 1)')
@click.option('-o', '--obj',
              type=click.IntRange(1), required=True,
              help='Number of objective functions (1 <= value <= dim)')
@click.option('-c', '--convexity',
              type=float, required=True,
              help='Front shape parameter')
@click.option('-e', '--noise',
              type=click.FloatRange(0.), required=True,
              help='Standard deviation of Gaussian noise (value >= 0)')
@click.option('-n', '--size',
              type=click.IntRange(1), required=True,
              help='Sample size to generate (value >= 1)')
@click.option('-s', '--seed',
              type=click.IntRange(0, 2**32 - 1), required=True,
              help='Random seed (0 <= value <= 2**32-1)')
def main(subproblem, dim, obj, convexity, noise, size, seed):
    click.echo(
        'subproblem = {}\n'
        'dim = {}\n'
        'obj = {}\n'
        'convexity = {}\n'
        'noise = {}\n'
        'size = {}\n'
        'seed = {}\n'
        .format(subproblem, dim, obj, convexity, noise, size, seed))
    np.random.seed(seed)

    med = Med(dim, obj, convexity)
    weights = runif_simplex(n=size, dim=obj,
                            face=[i - 1 for i in subproblem])
    click.echo('Generating Pareto set/front samples...')
    np.set_printoptions(formatter={'float': '{:5.3f}'.format})

    def isf(x):
        if x is None:
            return 'done.'
        return 'x={}'.format(x)

    with click.progressbar(weights, show_pos=True, item_show_func=isf) as bar:
        decisions = [x_star(w, med)
            + np.random.normal(scale=noise, size=med.dim()) for w in bar]
    objectives = [med(x) for x in decisions]
    s = '_'.join(str(i) for i in subproblem)
    fn = 'MED,d_{},o_{},c_{:1.0e},e_{:1.0e},n_{},s_{}'.format(dim, obj, convexity, noise, size, seed) + ',{}_' + s
    np.savetxt(fn.format('w'), weights)
    np.savetxt(fn.format('x'), decisions)
    np.savetxt(fn.format('f'), objectives)


if __name__ == '__main__':
    main()
