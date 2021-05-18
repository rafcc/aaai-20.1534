# -*- coding: utf-8 -*-
from itertools import combinations
from typing import Optional, Set

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import model
import sampling
import subfunction


class Dataset:
    """Dataset.
    >>> d = Dataset('test/ConstrEx.pf')
    >>> d
    array([[0.9991    , 1.00090081],
           [0.9973    , 1.00270731],
           [0.9955    , 1.00452034],
           ...,
           [0.9946    , 1.00542932],
           [0.9964    , 1.00361301],
           [0.9982    , 1.00180325]])
    """

    def __init__(self, filename: Optional[str] = None) -> None:
        if filename:
            self.read(filename)
        else:
            self.values = np.array([])

    def __str__(self) -> str:
        return str(self.values)

    def __repr__(self) -> str:
        return repr(self.values)

    def read(self, filename: str) -> None:
        """Read the dataset from a file.

        Examples
        ========
        >>> d = Dataset()
        >>> d.read('test/ConstrEx.pf')
        >>> d
        array([[0.9991    , 1.00090081],
               [0.9973    , 1.00270731],
               [0.9955    , 1.00452034],
               ...,
               [0.9946    , 1.00542932],
               [0.9964    , 1.00361301],
               [0.9982    , 1.00180325]])
        """
        self.values = np.loadtxt(filename)

    def write(self, filename: str) -> None:
        """Write the dataset to a file.

        Examples
        ========
        >>> d = Dataset('test/ConstrEx.pf')
        >>> d.write('test/ConstrEx.pf')
        """
        np.savetxt(filename, self.values)

    def unique(self) -> "Dataset":
        """Drop duplicated lines.

        Examples
        ========
        >>> d = Dataset()
        >>> d.values = np.array([[1,2],[1,2],[3,4]])
        >>> d
        array([[1, 2],
               [1, 2],
               [3, 4]])

        Drop the duplicated line
        >>> d.unique()
        array([[1, 2],
               [3, 4]])
        """
        return np.unique(self.values, axis=0)

    def union(self, other: "Dataset") -> "Dataset":
        """Concat a given dataset after this dataset.

        Examples
        ========
        >>> d1 = Dataset()
        >>> d1.values = np.array([[1,2],[3,4]])
        >>> d1
        array([[1, 2],
               [3, 4]])

        >>> d2 = Dataset()
        >>> d2.values = np.array([[5,6],[7,8]])
        >>> d2
        array([[5, 6],
               [7, 8]])

        Drop the duplicated line
        >>> d1.union(d2)
        array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])
        """
        return np.append(self.values, other.values, axis=0)

    def difference(self, other: "Dataset") -> "Dataset":
        """Remove elements in a given dataset from this dataset.

        Examples
        ========
        >>> d1 = Dataset()
        >>> d1.values = np.array([[1,2],[3,4]])
        >>> d1
        array([[1, 2],
               [3, 4]])

        >>> d2 = Dataset()
        >>> d2.values = np.array([[3,4],[2,1]])
        >>> d2
        array([[3, 4],
               [2, 1]])

        Drop the duplicated line
        >>> d1.difference(d2)
        array([[1, 2]])
        """
        return (
            pd.concat(
                [
                    pd.DataFrame(self.values),
                    pd.DataFrame(other.values),
                    pd.DataFrame(other.values),
                ]
            )
            .drop_duplicates(keep=False)
            .values
        )

    def sample(self, n: int) -> "Dataset":
        """Remove elements in a given dataset from this dataset.

        Examples
        ========
        >>> d = Dataset()
        >>> d.values = np.array([[1,2],[3,4]])
        >>> d
        array([[1, 2],
               [3, 4]])

        Setup random seed
        >>> np.random.seed(42)

        Sample
        >>> d.sample(1)
        array([[3, 4]])
        """
        return pd.DataFrame(self.values).sample(n).values


def weak_pareto_filter(costs: Dataset, subproblem: Set[int]) -> Dataset:
    """
    Filter weakly Pareto-optimal points.

    Parameters
    ==========
    costs: An (n_points, n_costs) array

    Returns
    =======
    A filtered dataset

    Examples
    ========
    >>> d = Dataset()
    >>> d.values = np.array([
    ...     [1, 2],
    ...     [1, 3],
    ...     [2, 1]])

    All points are weakly Pareto-optimal
    >>> weak_pareto_filter(d, {0, 1})
    array([[1, 2],
           [1, 3],
           [2, 1]])

    The points with smallest F1 are [1, 2] and [1, 3]
    >>> weak_pareto_filter(d, {0})
    array([[1, 2],
           [1, 3]])

    The point with smallest F2 is [2, 1]
    >>> weak_pareto_filter(d, {1})
    array([[2, 1]])
    """
    indices = list(subproblem)
    subcosts = costs.values[:, indices]
    is_efficient = np.ones(subcosts.shape[0], dtype=bool)
    for i, c in enumerate(subcosts):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                subcosts[is_efficient] <= c, axis=1
            )  # Remove dominated points
    d = Dataset()
    d.values = costs.values[is_efficient]
    return d


def pareto_filter(costs: Dataset, subproblem: Set[int], eps: float = 0.0) -> Dataset:
    """
    Filter weakly Pareto-optimal points.

    Parameters
    ==========
    costs: An (n_points, n_costs) array
    subproblem: An index set of subproblems
    eps: epsilon dominance

    Returns
    =======
    A filtered dataset

    Examples
    ========
    >>> d = Dataset()
    >>> d.values = np.array([
    ...     [1, 2],
    ...     [1, 3],
    ...     [2, 1]])

    The Pareto points are [1, 2] and [2, 1]
    >>> pareto_filter(d, {0, 1}, eps=0.0)
    array([[1, 2],
           [2, 1]])

    The points with smallest F1 are [1, 2] and [1, 3]
    >>> pareto_filter(d, {0})
    array([[1, 2],
           [1, 3]])

    The point with smallest F2 is [2, 1]
    >>> pareto_filter(d, {1})
    array([[2, 1]])
    """
    indices = list(subproblem)
    subcosts = costs.values[:, indices]
    is_efficient = np.ones(subcosts.shape[0], dtype=bool)
    for i, c in enumerate(subcosts):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                [
                    np.all(subcosts[is_efficient] == c - eps, axis=1),
                    np.any(subcosts[is_efficient] < c - eps, axis=1),
                ],
                axis=0,
            )  # Remove dominated points
    d = Dataset()
    d.values = costs.values[is_efficient]
    return d


class Normalizer:
    """Normalize each axis of data to the unit interval [0, 1].

    Examples
    ========
    >>> raw = Dataset('test/ConstrEx.pf')
    >>> t = Normalizer(raw)

    Normalize data to the unit interval [0, 1]:
    >>> normlized = t.normalize(raw)

    Check if data are correctly normalized:
    >>> normlized.values.min(axis=0) == np.zeros(2)
    array([ True,  True])
    >>> normlized.values.max(axis=0) == np.ones(2)
    array([ True,  True])

    Denormalize data to the original range:
    >>> raw2 = t.denormalize(normlized)

    Check if data are correctly denormalized:
    >>> raw2.values.min(axis=0) == raw.values.min(axis=0)
    array([ True,  True])
    >>> raw2.values.max(axis=0) == raw.values.max(axis=0)
    array([ True,  True])
    """

    def __init__(self, raw: Dataset) -> None:
        self.mins = raw.values.min(axis=0)
        self.maxs = raw.values.max(axis=0)

    def normalize(self, raw: Dataset) -> Dataset:
        """Normalize data to the unit interval.
        """
        d = Dataset()
        d.values = (raw.values - self.mins) / (self.maxs - self.mins)
        return d

    def denormalize(self, normalized: Dataset) -> Dataset:
        """Denormalize data to the original range.
        """
        d = Dataset()
        d.values = normalized.values * (self.maxs - self.mins) + self.mins
        return d


class SyntheticData:
    def __init__(self, degree, dimspace, dimsimplex):
        self.degree = degree
        self.dimspace = dimspace
        self.dimsimplex = dimsimplex
        self.objective_function_indices_list = [i for i in range(self.dimsimplex)]

        self.subproblem_indices_list = []
        for i in range(1, len(self.objective_function_indices_list) + 1):
            for c in combinations(self.objective_function_indices_list, i):
                self.subproblem_indices_list.append(c)

        # prepare class to generate data points on bezier simplex
        self.uniform_sampling = sampling.UniformSampling(dimension=self.dimsimplex)
        self.bezier_simplex = model.BezierSimplex(
            dimSpace=self.dimspace, dimSimplex=self.dimsimplex, degree=self.degree
        )
        self.monomial_degree_list = [
            i for i in subfunction.BezierIndex(dim=self.dimsimplex, deg=self.degree)
        ]
        # generate true control points
        generate_control_point = model.GenerateControlPoint(
            dimSpace=self.dimspace, dimSimplex=self.dimsimplex, degree=self.degree
        )
        self.control_point_true = generate_control_point.simplex()

    def sampling_borges(self, n, seed, sigma):
        param = self.uniform_sampling.subsimplex(
            num_sample=n,
            indices=self.objective_function_indices_list,
            seed=(seed + 1) * len(self.objective_function_indices_list),
        )
        data = self.bezier_simplex.generate_points(c=self.control_point_true, tt=param)
        epsilon = np.random.multivariate_normal(
            [0 for i in range(self.dimspace)],
            np.identity(self.dimspace) * (sigma ** 2),
            n,
        )
        data = data + epsilon
        return (param, data)

    def sampling_inductive(self, n, sample_size_list, seed, sigma):
        param = {}
        data = {}
        for c in self.subproblem_indices_list:
            if len(c) <= min(self.dimsimplex, self.degree):
                n = sample_size_list[len(c) - 1]
                seed = seed + 30
                z = self.uniform_sampling.subsimplex(indices=c, num_sample=n, seed=seed)
                param[c] = z
                b = self.bezier_simplex.generate_points(
                    c=self.control_point_true, tt=param[c]
                )
                epsilon = np.random.multivariate_normal(
                    [0 for i in range(self.dimspace)],
                    np.identity(self.dimspace) * (sigma ** 2),
                    n,
                )
                data[c] = b + epsilon
        return (param, data)


if __name__ == "__main__":
    from itertools import combinations

    eps = -0.1
    dimsimplex = 5
    d = Dataset("../data/raw/S3TD.pf")
    indices_list = [i for i in range(dimsimplex)]
    subproblem_indices_list = []
