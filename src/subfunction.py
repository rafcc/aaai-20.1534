# -*- coding: utf-8 -*-
import os

import numpy as np
import sympy
import yaml


def create_directory(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    else:
        pass


def nonzero_indices(a):
    return np.nonzero(a)[0]


def BezierIndex(dim, deg):
    """Iterator indexing control points of a Bezier simplex"""

    def iterate(c, r):
        if len(c) == dim - 1:
            yield c + (r,)
        else:
            for i in range(r, -1, -1):
                yield from iterate(c + (i,), r - i)

    yield from iterate((), deg)


def poly(i, n):
    eq = c(i)
    for k in range(n):
        eq *= T[k] ** i[k]
    return eq * (1 - sum(T[k] for k in range(n))) ** i[n]


def concat_data_to_arrray(d):
    index = 0
    for key in d:
        # if len(key)==1:
        #    d[key] == d[key].reshape((1,len(d[key])))
        if index == 0:
            d_array = d[key]
        else:
            d_array = np.r_[d_array, d[key]]
        index += 1
    return d_array


def extract_multiple_degree(degree_list, index_list):
    """
    return corresponding multiple indices as set
    indices: list or tuple
    """
    d = set()
    for key in degree_list:
        if set(np.array(nonzero_indices(key))).issubset(set(index_list)):
            d.add(key)
    return d


def calculate_l2_expected_error(true, pred):
    diff = true - pred
    # print(diff)
    # print(np.linalg.norm(diff,axis=1))
    # print((np.sum(np.linalg.norm(diff,axis=1)**2)))
    return (np.sum(np.linalg.norm(diff, axis=1) ** 2)) / diff.shape[0]


def define_monomial(deg_b, dim_b):
    global T
    global M
    global Mf
    global Mf_DIFF
    global Mf_DIFF2

    """T[dim]"""
    T = [sympy.Symbol("t" + str(i)) for i in range(dim_b - 1)]

    """M[dim][key]"""
    M = [
        {i: poly(i, n - 1) for i in BezierIndex(dim=n, deg=deg_b)}
        for n in range(2, dim_b + 1)
    ]
    Mf = []
    for n in range(2, dim_b + 1):
        Mf.append({})
        # print(n)
        for i in BezierIndex(dim=n, deg=deg_b):
            # print(i)
            f = poly(i, n - 1)
            b = compile(
                "Mf[-1][i] = lambda t0, t1=None, t2=None, t3=None: " + str(f),
                "<string>",
                "exec",
                optimize=2,
            )
            exec(b)
    """M_DIFF[dim][t][key]
        dim: 次元 dim=0 なら 2次元, dim=1 なら 3 次元
        t: 偏微分
        key: もとの単項式の次数
    """
    M_DIFF = [
        [{k: sympy.diff(v, t) for k, v in m.items()} for j, t in enumerate(T) if j <= i]
        for i, m in enumerate(M)
    ]
    Mf_DIFF = []
    for i, m in enumerate(M):
        Mf_DIFF.append([])
        # print(i,m)
        for j, t in enumerate(T):
            if j <= i:
                Mf_DIFF[-1].append({})
                for k, v in m.items():
                    f = sympy.diff(v, t)
                    b = compile(
                        "Mf_DIFF[-1][-1][k] = lambda t0, t1=None, t2=None, t3=None: "
                        + str(f),
                        "<string>",
                        "exec",
                        optimize=2,
                    )
                    exec(b)
    """M_DIFF2[dim][t][t][key]"""
    M_DIFF2 = [
        [
            [
                {k: sympy.diff(M_DIFF[i][j][k], t) for k, v in m.items()}
                for h, t in enumerate(T)
                if h <= i
            ]
            for j in range(i + 1)
        ]
        for i, m in enumerate(M)
    ]
    Mf_DIFF2 = []
    for i, m in enumerate(M):
        Mf_DIFF2.append([])
        for j in range(i + 1):
            Mf_DIFF2[-1].append([])
            for h, t in enumerate(T):
                if h <= i:
                    # print(i,j,h,t)
                    Mf_DIFF2[-1][-1].append({})
                    for k, v in m.items():
                        f = sympy.diff(M_DIFF[i][j][k], t)
                        b = compile(
                            "Mf_DIFF2[-1][-1][-1][k] = lambda t0, t1=None, t2=None, t3=None: "
                            + str(f),
                            "<string>",
                            "exec",
                            optimize=2,
                        )
                        exec(b)


def write_result(result, fname):
    f = open(fname, "w")
    f.write(yaml.dump(result, default_flow_style=False))
    f.close()
