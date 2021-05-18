import pickle
from functools import lru_cache

import numpy as np
import sympy

import subfunction


def c(ixs):
    return sum(range(1, sum(i > 0 for i in ixs) + 1))


def multinomial(lst):
    res, i = 1, 1
    for a in lst:
        for j in range(1, a + 1):
            res *= i
            res //= j
            i += 1
    return res


def count_nonzero(a):
    return np.count_nonzero(a)


def nonzero_indices(a):
    return np.nonzero(a)[0]


def construct_simplex_meshgrid(ng, dimSimplex):
    t_list = np.linspace(0, 1, ng)
    tmp = np.array(np.meshgrid(*[t_list for i in range(dimSimplex - 1)]))
    m = np.zeros([tmp[0].ravel().shape[0], dimSimplex])
    for i in range(dimSimplex - 1):
        m[:, i] = tmp[i].ravel()
    m[:, dimSimplex - 1] = 1 - np.sum(m, axis=1)
    return m[m[:, -1] >= 0, :]


class GenerateControlPoint:
    def __init__(self, dimSpace, dimSimplex, degree):
        self.dimSpace = dimSpace  # degree of bezier simplex
        self.dimSimplex = dimSimplex  # dimension of bezier simplex
        self.degree = degree  # dimension of constol point
        self.monomial_degree_list = list(
            subfunction.BezierIndex(dim=self.dimSimplex, deg=self.degree)
        )

    def simplex(self):
        control_point_true = {}
        for deg in self.monomial_degree_list:
            if count_nonzero(deg) == 1:
                index = int(nonzero_indices(deg)[0])
                control_point_true[deg] = np.zeros([self.dimSpace])
                control_point_true[deg][index] = 1
        for deg1 in self.monomial_degree_list:
            if deg1 not in control_point_true:
                control_point_true[deg1] = np.zeros([self.dimSpace])
                for deg2 in self.monomial_degree_list:
                    if count_nonzero(deg2) == 1:
                        # print(deg1,deg2)
                        index = int(nonzero_indices(deg2)[0])
                        control_point_true[deg1] = control_point_true[deg1]
                        control_point_true[deg1] = control_point_true[
                            deg1
                        ] + control_point_true[deg2] * (deg1[index] / self.degree)
        return control_point_true

    def squareroot(self):
        control_point_true = {}
        for deg in self.monomial_degree_list:
            if count_nonzero(deg) == 1:
                index = int(nonzero_indices(deg)[0])
                control_point_true[deg] = np.zeros([self.dimSpace])
                control_point_true[deg][index] = 1
        for deg1 in self.monomial_degree_list:
            if deg1 not in control_point_true:
                control_point_true[deg1] = np.zeros([self.dimSpace])
                for deg2 in self.monomial_degree_list:
                    if count_nonzero(deg2) == 1:
                        # print(deg1,deg2)
                        index = int(nonzero_indices(deg2)[0])
                        control_point_true[deg1] = control_point_true[deg1]
                        control_point_true[deg1] = control_point_true[
                            deg1
                        ] + control_point_true[deg2] * (
                            np.sqrt(deg1[index] / self.degree)
                        )
        return control_point_true


class BezierSimplex:
    def __init__(self, dimSpace, dimSimplex, degree):
        self.dimSpace = dimSpace  # degree of bezier simplex
        self.dimSimplex = dimSimplex  # dimension of bezier simplex
        self.degree = degree  # dimension of constol point
        self.define_monomial(dimSpace=dimSpace, dimSimplex=dimSimplex, degree=degree)

    def define_monomial(self, dimSpace, dimSimplex, degree):
        T = [sympy.Symbol("t" + str(i)) for i in range(self.dimSimplex - 1)]

        def poly(i, n):
            eq = multinomial(i)
            for k in range(n):
                eq *= T[k] ** i[k]
            return eq * (1 - sum(T[k] for k in range(n))) ** i[n]

        """M[multi_index]"""
        M = {
            i: poly(i, self.dimSimplex - 1)
            for i in subfunction.BezierIndex(dim=self.dimSimplex, deg=self.degree)
        }
        # for i in M:
        #    print(i,M[i])
        """Mf[multi_index]"""
        Mf = {}
        for i in subfunction.BezierIndex(dim=self.dimSimplex, deg=self.degree):
            f = poly(i, self.dimSimplex - 1)
            b = compile(
                "Mf[i] = lambda t0, t1=None, t2=None, t3=None, t4=None, t5=None, t6=None, t7=None: "
                + str(f),
                "<string>",
                "exec",
                optimize=2,
            )
            exec(b)
        """Mf_DIFF[multi_index][t]"""
        M_DIFF = [{k: sympy.diff(v, t) for k, v in M.items()} for j, t in enumerate(T)]
        Mf_DIFF = {}
        for k, v in M.items():
            Mf_DIFF[k] = []
            for j, t in enumerate(T):
                Mf_DIFF[k].append([])
                f = sympy.diff(v, t)
                b = compile(
                    "Mf_DIFF[k][-1] = lambda t0, t1=None, t2=None, t3=None, t4=None: "
                    + str(f),
                    "<string>",
                    "exec",
                    optimize=2,
                )
                exec(b)
        """Mf_DIFF2[multi_index][t][t]"""
        M_DIFF2 = [
            [
                {k: sympy.diff(M_DIFF[j][k], t) for k, v in M.items()}
                for h, t in enumerate(T)
            ]
            for j in range(self.dimSimplex - 1)
        ]
        Mf_DIFF2 = {}
        for k, v in M.items():
            Mf_DIFF2[k] = []
            for h, t in enumerate(T):
                Mf_DIFF2[k].append([])
                for j in range(self.dimSimplex - 1):
                    Mf_DIFF2[k][-1].append([])
                    f = sympy.diff(M_DIFF[j][k], t)
                    b = compile(
                        "Mf_DIFF2[k][-1][-1] = lambda t0, t1=None, t2=None, t3=None, t4=None: "
                        + str(f),
                        "<string>",
                        "exec",
                        optimize=2,
                    )
                    exec(b)
        Mf_all = {}
        for k, v in M.items():
            Mf_all[k] = {}
            Mf_all[k][None] = {}
            Mf_all[k][None][None] = Mf[k]
            for i in range(len(Mf_DIFF[k])):
                Mf_all[k][i] = {}
                Mf_all[k][i][None] = Mf_DIFF[k][i]
            for i in range(len(Mf_DIFF2[k])):
                for j in range(len(Mf_DIFF2[k][i])):
                    Mf_all[k][i][j] = Mf_DIFF2[k][i][j]
        self.Mf_all = Mf_all

    @lru_cache(maxsize=1000)
    def monomial_diff(self, multi_index, d0=None, d1=None):
        return self.Mf_all[multi_index][d0][d1]

    def sampling(self, c, t):
        """
        input:
            c: control point (dict)
            t: parameter ([t[0], t[1], t[1],t[3]])
        return:
            x
        """
        x = np.zeros(self.dimSpace)
        for key in subfunction.BezierIndex(dim=self.dimSimplex, deg=self.degree):
            for i in range(self.dimSpace):
                x[i] += (
                    self.monomial_diff(key, d0=None, d1=None)(
                        *t[0 : self.dimSimplex - 1]
                    )
                    * c[key][i]
                )
        return x

    def generate_points(self, c, tt):
        for i in range(tt.shape[0]):
            t = tt[i, :]
            if i == 0:
                x = self.sampling(c, t)
                xx = np.zeros([1, self.dimSpace])
                xx[i, :] = x
            else:
                x = self.sampling(c, t)
                x = x.reshape(1, self.dimSpace)
                xx = np.concatenate((xx, x), axis=0)
        return xx

    def meshgrid(self, c):
        tt = construct_simplex_meshgrid(21, self.dimSimplex)
        for i in range(tt.shape[0]):
            t = tt[i, :]
            if i == 0:
                x = self.sampling(c, t)
                xx = np.zeros([1, self.dimSpace])
                xx[i, :] = x
            else:
                x = self.sampling(c, t)
                x = x.reshape(1, self.dimSpace)
                xx = np.concatenate((xx, x), axis=0)
        return (tt, xx)

    def initialize_control_point(self, data):
        """initialize control point"""
        data_extreme_points = {}
        print(data.keys())
        for i in range(self.dimSimplex):
            # print(i)
            data_extreme_points[i + 1] = data[(i + 1,)]
        C = {}
        list_base_function_index = list(
            subfunction.BezierIndex(dim=self.dimSimplex, deg=self.degree)
        )
        list_extreme_point_index = [
            i for i in list_base_function_index if count_nonzero(i) == 1
        ]
        for key in list_extreme_point_index:
            index = int(nonzero_indices(key)[0])
            C[key] = data_extreme_points[index + 1]
        for key in list_base_function_index:
            if key not in C:
                C[key] = np.zeros(self.dimSpace)
                for key_extreme_points in list_extreme_point_index:
                    index = int(nonzero_indices(key_extreme_points)[0])
                    C[key] = C[key] + C[key_extreme_points] * (key[index] / self.degree)
        return C

    def read_control_point(self, filename):
        """ read control point """
        with open(filename, mode="rb") as f:
            c = pickle.load(f)
        return c

    def write_control_point(self, C, filename):
        """ output control point"""
        with open(filename, "wb") as f:
            pickle.dump(C, f)

    def write_meshgrid(self, C, filename):
        """ output meshgrid"""
        tt_, xx_ = self.meshgrid(C)
        np.savetxt(filename, xx_)
        return xx_


if __name__ == "__main__":
    import os
    import pickle
    from itertools import combinations

    import numpy as np
    import sympy

    import model
    import subfunction

    DEGREE = 3  # ベジエ単体の次数
    DIM_SIMPLEX = 5  # ベジエ単体の次元
    DIM_SPACE = 5  # 制御点が含まれるユークリッド空間の次元
    NG = 21
    NEWTON_ITR = 20
    MAX_ITR = 30  # 制御点の更新回数の上界

    # input data
    base_index = ["1", "2", "3", "4", "5"]
    subsets = []
    for i in range(len(base_index) + 1):
        for c in combinations(base_index, i):
            subsets.append(c)
    data = {}
    for e in subsets:
        if len(e) == 1:
            data[e] = np.loadtxt("../data/normalized_pf/normalized_5-MED.pf_" + e[0])
        if len(e) == 5:
            # data[e] = np.loadtxt('data/normalized_5-MED.pf_1_2_3_4_5')
            data[e] = np.loadtxt(
                "../data/normalized_pf/normalized_5-MED.pf_1_2_3_4_5_itr0"
            )

    bezier_simplex = model.BezierSimplex(
        dimSpace=DIM_SPACE, dimSimplex=DIM_SIMPLEX, degree=DEGREE
    )
    C_init = bezier_simplex.initialize_control_point(data)
    for key in C_init:
        print(key, C_init[key])
    x = bezier_simplex.sampling(C_init, [1, 0, 0, 0])
    print(x)
    tt, xx = bezier_simplex.meshgrid(C_init)
    print(tt.shape, xx.shape)
    bezier_simplex.write_meshgrid(C_init, "sample_mesghgrid")
