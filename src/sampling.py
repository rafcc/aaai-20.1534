import numpy as np  # type: ignore
from scipy.special import comb


class UniformSampling:
    def __init__(self, dimension):
        self.dimension = dimension  # dimension of simplex

    # generate training points and evaluation points
    def simplex(self, num_sample, seed, dimension):
        """
        num_sample: number of sample points
        dimension: dimension of simplex
        seed:
        return n\times d 2d ndarray
        """
        # print(seed)
        np.random.seed(seed)
        x = np.random.uniform(0, 1, [num_sample, dimension + 1])
        x[:, 0] = 0
        x[:, dimension] = 1
        x = np.sort(x, axis=1)
        z = np.zeros([num_sample, dimension])
        for col in range(dimension):
            z[:, col] = x[:, col + 1] - x[:, col]
        return z

    def subsimplex(self, num_sample, seed, indices):
        z = self.simplex(dimension=len(indices), num_sample=num_sample, seed=seed)
        m = np.zeros([num_sample, self.dimension])
        for i in range(len(indices)):
            col = indices[i]
            m[:, col] = z[:, i]
        return m


class GridSampling:
    def __init__(self, dimension):
        self.dimension = dimension  # dimension of simplex

    # generate training points and evaluation points
    def simplex(self, num_grid):
        t_list = np.linspace(0, 1, num_grid)
        tmp = np.array(np.meshgrid(*[t_list for i in range(self.dimension - 1)]))
        m = np.zeros([tmp[0].ravel().shape[0], self.dimension])
        for i in range(self.dimension - 1):
            m[:, i] = tmp[i].ravel()
        m[:, self.dimension - 1] = 1 - np.sum(m, axis=1)
        return m[m[:, -1] >= 0, :]


class CalcSampleSize:
    def __init__(self, dim_simplex, degree):
        self.dim_simplex = dim_simplex
        self.degree = degree

    def calc_sampling_ratio(self, opt_flag):
        # dict[opt_flag][degree][diimsimplex]
        dict_sampling_ratio = {}
        dict_sampling_ratio[1] = {
            2: {
                2: [1 - 0.586, 0.586],
                3: [1 - 0.739, 0.739],
                4: [1 - 0.771, 0.771],
                5: [1 - 0.772, 0.772],
                6: [1 - 0.767, 0.767],
                7: [1 - 0.762, 0.762],
                8: [1 - 0.758, 0.758],
            },
            3: {
                3: [1 - 0.587 - 0.314, 0.587, 0.314],
                4: [1 - 0.453 - 0.481, 0.453, 0.481],
                5: [1 - 0.386 - 0.547, 0.386, 0.547],
                6: [1 - 0.365 - 0.577, 0.365, 0.577],
                7: [1 - 0.359 - 0.589, 0.359, 0.589],
                8: [1 - 0.336 - 0.623, 0.336, 0.623],
            },
        }
        dict_sampling_ratio[0] = {
            2: {
                2: [1 - 0.500, 0.500],
                3: [1 - 0.500, 0.500],
                4: [1 - 0.500, 0.500],
                5: [1 - 0.500, 0.500],
                6: [1 - 0.500, 0.500],
                7: [1 - 0.500, 0.500],
                8: [1 - 0.500, 0.500],
            },
            3: {
                3: [1 - 0.333 - 0.333, 0.333, 0.333],
                4: [1 - 0.333 - 0.333, 0.333, 0.333],
                5: [1 - 0.333 - 0.333, 0.333, 0.333],
                6: [1 - 0.333 - 0.333, 0.333, 0.333],
                7: [1 - 0.333 - 0.333, 0.333, 0.333],
                8: [1 - 0.333 - 0.333, 0.333, 0.333],
            },
        }
        return dict_sampling_ratio[opt_flag][self.degree][self.dim_simplex]

    def get_sample_size_list(self, n, opt_flag):
        """
        calculate sample size for each dimension

        Parameters
        ----------
        n: int
            all sample size
        opt_flag : 0/1
            1 -> optimal strategy
            0 -> nonoptimal strategy

        Return
        ----------
        list : sample size list
        int  : entire sample size
        """
        round = lambda x: (x * 2 + 1) // 2
        s_list = []
        ratio = self.calc_sampling_ratio(opt_flag=opt_flag)
        for d in range(min(self.dim_simplex, self.degree)):
            num_skeleton = comb(self.dim_simplex, d + 1, exact=True)
            s_list.append(int(round(n * ratio[d] / num_skeleton)))
        return s_list
