# -*- coding: utf-8 -*-
import random
import time
from itertools import combinations

import yaml

import data
import model
import sampling
import subfunction
import trainer


def convert_params_to_string(
    dimension_simplex, dimension_space, degree, N, num_sample_train, sigma
):
    s = "synthetic_"
    s = s + "M." + str(dimension_simplex)
    s = s + "_L." + str(dimension_space)
    s = s + "_D." + str(degree)
    s = s + "_N." + str(N)
    s = s + "_sigma." + str(sigma)
    return s


def sampling_data_and_param(d, p, n, seed):
    random.seed(seed)
    s = list(range(d.shape[0]))
    s_ = random.sample(s, n)
    return (d[s_, :], p[s_, :])


def experiments_practical_instances(
    data_dir,
    trn_data,
    test_data,
    n,
    solution_type,
    dim_simplex,
    degree,
    method,
    seed,
    results_dir,
    opt_flag=1,
):
    """
    conduct experiments with synthetic data

    Parameters
    ----------
    data_dir : str
        the name of directory which include datasets
    trn_data : str
        dataname to be trained
    test_data : str
        dataname for test
    n : int
        number of samples to be trained
    dim_simplex : int
        dimension of bezier simplex
    degree : int
        max degree of bezier simplex fittng
    method: "borges"/"inductive"
    result_dir: str
        where to output results
    opt_flag : 0/1 (default is 1)
        0 : optimal sampling strategy for inductive skeleton fitting
        1 : nonoptimal sampling strategy inductive skeketon fitting
        "borges" does not care about this parameter.
    """
    # data preparation
    objective_function_indices_list = list(range(dim_simplex))
    subproblem_indices_list = []
    for i in range(1, len(objective_function_indices_list) + 1):
        for c in combinations(objective_function_indices_list, i):
            subproblem_indices_list.append(c)
    monomial_degree_list = list(
        subfunction.BezierIndex(dim=dim_simplex, deg=degree)
    )
    data_all = {}
    param_all = {}
    for e in subproblem_indices_list:
        if len(e) <= degree or len(e) == dim_simplex:
            string = "_".join(str(i + 1) for i in e)
            tmp = data.Dataset(
                data_dir + "/" + trn_data + "," + solution_type + "_" + string
            )
            data_all[e] = tmp.values

            tmp = data.Dataset(data_dir + "/" + trn_data + ",w_" + string)
            param_all[e] = tmp.values

    dim_space = data_all[(0, 1, 2,)].shape[1]
    # train
    if method == "borges":
        param_trn = {}
        data_trn = {}
        e = tuple(range(dim_simplex))
        data_trn[e], param_trn[e] = sampling_data_and_param(
            d=data_all[e], p=param_all[e], n=n, seed=seed
        )
        borges_pastva_trainer = trainer.BorgesPastvaTrainer(
            dim_space=dim_space, dim_simplex=dim_simplex, degree=degree
        )
        control_point = borges_pastva_trainer.update_control_point(
            t_mat=param_trn[e],
            data=data_trn[e],
            c={},
            indices_all=monomial_degree_list,
            indices_fix=[],
        )
    elif method == "inductive":
        # calculate sample size of each skeleton
        calc_sample_size = sampling.CalcSampleSize(degree=degree, dim_simplex=dim_simplex)
        train_sample_size_list = calc_sample_size.get_sample_size_list(
            n=n, opt_flag=opt_flag
        )
        # sampling
        data_trn = {}
        param_trn = {}
        for e in data_all:
            if len(e) <= degree:
                data_trn[e], param_trn[e] = sampling_data_and_param(
                    d=data_all[e],
                    p=param_all[e],
                    n=train_sample_size_list[len(e) - 1],
                    seed=seed + sum(e),
                )
        inductive_skeleton_trainer = trainer.InductiveSkeletonTrainer(
            dim_space=dim_space, dim_simplex=dim_simplex, degree=degree
        )
        control_point = inductive_skeleton_trainer.update_control_point(
            t_dict=param_trn,
            data_dict=data_trn,
            c={},
            indices_all=monomial_degree_list,
            indices_fix=[],
        )
    else:
        pass

    # evaluate empirical l2 risk
    e = tuple(range(dim_simplex))
    data_tst = data.Dataset(
        data_dir
        + "/"
        + test_data
        + ","
        + solution_type
        + "_"
        + "_".join(str(i + 1) for i in e)
    ).values
    param_tst = data.Dataset(
        data_dir + "/" + test_data + ",w_" + "_".join(str(i + 1) for i in e)
    ).values
    bezier_simplex = model.BezierSimplex(
        dim_space=dim_space, dim_simplex=dim_simplex, degree=degree
    )
    data_pred = bezier_simplex.generate_points(c=control_point, tt=param_tst)
    l2_risk = subfunction.calculate_l2_expected_error(true=data_tst, pred=data_pred)
    # output result
    settings = {}
    settings["trn_data"] = trn_data
    settings["tset_data"] = test_data
    settings["solution_type"] = solution_type
    settings["n"] = n
    settings["degree"] = degree
    settings["dim_space"] = dim_space
    settings["dim_simplex"] = dim_simplex
    settings["method"] = method
    settings["seed"] = seed
    settings["opt_flag"] = opt_flag
    results = {}
    results["l2_risk"] = "{:5E}".format(l2_risk)

    o = {}
    o["reults"] = results
    o["settings"] = settings

    ymlfilename = results_dir + "/" + trn_data + "solution_type." + solution_type + "/"
    subfunction.create_directory(dir_name=ymlfilename)
    for key in ["degree", "n", "method", "opt_flag", "seed"]:
        ymlfilename = ymlfilename + key + "." + str(settings[key]) + "_"
    ymlfilename = ymlfilename + ".yml"
    wf = open(ymlfilename, "w")
    wf.write(yaml.dump(o, default_flow_style=False))
    wf.close()


if __name__ == "__main__":

    results_dir = "../results_practical"
    subfunction.create_directory(dir_name=results_dir)

    solution_type_list = ["x", "f"]
    degree_list = [2, 3]
    n_list = [250, 500, 1000, 2000]
    start = time.time()
    seed_list = [i + 1 for i in range(20)]
    for degree in degree_list:
        for solution_type in solution_type_list:
            for n in n_list:
                for seed in seed_list:
                    start_lap = time.time()
                    print("(D,solution_type,n,seed)", degree, solution_type, n, seed)
                    experiments_practical_instances(
                        data_dir="../data",
                        trn_data="MED,d_100,o_3,c_2e+00,e_1e-01,n_10000,s_42",
                        test_data="MED,d_100,o_3,c_2e+00,e_1e-01,n_10000,s_43",
                        solution_type=solution_type,
                        n=n,
                        degree=degree,
                        dim_simplex=3,
                        seed=seed,
                        method="inductive",
                        opt_flag=1,
                        results_dir=results_dir,
                    )
                    experiments_practical_instances(
                        data_dir="../data",
                        trn_data="MED,d_100,o_3,c_2e+00,e_1e-01,n_10000,s_42",
                        test_data="MED,d_100,o_3,c_2e+00,e_1e-01,n_10000,s_43",
                        solution_type=solution_type,
                        n=n,
                        degree=degree,
                        dim_simplex=3,
                        seed=seed,
                        method="borges",
                        opt_flag=1,
                        results_dir=results_dir,
                    )
                    lap_time = time.time() - start_lap
                    elapsed_time = time.time() - start
                    print("laptime:", lap_time)
                    print("current:", elapsed_time)

    start = time.time()
    for degree in degree_list:
        for solution_type in solution_type_list:
            for n in n_list:
                for seed in seed_list:
                    start_lap = time.time()
                    print("(D,solution_type,n,seed)", degree, solution_type, n, seed)
                    experiments_practical_instances(
                        data_dir="../data",
                        trn_data="Birthwt6.csv,n_10000,r_1e+00,e_1e-01,m_0e+00,s_42,l_1e+00,t_1e-07,i_10000",
                        test_data="Birthwt6.csv,n_10000,r_1e+00,e_1e-01,m_0e+00,s_43,l_1e+00,t_1e-07,i_10000",
                        solution_type=solution_type,
                        n=n,
                        degree=degree,
                        dim_simplex=3,
                        seed=seed,
                        method="inductive",
                        opt_flag=1,
                        results_dir=results_dir,
                    )
                    experiments_practical_instances(
                        data_dir="../data",
                        trn_data="Birthwt6.csv,n_10000,r_1e+00,e_1e-01,m_0e+00,s_42,l_1e+00,t_1e-07,i_10000",
                        test_data="Birthwt6.csv,n_10000,r_1e+00,e_1e-01,m_0e+00,s_43,l_1e+00,t_1e-07,i_10000",
                        solution_type=solution_type,
                        n=n,
                        degree=degree,
                        dim_simplex=3,
                        seed=seed,
                        method="inductive",
                        opt_flag=1,
                        results_dir=results_dir,
                    )
                    lap_time = time.time() - start_lap
                    elapsed_time = time.time() - start
                    print("laptime:", lap_time)
                    print("current:", elapsed_time)
