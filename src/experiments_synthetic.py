import time

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


def experiments_synthetic_data(
    n, degree, dimspace, dimsimplex, sigma, method, seed, results_dir, opt_flag=1
):
    """
    conduct experiments with synthetic data

    Parameters
    ----------
    n : int
        number of sample points to be trained
    degree : int
        max degree of bezier simplex fittng
    dimspace : int
        the number of dimension of the Eucledian space
        where the bezier simplex is embedded
    dimsimplex : int
        the number of dimension of bezier simplex
    sigma : float
        the scale of noise.
        Noises are chosen form a normal distribution N(0,sigma^2I)
    method: "borges"/"inductive"
    result_dir: str
        where to output results
    opt_flag : 0/1 (default is 1)
        0 : optimal sampling strategy for inductive skeleton fitting
        1 : nonoptimal sampling strategy inductive skeketon fitting
        "borges" does not care about this parameter.
    """
    # data generation class
    synthetic_data = data.SyntheticData(
        degree=degree, dimspace=dimspace, dimsimplex=dimsimplex
    )

    # train
    if method == "borges":
        param_trn, data_trn = synthetic_data.sampling_borges(
            n=n, seed=seed, sigma=sigma
        )
        monomial_degree_list = list(
            subfunction.BezierIndex(dim=dimsimplex, deg=degree)
        )
        borges_pastva_trainer = trainer.BorgesPastvaTrainer(
            dimSpace=dimspace, dimSimplex=dimsimplex, degree=degree
        )
        control_point = borges_pastva_trainer.update_control_point(
            t_mat=param_trn,
            data=data_trn,
            c={},
            indices_all=monomial_degree_list,
            indices_fix=[],
        )
    elif method == "inductive":
        # calculate sample size of each skeleton
        calc_sample_size = sampling.CalcSampleSize(degree=degree, dimsimplex=dimsimplex)
        train_sample_size_list = calc_sample_size.get_sample_size_list(
            n=n, opt_flag=opt_flag
        )
        # data generation
        param_trn, data_trn = synthetic_data.sampling_inductive(
            n=n, seed=seed, sample_size_list=train_sample_size_list, sigma=sigma
        )
        monomial_degree_list = list(
            subfunction.BezierIndex(dim=dimsimplex, deg=degree)
        )
        inductive_skeleton_trainer = trainer.InductiveSkeletonTrainer(
            dimSpace=dimspace, dimSimplex=dimsimplex, degree=degree
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

    # generate test data which does not include gaussian noise
    param_tst, data_tst = synthetic_data.sampling_borges(
        n=10000, seed=seed * 2, sigma=0
    )
    bezier_simplex = model.BezierSimplex(
        dimSpace=dimspace, dimSimplex=dimsimplex, degree=degree
    )
    data_pred = bezier_simplex.generate_points(c=control_point, tt=param_tst)
    l2_risk = subfunction.calculate_l2_expected_error(true=data_tst, pred=data_pred)

    # output result
    settings = {}
    settings["n"] = n
    settings["degree"] = degree
    settings["dimspace"] = dimspace
    settings["dimsimplex"] = dimsimplex
    settings["sigma"] = sigma
    settings["method"] = method
    settings["seed"] = seed
    settings["opt_flag"] = opt_flag
    results = {}
    results["l2_risk"] = "{:5E}".format(l2_risk)

    o = {}
    o["reults"] = results
    o["settings"] = settings

    ymlfilename = results_dir + "/"
    for key in ["dimsimplex", "dimspace", "degree", "n", "method", "opt_flag", "seed"]:
        ymlfilename += key + "." + str(settings[key]) + "_"
    ymlfilename += ".yml"
    wf = open(ymlfilename, "w")
    wf.write(yaml.dump(o, default_flow_style=False))
    wf.close()


if __name__ == "__main__":
    degree_list = [2, 3]
    setting_tuple_list = [
        (250, 100, 8),  # n, dimspace, dimsimplex,
        (500, 100, 8),
        (1000, 100, 8),
        (2000, 100, 8),  ##
        (1000, 100, 3),
        (1000, 100, 4),
        (1000, 100, 5),
        (1000, 100, 6),
        (1000, 100, 7),  ##
        (1000, 8, 8),
        (1000, 25, 8),
        (1000, 50, 8),
    ]  # d, n, dimspace, dimsimplex,
    seed_list = [i + 1 for i in range(20)]

    results_dir = "../results_synthetic/"
    subfunction.create_directory(dir_name=results_dir)
    start = time.time()
    for degree in degree_list:
        for (n, dimspace, dimsimplex) in setting_tuple_list:
            for seed in seed_list:
                print("(D,N,L,M,seed):", degree, n, dimspace, dimsimplex, seed)
                start_lap = time.time()
                experiments_synthetic_data(
                    n=n,
                    degree=degree,
                    dimspace=dimspace,
                    dimsimplex=dimsimplex,
                    sigma=0.1,
                    seed=seed,
                    method="borges",
                    opt_flag=1,
                    results_dir=results_dir,
                )
                experiments_synthetic_data(
                    n=n,
                    degree=degree,
                    dimspace=dimspace,
                    dimsimplex=dimsimplex,
                    sigma=0.1,
                    seed=seed,
                    method="inductive",
                    opt_flag=1,
                    results_dir=results_dir,
                )
                experiments_synthetic_data(
                    n=n,
                    degree=degree,
                    dimspace=dimspace,
                    dimsimplex=dimsimplex,
                    sigma=0.1,
                    seed=seed,
                    method="inductive",
                    opt_flag=0,
                    results_dir=results_dir,
                )
                lap_time = time.time() - start_lap
                elapsed_time = time.time() - start
                print("laptime:", lap_time)
                print("current:", elapsed_time)
