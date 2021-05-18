import pickle
import time
from itertools import combinations

import numpy as np
import yaml

import model
import sampling
import subfunction
import trainer


def main(ymlfilename, resultdir, borges_flag=1):
    f = open(ymlfilename, "r+")
    params = yaml.load(f)

    subfunction.create_directory(resultdir)

    SEED = int(params["seed"])
    SIGMA = float(params["sigma"])
    DEGREE = int(params["degree"])
    DIMENSION_SIMPLEX = int(params["dimension_simplex"])
    DIMENSION_SPACE = int(params["dimension_space"])
    NUM_SAMPLE = params["num_sample_train"]
    NUM_SAMPLE_TEST = params["num_sample_test"]
    NUM_SAMPLE_ALL = params["num_sample_train_all"]
    SIMPLEX_TYPE = params["simplextype"]

    np.random.seed(SEED)
    objective_function_indices_list = list(range(DIMENSION_SIMPLEX))
    subproblem_indices_list = []
    for i in range(1, len(objective_function_indices_list) + 1):
        for c in combinations(objective_function_indices_list, i):
            subproblem_indices_list.append(c)
    uniform_sampling = sampling.UniformSampling(dimension=DIMENSION_SIMPLEX)
    bezier_simplex = model.BezierSimplex(
        dimSpace=DIMENSION_SPACE, dimSimplex=DIMENSION_SIMPLEX, degree=DEGREE
    )
    borges_pastva_trainer = trainer.BorgesPastvaTrainer(
        dimSpace=DIMENSION_SPACE, dimSimplex=DIMENSION_SIMPLEX, degree=DEGREE
    )
    monomial_degree_list = list(
        subfunction.BezierIndex(dim=DIMENSION_SIMPLEX, deg=DEGREE)
    )

    # generate true control points
    generate_control_point = model.GenerateControlPoint(
        dimSpace=DIMENSION_SPACE, dimSimplex=DIMENSION_SIMPLEX, degree=DEGREE
    )
    if SIMPLEX_TYPE == "linear":
        control_point_true = generate_control_point.simplex()
        SEED += 5
    elif SIMPLEX_TYPE == "squareroot":
        control_point_true = generate_control_point.squareroot()

    # generate training data
    print("generating data start")
    start = time.time()
    param_trn = {}
    data_trn = {}
    for c in subproblem_indices_list:
        if len(c) <= min(DIMENSION_SIMPLEX, DEGREE):
            n = NUM_SAMPLE[len(c) - 1]
            SEED += 30
            z = uniform_sampling.subsimplex(indices=c, num_sample=n, seed=SEED)
            param_trn[c] = z
            b = bezier_simplex.generate_points(c=control_point_true, tt=param_trn[c])
            epsilon = np.random.multivariate_normal(
                [0] * DIMENSION_SPACE,
                np.identity(DIMENSION_SPACE) * (SIGMA ** 2),
                n,
            )
            data_trn[c] = b + epsilon
    data_trn_array = subfunction.concat_data_to_arrray(d=data_trn)
    param_trn_array = subfunction.concat_data_to_arrray(d=param_trn)

    # todo create daata for borges
    if borges_flag == 1:
        param_trn_borges = uniform_sampling.subsimplex(
            indices=objective_function_indices_list,
            num_sample=NUM_SAMPLE_ALL,
            seed=(SEED + 1) * len(objective_function_indices_list),
        )
        data_trn_borges = bezier_simplex.generate_points(
            c=control_point_true, tt=param_trn_borges
        )
        epsilon = np.random.multivariate_normal(
            [0] * DIMENSION_SPACE,
            np.identity(DIMENSION_SPACE) * (SIGMA ** 2),
            NUM_SAMPLE_ALL,
        )
        data_trn_borges += epsilon
    print(
        "geenrating data finihed, elapsed_time:{0}".format(time.time() - start)
        + "[sec]"
    )

    # fitting by borges
    if borges_flag == 1:
        print("fitting by borges")
        start = time.time()
        control_point_borges = borges_pastva_trainer.update_control_point(
            t_mat=param_trn_borges,
            data=data_trn_borges,
            c={},
            indices_all=monomial_degree_list,
            indices_fix=[],
        )
        elapsed_time_borges = time.time() - start
        print(
            "fitting by borges finihed, elapsed_time:{0}".format(elapsed_time_borges)
            + "[sec]"
        )

    print("fitting by inductive")
    start = time.time()
    freeze_multiple_degree_set = set()
    control_point_inductive = {}
    for dim in range(1, DIMENSION_SIMPLEX + 1):
        for index in data_trn:
            if len(index) == dim:
                target_multiple_degree_set = subfunction.extract_multiple_degree(
                    degree_list=monomial_degree_list, index_list=index
                )
                target_multiple_degree_list = list(target_multiple_degree_set)
                freeze_multiple_degree_list = list(
                    freeze_multiple_degree_set.intersection(target_multiple_degree_set)
                )
                a = borges_pastva_trainer.update_control_point(
                    t_mat=param_trn[index],
                    data=data_trn[index],
                    c=control_point_inductive,
                    indices_all=target_multiple_degree_list,
                    indices_fix=freeze_multiple_degree_list,
                )
                control_point_inductive.update(a)
                freeze_multiple_degree_set = freeze_multiple_degree_set.union(
                    target_multiple_degree_set
                )
                if len(freeze_multiple_degree_set) == len(monomial_degree_list):
                    break
    elapsed_time_inductive = time.time() - start
    print(
        "fitting by inductive finihed, elapsed_time:{0}".format(elapsed_time_inductive)
        + "[sec]"
    )

    # calculate risk
    print("calc risks")
    start = time.time()

    if SIMPLEX_TYPE == "linear":
        SEED += 1
    else:
        SEED += 5
    param_tst_array = uniform_sampling.subsimplex(
        indices=objective_function_indices_list,
        num_sample=NUM_SAMPLE_TEST,
        seed=(SEED + 1) * (DIMENSION_SIMPLEX),
    )
    print("laptime array", time.time() - start)
    random_tst = bezier_simplex.generate_points(
        c=control_point_true, tt=param_tst_array
    )
    if borges_flag == 1:
        print("laptime test", time.time() - start)
        random_borges = bezier_simplex.generate_points(
            c=control_point_borges, tt=param_tst_array
        )
        print("laptime borges", time.time() - start)
    random_inductive = bezier_simplex.generate_points(
        c=control_point_inductive, tt=param_tst_array
    )
    print("laptime inductive", time.time() - start)
    if borges_flag == 1:
        random_l2risk_borges = subfunction.calculate_l2_expected_error(
            true=random_tst, pred=random_borges
        )
    random_l2risk_inductive = subfunction.calculate_l2_expected_error(
        true=random_tst, pred=random_inductive
    )
    # calc risk from grid sampled data points
    elapsed_time_risk = time.time() - start
    print("calc risk finihed, elapsed_time:{0}".format(elapsed_time_risk) + "[sec]")

    # print("random ",random_l2risk_borges,random_l2risk_inductive)
    results = {}
    results["random_l2risk"] = {}
    results["time"] = {}
    if borges_flag == 1:
        results["random_l2risk"]["borges"] = float(random_l2risk_borges)
        results["time"]["borges"] = elapsed_time_borges
    results["random_l2risk"]["inductive"] = float(random_l2risk_inductive)
    results["time"]["inductive"] = elapsed_time_inductive
    results["time"]["risk"] = elapsed_time_risk

    output_dict = {}
    output_dict["params"] = params
    output_dict["results"] = results
    subfunction.write_result(result=output_dict, fname=resultdir + "/output.yml")

    """
    visualize.plot_estimated_pairplot(d1=random_tst,
                                  d2=random_borges,
                                  d3=data_trn_borges,
                                  output_name=resultdir+'/random_borges.png')
    visualize.plot_estimated_pairplot(d1=random_tst,
                                  d2=random_inductive,
                                  d3=data_trn_array,
                                  output_name=resultdir+'/random_inductive.png')

    visualize.plot_graph3d(pareto=grid_tst,
                           simplex=grid_borges,
                           sample=data_trn_array,
                           loc='lower right',
                           output_name=resultdir+'/grid_borges.png')
    visualize.plot_graph3d(pareto=grid_tst,
                           simplex=grid_inductive,
                           sample=data_trn_array,
                           loc='lower right',
                           output_name=resultdir+'/grid_inductive.png')
    visualize.plot_graph3d(pareto=random_tst[:,1:4],
                           simplex=random_borges[:,1:4],
                           sample=data_trn_borges[:,1:4],
                           loc='lower right',
                           output_name=resultdir+'/3d_random_borges.png')
    visualize.plot_graph3d(pareto=random_tst[:,1:4],
                           simplex=random_inductive[:,1:4],
                           sample=data_trn_array[:,1:4],
                           loc='lower right',
                           output_name=resultdir+'/3d_random_inductive.png')
    """

    pi = {}
    pi["control_points"] = {}
    pi["train"] = {}
    pi["train"]["borges"] = {}
    pi["train"]["inductive"] = {}
    pi["test"] = {}
    pi["control_points"]["true"] = control_point_true
    if borges_flag == 1:
        pi["control_points"]["borges"] = control_point_borges
    pi["control_points"]["inductive"] = control_point_inductive
    pi["train"]["inductive"]["param"] = param_trn
    pi["train"]["inductive"]["val"] = data_trn
    pi["test"]["param"] = param_tst_array
    pi["test"]["val"] = random_tst

    output_name = resultdir + "/parameters.pkl"
    with open(output_name, "wb") as wf:
        pickle.dump(pi, wf)


if __name__ == "__main__":
    # target = 'n1.10_n2.5_n3.2_sigma.0.05'
    target = "test_L10"
    main(
        ymlfilename="../settings/" + target + "/0.yml",
        resultdir="../results/" + target + "/0",
    )
    main(
        ymlfilename="../settings/" + target + "/1.yml",
        resultdir="../results/" + target + "/1",
    )

    # generate_yaml.generate_yaml_instance(target=target)
    """
    main(ymlfilename='../settings/'+target+'_0.yml',resultdir='../results/'+target+'/0')
    main(ymlfilename='../settings/'+target+'_1.yml',resultdir='../results/'+target+'/1')
    main(ymlfilename='../settings/'+target+'_2.yml',resultdir='../results/'+target+'/2')
    main(ymlfilename='../settings/'+target+'_3.yml',resultdir='../results/'+target+'/3')
    main(ymlfilename='../settings/'+target+'_4.yml',resultdir='../results/'+target+'/4')
    main(ymlfilename='../settings/'+target+'_5.yml',resultdir='../results/'+target+'/5')
    main(ymlfilename='../settings/'+target+'_6.yml',resultdir='../results/'+target+'/6')
    main(ymlfilename='../settings/'+target+'_7.yml',resultdir='../results/'+target+'/7')
    main(ymlfilename='../settings/'+target+'_8.yml',resultdir='../results/'+target+'/8')
    main(ymlfilename='../settings/'+target+'_9.yml',resultdir='../results/'+target+'/9')
    """
