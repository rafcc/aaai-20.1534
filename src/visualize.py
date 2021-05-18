import glob
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def plot_pairplot(dname, ofname):
    d = np.loadtxt(dname)
    df = pd.DataFrame(d)
    g = sns.pairplot(df)
    # if not os.path.exists(odir):
    #    os.makedirs(odir)
    plt.savefig(ofname)
    plt.close()


def plot_estimated_pairplot(d1, d2, d3, output_name):
    df1 = pd.DataFrame(d1)
    df2 = pd.DataFrame(d2)
    df3 = pd.DataFrame(d3)
    df1["label"] = pd.Series("Pareto front", index=df1.index)
    df2["label"] = pd.Series("Bezier simplex", index=df2.index)
    df3["label"] = pd.Series("Training sample point", index=df3.index)
    df4 = pd.concat([df1, df2, df3])
    g = sns.pairplot(df4, hue="label")
    # if not os.path.exists(odir):
    #    os.makedirs(odir)
    plt.savefig(output_name)
    plt.close()


def plot_mesh_validation_pairplot(d1, d2, ofname):
    """d1: Pareto front, d2:bezier simplex"""
    df1 = pd.DataFrame(d1)
    df2 = pd.DataFrame(d2)
    df1["label"] = pd.Series("Pareto front", index=df1.index)
    df2["label"] = pd.Series("Bezier simplex", index=df2.index)
    df4 = pd.concat([df1, df2])
    g = sns.pairplot(df4, hue="label")
    # if not os.path.exists(odir):
    #    os.makedirs(odir)
    plt.savefig(ofname)
    plt.close()


def plot_graph3d(pareto, simplex, sample, output_name, loc, ymax=None):
    # pareto = np.loadtxt(pareto_name)
    # simplex = np.loadtxt(bezier_simplex_name)
    # sample = np.loadtxt(sample_name)

    fig = plt.figure(figsize=(6, 6))
    # fig.patch.set_facecolor('blue')
    # plt.style.use('ggplot')
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter3D(
        pareto[:, 0],
        pareto[:, 1],
        pareto[:, 2],
        c="blue",
        s=50,
        linewidth=1,
        edgecolors="black",
        alpha=0.5,
        depthshade=0,
        label="Pareto front",
    )
    ax.scatter3D(
        simplex[:, 0],
        simplex[:, 1],
        simplex[:, 2],
        c="red",
        s=10,
        linewidth=1,
        edgecolors="black",
        depthshade=0,
        label="Bezier simplex",
    )
    ax.scatter3D(
        sample[:, 0],
        sample[:, 1],
        sample[:, 2],
        c="green",
        s=40,
        linewidth=1,
        depthshade=0,
        edgecolors="black",
        label="Training points",
    )
    ax.set_xlabel("f1", fontsize=20)
    ax.set_ylabel("f2", fontsize=20)
    ax.set_zlabel("f3", fontsize=20)
    ax.legend(loc=loc, fontsize=20)
    ax.set_xlim([-0, 1])
    ax.set_ylim([-0, 1])
    ax.set_zlim([-0, 1.2])
    ax.view_init(elev=5, azim=45)
    fig.tight_layout()
    fig.savefig(output_name)
    plt.close()


if __name__ == "__main__":
    import glob

    """
    for dirname in glob.glob("../test/meshgrid/"):
        for fname in glob.glob(dirname + "meshgrid_itr*"):
            if ".png" not in fname:
                pngname = fname+'.png'
                print(pngname)
                plot_pairplot(fname,pngname)
    """
    """
    items = [str(i+1) for i in range(5)]
    subsets=[]
    for i in range(len(items) + 1):
        for c in combinations(items, i):
            subsets.append(c)
    for e  in subsets:
    """

    """
    for dirname in glob.glob("../test_tolerance_rss/5-MED/inductive/subproblem*"):
        for fname in glob.glob(dirname + "/meshgrid/meshgrid_itr*"):
            if ".png" not in fname:
                pngname = fname+'.png'
                print(pngname)
                plot_pairplot(fname,pngname)
    """
    # plot_pairplot("../result_test/5-MED/inductive/subproblem_1_2_3/meshgrid_itr009",
    #              "../result_test/5-MED/inductive/subproblem_1_2_3/meshgrid_itr009.png")
    # plot_pairplot("../result_test/5-MED/inductive/meshgrid_itr_025",
    #              "../result_test/5-MED/inductive/meshgrid_itr025.png")

    # plot_pairplot("../test/med5/meshgrid/meshgrid_itr_000","../test/med5/meshgrid_itr_000.png")
    # plot_pairplot("../test/med5/meshgrid/meshgrid_itr_001","../test/med5/meshgrid_itr_001.png")
    # plot_pairplot("../test/med5/meshgrid/meshgrid_itr_030","../test/med5/meshgrid_itr_030.png")

    # plot_pairplot("../test/meshgrid/meshgrid_itr_050","../test/meshgrid_itr_050.png")
    # plot_pairplot("../test/meshgrid/meshgrid_itr_100","../test/meshgrid_itr_100.png")
    # plot_pairplot("../test/meshgrid_itr100","../test/meshgrid_itr100.png")
    # plot_pairplot("../result_test/inductive/meshgrid_itr_003","../result_test/inductive/meshgrid_itr_003.png")
    # plot_pairplot("5med/meshgrid_all_loop_25","5med/meshgrid_25.png")
    # plot_pairplot("5med_pos/meshgrid_itr_025","5med_pos/meshgrid_itr_025.png")
    """
    plot_pairplot("../data/preprocessed/ConstrEx.pf","../fig/datasets/ConstrEx.png")
    plot_pairplot("../data/preprocessed/3-MED.pf","../fig/datasets/3-MED.png")
    plot_pairplot("../data/preprocessed/5-MED.pf","../fig/datasets/5-MED.png")
    plot_pairplot("../data/preprocessed/Osyczka2.pf","../fig/datasets/Osyczka2.png")
    plot_pairplot("../data/preprocessed/Schaffer.pf","../fig/datasets/Schaffer.png")
    plot_pairplot("../data/preprocessed/Viennet2.pf","../fig/datasets/Viennet2.png")
    """

    # plot_pairplot("../result_test/borges/meshgrid_itr063","borges.png")
    # plot_pairplot("../result_test/inductive/meshgrid_itr_007","inductive.png")
    # plot_pairplot("../data/preprocessed/3-MED.pf","MED3D.png")

    """

    items = [str(i+1) for i in range(5)]
    subsets=[]
    for i in range(len(items) + 1):
        for c in combinations(items, i):
            subsets.append(c)
    for e  in subsets:
        if len(e) == 2:
            d1 = '../data/normalized_pf/normalized_5-MED.pf_'+e[0]+'_'+e[1]
            for d2 in glob.glob('../result/estimated_bezier_curve_revised/normalized_5-MED.pf_'+e[0]+'_'+e[1]+'*'):
                #d2 =  '../result/estimated_bezier_curve/normalized_5-MED.pf_1_2_itr30'
                d2 = d2.replace('\\','/')
                tmp = d2.split('/')[-1]
                if not os.path.exists(o_dir):
                    os.makedirs(o_dir)
                ofname = o_dir+'/'+tmp+'.png'
                plot_estimated_pairplot(d1,d2,ofname)
    """

    """
    #o_dir = '../fig/pair_plot_estimated_bezier_surface_bottomup'
    d1 = '../data/normalized_pf/normalized_5-MED.pf_1_2_3_4_5'
    #d2 = '../result/estimated_bezier_simplex_allatonce/normalized_5-MED.pf_1_2_3_4_5_itr1'
    #d2 = '../result/estimated_bezier_simplex_bottomup/bottomup'
    #plot_estimated_pairplot(d1,d2,'../fig/pair_plot_estimated_allatonce/normalized_5-MED.pf_1_2_3_4_5_itr1.png')
    #plot_pairplot(d2,'estimated_pair_plot.png')

    items = [str(i+1) for i in range(5)]
    subsets=[]
    for i in range(len(items) + 1):
        for c in combinations(items, i):
            subsets.append(c)
    for e  in subsets:
        if len(e) == 5:
            d1 = '../data/normalized_pf/normalized_5-MED.pf_'+e[0]+'_'+e[1]+'_'+e[2]+'_'+e[3]+'_'+e[4]
            for d2 in glob.glob('../result/estimated_bezier_simplex_allatonce/normalized_5-MED.pf_'+e[0]+'_'+e[1]+'_'+e[2]+'_'+e[3]+'_'+e[4]+'*'):
                #d2 =  '../result/estimated_bezier_curve/normalized_5-MED.pf_1_2_itr30'
                print(d2)
                d2 = d2.replace('\\','/')
                tmp = d2.split('/')[-1]
                if not os.path.exists(o_dir):
                    os.makedirs(o_dir)
                ofname = o_dir+'/'+tmp+'.png'
                plot_estimated_pairplot(d1,d2,ofname)
    """
