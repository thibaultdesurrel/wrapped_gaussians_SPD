import sys, os
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
print("Path: ", path)

from wrapped_classifiers import Ho_WDA, He_WDA

from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

from pymanopt.optimizers import ConjugateGradient
from sklearn.model_selection import cross_val_score, KFold

import pandas as pd
from skimage import filters
import h5py
from pyriemann.estimation import Covariances
import time
from tqdm import tqdm

np.random.seed(42)

N_rep = 5

if __name__ == "__main__":

    try:
        filename = os.path.join(path, "textile/textile/train64.h5")
        f = h5py.File(filename, "r")
        my_data = pd.read_csv(path + "/textile/textile/train64.csv", delimiter=",")
        print("Successfully loaded the data.")
    except Exception as e:
        raise RuntimeError(
            "Failed to open the data, please download it from the link: https://drive.google.com/drive/folders/13IqVNr3HawAjPfjMqjtync3mDd98WGc9?usp=share_link"
        )

    labels = my_data["indication_type"]
    cool = f["images"]
    ind = np.where((labels == "good") | (labels == "cut"))
    coo = cool[ind[0], :, :, 0]
    lab = labels[ind[0]]

    Vals = []
    Covs = []
    imgs = []

    for i in tqdm(range(len(ind[0]))):

        # We normalize the images to prevent any bias based on lighting or intensity
        # prior to filtering and covariance matrix computation.
        g = (coo[i, 1:, 1:] - np.mean(coo[i, 1:, 1:])) / np.std(coo[i, 1:, 1:])
        Intensity = g.flatten()
        Xdif = np.diff(g, axis=0)
        Ydif = np.diff(g, axis=1)
        Xdiff = np.diff(g, axis=0).flatten()
        Ydiff = np.diff(g, axis=1).flatten()
        gauss = filters.gaussian(g, sigma=2)
        gauss1 = filters.gaussian(g, sigma=3)
        gauss2 = filters.gaussian(g, sigma=4)
        gaus = filters.gaussian(g, sigma=2).flatten()
        gaus1 = filters.gaussian(g, sigma=3).flatten()
        gaus2 = filters.gaussian(g, sigma=4).flatten()
        hess = filters.laplace(g).flatten()
        frang = filters.farid(g).flatten()
        hess1 = filters.laplace(gauss).flatten()
        frang1 = filters.farid(gauss).flatten()
        hess2 = filters.laplace(gauss1).flatten()
        frang2 = filters.farid(gauss1).flatten()

        imgs.append(Intensity)

        Val = np.vstack(
            [Intensity, gaus, gaus1, gaus2, hess, frang, hess1, hess2, frang1, frang2]
        )

        Vals.append(Val)

        Covs.append(np.cov(Val))
    all_cov = np.array(Covs)
    all_labels = np.array(lab)
    N = all_cov.shape[0]
    print(all_cov.shape)

    pipelines = {}
    pipelines["MDM"] = make_pipeline(MDM(metric="riemann"))
    # pipelines["TS_LDA_full"] = make_pipeline(TangentSpace(), LDA())
    # pipelines["TS_QDA_full"] = make_pipeline(TangentSpace(), QDA())

    pipelines["TS_LDA_diag"] = make_pipeline(
        TangentSpace(), LDA(solver="lsqr", shrinkage=1.0)
    )
    pipelines["TS_QDA_diag"] = make_pipeline(TangentSpace(), GaussianNB())
    # pipelines["Ho_WDA_full"] = make_pipeline(
    #     Ho_WDA(optimizer=ConjugateGradient, max_iterations=300000, max_time=3600)
    # )
    # pipelines["He_WDA_full"] = make_pipeline(
    #     He_WDA(optimizer=ConjugateGradient, max_iterations=300000, max_time=3600)
    # )

    pipelines["Ho_WDA_diagonal"] = make_pipeline(
        Ho_WDA(
            optimizer=ConjugateGradient,
            max_iterations=300000,
            max_time=3600,
            diagonal=True,
        )
    )

    pipelines["He_WDA_diagonal"] = make_pipeline(
        He_WDA(
            optimizer=ConjugateGradient,
            max_iterations=300000,
            max_time=3600,
            diagonal=True,
        )
    )

    # Définir le KFold et le nombre de splits
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_results = []
    # Évaluer chaque pipeline
    for name, pipeline in pipelines.items():
        print(f"Fitting the pipeline {name}")
        scores = cross_val_score(
            pipeline, all_cov, all_labels, cv=cv, n_jobs=5, verbose=0
        )
        mean_score = np.mean(scores)
        # print(f"The mean score for subject {subject} is {mean_score}")

        # Accumuler les résultats
        all_results.extend(
            [
                {
                    "pipeline": name,
                    "split": i,
                    "score": scores[i],
                }
                for i in range(n_splits)
            ]
        )

    results_df = pd.DataFrame(all_results)
    print(results_df.groupby("pipeline")["score"].agg(["mean", "var"]).reset_index())
    results_df.to_pickle(path + "/textile/results_textile_diagonal.pkl")
