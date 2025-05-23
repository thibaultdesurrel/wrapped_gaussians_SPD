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
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB

from pymanopt.optimizers import ConjugateGradient
from sklearn.model_selection import cross_val_score, KFold

import pandas as pd
import breizhcrops as bzh
from pyriemann.estimation import Covariances
import time
from tqdm import tqdm
from pyriemann.utils.mean import mean_riemann

np.random.seed(24)

N_rep = 5

if __name__ == "__main__":

    dataset = bzh.BreizhCrops(region="frh01")
    N = len(dataset)
    all_x = []
    all_y = []
    all_field_id = []
    for i in tqdm(range(N)):
        x, y, field_id = dataset[i]
        all_x.append(np.swapaxes(np.array(x), 0, 1))
        all_y.append(y.item())
        all_field_id.append(field_id)
    all_x = np.array(all_x)
    all_labels = np.array(all_y)
    all_cov = Covariances(estimator="oas").fit_transform(all_x)
    labels_unique, counts = np.unique(all_labels, return_counts=True)
    labels_to_dl = np.where(counts < 1000)[0]
    for lab in labels_to_dl:
        idx_to_delete = np.where(all_labels == lab)[0]
        all_labels = np.delete(all_labels, idx_to_delete)
        all_cov = np.delete(all_cov, idx_to_delete, axis=0)
    N = all_cov.shape[0]
    print("Number of classes: ", np.unique(all_labels).shape[0])
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

    # pipelines["Ho_WDA_full"] = make_pipeline(
    #     Ho_WDA(optimizer=ConjugateGradient, max_iterations=300000, max_time=3600)
    # )
    # pipelines["He_WDA_full"] = make_pipeline(
    #     He_WDA(optimizer=ConjugateGradient, max_iterations=300000, max_time=3600)
    # )
    # moyenne_riem = mean_riemann(all_cov)
    pipelines["Ho_WDA_diagonal"] = make_pipeline(
        Ho_WDA(
            optimizer=ConjugateGradient,
            max_iterations=1000000,
            max_time=2 * 3600,
            diagonal=True,
        )
    )

    pipelines["He_WDA_diagonal"] = make_pipeline(
        He_WDA(
            optimizer=ConjugateGradient,
            max_iterations=1000000,
            max_time=2 * 3600,
            diagonal=True,
        )
    )
    # Définir le KFold et le nombre de splits
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=24)
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
    results_df.to_pickle(path + "/Breizhcrops/results_Breizhcrops_frh01_diagonal_.pkl")
