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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import GaussianNB

from pymanopt.optimizers import ConjugateGradient
from sklearn.model_selection import cross_val_score, KFold

import pandas as pd
from pyriemann.estimation import Covariances
import time
from tqdm import tqdm
from pyriemann.utils.mean import mean_riemann
from data import read_indian_pines
from pipelines_utils import PCAImage, SlidingWindowVectorize, RemoveMeanImage
from numpy.lib.stride_tricks import sliding_window_view


def create_labels(labels, window_size):
    labels_sliding_windows = sliding_window_view(
        labels, window_shape=(window_size, window_size), axis=(0, 1)
    )
    labels_sliding_windows = labels_sliding_windows.reshape(
        (-1, window_size, window_size)
    )

    most_frequent_values = np.zeros(
        labels_sliding_windows.shape[0], dtype=labels_sliding_windows.dtype
    )

    for i in range(labels_sliding_windows.shape[0]):
        most_frequent_values[i] = np.bincount(
            labels_sliding_windows[i].ravel()
        ).argmax()
    return most_frequent_values


np.random.seed(42)

N_rep = 5
window_size = 25
if __name__ == "__main__":

    data, labels, labels_names = read_indian_pines(path + "/hyperspectral/datasets/")
    pipeline_to_cov = Pipeline(
        [
            ("remove_mean", RemoveMeanImage()),
            ("pca", PCAImage(n_components=5)),
            ("sliding_window", SlidingWindowVectorize(window_size)),
            ("scm", Covariances("scm")),
        ]
    )

    all_cov = pipeline_to_cov.fit_transform(data)
    print(all_cov.shape)
    all_labels = create_labels(labels, window_size)
    print("Number of classes: ", np.unique(all_labels).shape[0])

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
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=43)
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
    results_df.to_pickle(path + "/hyperspectral/results/results_Indiana_diagonal.pkl")
