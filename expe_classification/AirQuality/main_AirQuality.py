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
from sklearn.covariance import ledoit_wolf
import time
from tqdm import tqdm
from pyriemann.utils.mean import mean_riemann

np.random.seed(42)

N_rep = 5
window_size = 25
if __name__ == "__main__":

    try:
        df = pd.read_csv(
            path + "/AirQuality/Data/Hourly_data_of_Beijing_from_Jinxi_20210305.csv",
            delimiter=",",
        )
        latlon = pd.read_csv(path + "/AirQuality/Data/latlondata.csv", delimiter=",")
        print("Successfully loaded the data.")
    except Exception as e:
        raise RuntimeError(
            "Failed to open the data, please download it from the link: https://drive.google.com/drive/folders/1qdldmpP-25UozI_wVTRg3D3JZf_4aaGQ?usp=share_link"
        )

    df = df.interpolate(method="linear", axis=0)
    df1 = df.iloc[:, 11:17]

    # Indices for unique values
    Sites = df.Site.unique()
    Dates = df.Date.unique()
    Months = df.Month.unique()
    holi = df.Holiday.unique()
    hours = df.Hour.unique()
    dow = df.DOW.unique()

    Matrices = []
    labels = []
    Covs = []
    coord = []
    numz = []
    Cors = []
    for i in tqdm(Sites):
        for j in holi:
            Data = df1.loc[(df["Site"] == i) & (df["Holiday"] == j)]
            label = df.loc[(df["Site"] == i) & (df["Holiday"] == j)].iloc[:, 0]
            lat = latlon.loc[latlon["Site"] == i].iloc[:, 1]
            lon = latlon.loc[latlon["Site"] == i].iloc[:, 2]
            coord.append(np.hstack([lat, lon]))
            Matrices.append(Data.values)
            labels.append(str(j))
            Wcov = ledoit_wolf(Data.values)[0]
            Wcor = np.corrcoef(Data.values.T)
            Covs.append(Wcov)
            Cors.append(Wcor)

    for i in tqdm(Sites):
        for j in holi:
            if j == "holiday":
                numz.append(0)

            elif j == "weekday":
                numz.append(2)

            else:
                numz.append(1)
    all_cov = np.array(Covs)
    all_labels = np.array(labels)
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
    results_df.to_pickle(path + "/AirQuality/results_AirQuality_diagonal.pkl")
