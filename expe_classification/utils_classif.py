import sys, os
import numpy as np
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
print("Path: ", path)

from pyriemann.classification import MDM
from wrapped_classifiers import He_WDA, Ho_WDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import log_map_riemann
from wrapped_classifiers import vect_sigma

from source.sampling_wrapped_gaussian import sample_wrapped_gaussian


def compute_score(all_train, labels_train, all_test, labels_test):
    """Computes the scores of the different classifiers on the given data.
    This function is used in the synthetic experiements on the classification using of wrapped Gaussian distributions.

    Args:
        all_train (n_train, dim, dim) : The training set of SPD matrices
        labels_train (n_train, ): The labels of the training set
        all_test (n_test, dim, dim): The test set of SPD matrices
        labels_test (n_test, ): The labels of the test set

    Returns:
        tuple: The scores of the MDM, LDA, QDA, Ho_WDA and He_WDA classifiers on the training and test sets
    """
    MDM_ = MDM()
    MDM_.fit(all_train, labels_train)
    score_MDM = MDM_.score(all_test, labels_test)

    mean_point = mean_riemann(all_train)
    all_train_TS_vect = vect_sigma(
        log_map_riemann(all_train, mean_point, C12=True), mean_point
    )
    all_test_TS_vect = vect_sigma(
        log_map_riemann(all_test, mean_point, C12=True), mean_point
    )

    LDA_ = LDA()
    LDA_.fit(all_train_TS_vect, labels_train)
    score_LDA = LDA_.score(all_test_TS_vect, labels_test)

    QDA_ = QDA()
    QDA_.fit(all_train_TS_vect, labels_train)
    score_QDA = QDA_.score(all_test_TS_vect, labels_test)

    Ho_WDA_ = Ho_WDA(verbosity=0)
    Ho_WDA_.fit(all_train, labels_train)
    score_Ho_WDA = Ho_WDA_.score(all_test, labels_test)

    He_WDA_ = He_WDA(verbosity=0)
    He_WDA_.fit(all_train, labels_train)
    score_He_WDA = He_WDA_.score(all_test, labels_test)

    return (
        score_MDM * 100,
        score_LDA * 100,
        score_QDA * 100,
        score_Ho_WDA * 100,
        score_He_WDA * 100,
    )


def generate_dataset(
    p_alpha, mu_alpha, Sigma_alpha, p_beta, mu_beta, Sigma_beta, N_train, N_test
):
    """Generates a synthetic dataset of SPD matrices from wrapped Gaussian distributions given the parameters."""

    X_train_alpha = sample_wrapped_gaussian(N_train, p_alpha, mu_alpha, Sigma_alpha)
    X_train_beta = sample_wrapped_gaussian(N_train, p_beta, mu_beta, Sigma_beta)
    all_train = np.concatenate((X_train_alpha, X_train_beta))
    all_labels_train = np.array([0] * N_train + [1] * N_train)

    X_test_alpha = sample_wrapped_gaussian(N_test, p_alpha, mu_alpha, Sigma_alpha)
    X_test_beta = sample_wrapped_gaussian(N_test, p_beta, mu_beta, Sigma_beta)
    all_test = np.concatenate((X_test_alpha, X_test_beta))
    all_labels_test = np.array([0] * N_test + [1] * N_test)

    return all_train, all_labels_train, all_test, all_labels_test
