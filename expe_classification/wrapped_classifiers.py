import sys, os

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

import numpy as np

from source.estimation_autograd import estimation_wrapped_gaussian, estimation_Ho_WDA

from pyriemann.utils.base import invsqrtm, logm
from pyriemann.utils.tangentspace import log_map_riemann

from sklearn.base import BaseEstimator, ClassifierMixin
from pymanopt.optimizers import SteepestDescent


def vect_In(X):
    """Vectorize an array of symmetric matrices of size (m, n, n) into an array of size (m, D) where D = n(n+1)/2.
    It is the implementation of the homeomorphism between T_In P_n = S_n and R^D at point I_n described in the paper
    """
    m, n = X.shape[:2]
    idx = np.triu_indices(n, k=1)
    diag_elements = X[np.arange(m)[:, None], np.eye(n, dtype=bool)]
    other_elements = np.sqrt(2) * X[:, idx[0], idx[1]]
    return np.hstack((diag_elements, other_elements))


def vect_sigma(X, Sigma):
    """Vectorize an array of symmetric matrices of size (m, n, n) at point Sigma into an array of size (m, D) where D = n(n+1)/2.
    It is the implementation of the homeomorphism between T_Sigma P_n = S_n and R^D at point Sigma described in the paper
    """
    sigma_invsqrt = invsqrtm(Sigma)
    return vect_In(sigma_invsqrt @ X @ sigma_invsqrt)


class Ho_WDA(BaseEstimator, ClassifierMixin):
    """A Homogeneous Wrapped Discriminant Analysis classifier for SPD matrices based on the Wrapped Gaussian distribution.
    Each class k is modeled by (p_k ,mu_k, Sigma) where Sigma is shared amoungst all the classes.
    """

    def __init__(
        self,
        verbosity=1,
        max_iterations=10000,
        max_time=1000,
        optimizer=SteepestDescent,
        diagonal=False,
    ):
        """Init.

        Args:
            verbosity (int, optional): The verbosity of the optimization when fitting the classifier. Defaults to 1.
            max_iterations (int, optional): The maximum number of iterations for the optimization. Defaults to 10000.
            optimizer (pymanopt.optimizer, optional): The optimizer to use for the optimization. Defaults to SteepestDescent.
            diagonal (bool, optional): Whether the covariance matrix is diagonal or not. Defaults to False.
        """
        self.verbosity = verbosity
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.optimizer = optimizer
        self.diagonal = diagonal
        self.classes = None
        self.p = None
        self.mu = None
        self.Sigma = None
        self.stopping_criterion = None

    def fit(self, X, y):
        """Fit the classifier.

        Args:
            X (num, dim, dim): The training set of SPD matrices of size dim x dim
            y (num, ): The labels of the training set

        Returns:
            The fitted classifier
        """
        self.classes_ = np.unique(y)
        N_total = X.shape[0]
        # We to learn the parameters of the wrapped gaussians
        res_optim = estimation_Ho_WDA(
            X,
            y,
            verbosity=self.verbosity,
            max_iterations=self.max_iterations,
            max_time=self.max_time,
            optimizer=self.optimizer,
            diagonal=self.diagonal,
        )

        self.p = res_optim.point[0]
        self.mu = res_optim.point[1]
        if self.diagonal:
            self.Sigma = np.diag(res_optim.point[2] ** 2)
        else:
            self.Sigma = res_optim.point[2]
        self.pi = []
        for c in self.classes_:
            self.pi.append(X[y == c].shape[0] / N_total)

        self.stopping_criterion = res_optim.stopping_criterion
        return self

    def compute_log_likelihood(self, X, p, mu, Sigma):
        """Compute the log-likelihood of the observations X according to a Wrapped Gaussian WG(p; mu, Sigma).

        Args:
            X (n, dim, dim): A set of SPD matrices of size dim x dim
            p (dim, dim): The parameter p of the Wrapped Gaussian
            mu (dim*(dim+1)/2): The parameter mu of the Wrapped Gaussian
            Sigma (dim*(dim+1)/2, dim*(dim+1)/2): The parameter Sigma of the Wrapped Gaussian

        Returns:
            n, : The log-likelihood of each observation
        """
        n = X.shape[1]  # Dimension of the sample matrices
        d = (n * (n + 1)) / 2  # Dimension of the tangent space
        log_vect = vect_sigma(log_map_riemann(X, p, C12=True), p)

        # Density of the multivariate normal on the tangent space
        expo = np.exp(
            -np.sum(((log_vect - mu) @ np.linalg.inv(Sigma)) * (log_vect - mu), axis=1)
            / 2
        )
        const_norm = 1.0 / np.sqrt(np.power(2.0 * np.pi, d) * np.linalg.det(Sigma))
        h_ = const_norm * expo

        # The Jacobian of the exponential
        # (without the constante in front but it is not importante for the MLE)
        p_invsqrt = invsqrtm(p)
        eigenvalues, _ = np.linalg.eig(logm(p_invsqrt @ X @ p_invsqrt))

        # Creating an index grid for differences
        indices = np.arange(n)

        # Use broadcasting to compute all eigenvalue differences
        diff_matrix = eigenvalues[:, :, np.newaxis] - eigenvalues[:, np.newaxis, :]

        # Take only the differences for i < j
        triu_indices = np.triu_indices(n, k=1)
        diff_triu = diff_matrix[:, triu_indices[0], triu_indices[1]]

        # Compute sinh((lambda_i - lambda_j) / 2) / (lambda_i - lambda_j)
        sinh_term = np.sinh(diff_triu / 2) / diff_triu

        # Calculate the product of all terms for each matrix
        J_ = np.prod(sinh_term, axis=1)
        return np.log(h_ / J_)

    def predict(self, X):
        """Predict the class of the input matrices.

        Args:
            X (n, dim, dim): A set of SPD matrices of size dim x dim to classify

        Returns:
            n, : The predicted class of each input matrix
        """

        if self.p is None:
            raise ValueError("The classifier has not been fitted yet.")

        nb_samples = X.shape[0]  # Number of matrices to classify
        n = X.shape[1]  # Dimension of the sample matrices
        d = (n * (n + 1)) / 2  # Dimension of the tangent space

        log_likelihoods = np.zeros((nb_samples, len(self.classes_)))
        for i in range(len(self.classes_)):
            log_likelihoods[:, i] = self.compute_log_likelihood(
                X, self.p[i], self.mu[i], self.Sigma
            ) + np.log(self.pi[i])
        self.log_likelihoods = log_likelihoods
        return self.classes_[np.argmax(log_likelihoods, axis=1)]

    def predict_proba(self, X):
        """Predict the probabilities using softmax of the log-likelihoods

        Args:
            X (n, dim, dim): A set of SPD matrices of size dim x dim to classify

        Returns:
            n, : The probabilities of each input matrix under the fitted model
        """
        if self.p is None:
            raise ValueError("The classifier has not been fitted yet.")

        # Number of samples
        nb_samples = X.shape[0]
        # Initialize log-likelihoods array
        log_likelihoods = np.zeros((nb_samples, len(self.classes_)))
        for i in range(len(self.classes_)):
            log_likelihoods[:, i] = self.compute_log_likelihood(
                X, self.p[i], self.mu[i], self.Sigma
            ) + np.log(self.pi[i])
        # Apply softmax to convert log-likelihoods to probabilities
        max_log_likelihoods = np.max(log_likelihoods, axis=1, keepdims=True)
        exp_log_likelihoods = np.exp(log_likelihoods - max_log_likelihoods)
        probabilities = exp_log_likelihoods / np.sum(
            exp_log_likelihoods, axis=1, keepdims=True
        )

        return probabilities


class He_WDA(BaseEstimator, ClassifierMixin):
    """A Quadratic Discriminant Analysis classifier for SPD matrices based on the Wrapped Gaussian distribution.
    Each class has its own set of parameters (p, mu, Sigma).
    """

    def __init__(
        self,
        verbosity=1,
        max_iterations=10000,
        max_time=1000,
        optimizer=SteepestDescent,
        diagonal=False,
    ):
        """Init.

        Args:
            verbosity (int, optional): The verbosity of the optimization when fitting the classifier. Defaults to 1.
            max_iterations (int, optional): The maximum number of iterations for the optimization. Defaults to 10000.
            optimizer (pymanopt.optimizer, optional): The optimizer to use for the optimization. Defaults to SteepestDescent.
            diagonal (bool, optional): Whether the covariance matrix is diagonal or not. Defaults to False.
        """
        self.verbosity = verbosity
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.optimizer = optimizer
        self.diagonal = diagonal
        self.classes = None
        self.p = None
        self.mu = None
        self.Sigma = None
        self.stopping_criterion = None

    def estim_param(self, X):
        """Estimate the parameters of the wrapped gaussian distribution.

        Args:
            X (n, dim, dim): The set of SPD matrices of the initial training set that belong to the same class

        Returns:
            tuple: The estimated parameters (p, mu, Sigma) of the wrapped gaussian distribution for the given class
        """
        d = X.shape[1]
        res_optim = estimation_wrapped_gaussian(
            X,
            verbosity=self.verbosity,
            max_iterations=self.max_iterations,
            max_time=self.max_time,
            optimizer=self.optimizer,
            diagonal=self.diagonal,
        )
        return (
            res_optim.point[0],
            res_optim.point[1],
            res_optim.point[2],
            res_optim.stopping_criterion,
        )

    def fit(self, X, y):
        """Fit the classifier.

        Args:
            X (num, dim, dim): The training set of SPD matrices of size dim x dim
            y (num, ): The labels of the training set

        Returns:
            The fitted classifier
        """
        self.classes_ = np.unique(y)
        N_total = X.shape[0]
        # For each class, we need to learn the parameters of the wrapped gaussian
        self.p = []
        self.mu = []
        self.Sigma = []
        self.pi = []
        self.stopping_criterion = []
        for c in self.classes_:
            p, mu, Sigma, stopping_criterion = self.estim_param(X[y == c])
            self.p.append(p)
            self.mu.append(mu)
            if self.diagonal:
                Sigma = np.diag(Sigma**2)
            self.Sigma.append(Sigma)
            self.pi.append(X[y == c].shape[0] / N_total)
            self.stopping_criterion.append(stopping_criterion)

        return self

    def compute_log_likelihood(self, X, p, mu, Sigma):
        """Compute the log-likelihood of the observations X according to a Wrapped Gaussian WG(p; mu, Sigma).

        Args:
            X (n, dim, dim): A set of SPD matrices of size dim x dim
            p (dim, dim): The parameter p of the Wrapped Gaussian
            mu (dim*(dim+1)/2): The parameter mu of the Wrapped Gaussian
            Sigma (dim*(dim+1)/2, dim*(dim+1)/2): The parameter Sigma of the Wrapped Gaussian

        Returns:
            n, : The log-likelihood of each observation
        """
        n = X.shape[1]  # Dimension of the sample matrices
        d = (n * (n + 1)) / 2  # Dimension of the tangent space
        log_vect = vect_sigma(log_map_riemann(X, p, C12=True), p)

        # Density of the multivariate normal on the tangent space
        expo = np.exp(
            -np.sum(((log_vect - mu) @ np.linalg.inv(Sigma)) * (log_vect - mu), axis=1)
            / 2
        )
        const_norm = 1.0 / np.sqrt(np.power(2.0 * np.pi, d) * np.linalg.det(Sigma))
        h_ = const_norm * expo

        # The Jacobian of the exponential
        # (without the constante in front but it is not importante for the MLE)
        p_invsqrt = invsqrtm(p)
        eigenvalues, _ = np.linalg.eig(logm(p_invsqrt @ X @ p_invsqrt))

        # Creating an index grid for differences
        indices = np.arange(n)

        # Use broadcasting to compute all eigenvalue differences
        diff_matrix = eigenvalues[:, :, np.newaxis] - eigenvalues[:, np.newaxis, :]

        # Take only the differences for i < j
        triu_indices = np.triu_indices(n, k=1)
        diff_triu = diff_matrix[:, triu_indices[0], triu_indices[1]]

        # Compute sinh((lambda_i - lambda_j) / 2) / (lambda_i - lambda_j)
        sinh_term = np.sinh(diff_triu / 2) / diff_triu

        # Calculate the product of all terms for each matrix
        J_ = np.prod(sinh_term, axis=1)
        return np.log(h_ / J_)

    def predict(self, X):
        """Predict the class of the input matrices.

        Args:
            X (n, dim, dim): A set of SPD matrices of size dim x dim to classify

        Returns:
            n, : The predicted class of each input matrix
        """
        if self.p is None:
            raise ValueError("The classifier has not been fitted yet.")

        nb_samples = X.shape[0]  # Number of matrices to classify
        n = X.shape[1]  # Dimension of the sample matrices
        d = (n * (n + 1)) / 2  # Dimension of the tangent space

        log_likelihoods = np.zeros((nb_samples, len(self.classes_)))
        for i in range(len(self.classes_)):
            log_likelihoods[:, i] = self.compute_log_likelihood(
                X, self.p[i], self.mu[i], self.Sigma[i]
            )
        self.log_likelihoods = log_likelihoods
        return self.classes_[np.argmax(log_likelihoods, axis=1)]

    def predict_proba(self, X):
        """Predict the probabilities using softmax of the log-likelihoods

        Args:
            X (n, dim, dim): A set of SPD matrices of size dim x dim to classify

        Returns:
            n, : The probabilities of each input matrix under the fitted model
        """
        if self.p is None:
            raise ValueError("The classifier has not been fitted yet.")

        # Number of samples
        nb_samples = X.shape[0]
        # Initialize log-likelihoods array
        log_likelihoods = np.zeros((nb_samples, len(self.classes_)))
        for i in range(len(self.classes_)):
            log_likelihoods[:, i] = self.compute_log_likelihood(
                X, self.p[i], self.mu[i], self.Sigma[i]
            )
        # Apply softmax to convert log-likelihoods to probabilities
        max_log_likelihoods = np.max(log_likelihoods, axis=1, keepdims=True)
        exp_log_likelihoods = np.exp(log_likelihoods - max_log_likelihoods)
        probabilities = exp_log_likelihoods / np.sum(
            exp_log_likelihoods, axis=1, keepdims=True
        )

        return probabilities
