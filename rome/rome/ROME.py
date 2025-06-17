import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.cluster._optics import cluster_optics_dbscan, cluster_optics_xi
from sklearn.neighbors import KernelDensity

from sklearn.decomposition import PCA
import scipy as sp


def _silhouette_multiple_clusterings(X, clusterings):
    """
    Calculates the silhouette score for multiple clusterings and returns the best one.

    Parameters
    ----------
    X: np.ndarray
        data to calculate the silhouette score for

    clusterings: np.ndarray
        clusterings to calculate the silhouette score for

    Returns
    -------
    best_cluster: np.ndarray
        the best clustering according to the silhouette score
    """

    num_samples, _ = X.shape
    assert len(X.shape) == 2
    assert len(clusterings.shape) == 2

    # each row in clusterings is a different approach
    assert clusterings.shape[0] == num_samples

    # Get valid clusterings
    useful = (clusterings.max(0) - clusterings.min(0)) > 0

    # get distances
    Dist = np.sqrt(((X[:, np.newaxis] - X[np.newaxis]) ** 2).sum(-1))

    # Intialize best score (worst score of valid clustering is -1)
    values = -1.1 * np.ones(clusterings.shape[1])

    for i in range(clusterings.shape[1]):
        if not useful[i]:
            continue

        test_labels = clusterings[:, i]
        num_noise_samples = (test_labels == -1).sum()
        silhouette_labels = test_labels.copy()
        silhouette_labels[test_labels == -1] = test_labels.max() + 1 + np.arange(num_noise_samples)

        # Treat noise as separate cluster
        test_score_noise_separate = silhouette_score(Dist, silhouette_labels, metric="precomputed")
        test_score_noise_combined = silhouette_score(Dist, test_labels, metric="precomputed")

        noise_fac = num_noise_samples / len(X)
        values[i] = noise_fac * test_score_noise_separate + (1 - noise_fac) * test_score_noise_combined

    # Get best clustering
    best_cluster = np.argmax(values)
    return clusterings[:, best_cluster]


class ROME:
    """
    A class for the Robust Multi-modal density Estimator (ROME).
    It creates a probability density function using nonparametric methods.

    Spcecifically, it involves a multi-step process, where first clustering is
    performed using the OPTICS algorithm.

    For each cluster, the samples are then decorrelated using principal component
    analysis, and are normalised thereafter.

    Finally, for each cluster one then performs kernel density estimation, according
    to which one can calcualte probability density values or sample from.

    ...

    Attributes
    ----------

    fitted: bool
        a boolean indicating if the density estimator has been fitted to data

    min_std: float
        minimum standard deviation the estimator assumes is present in the data
        (default is 0.1)

    num_features: int
        number of features in the data

    Models: list
        list of fitted kernel density estimators for each cluster

    means: np.ndarray
        array of means for each cluster

    T_mat: np.ndarray
        array of transformation matrices for each cluster

    log_det_T_mat: np.ndarray
        array of log determinants of the transformation matrices for each cluster

    probs: np.ndarray
        array of probabilities for each cluster

    Methods
    -------

    fit(X, clusters=None)
        Fits the density estimator to the data X. If clusters are provided, they are used
        to fit the estimator, otherwise they are calculated using OPTICS.

    score_samples(X)
        Returns the log probability of the samples in X.

    sample(num_samples=1, random_state=0)
        Samples num_samples from the density estimator.



    """

    def __init__(self, min_std=0.1):
        """
        Parameters
        ----------
        min_std: float
            minimum standard deviation the estimator assumes is present in the data
            (default is 0.1)
        """
        self.fitted = False
        self.min_std = min_std

    def fit(self, X, clusters=None):
        """
        Fits the density estimator to the data X. If clusters are provided, they are used
        to fit the estimator, otherwise they are calculated using OPTICS.

        Parameters
        ----------
        X: np.ndarray
            data to fit the estimator to

        clusters: np.ndarray
            clusters to use for fitting the estimator (default is None)

        Returns
        -------
        self: ROME
            the fitted ROME object
        """

        assert len(X.shape) == 2

        self.num_features = X.shape[1]

        if clusters is None:
            if len(X) >= 5:
                num_min_samples = X.shape[0] * self.num_features / 400
                num_min_samples = int(np.clip(num_min_samples, min(5, X.shape[0]), min(20, X.shape[0])))

                # Get reachability plot
                self.optics = OPTICS(min_samples=num_min_samples)
                self.optics.fit(X)

                reachability = self.optics.reachability_[np.isfinite(self.optics.reachability_)]

                # Test potential cluster extractions
                Eps = np.linspace(0, 1, 100) ** 2
                Eps = Eps * (reachability.max() - reachability.min()) + reachability.min()
                Xi = np.linspace(0.01, 0.99, 99)

                Method = np.repeat(np.array(["Eps", "Xi"]), (len(Eps), len(Xi)))
                Params = np.concatenate((Eps, Xi), axis=0)

                # Initializes empty clusters
                Clustering = np.zeros((len(X), len(Method)), int)

                # Iterate over all potential cluster extractions
                for i in range(len(Method)):
                    method = Method[i]
                    param = Params[i]
                    # Cluster using dbscan
                    if method == "Eps":
                        eps = param
                        test_labels = cluster_optics_dbscan(
                            reachability=self.optics.reachability_,
                            core_distances=self.optics.core_distances_,
                            ordering=self.optics.ordering_,
                            eps=eps,
                        )

                    # Cluster using xi
                    elif method == "Xi":
                        xi = param
                        test_labels, _ = cluster_optics_xi(
                            reachability=self.optics.reachability_,
                            predecessor=self.optics.predecessor_,
                            ordering=self.optics.ordering_,
                            min_samples=num_min_samples,
                            min_cluster_size=2,
                            xi=xi,
                            predecessor_correction=self.optics.predecessor_correction,
                        )
                    else:
                        raise ValueError("Clustering method not recognized")

                    # Check for improvement
                    if len(np.unique(test_labels)) > 1:
                        # Check if there are lusters of size one
                        test_clusters, test_size = np.unique(test_labels, return_counts=True)

                        noise_clusters = test_clusters[test_size == 1]
                        test_labels[np.isin(test_labels, noise_clusters)] = -1
                        test_labels[test_labels > -1] = np.unique(test_labels[test_labels > -1], return_inverse=True)[1]

                        Clustering[:, i] = test_labels

                self.labels_ = _silhouette_multiple_clusterings(X, Clustering)

            else:
                self.labels_ = np.zeros(len(X))
        else:
            self.labels_ = clusters.copy()

        unique_labels, cluster_size = np.unique(self.labels_, return_counts=True)

        # Fit distribution to each cluster of data
        self.Models = [None] * len(unique_labels)

        # initialise rotation matrix for PCA
        self.means = np.zeros((len(unique_labels), self.num_features))
        self.T_mat = np.zeros((len(unique_labels), self.num_features, self.num_features))

        # Get probability adjustment
        self.log_det_T_mat = np.zeros(len(unique_labels))

        Stds = np.zeros((len(unique_labels), self.num_features))

        for i, label in enumerate(unique_labels):
            if label == -1:
                continue
            # Get cluster data
            X_label = X[self.labels_ == label]
            num_samples = len(X_label)
            assert num_samples == cluster_size[i]

            # Get mean and std
            self.means[i] = X_label.mean(0)
            Stds[i] = X_label.std(0)

            # Shift coordinate system origin to mean
            X_label_stand = X_label - self.means[[i]]

            # Decorrelate samples
            # Repeat data if not enough samples are available
            if num_samples < self.num_features:
                c = np.tile(X_label_stand, (int(np.ceil(self.num_features / num_samples)), 1))
            else:
                c = X_label_stand.copy()

            # calculate PCA on X_label_stand -> get rot matrix and std
            # Stabalize correlation matrix if necessary
            attempt = 0
            successful_pca = False
            while not successful_pca:
                if attempt < 10:
                    try:
                        pca = PCA(random_state=attempt).fit(c)
                        successful_pca = True
                    except:
                        e_fac = 10 ** (0.5 * attempt - 6)
                        c[: self.num_features] += np.eye(self.num_features) * e_fac

                        # Prepare next attempt
                        attempt += 1

                    if not successful_pca:
                        print("PCA failed, was done again with different random start.")
                else:
                    pca = PCA(random_state=attempt).fit(c)

            # Exctract components std
            pca_std = np.sqrt(pca.explained_variance_)

            # Extract roation matrix
            self.T_mat[i] = pca.components_.T

            # Initiallize probability adjustment
            self.log_det_T_mat[i] = np.log(np.abs(np.linalg.det(self.T_mat[i])))

            # Apply standardization
            # Apply minimum std levels
            pca_std = pca_std * (pca_std.max() - self.min_std) / pca_std.max() + self.min_std

            # Adjust T_mat accordingly
            self.T_mat[i] /= pca_std[np.newaxis]
            self.log_det_T_mat[i] -= np.log(pca_std).sum()

            # Apply transformation matrix
            X_label_pca = X_label_stand @ self.T_mat[i]  # @ is matrix multiplication

            # Fit Surrogate distribution
            model = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(X_label_pca)

            self.Models[i] = model

        # consider noise values
        if unique_labels[0] == -1:
            X_noise = X[self.labels_ == -1]

            # assume that no rotation is necessary rot_mat_pca[0] to be an identity matrix
            if len(self.T_mat) > 1:
                stds = np.maximum(Stds[1:].mean(0), self.min_std)
                self.T_mat[0] = np.diag(1 / stds)
                self.log_det_T_mat[0] = -np.sum(np.log(stds))
            else:
                self.T_mat[0] = np.eye(self.num_features)
                self.log_det_T_mat[0] = 0.0

            # Apply transformation matrix
            X_noise_stand = (X_noise - self.means[[0]]) @ self.T_mat[0]

            # Fit Surrogate distribution
            # We assume each noise point is its own cluster

            # calculate silverman rule assuming only 1 sample
            bandwidth = ((self.num_features + 2) / 4) ** (-1 / (self.num_features + 4))
            model_noise = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_noise_stand)

            self.Models[0] = model_noise

        # Get cluster probabilities
        self.cluster_weights = cluster_size / cluster_size.sum()

        self.fitted = True

        return self

    def _prob(self, X):
        assert self.fitted, "The model was not fitted yet"

        assert len(X.shape) == 2
        assert X.shape[1] == self.num_features

        # calculate logarithmic probability
        log_probs = np.zeros((len(X), len(self.Models)), dtype=np.float64)

        for i, model in enumerate(self.Models):
            X_stand = (X - self.means[[i]]) @ self.T_mat[i]

            # Get in model log probabilty
            if isinstance(model, KernelDensity):
                log_probs[:, i] = model.score_samples(X_stand)
            else:
                raise ValueError("Estimator not recognized")

            # adjust log prob for transformation
            log_probs[:, i] += self.log_det_T_mat[i]

            # adjust log prob for cluster likelihood
            log_probs[:, i] += np.log(self.cluster_weights[i])

        return log_probs

    def score_samples(self, X):
        """
        Returns the log probability of the samples in X.

        Parameters
        ----------
        X: np.ndarray
            data to calculate the log probability for

        Returns
        -------
        l_probs: np.ndarray
            array of log probabilities for each sample in X
        """
        log_probs = self._prob(X)
        l_probs = sp.special.logsumexp(log_probs, axis=-1)
        return l_probs

    def sample(self, num_samples=1, random_state=0):
        """
        Samples num_samples from the density estimator.

        Parameters
        ----------
        num_samples: int
            number of samples to generate (default is 1)

        random_state: int
            random seed to use for sampling (default is 0)

        Returns
        -------
        samples: np.ndarray
            array of samples generated from the density estimator
        """
        assert self.fitted, "The model was not fitted yet"

        # Determine cluster belonging
        np.random.seed(random_state)
        labels = np.random.choice(np.arange(len(self.Models)), num_samples, p=self.cluster_weights)

        samples = []

        # generate from different clusters
        for label in np.unique(labels):
            # Get number of samples from cluster
            num = (label == labels).sum()

            # Reset radnom seed to be sure
            np.random.seed(random_state)

            # Sample transformed samples from model
            if isinstance(self.Models[label], KernelDensity):
                X_label_stand = self.Models[label].sample(num, random_state)
            else:
                raise ValueError("Estimator not considered.")

            # Apply inverse transformation to get original coordinate samples
            X_label = X_label_stand @ np.linalg.inv(self.T_mat[label]) + self.means[[label]]

            # Add samples to output set
            samples.append(X_label)

        samples = np.concatenate(samples, axis=0)

        # Shuffle samples
        np.random.shuffle(samples)

        return samples
