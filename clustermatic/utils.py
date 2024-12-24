from sklearn.cluster import (
    KMeans,
    DBSCAN,
    MiniBatchKMeans,
    AgglomerativeClustering,
    SpectralClustering,
    OPTICS,
)
from scipy.stats import uniform, randint


search_spaces_bayes = {
    KMeans: {
        "n_clusters": (2, 10),
        "init": ["k-means++", "random"],
        "n_init": (10, 20),
        "tol": (1e-4, 1e-2),
    },
    DBSCAN: {
        "eps": (0.1, 10.0),
        "min_samples": (2, 20),
        "metric": ["euclidean", "manhattan", "cosine"],
    },
    MiniBatchKMeans: {
        "n_clusters": (2, 10),
        "init": ["k-means++", "random"],
        "n_init": (10, 20),
        "tol": (1e-4, 1e-2),
        "max_iter": (100, 300),
        "batch_size": (10, 100),
    },
    AgglomerativeClustering: {
        "n_clusters": (2, 10),
        "metric": ["euclidean", "l1", "l2", "manhattan", "cosine"],
        "linkage": ["complete", "average", "single"],
    },
    OPTICS: {
        "min_samples": (2, 20),
        "xi": (0.01, 0.5),
        "min_cluster_size": (0.01, 0.5),
    },
    SpectralClustering: {
        "n_clusters": (2, 10),
        "eigen_solver": ["arpack", "lobpcg", "amg"],
        "affinity": ["nearest_neighbors", "rbf"],
        "n_init": (10, 20),
        "assign_labels": ["kmeans", "discretize"],
    },
}

search_spaces_random = {
    KMeans: {
        "n_clusters": randint(2, 11),
        "init": ["k-means++", "random"],
        "n_init": randint(10, 21),
        "tol": uniform(1e-4, 1e-2 - 1e-4),
    },
    DBSCAN: {
        "eps": uniform(0.1, 10.0 - 0.1),
        "min_samples": randint(2, 21),
        "metric": ["euclidean", "manhattan", "cosine"],
    },
    MiniBatchKMeans: {
        "n_clusters": randint(2, 11),
        "init": ["k-means++", "random"],
        "n_init": randint(10, 21),
        "tol": uniform(1e-4, 1e-2 - 1e-4),
        "max_iter": randint(100, 301),
        "batch_size": randint(10, 101),
    },
    AgglomerativeClustering: {
        "n_clusters": randint(2, 11),
        "metric": [
            "euclidean",
            "l1",
            "l2",
            "manhattan",
            "cosine",
        ],
        "linkage": ["complete", "average", "single"],
    },
    OPTICS: {
        "min_samples": randint(2, 21),
        "xi": uniform(0.01, 0.5 - 0.01),
        "min_cluster_size": uniform(0.01, 0.5 - 0.01),
    },
    SpectralClustering: {
        "n_clusters": randint(2, 11),
        "eigen_solver": ["arpack", "lobpcg", "amg"],
        "affinity": ["nearest_neighbors", "rbf"],
        "n_init": randint(10, 21),
        "assign_labels": ["kmeans", "discretize"],
    },
}
