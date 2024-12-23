from sklearn.cluster import (
    KMeans,
    DBSCAN,
    MiniBatchKMeans,
    AgglomerativeClustering,
    SpectralClustering,
    OPTICS,
)

search_spaces = {
    KMeans: {
        "n_clusters": (2, 10),
        "init": ["k-means++", "random"],
        "n_init": (10, 20),
        "tol": (1e-4, 1e-2, "log-uniform"),
    },
    DBSCAN: {
        "eps": (0.1, 10.0, "log-uniform"),
        "min_samples": (2, 20),
        "metric": ["euclidean", "manhattan", "cosine"],
    },
    MiniBatchKMeans: {
        "n_clusters": (2, 10),
        "init": ["k-means++", "random"],
        "n_init": (10, 20),
        "tol": (1e-4, 1e-2, "log-uniform"),
        "max_iter": (100, 300),
        "batch_size": (10, 100),
    },
    AgglomerativeClustering: {
        "n_clusters": (2, 10),
        "metric": ["euclidean", "l1", "l2", "manhattan", "cosine"],
        "linkage": ["complete", "average", "single"],
    },
    SpectralClustering: {
        "n_clusters": (2, 10),
        "eigen_solver": ["arpack", "lobpcg", "amg"],
        "affinity": ["nearest_neighbors", "rbf"],
        "n_init": (10, 20),
        "assign_labels": ["kmeans", "discretize"],
    },
    OPTICS: {
        "min_samples": (2, 20),
        "xi": (0.01, 0.5, "log-uniform"),
        "min_cluster_size": (0.01, 0.5, "log-uniform"),
    },
}
