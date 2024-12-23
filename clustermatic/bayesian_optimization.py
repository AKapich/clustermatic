from skopt import BayesSearchCV
import pandas as pd
from clustermatic.scoring import (
    silhouette_scorer,
    davies_bouldin_scorer,
    calinski_harabasz_scorer,
)
from clustermatic.utils import search_spaces


class BayesianOptimizer:
    def __init__(self, n_iterations=30, score_metric="silhouette"):
        self.n_iterations = n_iterations
        self.scorers = {
            "silhouette": silhouette_scorer,
            "davies_bouldin": davies_bouldin_scorer,
            "calinski_harabasz": calinski_harabasz_scorer,
        }
        assert (
            score_metric in self.scorers
        ), "Invalid score metric. Choose from 'silhouette', 'davies_bouldin', or 'calinski_harabasz'."
        self.scorer = self.scorers[score_metric]

    def bayesian_search(self, X):
        results = []
        best_models = {}

        for algorithm, params in search_spaces.items():
            search = BayesSearchCV(
                algorithm(),
                params,
                n_iter=self.n_iterations,
                scoring=self.scorer,
                n_jobs=-1,
            )
            search.fit(X)
            best_model = search.best_estimator_
            best_score = search.best_score_
            best_params = search.best_params_
            results.append(
                {
                    "Algorithm": algorithm.__name__,
                    "Best Score": best_score,
                    "Best Params": best_params,
                }
            )

            best_models[algorithm.__name__] = best_model

        report = pd.DataFrame(results)
        self.report = report

        return best_models, report
