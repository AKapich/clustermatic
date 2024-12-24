from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from clustermatic.scoring import (
    silhouette_scorer,
    davies_bouldin_scorer,
    calinski_harabasz_scorer,
)
from clustermatic.utils import search_spaces_bayes, search_spaces_random


class Optimizer:
    def __init__(
        self, optimization_method="bayes", n_iterations=30, score_metric="silhouette"
    ):
        assert optimization_method in [
            "bayes",
            "random",
        ], "Invalid optimization method. Choose from 'bayes' or 'random'."
        self.optimization_method = optimization_method
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

    def optimize(self, X):
        results = []
        best_models = {}
        search_spaces = (
            search_spaces_bayes
            if self.optimization_method == "bayes"
            else search_spaces_random
        )
        search_class = (
            BayesSearchCV if self.optimization_method == "bayes" else RandomizedSearchCV
        )

        for algorithm, params in search_spaces.items():
            search = search_class(
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

        sort_ascending = (
            False
            if self.scorer in [silhouette_scorer, calinski_harabasz_scorer]
            else True
        )
        report = (
            pd.DataFrame(results)
            .sort_values(by="Best Score", ascending=sort_ascending)
            .reset_index(drop=True)
        )
        self.report = report

        return best_models, report
