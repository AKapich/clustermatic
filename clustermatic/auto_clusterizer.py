from clustermatic.preprocessing import Preprocessor
from clustermatic.optimization import Optimizer
from clustermatic.model_saver import ModelSaver
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


class AutoClusterizer:
    def __init__(
        self,
        optimization_method="bayes",
        n_iterations=30,
        score_metric="silhouette",
        numerical_impute_strategy="mean",
        categorical_impute_strategy="constant",
        numerical_scaling_strategy="standard",
        categorical_encoding_strategy="onehot",
        reduce_dim=False,
        seed=None,
    ):
        self.preprocessor = Preprocessor(
            numerical_impute_strategy=numerical_impute_strategy,
            categorical_impute_strategy=categorical_impute_strategy,
            numerical_scaling_strategy=numerical_scaling_strategy,
            categorical_encoding_strategy=categorical_encoding_strategy,
            reduce_dim=reduce_dim,
        )
        self.optimizer = Optimizer(
            optimization_method=optimization_method,
            n_iterations=n_iterations,
            score_metric=score_metric,
            seed=seed,
        )
        self.best_model = None
        self.best_models = None
        self.report = None

    def fit(self, X, save_model=True):
        X = self.preprocessor.process(X)
        best_models, report = self.optimizer.optimize(X)
        self.best_models = best_models
        self.report = report
        self.best_model = best_models[report.iloc[0]["Algorithm"]]
        if save_model:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"clustermatic/best_model_{report.iloc[0]['Algorithm']}_{current_time}.pkl"
            model_saver = ModelSaver(self.best_model, model_path)
            model_saver.save_model()

    def cluster(self, X):
        assert hasattr(
            self, "best_model"
        ), "Model not trained. Run fit method before cluster."
        return self.best_model.fit_predict(X)

    def fit_cluster(self, X, save_model=True):
        self.fit(X, save_model)
        return self.cluster(X)
