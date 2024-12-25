from clustermatic.preprocessing import Preprocessor
from clustermatic.optimization import Optimizer
from clustermatic.model_saver import ModelSaver
from clustermatic.evaluation import Evaluator
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
        self.evaluator = Evaluator()
        self.best_model = None
        self.best_models = None
        self.report = None
        self.preprocessed_data_ = None

    def fit(self, X, save_model=True):
        self.preprocessed_data_ = self.preprocessor.process(X)
        best_models, report = self.optimizer.optimize(self.preprocessed_data_)
        self.best_models = best_models
        self.report = report
        self.best_model = best_models[report.iloc[0]["Algorithm"]]
        self.scores_ = self.optimizer.scores_
        if save_model:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"clustermatic/best_model_{report.iloc[0]['Algorithm']}_{current_time}.pkl"
            model_saver = ModelSaver(self.best_model, model_path)
            model_saver.save_model()

    def cluster(self, X=None):
        assert hasattr(
            self, "best_model"
        ), "Model not trained. Run fit method before cluster."
        if X is not None:
            X = self.preprocessor.process(X)
        else:
            X = self.preprocessed_data_
        return self.best_model.fit_predict(X)

    def fit_cluster(self, X, save_model=True):
        self.fit(X, save_model)
        return self.cluster()

    def evaluate(self, X):
        assert hasattr(
            self, "scores_"
        ), "Model not trained. Run fit method before evaluate."
        self.evaluator.evaluate(
            scores=dict(self.scores_),
            report=self.report,
            best_model=self.best_model,
            data=self.preprocessed_data_,
        )
