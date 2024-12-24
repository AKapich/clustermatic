from clustermatic.preprocessing import Preprocessor
from clustermatic.optimization import Optimizer
import warnings

warnings.filterwarnings("ignore")


class AutoClusterizer:
    def __init__(
        self,
        mode="basic",
        optimization_method="bayes",
        n_iterations=30,
        score_metric="silhouette",
        numerical_impute_strategy="mean",
        categorical_impute_strategy="constant",
        numerical_scaling_strategy="standard",
        categorical_encoding_strategy="onehot",
        reduce_dim=False,
    ):
        self.mode = mode
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
        )
