from clustermatic.preprocessing import Preprocessor


class AutoClusterizer:
    def __init__(self, mode="basic", n_iterations=30):
        self.mode = mode
        self.n_iterations = n_iterations
        self.preprocessor = Preprocessor()
