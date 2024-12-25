import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


class Evaluator:
    def __init__(self):
        pass

    def boxplot(self, scores):
        model_names = scores.keys()
        model_scores = scores.values()
        plt.figure(figsize=(14, 7))
        sns.set_theme(style="whitegrid")

        palette = sns.color_palette("pastel", len(model_names))

        box = plt.boxplot(
            model_scores,
            labels=model_names,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"color": "gray", "linewidth": 1.5},
            capprops={"color": "gray", "linewidth": 1.5},
            flierprops={"marker": "o", "color": "gray", "alpha": 0.7},
        )

        for patch, color in zip(box["boxes"], palette):
            patch.set_facecolor(color)

        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.title("Model performance", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Models", fontsize=14, labelpad=10)
        plt.ylabel("Scores", fontsize=14, labelpad=10)

        for i, color in enumerate(palette):
            plt.scatter([], [], c=[color], label=list(model_names)[i])
        plt.legend(title="Models", loc="lower right", frameon=True)

        plt.tight_layout()
        plt.show()

    def cummulative_plot(self, scores):
        cumulative_best_scores = {
            model: [max(scores_list[: i + 1]) for i in range(len(scores_list))]
            for model, scores_list in scores.items()
        }

        plt.figure(figsize=(14, 7))
        sns.set_theme(style="whitegrid")

        for model, cumulative_scores in cumulative_best_scores.items():
            plt.plot(
                range(1, len(cumulative_scores) + 1),
                cumulative_scores,
                label=model,
                marker="o",
                linewidth=2,
            )

        plt.title(
            "Best Performance Over Iterations", fontsize=16, fontweight="bold", pad=20
        )
        plt.xlabel("Iteration", fontsize=14, labelpad=10)
        plt.ylabel("Best Score", fontsize=14, labelpad=10)
        plt.xticks(range(1, len(next(iter(cumulative_best_scores.values()))) + 1))
        plt.legend(title="Models", loc="lower right", fontsize=12, frameon=True)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    def clusters_plot(self, best_model, data):
        labels = best_model.fit_predict(data)
        plt.figure(figsize=(14, 7))
        if data.shape[1] == 2:
            plot_data = data
            x_label, y_label = "Feature 1", "Feature 2"
            title = "Cluster Plot"
        else:
            pca = PCA(n_components=2)
            plot_data = pca.fit_transform(data)
            x_label, y_label = "Principal Component 1", "Principal Component 2"
            title = "Cluster Plot (PCA)"

        sns.scatterplot(
            x=plot_data[:, 0], y=plot_data[:, 1], hue=labels, palette="viridis"
        )
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.xlabel(x_label, fontsize=14, labelpad=10)
        plt.ylabel(y_label, fontsize=14, labelpad=10)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def evaluate(self, scores, report, best_model, data):
        print(report)
        self.boxplot(scores)
        self.cummulative_plot(scores)
        self.clusters_plot(best_model, data)
