import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, validation_curve


def plot_validation_curve(X: np.ndarray, y: np.ndarray) -> None:
    clf = RandomForestClassifier()
    param_range = np.arange(1, 30, 2)

    train_scores, test_scores = validation_curve(
        clf,
        X,
        y,
        param_name="max_depth",
        param_range=param_range,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation Curve with RandomForestClassifier")
    plt.xlabel("max_depth")
    plt.ylabel("ROC AUC")
    plt.ylim(0.7, 1.01)
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="r",
    )
    plt.plot(
        param_range,
        test_scores_mean,
        label="Cross-validation score",
        color="g",
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="g",
    )
    plt.legend(loc="best")
    plt.grid()


def plot_learning_curve(X: np.ndarray, y: np.ndarray) -> None:
    clf = RandomForestClassifier(max_depth=12)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        clf,
        X,
        y,
        cv=10,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        return_times=True,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title("Learning Curve")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    axes[0].plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color="g",
        label="Cross-validation score",
    )
    axes[0].legend(loc="best")

    axes[1].set_title("Scalability of the model")
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit times")
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )

    axes[2].set_title("Performance of the model")
    axes[2].set_xlabel("Fit times")
    axes[2].set_ylabel("Score")
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )


def main() -> None:
    X, y = make_classification(
        n_samples=100000,
        n_features=30,
        n_informative=10,
        flip_y=0.2,
    )

    plot_validation_curve(X, y)
    plot_learning_curve(X, y)
    plt.show()


if __name__ == "__main__":
    main()
