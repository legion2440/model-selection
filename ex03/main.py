import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split


def print_cv_results(gridsearch: GridSearchCV) -> None:
    results = pd.DataFrame(gridsearch.cv_results_)
    results = results[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].sort_values("rank_test_score")

    with pd.option_context(
        "display.max_colwidth",
        None,
        "display.max_columns",
        None,
        "display.width",
        120,
    ):
        print(results.to_string(index=False))


def main() -> None:
    housing = fetch_california_housing()
    X, y = housing["data"], housing["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        shuffle=True,
        random_state=43,
    )

    parameters = {
        "n_estimators": [10, 50, 75],
        "max_depth": [4, 7, 10],
    }

    rf = RandomForestRegressor()
    gridsearch = GridSearchCV(
        rf,
        parameters,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )

    gridsearch.fit(X_train, y_train)

    best_neg_mse = gridsearch.best_score_
    test_neg_mse = gridsearch.score(X_test, y_test)

    print("Best neg_mean_squared_error:")
    print(best_neg_mse)
    print()
    print("Corresponding best MSE:")
    print(-best_neg_mse)
    print()

    print("Best params:")
    print(gridsearch.best_params_)
    print()

    print("CV results:")
    print_cv_results(gridsearch)
    print()

    print("Test neg_mean_squared_error:")
    print(test_neg_mse)
    print()
    print("Corresponding test MSE:")
    print(-test_neg_mse)


if __name__ == "__main__":
    main()
