from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split


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

    print(gridsearch.best_score_)
    print(gridsearch.best_params_)
    print(gridsearch.cv_results_)
    print(gridsearch.score(X_test, y_test))


if __name__ == "__main__":
    main()
