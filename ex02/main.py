import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    housing = fetch_california_housing()
    X, y = housing["data"], housing["target"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.1,
        shuffle=True,
        random_state=43,
    )

    pipeline = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ]
    pipe = Pipeline(pipeline)

    scores = cross_validate(pipe, X_train, y_train, cv=10, scoring="r2")
    validation_scores = scores["test_score"]

    print("Scores on validation sets:")
    print("", validation_scores)
    print()
    print("Mean of scores on validation sets:")
    print("", np.mean(validation_scores))
    print()
    print("Standard deviation of scores on validation sets:")
    print("", np.std(validation_scores))


if __name__ == "__main__":
    main()
