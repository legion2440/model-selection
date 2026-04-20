import numpy as np
from sklearn.model_selection import KFold


def main() -> None:
    X = np.array(np.arange(1, 21).reshape(10, -1))
    y = np.array(np.arange(1, 11))

    kf = KFold(n_splits=5)

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        print("Fold: ", fold)
        print(f"TRAIN: {train_index} TEST: {test_index}")
        if fold < 5:
            print()


if __name__ == "__main__":
    main()
