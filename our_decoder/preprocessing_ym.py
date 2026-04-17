import numpy as np
from scipy.io import loadmat
import pickle
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

# with open("../data/example_data_m1.pickle", "rb") as f:
#     data = pickle.load(f, encoding="latin1")
#
# print(type(data))
# print(len(data))
# for i, item in enumerate(data):
#     if hasattr(item, "shape"):
#         print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")
#     else:
#         print(f"  [{i}]: type={type(item)}, value={item}")

def load_dataset(path):
    # data = loadmat(path)
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data[0], data[1]

def preprocess(neural_data, y, bins_before=13, bins_current=1, bins_after=0,
               train_frac=0.8):
    X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)

    if bins_after > 0:
        X = X[bins_before:-bins_after]
        y = y[bins_before:-bins_after]
    else:
        X = X[bins_before:]
        y = y[bins_before:]

    print(f"X: {X.shape}, y: {y.shape}")

    assert X.shape[0] == y.shape[0], "X and y sample mismatch"

    split = int(train_frac * X.shape[0])

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Normalize using training stats only
    X_mean = np.nanmean(X_train.reshape(-1, X_train.shape[-1]), axis=0)
    X_std = np.nanstd(X_train.reshape(-1, X_train.shape[-1]), axis=0)
    X_std[X_std == 0] = 1

    y_mean = np.nanmean(y_train, axis=0)
    y_std = np.nanstd(y_train, axis=0)
    y_std[y_std == 0] = 1

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_mean": X_mean, "X_std": X_std,
        "y_mean": y_mean, "y_std": y_std,
    }