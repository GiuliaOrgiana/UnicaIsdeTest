import numpy as np
import matplotlib.pyplot as plt

from utils import load_mnist_data, split_data, plot_ten_digits, load_mnist_data_openml
from nmc import NMC

from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit


x, y = load_mnist_data()

splitter = ShuffleSplit(n_splits=5, train_size=.5, random_state=658131)

test_error = np.zeros(shape=(splitter.n_splits,))
# clf = NMC()
# clf = NearestCentroid()
clf = SVC(C=1, kernel='linear')

# clf.robust_estimation = True


for i, (tr_idx, ts_idx) in enumerate(splitter.split(x, y)):
    # x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=1000)
    x_tr, y_tr = x[tr_idx, :], y[tr_idx]
    x_ts, y_ts = x[ts_idx, :], y[ts_idx]
    clf.fit(x_tr, y_tr)
    # plot_ten_digit(clf.centroids)
    ypred = clf.predict(x_ts)
    test_error[i] = (ypred != y_ts).mean()
    print(test_error)
# plot_ten_digit(clf.centroids)

print(test_error.mean(), test_error.std())
