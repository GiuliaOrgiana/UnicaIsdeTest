import numpy as np
import matplotlib.pyplot as plt

from utils import load_mnist_data, split_data, plot_ten_digit, load_mnist_data_openml
from nmc import NMC

from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
n_rep = 5
x, y = load_mnist_data()

test_error = np.zeros(shape=(n_rep,))
#clf = NMC()
#clf = NearestCentroid()
clf = SVC(C=100, kernel='linear')

#clf.robust_estimation = True
for r in range(n_rep):
    print(r)
    x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=1000)
    clf.fit(x_tr, y_tr)
    #plot_ten_digit(clf.centroids)
    ypred = clf.predict(x_ts)
    test_error[r] = (ypred != y_ts).mean()

#plot_ten_digit(clf.centroids)

print(test_error.mean(), test_error.std())
