import numpy as np
import matplotlib.pyplot as plt

from utils import load_mnist_data, split_data, plot_ten_digits
from nmc import NMC

from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

from data_perturb import CDataPerturbUniform, CDataPerturbGaussian

def robustness_test(clf, data_pert, param_name, param_values):

    test_accuracies = np.zeros(shape=param_values.shape)
    for i, k in enumerate(param_values):
        # perturb ts
        setattr(data_pert, param_name, k)
        # data_pert.set(param_name, k)
        data_pert.K = k
        xp = data_pert.perturb_dataset(x_ts)
        # plot_ten_digits(xp, y)
        # compute predicted labels on the perturbed ts
        y_pred = clf.predict(xp)
        # compute classification accuracy using y_pred
        clf_acc = np.mean(y_ts == y_pred)
        print("test accuracy (K=:", k, "): ", int(clf_acc * 1000) / 10, "%")
        test_accuracies[i] = clf_acc
    return test_accuracies


x, y = load_mnist_data()

# implementing perturb_dataset(x) --> xp (perturbed dataset)
# 1.first I have to create Xp for example with all 0 values
# 2. loop over the X matrix rows
# 3. so we take the samples from the given row at each iteration
# 4. we apply the data perturbation function
# 5. we copy the result (so the perturbed samples) in Xp


# plot_ten_digits(x, y)
# plot_ten_digits(xp, y)


# split MNITS data into 60% trainingset and 40% test set
n_tr = int(0.6 * x.shape[0])
print("Number of total samples: ", x.shape[0], "\nNumber of training samples: ", n_tr)
x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=n_tr)

param_values = np.array([0, 10, 20, 50, 100, 200, 330, 400, 500])

clf_list = [NMC(), SVC(kernel='linear')]
clf_names = ['NMC', 'SVC']

plt.figure(figsize=(10,5))

for i, clf in enumerate(clf_list):

    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_ts)
    clf_acc = np.mean(y_ts == y_pred)
    print("test accuracy: ", int(clf_acc * 1000) / 10, "%")

    test_accuracies = robustness_test(
        clf, CDataPerturbUniform(), param_name='K', param_values=param_values)
    plt.subplot(1,2,1)
    plt.plot(param_values, test_accuracies, label=clf_names[i])
    plt.xlabel('K')
    plt.ylabel('Test accuracy(K)')
    plt.legend()

    test_accuracies = robustness_test(
        clf, CDataPerturbGaussian(), param_name='sigma', param_values=param_values)
    plt.subplot(1,2,2)
    plt.plot(param_values, test_accuracies, label=clf_names[i])
    plt.xlabel(r'$\sigma$')
    plt.ylabel('Test accuracy($\sigma$)')
    plt.legend()

plt.show()