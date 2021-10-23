import numpy as np
import matplotlib.pyplot as plt

from utils import load_mnist_data, split_data, plot_ten_digits
from nmc import NMC

from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

from data_perturb import CDataPerturbRandom

def data_perturbation(x):
    return x

x, y = load_mnist_data()

# implementing perturb_dataset(x) --> xp (perturbed dataset)
# 1.first I have to create Xp for example with all 0 values
# 2. loop over the X matrix rows
# 3. so we take the samples from the given row at each iteration
# 4. we apply the data perturbation function
# 5. we copy the result (so the perturbed samples) in Xp

data_pert = CDataPerturbRandom()

xp = data_pert.perturb_dataset(x)

plt.imshow(x[0, :].reshape(28,28))
plt.show()
plt.imshow(x[1, :].reshape(28,28))
plt.show()

plt.imshow(xp[0, :].reshape(28,28))
plt.show()
plt.imshow(xp[1, :].reshape(28,28))
plt.show()

plt.imshow(x[0, :].reshape(28,28))
plt.show()
plt.imshow(x[1, :].reshape(28,28))
plt.show()