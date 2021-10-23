from abc import ABC, abstractmethod
import numpy as np


class CDataPerturb(ABC):
    """Abstract interface to define data perturbation models"""

    def __init__(self):
        pass

    @abstractmethod
    def data_perturbation(self, x):
        """
        Parameters
        ----------
        x: flattened vector to be perturbed
        Returns
        -------
        xp: the perturbed version of x
        """
        raise NotImplementedError("data_perturbation not implemented!")

    def perturb_dataset(self, x):
        """
        Parameters
        ----------
        x: matrix ...
        Returns
        -------
        xp: ...
        """
        xp = np.zeros(shape=x.shape)
        for i in range(xp.shape[0]):
            xp[i, :] = self.data_perturbation(x[i, :])
        return xp