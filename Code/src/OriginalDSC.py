import numpy as np
from typing import List
from src.DataStructureCalculator import DataStructureCalculator


class OriginalDSC(DataStructureCalculator):
    @staticmethod
    def computeDS(weightMatrixList: List[np.ndarray]) -> np.ndarray:
        """
        Computes the chosen data structure to compute simplexes from weightMatrixList.

        Parameters
        ---------
        weightMatrix : List[np.ndarray]
            Neural network's weight matrix list.

        Returns
        -------
        : np.ndarray
            The chosen data structure.
        """
        weightMatrixList.reverse()
        shapes = [a.shape[1] for a in weightMatrixList]
        mat = np.identity(sum(shapes))
        c1 = 0
        c2 = 0
        for k in range(len(shapes) - 1):
            c1 += shapes[k]

            for j in range(0, shapes[k]):
                normalize_factor = 0
                weight_plus = weightMatrixList[k] * (weightMatrixList[k] > 0)

                for i in range(0, shapes[k + 1]):
                    normalize_factor += weight_plus[i][j]
                for i in range(0, shapes[k + 1]):
                    mat[i + c1][j + c2] = weight_plus[i][j] / normalize_factor
            c2 += shapes[k]
        return mat
