import numpy as np
from typing import List, Dict
from src.DataStructureCalculator import DataStructureCalculator


class NewDSC(DataStructureCalculator):
    @staticmethod
    def computeDS(weightMatrixList: List[np.ndarray]) -> Dict[int, Dict[int, float]]:
        """
        Computes the chosen data structure to compute simplexes from weightMatrixList.

        Parameters
        ---------
        weightMatrix : List[np.ndarray]
            Neural network's weight matrix list.

        Returns
        -------
        : Dict[int, Dict[int, float]]
            The chosen data structure.
        """
        weightMatrixList.reverse()
        shapes = [a.shape[1] for a in weightMatrixList]
        l_res = [[] for _ in range(sum(shapes))]
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
                    w = weight_plus[i][j] / normalize_factor
                    if w > 0:
                        l_res[i + c1].append([j + c2, w])
            c2 += shapes[k]
        return {k: dict(v) for k, v in enumerate(l_res)}
