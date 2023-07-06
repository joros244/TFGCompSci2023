import abc
from typing import Any, List


class DataStructureCalculator(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "computeDS")
            and callable(subclass.computeDS)
            or NotImplemented
        )

    @staticmethod
    @abc.abstractmethod
    def computeDS(weightMatrix: List[List[float]]) -> Any:
        """
        Computes the chosen data structure to compute simplexes from weightMatrix.

        Parameters
        ---------
        weightMatrix : List[float]
            Neural network's weight matrix.

        Returns
        -------
        : Any
            The chosen data structure. (A matrix, a dictionary, a list, ...)
        """
        pass
