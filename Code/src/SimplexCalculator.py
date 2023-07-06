import abc
from typing import List


class SimplexCalculator(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "compute")
            and callable(subclass.compute)
            and hasattr(subclass, "compute_full")
            and callable(subclass.compute_full)
            or NotImplemented
        )

    @abc.abstractmethod
    def compute(self, t: float, save_path: str):
        """
        Computes the simplicial complex of a DAG using t as a threshold,
        then it saves it in save_path.

        Parameters
        ---------
        t : float
            Threshold value.
        save_path : str
            Path to save the computed simplicial complex.
        """
        pass

    @abc.abstractmethod
    def compute_full(self, t: List[float], save_path: str):
        """
        Computes the simplicial complex of a DAG using t as a list of threshold
        values, then it saves it in save_path.

        Parameters
        ---------
        t : List[float]
            List of threshold values.
        save_path : str
            Path to save the computed simplicial complex.

        """
        pass
