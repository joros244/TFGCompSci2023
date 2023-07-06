import abc
from src.DataStructureCalculator import DataStructureCalculator


class Preprocessor(abc.ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "doPreproc")
            and hasattr(subclass, "__init__")
            and callable(subclass.doPreproc)
            and callable(subclass.__init__)
            or NotImplemented
        )

    @abc.abstractmethod
    def __init__(self, path: str, calc: DataStructureCalculator) -> None:
        """
        A Preprocessor is defined from a path to a DNN's model file and a
        DataStructureCalculator to compute the chosen data structure.

        Parameters
        ---------
        path : src
            Neural network's model file.
        calc : DataStructureCalculator
            DataStructureCalculator to compute the data structure.

        """
        super().__init__()
        self.modelPath = path
        self.calculator = calc

    @abc.abstractmethod
    def doPreproc(self, savePath: str):
        """
        Computes the chosen data structure and saves it in "savePath".

        Parameters
        ---------
        savePath : str
            Path to save the computed data structure.

        """
        pass
