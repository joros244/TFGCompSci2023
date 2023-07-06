import pickle
import keras.layers

from src.Preprocessor import Preprocessor
from src.DataStructureCalculator import DataStructureCalculator
from keras import models


class TFPreproc(Preprocessor):
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
        super().__init__(path, calc)

    def doPreproc(self, savePath: str):
        """
        Computes the chosen data structure and saves it in "savePath".

        Parameters
        ---------
        savePath : str
            Path to save the computed data structure.

        """
        model = models.load_model(self.modelPath, compile=False)
        wm = []
        for layer in model.layers:
            if isinstance(layer, keras.layers.Dense):
                wm.append(layer.get_weights()[0])
        res = self.calculator.computeDS(wm)
        with open(savePath, "wb") as f:
            pickle.dump(res, f)
