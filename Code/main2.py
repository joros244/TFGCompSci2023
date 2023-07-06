import pickle
import time
import logging
import os
from src.TFPreproc import TFPreproc
from src.NewDSC import NewDSC
from src.OriginalDSC import OriginalDSC
from src.V5P1.TFGSC_5P1 import TFGSC_5P1
from src.V5P1.WYSC_2P1 import WYSC_2P1
from src.V5P2.TFGSC_5P2 import TFGSC_5P2
from src.V5P2.WYSC_2P2 import WYSC_2P2
from src.PDiagramPlotter import PDiagramPlotter

ROOT_DIR = os.path.dirname(__file__)
logging.basicConfig(
    filename=ROOT_DIR + "/app.log",
    filemode="w",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)
METADATA_PATH = ROOT_DIR + "/data/metadata.csv"
SIMPLEX_PATH = ROOT_DIR + "/data/simplex.pkl"

if __name__ == "__main__":
    logging.info("Starting session...")

    pre = TFPreproc(
        ROOT_DIR
        + "/models/firstLayerSize300secondLayerSize100outputNeuron10class[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].h5",
        NewDSC(),
    )
    pre.doPreproc(ROOT_DIR + "/models/list/list_MNIST.pkl")

    pre = TFPreproc(
        ROOT_DIR
        + "/models/firstLayerSize300secondLayerSize100outputNeuron10class[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].h5",
        OriginalDSC(),
    )
    pre.doPreproc(ROOT_DIR + "/models/matrix/matrix_MNIST.pkl")

    logging.info("Preprocessing has finished")

    r = [
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.3,
        0.2,
        1.0e-1,
        0.9e-1,
        0.8e-1,
        0.7e-1,
        0.6e-1,
        0.5e-1,
        0.4e-1,
        0.3e-1,
        0.2e-1,
        1.0e-2,
        0.9e-2,
        0.8e-2,
        0.7e-2,
        0.6e-2,
        0.5e-2,
        0.4e-2,
        0.3e-2,
        0.2e-2,
        1.0e-3,
        0.9e-3,
        0.8e-3,
        0.7e-3,
        0.6e-3,
        0.5e-3,
        0.4e-3,
        0.3e-3,
        0.2e-3,
        1.0e-4,
        0.9e-4,
        0.8e-4,
        0.7e-4,
        0.6e-4,
        0.5e-4,
        0.4e-4,
        0.3e-4,
        0.2e-4,
        1.0e-5,
        0.9e-5,
        0.8e-5,
        0.7e-5,
        0.6e-5,
        0.5e-5,
        0.4e-5,
        0.3e-5,
        0.2e-5,
        1.0e-6,
        0.9e-6,
        0.8e-6,
        0.7e-6,
        0.6e-6,
        0.5e-6,
        0.4e-6,
        0.3e-6,
        0.2e-6,
        1.0e-7,
    ]

    with open(ROOT_DIR + "/models/list/list_MNIST.pkl", "rb") as f:
        l = pickle.load(f)
    with open(ROOT_DIR + "/models/matrix/matrix_MNIST.pkl", "rb") as f:
        mat = pickle.load(f)

    logging.info("Data loaded")
    logging.info("Computing simplices...")

    ti = time.time()
    # sc = TFGSC_5P1(l, r)
    sc = TFGSC_5P2(l, r)
    sc.compute_full(r, SIMPLEX_PATH)
    print(time.time() - ti)
    pd = PDiagramPlotter([SIMPLEX_PATH])
    pd.plotPD([ROOT_DIR + "/diagrams/MNIST300_TFGSC5_Example.png"])

    ti = time.time()
    # sc = WYSC_2P1(mat)
    sc = WYSC_2P2(mat)
    sc.compute_full(r, SIMPLEX_PATH)
    print(time.time() - ti)
    pd = PDiagramPlotter([SIMPLEX_PATH])
    pd.plotPD([ROOT_DIR + "/diagrams/MNIST300_WYSC2_Example.png"])

    logging.info("Computing has finished")
