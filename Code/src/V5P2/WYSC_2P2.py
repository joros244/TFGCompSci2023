import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import pickle
from typing import List
import copy
from SimplexCalculator import SimplexCalculator
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


class WYSC_2P2(SimplexCalculator):
    def __init__(self, mat: List[List[int]]):
        """
        WYSC_2P2 is defined by a weight matrix.
        """
        self.__matrix = mat

    @staticmethod
    def __comb(seq: List[int]) -> List[int]:
        """
        It computes the power set of the sequence 'seq'.

        Parameters
        ----------
        seq : List[int]
            Sequence to compute its power set.

        Returns
        -------
        List[int]
            The power set of 'seq'.

        """
        res = []
        for L in range(1, len(seq) + 1):
            res.extend(list(map(list, itertools.combinations(seq, L))))
        return res

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
        simplicial_complex = []
        with ThreadPool(cpu_count() - 1) as p:
            parallel_list = p.map(
                lambda x: self.__get_simplex([x], t), range(len(self.__matrix))
            )
        for e in parallel_list:
            simplicial_complex.extend(e)

        with open(save_path, "wb") as f:
            pickle.dump([t, simplicial_complex], f)

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
        result = []
        for thresh in t:
            simplicial_complex = []
            with ThreadPool(cpu_count() - 1) as p:
                parallel_list = p.map(
                    lambda x: self.__get_simplex([x], thresh), range(len(self.__matrix))
                )
            for e in parallel_list:
                simplicial_complex.extend(e)
            result.append([t, simplicial_complex])

        with open(save_path, "wb") as f:
            pickle.dump(result, f)

    def __get_simplex(self, current_path: List[int], t: float) -> List[List[int]]:
        """
        Computes a simplicial complex on a DAG (represented by 'self.__matrix')
        using 't' as threshold value and the last vertex of 'current_path' as
        the starting vertex.

        Parameters
        ---------
        current_path : List[int]
            List of visited vertices.
        t : float
            Threshold value.

        Returns
        -------
        List[int]
            Simplicial complex of the DAG whose first vertex is current_path[-1]

        """
        rel = 1.0
        result = []
        origin = current_path[0]
        for destination in current_path:
            rel = rel * self.__matrix[origin][destination]
            origin = destination
        if rel >= t:
            for e in self.__comb(current_path):
                result.append(e)
            last_vert = current_path[-1]
            for i in range(last_vert):
                if self.__matrix[last_vert][i] > 0:
                    copy_path = copy.deepcopy(current_path)
                    copy_path.append(i)
                    k = self.__get_simplex(copy_path, t)
                    for e in k:
                        result.append(e)
        return list(map(list, set(map(tuple, result))))
