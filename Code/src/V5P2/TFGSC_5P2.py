import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools as it
import pickle
from typing import List, Dict
from SimplexCalculator import SimplexCalculator
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


class TFGSC_5P2(SimplexCalculator):
    def __init__(self, d: Dict[int, Dict[int, float]], t: List[float]):
        """
        TFGSC_5P2 is defined by a list of threshold values, the dict which
        contains adjacency data and a list of simplices.
        """
        self.__thresh = t
        self.__data = dict(
            map(
                lambda e: (
                    e[0],
                    dict(sorted(e[1].items(), reverse=True, key=lambda c: c[1])),
                )
                if e[1]
                else (e[0], {}),
                d.items(),
            )
        )
        self.__simp = []
        self.__compute_simplices()
        self.__simp.sort(reverse=True, key=lambda simpl: simpl[1])

    def compute(self, t: float, save_path: str):
        """
        Computes the simplicial complex of the TFGSC_5P2 using t as a
        threshold,then it saves it in save_path.

        Parameters
        ---------
        t : float
            Threshold value.
        save_path : str
            Path to save the computed simplicial complex.
        """
        res = []

        for s in self.__simp:
            if s[1] < t:
                break
            for k in range(2, len(s[0]) + 1):
                res.extend(it.combinations(s[0], k))

        res = list(map(list, set(res)))
        res.extend([[i] for i in range(len(self.__data))])

        with open(save_path, "wb") as f:
            pickle.dump([t, res], f)

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
        res = []
        for tr in t:
            pres = []

            for s in self.__simp:
                if s[1] < tr:
                    break
                for k in range(2, len(s[0]) + 1):
                    pres.extend(it.combinations(s[0], k))

            pres = list(map(list, set(pres)))
            pres.extend([[i] for i in range(len(self.__data))])

            res.append([t, pres])

        with open(save_path, "wb") as f:
            pickle.dump(res, f)

    def __search_simplex(self, i: Dict[int, float], c: List[int] = [], q: float = 1):
        """
        Performs a DFS over TFGSC_5P2's DAG to find the maximal simplices from the
        vertex i.

        Parameters
        ---------
        i: Dict[int, float]
            Adjacency dict of vertex 'i'.
        c: List[int]
            Current path.
        q: float
            Current threshold.

        """
        lim = self.__thresh[-1] / q
        for vertex, t in i.items():
            if t >= lim:
                self.__search_simplex(self.__data[vertex], c + [vertex], q * t)
            else:
                break
        if len(c) > 1:
            self.__simp.append([c, q])

    def __compute_simplices(self):
        """
        It stores the incomplete simplicial complex of the graph in 'self.__simp'. It uses the last value of the
        thresholds' list as the threshold value.
        """
        with ThreadPool(cpu_count() - 1) as p:
            p.map(
                lambda x: self.__search_simplex(self.__data[x], [x]),
                list(self.__data.keys()),
            )
