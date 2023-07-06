import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools as it
import pickle
from typing import List, Dict
from SimplexCalculator import SimplexCalculator


class TFGSC_4(SimplexCalculator):
    def __init__(self, d: Dict[int, Dict[int, float]]):
        """
        TFGSC_4 is defined by a collection of dicts.
        """
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

    def compute(self, t: float, save_path: str):
        """
        Computes the simplicial complex of the TFGSC_4 using t as a
        threshold,then it saves it in save_path.

        Parameters
        ---------
        t : float
            Threshold value.
        save_path : str
            Path to save the computed simplicial complex.
        """
        simplices = self.__compute_simplices(t)
        res = []

        for s in simplices:
            for L in range(2, len(s) + 1):
                res.extend(it.combinations(s, L))

        res.extend(map(tuple, [[i] for i in range(len(self.__data))]))

        with open(save_path, "wb") as f:
            pickle.dump([t, list(map(list, set(res)))], f)

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
            simplices = self.__compute_simplices(thresh)
            pres = []

            for s in simplices:
                for L in range(2, len(s) + 1):
                    pres.extend(list(it.combinations(s, L)))

            pres.extend(map(tuple, [[i] for i in range(len(self.__data))]))

            result.append([t, list(map(list, set(pres)))])

        with open(save_path, "wb") as f:
            pickle.dump(result, f)

    def __search_simplex(
        self,
        v: Dict[int, float],
        t: float,
        c: List[int],
        r: List[List[int]],
        q: float = 1,
    ):
        """
        Performs a DFS over TFGSC_4's DAG to find the maximal simplices from the
        vertex v.

        Parameters
        ---------
        v: Dict[int, float]
            Adjacency dict of vertex 'v'.
        t: float
            Threshold value.
        c: List[int]
            Current path.
        r: List[List[int]]
            Result list.
        q: float
            Current threshold.

        """
        lim = t / q
        b = False
        for vertex, thresh in v.items():
            if thresh >= lim:
                b = True
                self.__search_simplex(
                    self.__data[vertex], t, c + [vertex], r, q * thresh
                )
            else:
                break
        if len(c) > 1 and not b:
            r.append(c)

    def __compute_simplices(self, t: float) -> List[List[int]]:
        """
        It returns the incomplete simplicial complex of the graph using t as a
        threshold.

        Parameters
        ---------
        t : float
            Threshold value.

        Returns
        -------
        List[List[int]]
            The incomplete simplicial complex.
        """
        res = []
        for i in range(len(self.__data)):
            ilist = []
            self.__search_simplex(self.__data[i], t, [i], ilist)
            res.extend(ilist)
        return res
