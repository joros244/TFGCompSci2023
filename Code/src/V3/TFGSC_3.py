from __future__ import annotations
from typing import List
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import copy
from SimplexCalculator import SimplexCalculator

import itertools as it


class DAG:
    def __init__(self, V: int):
        """
        A directed acyclic graph (DAG) is defined by its vertex number and an
        adjacency list.
        """
        self.__V = V
        self.adj = []
        for _ in range(0, V):
            self.adj.append([])

    def __is_vertex(self, v: int) -> bool:
        """
        Checks if 'v' is a vertex of the DAG.

        Parameters
        ----------
        v : int
            Vertex to check.

        Returns
        -------
        bool
            Returns true if 'v' is in the DAG, otherwise it returns false.

        """
        return v >= 0 and v <= self.__V

    def create_edge(self, o: int, d: int):
        """
        Creates an edge between the vertices 'o' and 'd'.

        Parameters
        ----------
        o : int
            Origin vertex.
        d : int
            Destination vertex.
        """
        if self.__is_vertex(o) and self.__is_vertex(d) and o > d:
            self.adj[o].append(d)
            self.adj[o].sort(reverse=True)

    def transitive_closure(self) -> DAG:
        """
        Computes the transitive closure of the DAG.

        Returns
        -------
        DAG
            The transitive closure.
        """
        C = DAG(self.__V)

        for i in range(0, self.__V):
            reach = []
            self.__reachable_vertices(i, reach)

            for j in reach:
                C.create_edge(i, j)

        return C

    def __reachable_vertices(self, o: int, res: List[int] = []):
        """
        Computes the reachable vertices from 'o' and stores them in 'res'.

        Parameters
        ----------
        o : int
            Origin vertex.
        res : List[int]
            List of reachable vertices from 'o'.
        """
        res.append(o)

        for v in self.adj[o]:
            if v not in res:
                self.__reachable_vertices(v, res)


class TFGSC_3(SimplexCalculator):
    def __init__(self, M: List[List[int]]):
        """
        A TFGSC_3 is defined as a DAG with a weight matrix.
        """
        self.G = DAG(len(M))
        self.M = M
        for i in range(len(self.M)):
            for j in range(i):
                if self.M[i][j] > 0:
                    self.G.create_edge(i, j)

    def compute(self, t: float, save_path: str):
        """
        Computes the simplicial complex of the TFGSC_3 using t as a
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
                res.extend(list(it.combinations(s, L)))

        res.extend(map(tuple, [[i] for i in range(len(self.M))]))

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

            pres.extend(map(tuple, [[i] for i in range(len(self.M))]))

            result.append([t, list(map(list, set(pres)))])

        with open(save_path, "wb") as f:
            pickle.dump(result, f)

    def __search_simplex(self, t: float, c: List[int], q: float, l: List[List[int]]):
        """
        Performs a DFS over TFGSC_3's DAG to find the maximal simplices from the
        vertex 'c[-1]'.

        Parameters
        ----------
        t: float
            Threshold value.
        c: List[int]
            Current path.
        q: float
            Current threshold value.
        l: List[List[int]]
            Result list.
        """
        a = c[-1]
        for b in self.G.adj[a]:
            if q * self.M[a][b] >= t:
                self.__search_simplex(t, c + [b], q * self.M[a][b], l)
        l.append(c)

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
        simp = []
        for i in range(len(self.M)):
            for j in self.G.adj[i]:
                if self.M[i][j] >= t:
                    c = [i, j]
                    l = []
                    self.__search_simplex(t, c, self.M[i][j], l)
                    simp.extend(l)
        return simp
