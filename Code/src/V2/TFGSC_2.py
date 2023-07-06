from __future__ import annotations
from typing import List, Tuple
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


class TFGSC_2(SimplexCalculator):
    def __init__(self, M: List[List[int]]):
        """
        A TFGSC_2 is defined as a DAG with a weight matrix.
        """
        self.G = DAG(len(M))
        self.M = M
        for i in range(len(self.M)):
            for j in range(i):
                if self.M[i][j] > 0:
                    self.G.create_edge(i, j)

    def transitive_closure(self) -> TFGSC_2:
        """
        Computes the transitive closure of the TFGSC_2.

        Returns
        -------
        TFGSC_2
            The transitive closure of the TFGSC_2 defined by the
            transitive closure of the underlying DAG and the completed matrix.
        """
        T = copy.deepcopy(self)
        T.G = T.G.transitive_closure()
        return T

    def filtration(self, r: float) -> TFGSC_2:
        """
        Computes a new TFGSC_2 by deleting the entries of the matrix which
        are less than 'r'.

        Parameters
        ---------
        r : float
            Threshold value.

        Returns
        -------
        TFGSC_2
            Filtered TFGSC_2.
        """
        F = copy.deepcopy(self.M)
        for i in range(len(self.M)):
            for j in range(i):
                if self.M[i][j] < r:
                    F[i][j] = 0
        return TFGSC_2(F)

    def compute(self, t: float, save_path: str):
        """
        Computes the simplicial complex of the TFGSC_2 using t as a
        threshold,then it saves it in save_path.

        Parameters
        ---------
        t : float
            Threshold value.
        save_path : str
            Path to save the computed simplicial complex.
        """
        simplices = self.__compute_simplices(t)

        simplices.extend([[i] for i in range(len(self.M))])

        for s in simplices:
            for L in range(2, len(s) + 1):
                for sub in it.combinations(s, L):
                    if list(sub) not in simplices:
                        simplices.append(list(sub))

        with open(save_path, "wb") as f:
            pickle.dump([t, simplices], f)

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
        for thresh in t:
            simplices = self.__compute_simplices(thresh)
            simplices.extend([[i] for i in range(len(self.M))])

            for s in simplices:
                for L in range(2, len(s) + 1):
                    for sub in it.combinations(s, L):
                        if list(sub) not in simplices:
                            simplices.append(list(sub))
            res.append([t, simplices])

        with open(save_path, "wb") as f:
            pickle.dump(res, f)

    def search_weights(
        self,
        o: int,
        d: int,
        p: List[Tuple[float, List[int]]] = [],
        q: float = 1,
        c: List[int] = [],
    ):
        """
        It looks for the weight between 'o' and 'd' and it stores in p the
        path which gives such weight and the weight.

        Parameters
        ----------
        o: int
            Origin vertex.
        d: int
            Destination vertex.
        p: List[Tuple[float, List[int]]]
            Result of the search.
        q: float
            Threshold value for the search.
        c: List[int]
            Path which is currently being inspected.

        """
        if o == d:
            c.append(o)
            p.append((q, c))
            q = 1
            c = []
        else:
            for j in range(d, o):
                if self.M[o][j] != 0:
                    self.search_weights(j, d, p, q * self.M[o][j], c + [o])

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
        P = self.filtration(t)
        T = P.transitive_closure()
        for i in range(len(self.M)):
            for j in T.G.adj[i]:
                if self.M[i][j] == 0:
                    p = []
                    self.search_weights(i, j, p)
                    if len(p) > 0:
                        l = [k[1] for k in p if k[0] >= t]
                        simp.extend(l)
                elif self.M[i][j] >= t:
                    simp.append([i, j])
        return simp
