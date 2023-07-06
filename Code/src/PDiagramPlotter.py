import pickle
from typing import List

import dionysus as d
import gudhi as g
import matplotlib.pyplot as plt


class PDiagramPlotter:
    def __init__(self, simplex_path: List[str]):
        """
        PDiagramPlotter is defined by a collection of paths to simplex data.
        """
        self.__simp_path = simplex_path

    def plotPD(self, save_Path: List[str]):
        """
        It plots persistence diagrams of the simplicial complexes saved.

        Parameters
        ----------
        save_Path: List[str]
            Path to save the diagram including filename and extension (.png)
        """
        if len(save_Path) != len(self.__simp_path):
            raise Exception("Length mismatch")

        for file in self.__simp_path:
            s = []
            with open(file, "rb") as f:
                s = pickle.load(f)

            fil = d.Filtration()

            for e in s:
                for simp in e[1]:
                    fil.add(d.Simplex(simp, s.index(e)))

            fil.sort()
            m = d.homology_persistence(fil)
            dgms = d.init_diagrams(m, fil)

            list_pers = [[] for _ in range(len(dgms))]
            pers_num = [0 for _ in range(len(dgms))]

            for i, dgm in enumerate(dgms):
                for p in dgm:
                    birth = p.birth
                    death = p.death
                    list_pers[i].append([i, [birth, death]])
                    if death == float("inf"):
                        pers_num[i] += 1

            list_pers.sort(reverse=True)

            data = ""
            sumpers = 0
            for w in range(len(list_pers)):
                sumpers += len(list_pers[w])
                data += str(len(list_pers[w])) + ","

            data += str(sumpers) + ","

            for h in range(len(pers_num)):
                data += str(pers_num[h]) + ","

            with open(
                save_Path[self.__simp_path.index(file)].split(".")[0] + ".csv", "w"
            ) as f:
                f.write(data)

            ax1 = plt.axes()
            ax1.use_sticky_edges = False
            ax1.set_aspect("auto")
            list_pers = [item for sub in list_pers for item in sub]
            g.plot_persistence_diagram(
                list_pers, legend=True, axes=ax1, max_intervals=10000000
            )
            plt.title(
                "Diagrama de persistencia",
                fontdict={
                    "fontsize": 16,
                    "fontweight": "bold",
                    "color": "black",
                    "verticalalignment": "baseline",
                    "horizontalalignment": "center",
                },
            )
            plt.xlabel(
                "Nacimiento",
                fontdict={
                    "fontsize": 16,
                    "fontweight": "bold",
                    "color": "black",
                    "horizontalalignment": "center",
                },
            )
            plt.ylabel(
                "Muerte",
                fontdict={
                    "fontsize": 16,
                    "fontweight": "bold",
                    "color": "black",
                    "horizontalalignment": "center",
                },
            )
            plt.savefig(save_Path[self.__simp_path.index(file)])
            plt.clf()
            plt.close("all")
