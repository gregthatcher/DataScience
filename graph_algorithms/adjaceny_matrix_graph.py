'''
Represent a graph using an Adjacency matrix
Ideas from https://app.pluralsight.com/course-player?clipId=70897c13-9ed7-4067-a56e-06ba83dfe681

If a row and column has a nonzero entry, then there is an edge from that
row index to that column index

Bidirectional graphs have a symmetric matrix
'''

import numpy as np
from . import graph_base


class AdjacencyMatrixGraph(graph_base.GraphBase):

    def __init__(self, num_vertices, directed=False):
        super().__init__(num_vertices, directed)

        self.matrix = np.zeros((num_vertices, num_vertices))

    # Add edge from v1 _to_ v2 with weight;  if its an undirected
    # graph, also add the mirror (v2 _to_ v1) edge
    def add_edge(self, v1, v2, weight=1):
        self._check_two_vertices(v1, v2)
        self._check_weight(weight)

        self.matrix[v1][v2] = weight

        # If it's _not_ a directed graph, then we
        # need to make mirror image weight
        if self.directed is False:
            self.matrix[v2][v1] = weight

    # Find all nodes which "flow" into v
    # For undirected graphs, there is no "flow" direction
    # Since undirected graphs have a symmetri matrix,
    # we can just count the "incoming"
    # For a directed graph, matrix[i][j] > 0 indicates
    # that the graph has an edge _from_ i _to_ j
    # That is, "adjacent" means "child"
    def get_adjacent_vertices(self, v):
        self._check_one_vertex(v)

        adjacent_vertices = []
        for i in range(self.num_vertices):
            if self.matrix[v][i] > 0:
                adjacent_vertices.append(i)

        return adjacent_vertices

    # Count all nodes which "flow" into v
    def get_indegree(self, v):
        self._check_one_vertex(v)
        indegree = 0
        for i in range(self.num_vertices):
            if (self.matrix[i, v] > 0):
                indegree += 1
        return indegree

    def get_edge_weight(self, v1, v2):
        self._check_two_vertices(v1, v2)
        return self.matrix[v1, v2]

    def display(self):
        print("\nMatrix:")
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                print(self.matrix[i, j], " ", sep=" ", end="")
            print()

        print("\nEdges:")
        for i in range(self.num_vertices):
            for v in self.get_adjacent_vertices(i):
                print(i, "-->", v)

        print("\nAdjacent Vertices:")
        for i in range(self.num_vertices):
            print("Adjacent to: ", i, self.get_adjacent_vertices(i))

        print("\nIndegrees:")
        for i in range(self.num_vertices):
            print("Indegree :", i, self.get_indegree(i))

        print("\nWeights:")
        for i in range(self.num_vertices):
            for j in self.get_adjacent_vertices(i):
                print("Edge Weight: ", i, " ", j,
                      " weight: ", self.get_edge_weight(i, j))

    def _check_two_vertices(self, v1, v2):
        self._check_one_vertex(v1)
        self._check_one_vertex(v2)

    def _check_one_vertex(self, v):
        if (v < 0 or v >= self.num_vertices):
            raise ValueError("Vertex {v} does not exist!")

    def _check_weight(self, weight):
        if weight < 1:
            raise ValueError("Weight must be >= 1.")


if __name__ == "__main__":
    g = AdjacencyMatrixGraph(4)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(2, 3)

    g.display()
