import graph_algorithms.adjaceny_matrix_graph as matrix

g = matrix.AdjacencyMatrixGraph(4)

g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(2, 3)

g.display()