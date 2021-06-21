from numpy.core.fromnumeric import sort
import graph_algorithms.topological_sort as t_sort
import graph_algorithms.adjaceny_matrix_graph as graph

# Note that this is a "DAG"
g = graph.AdjacencyMatrixGraph(9, directed=True)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 7)
g.add_edge(2, 4)
g.add_edge(2, 3)
g.add_edge(1, 5)
g.add_edge(5, 6)
g.add_edge(3, 6)
g.add_edge(3, 4)
g.add_edge(6, 8)

sorted_list = t_sort.topological_sort(g)

print(sorted_list)