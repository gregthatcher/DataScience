import graph_algorithms.find_shortest_path as find_shortest_path
import graph_algorithms.adjacency_set_graph as adjacency_graph
import graph_algorithms.adjaceny_matrix_graph as matrix_graph

print("Find shortest path for undirected, unweighted graph.")
g = adjacency_graph.AdjacencySetGraph(8, directed=False)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(2, 3)
g.add_edge(1, 4)
g.add_edge(3, 5)
g.add_edge(5, 4)
g.add_edge(3, 6)
g.add_edge(6, 7)
g.add_edge(0, 7)

print(find_shortest_path.shortest_path(g, 0, 5))
print(find_shortest_path.shortest_path(g, 0, 6))
print(find_shortest_path.shortest_path(g, 7, 4))
print()

print("Find shortest path for directed, unweighted graph.")
g = adjacency_graph.AdjacencySetGraph(8, directed=True)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(2, 3)
g.add_edge(1, 4)
g.add_edge(3, 5)
g.add_edge(5, 4)
g.add_edge(3, 6)
g.add_edge(6, 7)
g.add_edge(0, 7)

print(find_shortest_path.shortest_path(g, 0, 5))
print(find_shortest_path.shortest_path(g, 0, 6))
print(find_shortest_path.shortest_path(g, 7, 4))
