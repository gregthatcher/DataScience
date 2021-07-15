'''
Find shortest path of a _weighted_ graph.
(Use "shortest path" Algorithm for unweighted graphs.)
Ideas from:
https://app.pluralsight.com/course-player?clipId=89220a1d-cbb8-4b0f-86c6-163dc0c9d8ac
'''

from graph_library.find_shortest_path import shortest_path_dijkstra
import graph_library.adjaceny_matrix_graph as matrix_graph

print("Find shortest path for _undirected_, weighted, adjacency matrix graph.")
# We have to use matrix graph, as adjacency graph can't (currently?)
# handle weights
g = matrix_graph.AdjacencyMatrixGraph(8, directed=False)
g.add_edge(0, 1, 1)
g.add_edge(1, 2, 2)
g.add_edge(1, 3, 6)
g.add_edge(2, 3, 2)
g.add_edge(1, 4, 3)
g.add_edge(3, 5, 1)
g.add_edge(5, 4, 5)
g.add_edge(3, 6, 1)
g.add_edge(6, 7, 1)
g.add_edge(0, 7, 8)

print(shortest_path_dijkstra(g, 0, 6))
print(shortest_path_dijkstra(g, 4, 7))
print(shortest_path_dijkstra(g, 7, 0))
print()

print("Find shortest path for _directed_, weighted, adjacency matrix graph.")
# We have to use matrix graph, as adjacency graph can't (currently?)
# handle weights
g = matrix_graph.AdjacencyMatrixGraph(8, directed=True)
g.add_edge(0, 1, 1)
g.add_edge(1, 2, 2)
g.add_edge(1, 3, 6)
g.add_edge(2, 3, 2)
g.add_edge(1, 4, 3)
g.add_edge(3, 5, 1)
g.add_edge(5, 4, 5)
g.add_edge(3, 6, 1)
g.add_edge(6, 7, 1)
g.add_edge(0, 7, 8)

print(shortest_path_dijkstra(g, 0, 6))
print(shortest_path_dijkstra(g, 4, 7))
print(shortest_path_dijkstra(g, 7, 0))
print()
