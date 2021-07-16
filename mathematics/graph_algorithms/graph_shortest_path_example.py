'''
Find shortest path of an _unweighted_ graph.
(Use Dijkstra's Algorithm for weighted graphs.)
Ideas from:
https://app.pluralsight.com/course-player?clipId=493787bc-6d05-4eba-bf84-710d07e3ddf6
TODO: Show image of graph, see
https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python
'''

import graph_library.find_shortest_path as find_shortest_path
import graph_library.adjacency_set_graph as adjacency_graph
import graph_library.adjaceny_matrix_graph as matrix_graph

print("Find shortest path for undirected, unweighted, adjacency set graph.")
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

print("Find shortest path for directed, unweighted, adjacency set graph.")
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

print("Find shortest path for undirected, unweighted, adjacency matrix graph.")
g = matrix_graph.AdjacencyMatrixGraph(8, directed=False)
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

print("Find shortest path for directed, unweighted, adjacency matrix graph.")
g = matrix_graph.AdjacencyMatrixGraph(8, directed=True)
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
