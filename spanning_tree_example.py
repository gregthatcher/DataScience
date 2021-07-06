'''
Ideas from: https://app.pluralsight.com/course-player?clipId=bf5077d9-deff-4529-b926-5e8ec9e9ddc5
'''


import graph_algorithms.adjaceny_matrix_graph as matrix
from graph_algorithms.minimal_spanning_tree import spanning_tree_pims

g = matrix.AdjacencyMatrixGraph(8, directed=False)

g.add_edge(0, 1, 1)
g.add_edge(1, 2, 2)
g.add_edge(1, 3, 2)
g.add_edge(2, 3, 2)
g.add_edge(1, 4, 3)
g.add_edge(3, 5, 1)
g.add_edge(5, 4, 3)
g.add_edge(3, 6, 1)
g.add_edge(6, 7, 1)
g.add_edge(7, 0, 1)

print("Spanning tree from node 1")
print(spanning_tree_pims(g, 1))
print()

print("Spanning tree from node 3")
print(spanning_tree_pims(g, 3))
print()