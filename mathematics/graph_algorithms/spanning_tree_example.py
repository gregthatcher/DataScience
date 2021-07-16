'''
Ideas from: https://app.pluralsight.com/course-player?clipId=bf5077d9-deff-4529-b926-5e8ec9e9ddc5
'''


import graph_library.adjaceny_matrix_graph as matrix
from graph_library.minimal_spanning_tree import spanning_tree_pims, \
    spanning_tree_krushal, print_krushal_spanning_tree

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

print("Spanning tree from node 1 using Pim's Algorithm:")
print(spanning_tree_pims(g, 1))
print()

print("Spanning tree from node 3 using Pim's Algorithm:")
print(spanning_tree_pims(g, 3))
print()

print("Spanning tree using Krushal's Algorithm:")
print_krushal_spanning_tree(spanning_tree_krushal(g))
