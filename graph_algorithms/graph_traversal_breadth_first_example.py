'''
Ideas from: https://app.pluralsight.com/course-player?clipId=894c22e6-04aa-4002-9cc8-48201a9e92b5
'''

from queue import Queue
import numpy as np

import graph_library.adjaceny_matrix_graph as matrix_graph
import graph_library.adjacency_set_graph as set_graph
from graph_library.graph_traversal import breadth_first


g = matrix_graph.AdjacencyMatrixGraph(9)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 7)
g.add_edge(2, 4)
g.add_edge(2, 3)
g.add_edge(1, 5)
g.add_edge(5, 6)
g.add_edge(6, 3)
g.add_edge(3, 4)
g.add_edge(6, 8)

print("Traverse Adjacency Matrix graph from node 2 Breadth First:")
visits = breadth_first(g, 2)
print(visits)
print("Traverse Adjacency Matrix Graph from node 0 Breadth First:")
visits = breadth_first(g, 0)
print(visits)

g = set_graph.AdjacencySetGraph(9)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 7)
g.add_edge(2, 4)
g.add_edge(2, 3)
g.add_edge(1, 5)
g.add_edge(5, 6)
g.add_edge(6, 3)
g.add_edge(3, 4)
g.add_edge(6, 8)

print("Traverse Adjacency Set graph from node 2 Breadth First:")
visits = breadth_first(g, 2)
print(visits)
print("Traverse Adjancey Set Graph from node 0 Breadth First:")
visits = breadth_first(g, 0)
print(visits)
