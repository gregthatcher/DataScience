import graph_library.adjacency_set_graph as adjacency_set


g = adjacency_set.AdjacencySetGraph(4)

print("Undirected Graph (Adjacency Set)")
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(2, 3)

g.display()

print("Directed Graph (Adjacency Set)")
g = adjacency_set.AdjacencySetGraph(4, True)

g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(2, 3)

g.display()
