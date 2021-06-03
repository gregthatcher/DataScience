'''
Adjacency Sets are an alternative to Adjaceny Matrices
Note that we can also make an "Adjaceny List", but
lists aren't as effective as sets since sets
can delete and search items faster.

This type of representation is good for large, sparsely 
connected graphs because it saves space
'''


from . import graph_base


# A single node in a graph represented by an adjacency set
# Every node has a vertex id, and is associated with a set
# of adjacent vertices
class Node:
    def __init__(self, vertex_id):
        self.vertex_id = vertex_id
        self.adjacency_set = set()

    def add_edge(self, v):
        if self.vertex_id == v:
            raise ValueError(f"The vertex {v} cannot be adjacent to itself.")
        self.adjacency_set.add(v)

    def get_adjacency_set(self):
        return sorted(self.adjacency_set)


class AdjacencySetGraph(graph_base.GraphBase):
    def __init__(self, num_vertices, directed=False):
        super().__init__(num_vertices, directed)

        self.vertex_list = []
        for i in range(num_vertices):
            self.vertex_list.append(Node(i))
    
    def add_edge(self, v1, v2, weight=1):
        super()._check_two_vertices(v1, v2)
        super()._check_weight(weight)

        if weight != 1:
            raise ValueError("An adjacency set cannot (currently) handle edge weights > 1")
        
        self.vertex_list[v1].add_edge(v2)

    def get_adjacent_vertices(self, v):
        super()._check_one_vertex(v)

        return self.vertex_list[v].get_adjacency_set()

    def get_indegree(self, v):
        super()._check_one_vertex(v)

        indegree = 0
        for i in range(self.num_vertices):
            if v in self.get_adjacent_vertices(i):
                indegree += 1

        return indegree

    def get_edge_weight(self, v1, v2):
        # We currently only support unweighted graphs
        return 1

    def display(self):
        print("\nSets:")
        for i in range(self.num_vertices):
            for j in self.get_adjacent_vertices(i):
                print(i, "-->", j)
        print()

        super().display()
        
