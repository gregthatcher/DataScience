'''
An abstract-like class for graphing algorithms
'''


class GraphBase():

    def __init__(self, num_vertices, directed=False):
        self.num_vertices = num_vertices
        self.directed = directed

    # Abstract methods
    def add_edge(self, v1, v2, weight):
        raise NotImplementedError("add_edge not implemented!")

    def get_adjacent_vertices(self, v):
        raise NotImplementedError("get_adjacent_vertices not implemented!")

    # How many edges are directed towards this vertex
    def get_indegree(self, v):
        raise NotImplementedError("get_indegree not implemented!")

    def get_edge_weight(self, v1, v2):
        raise NotImplementedError("get_edge_weight not implemented!")

    # Used for debugging
    def display(self):
        raise NotImplementedError("display not implemented!")
