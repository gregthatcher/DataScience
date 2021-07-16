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
        print("\nEdges:")
        for i in range(self.num_vertices):
            for v in self.get_adjacent_vertices(i):
                print(i, "-->", v)

        print("\nAdjacent Vertices:")
        for i in range(self.num_vertices):
            print("Adjacent to: ", i, self.get_adjacent_vertices(i))

        print("\nIndegrees:")
        for i in range(self.num_vertices):
            print("Indegree :", i, self.get_indegree(i))

        print("\nWeights:")
        for i in range(self.num_vertices):
            for j in self.get_adjacent_vertices(i):
                print("Edge Weight: ", i, " ", j,
                      " weight: ", self.get_edge_weight(i, j))

    def _check_two_vertices(self, v1, v2):
        self._check_one_vertex(v1)
        self._check_one_vertex(v2)

    def _check_one_vertex(self, v):
        if (v < 0 or v >= self.num_vertices):
            raise ValueError("Vertex {v} does not exist!")

    def _check_weight(self, weight):
        if weight < 1:
            raise ValueError("Weight must be >= 1.")

