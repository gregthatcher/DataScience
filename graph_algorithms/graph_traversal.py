import numpy as np
from queue import Queue


def breadth_first(graph, start=0):
    queue = Queue()
    queue.put(start)

    visits_in_order = []

    visited = np.zeros(graph.num_vertices)

    while not queue.empty():
        vertex = queue.get()

        if visited[vertex] == 1:
            continue

        # print("Visited:", vertex)
        visits_in_order.append(vertex)
        visited[vertex] = 1

        for v in graph.get_adjacent_vertices(vertex):
            if (visited[v] != 1):
                queue.put(v)

    return visits_in_order
