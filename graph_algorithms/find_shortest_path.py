'''
Use the Shortest Path algorithm to find the shortest path between
two nodes in an unweighted graph.  Graph can be directed or
undirected.
Use Dijkstra's algorithm for weighted graphs
Ideas from:
https://app.pluralsight.com/course-player?clipId=493787bc-6d05-4eba-bf84-710d07e3ddf6
'''

from queue import Queue


def _build_distance_table(graph, source):
    # Table contains a key for each vertex, and a
    # value which is a tuple containing the shortest distance
    # from the source node and the previous node
    # which is closest to the source
    distance_table = {}

    for i in range(graph.num_vertices):
        distance_table[i] = (None, None)

    # We know the distance from source to source is zero
    distance_table[source] = (0, source)

    # We keep a queue of all nodes which still need processing
    queue = Queue()
    # and we start with the source node
    queue.put(source)

    while not queue.empty():
        current_vertex = queue.get()

        current_distance = distance_table[current_vertex][0]

        for neighbor in graph.get_adjacent_vertices(current_vertex):
            # if we haven't processed this vertex previously
            if distance_table[neighbor][0] is None:
                distance_table[neighbor] = (current_distance+1, current_vertex)
                # if this new vertex has neighbors, we need to put it
                # in the queue for processing.
                if len(graph.get_adjacent_vertices(neighbor)) > 0:
                    queue.put(neighbor)

    return distance_table


def shortest_path(graph, source, destination):
    # Note that each source has a different distance table
    distance_table = _build_distance_table(graph, source)

    # We'll use a list for the queue
    # This uses contiguous memory (which can be bad)
    # but is faster than collections.deqeue
    # (pronounced "deck")
    path = [destination]
    previous_vertex = distance_table[destination][1]

    while previous_vertex is not None and previous_vertex is not source:
        #path = [previous_vertex] + path
        path.insert(0, previous_vertex)
        previous_vertex = distance_table[previous_vertex][1]
        
    if previous_vertex is None:
        print(f"There is no path from {source} to {destination}")
        return []
    else:
        path.insert(0, source)

    return path
