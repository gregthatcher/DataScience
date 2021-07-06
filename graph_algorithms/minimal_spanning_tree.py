'''
Find the minimum spanning tree for a connected, weighted, undirected graph.
That is, for a fully connected, weighted graph, find the minimum
number of edges needed to connect all nodes with a minimum weight of edges.
The first algorithm we use, Prim's algorithem, is very similar to Dijkstra's
Algorithm.  
Both Pims and Krushkal's algorithms are "greedy" algorithms, so it may not find
the "best" minimal spanning tree.
Note that Prim's algorithm does not work for disconnected graphs
(use Krushkal's algorithm for this case).
Like Dijkstra's algorithm, we use a distance table, but the
edge weight is the distance (instead of the total distance from source).
'''

# This is used for our "priority dictionary"
from pqdict import pqdict


def spanning_tree_pims(graph, source):
    # A dictionary mapping from vertex number to tuple of
    # (distance from last vertex, last vertex)
    # Note that for "shortest path", we use the full distance
    # from the source, whereas for "spanning tree", we use
    # the distance from the previous node
    distance_table = {}

    for i in range(graph.num_vertices):
        distance_table[i] = (None, None)

    distance_table[source] = (0, source)
        # We keep a queue of all nodes which still need processing,
    # but will use a priority queue, so that we can "pop"
    # nodes based on their minimum distance from source node
    # Keys are vertex numbers, values are distance from source
    # Highest priority (lowest distance) items can be accessed first
    priority_queue = pqdict()

    priority_queue[source] = 0

    visited_vertices = set()

    # Set of edges where each edge is represented as a string
    # e.g. "1->2" is an edge between vertices 1 and 2
    spanning_tree = set()

    while len(priority_queue.keys()) > 0:
        current_vertex = priority_queue.pop()

        # If we've visited a vertex already then we
        # have all outbound edges from it
        if current_vertex in visited_vertices:
            continue

        visited_vertices.add(current_vertex)

        # If the current vertex is the source,
        # then we don't have any edges to process (yet)
        if current_vertex != source:
            last_vertex = distance_table[current_vertex][1]   
            edge = str(last_vertex) + "->" + str(current_vertex)

            spanning_tree.add(edge)         

        for neighbor in graph.get_adjacent_vertices(current_vertex):
            distance = graph.get_edge_weight(current_vertex, neighbor)
            # In case we already have a distance for this neighbor
            # from another edge
            neighbor_distance = distance_table[neighbor][0]

            # If we haven't seen this before
            # or if old distance is bigger
            if neighbor_distance is None or neighbor_distance > distance:
                distance_table[neighbor] = (distance, current_vertex)
                priority_queue[neighbor] = distance
    
    return spanning_tree
