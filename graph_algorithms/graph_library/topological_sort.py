'''
Topological Sorting is used when we wish to find precedence relationships
This is good for:
1.) Scheduling Tasks (which tasks need to be done first)
2.) Evalulating (Mathematical) Expressions (which sub-expressions
        need to be evaluated first.
3.) Building neural network models (which nodes come first in 
        feed forward networks)
Topological Sorts are typically used on Directed Acyclic Graphs (DAGs)
"Any ordering of all nodes that satisfies all relationships is a 
topological sort" -> 
https://app.pluralsight.com/course-player?clipId=0ea336bd-dd7b-4e76-925b-07e89280917c
"A Topilogical Sort is any ordering of all the graph's vertices that 
satisfies all precedence relationships" ->
https://app.pluralsight.com/course-player?clipId=f01c97a5-2e43-4d44-b24a-29c0cb207948
'''

from queue import Queue


def topological_sort(graph):
    queue = Queue()

    indegree_map = {}

    for i in range(graph.num_vertices):
        indegree_map[i] = graph.get_indegree(i)

        # If we find a node with no dependencies,
        # (no edges coming in), then we queue it
        if indegree_map[i] == 0:
            queue.put(i)

    sorted_list = []
    while not queue.empty():
        vertex = queue.get()
        sorted_list.append(vertex)

        # Now, we simulate removing that node
        # by decrementing indegree counts
        # from all adjacent nodes
        for v in graph.get_adjacent_vertices(vertex):
            indegree_map[v] -= 1
            if indegree_map[v] == 0:
                queue.put(v)

    if len(sorted_list) != graph.num_vertices:
        raise ValueError("This graph has a cycle.")

    return sorted_list



        
