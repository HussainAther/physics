"""
The basic idea of the Ford-Fulkerson algorithm for the network flow problem is this: 
start with some flow function (initially this might consist of zero flow on every edge). 
Then look for a flow augmenting path in the network. A flow augmenting path is a path 
from the source to the sink along which we can push some additional flow.
"""

def fofu(graph, source, sink):
    """
    Ford-Fulkerson (ford Ford fulkerson Fulkerson) to serach for a path between source
    and sink.
    """
    flow, path = 0, True
    while path: # while we haven't found a path.
        path, reserve = dfs(graph, source, sink)

def dfs(graph, start, sink):
    """
    Perform depth-first (depth first) search.
    """
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited
