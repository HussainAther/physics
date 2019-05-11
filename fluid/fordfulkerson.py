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
        flow += reserve
    # increase flow along the path
    for v, u in zip(path, path[1:]):
        if graph.has_edge(v, u):
            graph[v][u]["flow"] += reserve
        else:
            graph[u][v]["flow"] -= reserve

def dfs(graph, source, sink):
    """
    Perform depth-first (depth first) search and its
    corresponding flow reserve. 
    """
    undirected = graph.to_undirected()
    explored = {source}
    stack = [(source, 0, undirected[source])]
    while stack:
        v, _, neighbours = stack[-1]
        if v == sink:
            break
        # search the next neighbour
        while neighbours:
            u, e = neighbours.popitem()
            if u not in explored:
                break
        else:
            stack.pop()
            continue
        # current flow and capacity
        in_direction = graph.has_edge(v, u)
        capacity = e["capacity"]
        flow = e["flow"]
        # increase or redirect flow at the edge
        if in_direction and flow < capacity:
            stack.append((u, capacity - flow, undirected[u]))
            explored.add(u)
        elif not in_direction and flow:
            stack.append((u, flow, undirected[u]))
            explored.add(u)
    # (source, sink) path and its flow reserve
    reserve = min((f for _, f, _ in stack[1:]), default=0)
    path = [v for v, _, _ in stack]
    return path, reserve
