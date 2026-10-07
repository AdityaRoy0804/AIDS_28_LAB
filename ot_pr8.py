INF = float("inf")

# Weighted directed graph (adjacency matrix), vertices 1 to 5
graph = [
    [0,   3,   8, INF,   7],
    [INF, 0, INF,   1, INF],
    [INF, 4,   0, INF, INF],
    [2, INF,   5,   0, INF],
    [INF, INF, INF,  6,  0],
]
n = len(graph)

def show(matrix, title):
    print(title)
    print("      " + "".join(f"{j+1:>5}" for j in range(n)))
    for i in range(n):
        row = ""
        for v in matrix[i]:
            row += f"{('INF' if v == INF else int(v)):>5}"
        print(f"{i+1:>4}  {row}")
    print()

# Distance matrix and next-vertex matrix (for path reconstruction)
dist = [row[:] for row in graph]
nxt = [[j if graph[i][j] != INF and i != j else None
        for j in range(n)] for i in range(n)]

show(dist, "Initial Distance Matrix:")

# Floyd-Warshall algorithm (k = intermediate vertex)
for k in range(n):
    for i in range(n):
        for j in range(n):
            if dist[i][k] + dist[k][j] < dist[i][j]:
                dist[i][j] = dist[i][k] + dist[k][j]
                nxt[i][j] = nxt[i][k]

show(dist, "Final Shortest Distance Matrix:")

# Negative cycle check
if any(dist[i][i] < 0 for i in range(n)):
    print("Graph contains a negative cycle")

# Path reconstruction
def get_path(u, v):
    if nxt[u][v] is None:
        return None
    path = [u + 1]
    while u != v:
        u = nxt[u][v]
        path.append(u + 1)
    return path

print("Shortest Paths:")
for i in range(n):
    for j in range(n):
        if i != j:
            p = get_path(i, j)
            if p is None:
                print(f"{i+1} -> {j+1}: no path")
            else:
                route = " -> ".join(map(str, p))
                print(f"{i+1} -> {j+1}: cost = {int(dist[i][j])}, "
                      f"path = {route}")
