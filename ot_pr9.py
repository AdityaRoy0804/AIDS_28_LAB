import math
from scipy.optimize import linprog

# Maximize Z = 5*x1 + 4*x2
# subject to  x1 + x2 <= 5
#             10*x1 + 6*x2 <= 45
#             x1, x2 >= 0 and integer
c = [-5, -4]          # negated because linprog minimizes
A = [[1, 1], [10, 6]]
b = [5, 45]

best_value = -math.inf
best_sol = None
node_count = 0

def branch_and_bound(bounds, parent, cond):
    global best_value, best_sol, node_count
    node_count += 1
    node = node_count
    if parent is None:
        label = f"Node {node} (root LP)"
    else:
        label = f"Node {node} (from Node {parent}, {cond})"
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        print(f"{label}: infeasible -> pruned")
        return
    z = -res.fun
    x1, x2 = res.x
    print(f"{label}: x1 = {x1:.3f}, x2 = {x2:.3f}, Z = {z:.3f}")
    if z <= best_value + 1e-9:
        print("   Bound <= best integer value -> pruned")
        return
    # Find a fractional variable
    frac = [i for i, v in enumerate(res.x)
            if abs(v - round(v)) > 1e-6]
    if not frac:
        print("   Integer solution found -> new best")
        best_value, best_sol = z, [round(v) for v in res.x]
        return
    i = frac[0]
    v = res.x[i]
    name = f"x{i+1}"
    lo, hi = bounds[i]
    # Left branch: xi <= floor(v)
    left = list(bounds); left[i] = (lo, math.floor(v))
    branch_and_bound(left, node, f"{name} <= {math.floor(v)}")
    # Right branch: xi >= ceil(v)
    right = list(bounds); right[i] = (math.ceil(v), hi)
    branch_and_bound(right, node, f"{name} >= {math.ceil(v)}")

print("Branch and Bound Tree:")
branch_and_bound([(0, None), (0, None)], None, "")

print("\nTotal LP problems solved:", node_count)
print("Optimal Integer Solution: x1 = %d, x2 = %d" % tuple(best_sol))
print("Maximum Value of Z:", round(best_value))
