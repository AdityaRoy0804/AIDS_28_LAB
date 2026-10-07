import sympy as sp
import itertools

# define variables
x, y = sp.symbols("x y", real=True)
mu1, mu2 = sp.symbols("mu1 mu2", real=True)

# Objective function (minimize)
f = (x - 3)**2 + (y - 2)**2

# Inequality constraints in the form g(x, y) <= 0
g1 = x + y - 4      # x + y <= 4
g2 = -x             # x >= 0
g = [g1, g2]
mu = [mu1, mu2]

# Lagrangian function
L = f + mu1*g1 + mu2*g2
print("Lagrangian Function:")
print(L)

# Stationarity conditions
dLdx = sp.diff(L, x)
dLdy = sp.diff(L, y)
print("\nStationarity Conditions:")
print("dL/dx =", dLdx)
print("dL/dy =", dLdy)

# Check every combination of active / inactive constraints
print("\nCase Analysis:")
best = None
cases = itertools.product([0, 1], repeat=2)
for case_no, active in enumerate(cases, start=1):
    eqs = [dLdx, dLdy]
    for i in range(2):
        if active[i]:
            eqs.append(g[i])      # active: g_i = 0
        else:
            eqs.append(mu[i])     # inactive: mu_i = 0
    sol = sp.solve(eqs, [x, y, mu1, mu2], dict=True)
    print(f"\nCase {case_no}: g1 active = {bool(active[0])}, "
          f"g2 active = {bool(active[1])}")
    if not sol:
        print("  No solution")
        continue
    s = sol[0]
    xv, yv = s[x], s[y]
    m1, m2 = s.get(mu1, 0), s.get(mu2, 0)
    feasible = all(gi.subs({x: xv, y: yv}) <= 0 for gi in g)
    dual_ok = (m1 >= 0) and (m2 >= 0)
    print(f"  x = {xv}, y = {yv}, mu1 = {m1}, mu2 = {m2}")
    print(f"  Primal feasible: {feasible}, "
          f"Dual feasible (mu >= 0): {dual_ok}")
    if feasible and dual_ok:
        print("  --> KKT conditions satisfied")
        best = (xv, yv, m1, m2)

# Optimal solution
xv, yv, m1, m2 = best
print("\nOptimal Solution:")
print("x =", xv)
print("y =", yv)
print("mu1 =", m1)
print("mu2 =", m2)
print("\nMinimum Value of Objective Function:")
print(f.subs({x: xv, y: yv}))
