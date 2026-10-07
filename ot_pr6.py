import numpy as np
import sympy as sp

# define variables
x, y = sp.symbols("x y")

# Objective function
f = x**4 + 2*y**2 - 4*x

# Gradient and Hessian
grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
hess = sp.hessian(f, (x, y))
print("Objective Function:", f)
print("Gradient:", list(grad))
print("Hessian:", hess.tolist())

# Convert to numerical functions
f_num = sp.lambdify((x, y), f, "numpy")
g_num = sp.lambdify((x, y), grad, "numpy")
h_num = sp.lambdify((x, y), hess, "numpy")

# Newton's method
X = np.array([2.0, 1.0])     # initial point
tol = 1e-6
max_iter = 50

print("\nIter        x          y          f(x,y)      ||grad||")
for k in range(max_iter):
    g = np.array(g_num(*X), dtype=float).flatten()
    H = np.array(h_num(*X), dtype=float)
    gn = np.linalg.norm(g)
    print(f"{k:3d}   {X[0]:10.6f} {X[1]:10.6f} "
          f"{f_num(*X):12.6f} {gn:12.6f}")
    if np.linalg.norm(g) < tol:
        print(f"\nConverged in {k} iterations")
        break
    # Newton step: solve H * d = -grad
    d = np.linalg.solve(H, -g)
    X = X + d

# Check second-order condition
eig = np.linalg.eigvals(np.array(h_num(*X), dtype=float))
print("\nOptimal Point: x = %.6f, y = %.6f" % (X[0], X[1]))
print("Minimum Value of f: %.6f" % f_num(*X))
print("Eigenvalues of Hessian:", np.round(eig, 4))
print("Hessian is positive definite:", bool(np.all(eig > 0)))
