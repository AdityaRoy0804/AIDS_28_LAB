import numpy as np
from scipy.optimize import curve_fit

# Observed data
x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
y = np.array([2.0, 3.4, 5.3, 9.1, 14.5, 24.6])

# Model to be fitted: y = a * exp(b * x)
def model(x, a, b):
    return a * np.exp(b * x)

# Nonlinear least squares fit (initial guess a = 1, b = 0.1)
params, cov = curve_fit(model, x, y, p0=[1.0, 0.1])
a, b = params

# Fitted values and residuals
y_fit = model(x, a, b)
residuals = y - y_fit
sse = np.sum(residuals**2)
sst = np.sum((y - np.mean(y))**2)
r2 = 1 - sse/sst

print("Model: y = a * exp(b * x)")
print("\nx    y(actual)   y(fitted)   residual")
for xi, yi, fi, ri in zip(x, y, y_fit, residuals):
    print(f"{xi:.0f}    {yi:8.3f}    {fi:8.3f}    {ri:8.3f}")

print("\nEstimated Parameters:")
print("a = %.4f" % a)
print("b = %.4f" % b)
print("\nSum of Squared Errors (SSE) = %.4f" % sse)
print("R-squared = %.4f" % r2)
print("\nBest-fit curve: y = %.4f * exp(%.4f * x)" % (a, b))
