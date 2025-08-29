import os
try:
    import pymanopt
except ImportError:
    os.system('pip install pymanopt')
from pymanopt import Problem
from pymanopt.manifolds import Sphere
from pymanopt.optimizers import ConjugateGradient
from pymanopt.function import autograd as pymanopt_autograd

try:
    import autograd.numpy as anp
except ImportError:
    os.system('pip install autograd')

import autograd.numpy as anp

import numpy as np

def compute_phi(y, A, B, c):
    Ay = A @ y
    By = B @ y
    M = anp.outer(By, anp.conj(By)) + c * anp.eye(A.shape[0])
    return anp.real(anp.conj(Ay).T @ anp.linalg.solve(M, Ay))


def compute_gradient(y, A, B, c):
    # Gradient of phi(y) w.r.t y (Euclidean)
    Ay = A @ y
    By = B @ y
    n = A.shape[0]
    M = anp.outer(By, anp.conj(By)) + c * anp.eye(n)
    M_inv = anp.linalg.inv(M)

    # Compute core components
    w = M_inv @ Ay
    grad = 2 * anp.real(anp.conj(A).T @ w)  # derivative through Ay
    # Derivative through By and M
    alpha = anp.vdot(By, M_inv @ Ay)
    temp = (anp.outer(w, anp.conj(By)) + anp.outer(By, anp.conj(w))) @ M_inv @ Ay
    correction = -2 * anp.real(anp.conj(B).T @ temp)
    grad += correction
    return grad


def solve_with_random_restarts(A, B, c=1.0, restarts=10):
    n = A.shape[1]
    manifold = Sphere(n)
    best_val = -anp.inf
    best_y = None

    # Define cost and gradient for pymanopt
    @pymanopt_autograd(manifold)
    def cost(y):
        return -compute_phi(y, A, B, c)  # negative for maximization

    optimizer = ConjugateGradient(verbosity=0)

    for r in range(restarts):
        y0 = anp.random.randn(n) + 1j * anp.random.randn(n)
        y0 /= anp.linalg.norm(y0)

        problem = Problem(manifold=manifold, cost=cost)
        result = optimizer.run(problem, initial_point=y0)

        val = -result.cost
        if val > best_val:
            best_val = val
            best_y = result.point

    # Compute x* for best y
    Ay = A @ best_y
    By = B @ best_y
    M = anp.outer(By, anp.conj(By)) + c * anp.eye(A.shape[0])
    x_star = anp.linalg.solve(M, Ay)
    x_star /= anp.linalg.norm(x_star)

    return best_y, x_star, best_val

def normalize(z):
    nrm = np.linalg.norm(z)
    if nrm == 0:
        return z
    return z / nrm

def objective(x, A, B, c):
    # f(x) = |x^H A x|^2 / (|x^H B x|^2 + c), with ||x||=1
    a = np.real(np.vdot(x, A @ x))             # since A is Hermitian, a is real
    b = np.vdot(x, B @ x)                       # complex in general
    denom = (np.abs(b) ** 2) + c
    return (a * a) / denom

def euclidean_grad(x, A, B, c):
    """
    Euclidean gradient of f wrt x in C^n, using CR-calculus.
    For a real-valued f, ∇_x f = 2 * ∂f/∂x*.
    Let a = x^H A x (real), b = x^H B x (complex), h = |b|^2 + c
    ∂f/∂x* = ((2 a A x) * h - (a^2) * (conj(b) B x)) / h^2
    => ∇ f = 2 * ∂f/∂x*.
    """
    Ax = A @ x
    Bx = B @ x
    a = np.real(np.vdot(x, Ax))
    b = np.vdot(x, Bx)
    h = (np.abs(b) ** 2) + c
    # Safeguard
    if h <= 1e-16:
        h = 1e-16
    grad = (4.0 * a / h) * Ax - (2.0 * (a ** 2) * np.conj(b) / (h ** 2)) * Bx
    return grad

def riemannian_grad(x, A, B, c):
    """
    Projection onto the tangent space of the complex sphere with metric Re(<.,.>):
    Proj_x(g) = g - x * Re(x^H g).
    """
    g = euclidean_grad(x, A, B, c)
    return g - x * np.real(np.vdot(x, g))

# ---------- Riemannian gradient ascent with Armijo backtracking ----------

def rgrad_ascent(
    x0, A, B, c,
    max_iter=2000, tol=1e-8,
    step0=1.0, backtrack_beta=0.5, armijo_sigma=1e-4
):
    """
    Returns: x, fval, iters
    """
    x = normalize(x0.astype(complex))
    f = objective(x, A, B, c)

    for it in range(1, max_iter + 1):
        gradR = riemannian_grad(x, A, B, c)
        grad_norm = np.linalg.norm(gradR)

        if grad_norm < tol:
            break

        # Armijo backtracking line search on manifold (retraction is normalization)
        t = step0
        # ascent direction is +gradR
        while True:
            x_new = normalize(x + t * gradR)
            f_new = objective(x_new, A, B, c)
            # Armijo condition for ascent: f_new >= f + sigma * t * <gradR, gradR>_R,
            # where metric is Re(<.,.>):
            lhs = f_new
            rhs = f + armijo_sigma * t * (grad_norm ** 2)
            if lhs >= rhs or t < 1e-16:
                break
            t *= backtrack_beta

        x, f = x_new, f_new

    return x, f, it

# ---------- Multiple random restarts ----------

def solve_x_equals_y_fast(A, B, c=1.0, restarts=10, seed=None):
    """
    Maximize |x^H A x|^2 / (|x^H B x|^2 + c), ||x||=1
    using Pymanopt (Riemannian conjugate gradient with restarts).
    """
    if seed is not None:
        np.random.seed(seed)

    # Symmetrize A to be exactly Hermitian (for numerical safety)
    A = 0.5 * (A + A.conj().T)
    n = A.shape[0]
    manifold = Sphere(n)

    @pymanopt_autograd(manifold)
    def cost(x):
        # Convert to autograd operations for complex matrices
        Ax = A @ x
        Bx = B @ x
        # For complex inner products, use explicit real/imag parts
        a_real = anp.real(anp.sum(anp.conj(x) * Ax))  # Re(x^H A x)
        b_complex = anp.sum(anp.conj(x) * Bx)         # x^H B x
        denom = anp.real(b_complex) ** 2 + anp.imag(b_complex) ** 2 + c
        return -(a_real * a_real) / denom

    optimizer = ConjugateGradient(verbosity=0)
    best_val, best_x = -np.inf, None

    for _ in range(restarts):
        x0 = anp.random.randn(n) + 1j * anp.random.randn(n)
        x0 /= anp.linalg.norm(x0)
        problem = Problem(manifold=manifold, cost=cost)
        try:
            result = optimizer.run(problem, initial_point=x0)
            val = -result.cost
            if val > best_val:
                best_val, best_x = val, result.point
        except Exception as e:
            print(f"Optimization failed for restart: {e}")
            continue

    return best_x, best_val

# Example usage
# if __name__ == "__main__":
#     print("Testing manifold optimization...")
    
#     np.random.seed(0)
#     n = 6
#     # Construct a random Hermitian A and arbitrary B
#     M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
#     A = 0.5 * (M + M.conj().T)               # Hermitian
#     B = np.random.randn(n, n) + 1j * np.random.randn(n, n)
#     c = 0.3

#     print("Testing pymanopt implementation...")
#     try:
#         x_star, fval = solve_x_equals_y_fast(A, B, c, restarts=10, seed=123)
#         print("Fast method - Best objective:", fval)
#         print("Fast method - ||x_star||:", np.linalg.norm(x_star))
#         print("Fast method - x^H A x (real):", np.real(np.vdot(x_star, A @ x_star)))
#         print("Fast method - |x^H B x|:", np.abs(np.vdot(x_star, B @ x_star)))
#     except Exception as e:
#         print(f"Fast method failed: {e}")

#     print("\nTesting original solve_with_random_restarts...")
#     try:
#         y_star, x_star_orig, val_orig = solve_with_random_restarts(A, B, c, restarts=5)
#         print("Original method - Best value:", val_orig)
#         print("Original method - ||y||:", np.linalg.norm(y_star), "||x||:", np.linalg.norm(x_star_orig))
#     except Exception as e:
#         print(f"Original method failed: {e}")

#     print("\nAll tests completed!")
