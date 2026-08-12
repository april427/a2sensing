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

# ---------- Complex unit sphere via pymanopt's real Sphere ----------
#
# pymanopt 2.x has no ComplexSphere, and Sphere(n) is a REAL manifold: for a complex
# point it reports dim = n-1 instead of 2n-1 and its tangent projection is wrong, so
# ConjugateGradient stalls after a couple of iterations ("min step_size reached").
# The complex unit sphere {x in C^n : ||x||=1} is isometric to the real sphere
# S^(2n-1) under the metric Re<.,.>, so we optimize over z = [Re(x); Im(x)] on
# Sphere(2n) and keep every operation in real arithmetic (autograd then differentiates
# a real function of real inputs, with no complex-conjugation convention to get wrong).

def _split(z, n):
    'z = [Re(x); Im(x)] -> (xr, xi)'
    return z[:n], z[n:]


def _matvec(Mr, Mi, xr, xi):
    '(Mr + 1j*Mi) @ (xr + 1j*xi) -> (real, imag)'
    return Mr @ xr - Mi @ xi, Mr @ xi + Mi @ xr


def _quad_form(Mr, Mi, xr, xi):
    'Re and Im of x^H M x'
    ur, ui = _matvec(Mr, Mi, xr, xi)
    return (anp.sum(xr * ur) + anp.sum(xi * ui),
            anp.sum(xr * ui) - anp.sum(xi * ur))


def _random_sphere_point(n):
    z = np.random.randn(2 * n)
    return z / np.linalg.norm(z)


def _run_from_random_start(optimizer, problem, n):
    """One CG run from a random point on the complex unit sphere.

    pymanopt's default Hestenes-Stiefel beta rule evaluates
    <Pnewgrad, diff> / <diff, descent_direction> with diff = newgrad - oldgrad.
    Once the line search stops making progress diff is exactly zero, so numpy
    computes 0.0/0.0 and emits "invalid value encountered in divide"; pymanopt's
    own guard there only catches Python's ZeroDivisionError, which numpy never
    raises. The resulting nan is absorbed by max(0, nan) == 0, so beta simply
    resets to steepest descent and the run terminates normally. The warning is
    therefore cosmetic, and is silenced only for this call.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return optimizer.run(problem, initial_point=_random_sphere_point(n))


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
    """
    Maximize over y (transmit, ||y||=1) of

        phi(y) = max_x |x^H A y|^2 / (|x^H B y|^2 + c ||x||^2)
               = (Ay)^H M^-1 (Ay),   M = (By)(By)^H + c I

    i.e. the receive beamformer x is the closed-form MVDR/MMSE solution for each y.
    Only y is on the manifold. M is rank-one-plus-identity, so Sherman-Morrison
    replaces the linear solve with

        phi(y) = ( ||Ay||^2 - |<By, Ay>|^2 / (c + ||By||^2) ) / c

    which is cheaper and keeps the cost differentiable in real arithmetic.
    """
    n = A.shape[1]
    Ar, Ai = np.real(A), np.imag(A)
    Br, Bi = np.real(B), np.imag(B)
    manifold = Sphere(2 * n)
    best_val = -np.inf
    best_y = None

    @pymanopt_autograd(manifold)
    def cost(z):
        yr, yi = _split(z, n)
        ar, ai = _matvec(Ar, Ai, yr, yi)   # A y
        br, bi = _matvec(Br, Bi, yr, yi)   # B y
        a_sq = anp.sum(ar ** 2 + ai ** 2)
        b_sq = anp.sum(br ** 2 + bi ** 2)
        ip_re = anp.sum(br * ar + bi * ai)  # Re<By, Ay>
        ip_im = anp.sum(br * ai - bi * ar)  # Im<By, Ay>
        phi = (a_sq - (ip_re ** 2 + ip_im ** 2) / (c + b_sq)) / c
        return -phi

    optimizer = ConjugateGradient(verbosity=0)
    problem = Problem(manifold=manifold, cost=cost)

    for r in range(restarts):
        try:
            result = _run_from_random_start(optimizer, problem, n)
        except Exception as e:
            print(f"Optimization failed for restart: {e}")
            continue
        val = -result.cost
        if np.isfinite(val) and val > best_val:
            zr, zi = _split(result.point, n)
            best_val = val
            best_y = zr + 1j * zi

    # Closed-form receive beamformer x* = M^-1 (A y), again via Sherman-Morrison
    Ay = A @ best_y
    By = B @ best_y
    x_star = (Ay - By * (np.vdot(By, Ay) / (c + np.vdot(By, By).real))) / c
    x_star /= np.linalg.norm(x_star)

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

    # NOTE: A must NOT be Hermitian-symmetrized here. For a reciprocal round-trip
    # channel A = a a^T is complex symmetric, and replacing it by 0.5*(A + A^H)
    # together with Re(x^H A x)^2 optimizes a different objective than the
    # |x^H A x|^2 that callers actually evaluate.
    n = A.shape[0]
    Ar, Ai = np.real(A), np.imag(A)
    Br, Bi = np.real(B), np.imag(B)
    manifold = Sphere(2 * n)

    @pymanopt_autograd(manifold)
    def cost(z):
        xr, xi = _split(z, n)
        a_re, a_im = _quad_form(Ar, Ai, xr, xi)   # x^H A x
        b_re, b_im = _quad_form(Br, Bi, xr, xi)   # x^H B x
        return -(a_re ** 2 + a_im ** 2) / (b_re ** 2 + b_im ** 2 + c)

    optimizer = ConjugateGradient(verbosity=0)
    problem = Problem(manifold=manifold, cost=cost)
    best_val, best_x = -np.inf, None

    for _ in range(restarts):
        try:
            result = _run_from_random_start(optimizer, problem, n)
            val = -result.cost
            if np.isfinite(val) and val > best_val:
                zr, zi = _split(result.point, n)
                best_val, best_x = val, zr + 1j * zi
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
