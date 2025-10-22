""" Optimization for maximizing |x^T A x|^2 / (|x^T B x|^2 + c), ||x||=1
    using manifold optimization with transpose instead of conjugate transpose."""


import numpy as np
from numba import jit, complex128, float64
import warnings

@jit(nopython=True)
def normalize_numba(z):
    """Fast normalization using numba"""
    nrm = np.sqrt(np.sum(np.abs(z)**2))
    if nrm < 1e-16:
        return z
    return z / nrm

@jit(nopython=True)
def objective_transpose_numba(x, A, B, c):
    """Fast objective computation using numba"""
    # x^T A x and x^T B x
    a = np.sum(x * (A @ x))  # equivalent to x.T @ A @ x for vectors
    b = np.sum(x * (B @ x))  # equivalent to x.T @ B @ x for vectors
    
    # |a|^2 and |b|^2
    a_abs_sq = a.real**2 + a.imag**2
    b_abs_sq = b.real**2 + b.imag**2
    
    return a_abs_sq / (b_abs_sq + c)

@jit(nopython=True)
def euclidean_grad_transpose_numba(x, A, B, c):
    """Fast gradient computation using numba"""
    # Compute A@x and B@x once
    Ax = A @ x
    Bx = B @ x
    
    # x^T A x and x^T B x
    a = np.sum(x * Ax)
    b = np.sum(x * Bx)
    
    # Gradients: d/dx (x^T A x) = (A + A^T) x
    grad_a = Ax + (A.T @ x)
    grad_b = Bx + (B.T @ x)
    
    # Apply chain rule
    b_abs_sq = b.real**2 + b.imag**2
    denom = b_abs_sq + c
    
    if denom <= 1e-16:
        denom = 1e-16
    
    # For complex a: d/dx |a|^2 = 2 * Re(conj(a) * grad_a)
    num_grad = 2.0 * (a.real * grad_a.real + a.imag * grad_a.imag)
    denom_grad = 2.0 * (b.real * grad_b.real + b.imag * grad_b.imag)
    
    # Quotient rule
    a_abs_sq = a.real**2 + a.imag**2
    grad = (denom * num_grad - a_abs_sq * denom_grad) / (denom * denom)
    
    return grad

@jit(nopython=True)
def riemannian_grad_transpose_numba(x, A, B, c):
    """Fast Riemannian gradient using numba"""
    g = euclidean_grad_transpose_numba(x, A, B, c)
    # For complex: Proj_x(g) = g - x * Re(x^H g)
    dot_product = np.sum(x.conj() * g)
    return g - x * dot_product.real

@jit(nopython=True)
def rgrad_ascent_transpose_fast(
    x0, A, B, c,
    max_iter=1000, tol=1e-6,
    step0=0.1, backtrack_beta=0.8, armijo_sigma=1e-4
):
    """Optimized Riemannian gradient ascent"""
    x = normalize_numba(x0)
    f = objective_transpose_numba(x, A, B, c)
    
    # Adaptive step size
    step = step0
    
    for it in range(1, max_iter + 1):
        gradR = riemannian_grad_transpose_numba(x, A, B, c)
        grad_norm_sq = np.sum(np.abs(gradR)**2)
        grad_norm = np.sqrt(grad_norm_sq)
        
        if grad_norm < tol:
            break
        
        # Armijo backtracking with fewer iterations
        t = step
        max_backtrack = 10  # Limit backtracking iterations
        backtrack_count = 0
        
        while backtrack_count < max_backtrack:
            x_new = normalize_numba(x + t * gradR)
            f_new = objective_transpose_numba(x_new, A, B, c)
            
            # Armijo condition for ascent
            if f_new >= f + armijo_sigma * t * grad_norm_sq or t < 1e-12:
                break
            t *= backtrack_beta
            backtrack_count += 1
        
        # Update with momentum-like adaptation
        if f_new > f:
            step = min(1.2 * t, 1.0)  # Increase step size if successful
        else:
            step = max(0.5 * t, 1e-8)  # Decrease if not improving
        
        x, f = x_new, f_new
    
    return x, f, it

def solve_transpose_fast_vectorized(A, B, c=1.0, restarts=10, seed=None):
    """
    Fast vectorized version with multiple optimizations
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = A.shape[0]
    
    # Pre-compute A + A.T and B + B.T for efficiency
    A_sym = A + A.T
    B_sym = B + B.T
    
    # Generate all initial points at once
    if np.iscomplexobj(A) or np.iscomplexobj(B):
        X0 = np.random.randn(n, restarts) + 1j * np.random.randn(n, restarts)
    else:
        X0 = np.random.randn(n, restarts)
    
    # Normalize all initial points
    X0 = X0 / np.linalg.norm(X0, axis=0, keepdims=True)
    
    best_val = -np.inf
    best_x = None
    
    # Run optimization for each restart
    for i in range(restarts):
        x0 = X0[:, i]
        try:
            x_opt, val, iters = rgrad_ascent_transpose_fast(
                x0, A, B, c, 
                max_iter=500,  # Reduced iterations
                tol=1e-6,
                step0=0.1
            )
            if val > best_val:
                best_val = val
                best_x = x_opt
        except:
            continue
    
    return best_x, best_val

def solve_transpose_eigen_init(A, B, c=1.0, restarts=5, seed=None):
    """
    Smart initialization using generalized eigenvalue problem
    Combined with random restarts for robustness
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = A.shape[0]
    best_val = -np.inf
    best_x = None
    
    # Smart initialization: use leading generalized eigenvector
    try:
        # For the problem max |x^T A x|^2 / (|x^T B x|^2 + c)
        # This is related to the generalized eigenvalue problem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Try different formulations for smart initialization
            candidates = []
            
            # Method 1: Use A as starting point
            if np.allclose(A, A.conj().T):  # If A is Hermitian
                eigvals, eigvecs = np.linalg.eigh(A)
                candidates.append(eigvecs[:, -1])  # Largest eigenvalue
            
            # Method 2: Use B^{-1}A if B is invertible
            try:
                B_reg = B + 1e-8 * np.eye(n)
                if np.allclose(B_reg, B_reg.conj().T):
                    eigvals, eigvecs = np.linalg.eig(np.linalg.solve(B_reg, A))
                    idx = np.argmax(np.real(eigvals))
                    candidates.append(eigvecs[:, idx])
            except:
                pass
            
            # Method 3: Random with better distribution
            for _ in range(restarts):
                if np.iscomplexobj(A) or np.iscomplexobj(B):
                    x0 = np.random.randn(n) + 1j * np.random.randn(n)
                else:
                    x0 = np.random.randn(n)
                candidates.append(x0)
            
            # Test all candidates
            for x0 in candidates:
                x0 = x0 / np.linalg.norm(x0)
                try:
                    x_opt, val, _ = rgrad_ascent_transpose_fast(
                        x0, A, B, c,
                        max_iter=300,  # Fewer iterations per restart
                        tol=1e-6
                    )
                    if val > best_val:
                        best_val = val
                        best_x = x_opt
                except:
                    continue
                    
    except Exception as e:
        print(f"Smart initialization failed: {e}, falling back to random")
        return solve_transpose_fast_vectorized(A, B, c, restarts, seed)
    
    return best_x, best_val

# Drop-in replacement for your current function
def solve_transpose_with_manual_restarts(A, B, c=1.0, restarts=10, seed=None):
    """
    Fast drop-in replacement for the original function
    """
    # Use the smart initialization method first (fewer restarts)
    x_opt, val = solve_transpose_eigen_init(A, B, c, restarts=min(restarts, 5), seed=seed)
    
    # If we have extra restarts, use the fast vectorized method
    if restarts > 5:
        remaining_restarts = restarts - 5
        x_opt2, val2 = solve_transpose_fast_vectorized(A, B, c, remaining_restarts, seed)
        if val2 > val:
            x_opt, val = x_opt2, val2
    
    return x_opt, val