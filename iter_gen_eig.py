"""Iterative Optimization """
import numpy as np

def normalize(z, eps=1e-16):
    nrm = np.linalg.norm(z)
    if nrm < eps:
        # random unit vector fallback
        z = np.random.randn(*z.shape) + 1j*np.random.randn(*z.shape)
        nrm = np.linalg.norm(z)
    return z / nrm

def rank1_inv_times_vec(v, a, c):
    """
    Compute (v v^H + c I)^{-1} a efficiently via Sherman-Morrison:
    (c I + v v^H)^{-1} a = (1/c) * (a - v * (v^H a) / (c + v^H v))
    """
    vvHa = np.vdot(v, a)               # v^H a (scalar)
    v_norm2 = np.vdot(v, v)            # v^H v (scalar, real >=0)
    denom = c + v_norm2
    # numeric safeguard
    if np.abs(denom) < 1e-16:
        denom = 1e-16
    return (a - v * (vvHa / denom)) / c

def alternating_xy(A, B, c=1e-3, y0=None, max_iter=200, tol=1e-9, verbose=False):
    """
    Alternating maximization for z = |x^H A y|^2 / (|x^H B y|^2 + c)
    A, B: (m,n) complex
    x in C^m, y in C^n
    returns best (x,y,z)
    """
    m, n = A.shape[0], A.shape[1]
    if y0 is None:
        y = np.random.randn(n) + 1j*np.random.randn(n)
    else:
        y = y0.copy()
    y = normalize(y)

    # initialize x randomly
    x = np.random.randn(m) + 1j*np.random.randn(m)
    x = normalize(x)

    def objective(x, y):
        num = np.vdot(x, A @ y)      # x^H A y
        den = np.vdot(x, B @ y)      # x^H B y
        return (np.abs(num)**2) / (np.abs(den)**2 + c)

    z_prev = objective(x, y)

    for k in range(max_iter):
        # --- x-update ---
        a = A @ y        # (m,)
        v = B @ y        # (m,)
        if np.linalg.norm(a) < 1e-16:
            x = np.random.randn(m) + 1j*np.random.randn(m)
            x = normalize(x)
        else:
            x_new_unnorm = rank1_inv_times_vec(v, a, c)  # (V+cI)^{-1} a
            x = normalize(x_new_unnorm)

        # --- y-update ---
        e = A.conj().T @ x    # (n,)
        w = B.conj().T @ x    # (n,)
        if np.linalg.norm(e) < 1e-16:
            y = np.random.randn(n) + 1j*np.random.randn(n)
            y = normalize(y)
        else:
            y_new_unnorm = rank1_inv_times_vec(w, e, c)  # (D+cI)^{-1} e
            y = normalize(y_new_unnorm)

        z_curr = objective(x, y)
        if verbose:
            print(f"iter {k+1}: z = {z_curr:.6e}")

        # stopping
        if np.abs(z_curr - z_prev) <= tol * max(1.0, np.abs(z_prev)):
            break
        z_prev = z_curr

    return x, y, z_curr
