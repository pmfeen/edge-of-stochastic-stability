import numpy as np
import torch
from functools import wraps

"""Native PyTorch implementation of the LOBPCG algorithm for eigenvalue computation.

This code was copied (and translated into PyTorch) from Google's JAX implementation:
https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py#L37-L105
"""


def _project_out(basis, U):
    for _ in range(2):
        U -= _mm(basis, _mm(basis.T, U))
        U = _orthonormalize(U)
    for _ in range(2):
        U -= _mm(basis, _mm(basis.T, U))
    normU = torch.linalg.norm(U, ord=2, dim=0, keepdims=True)
    U *= normU >= 0.99

    return U


def _rayleigh_ritz_orth(A, S):
    SAS = _mm(S.T, A(S))
    return _eigh_ascending(SAS)


def _orthonormalize(basis):
    for _ in range(2):
        basis = _svqb(basis)
    return basis


def _mm(a, b):
    return torch.matmul(a, b)


def _eigh_ascending(A):
    try:
        w, V = torch.linalg.eigh(A)
    except:
        print("uh oh")
        raise Exception("eigh failed")
    idx = torch.argsort(-w)
    return w[idx], V[:, idx]


def _svqb(X):
    norms = torch.linalg.norm(X, ord=2, dim=0, keepdims=True)
    X /= torch.where(norms == 0, 1.0, norms)
    inner = _mm(X.T, X)
    w, V = _eigh_ascending(inner)
    tau = torch.finfo(X.dtype).eps * w[0]
    padded = torch.maximum(w, tau)
    sqrted = torch.where(tau > 0, padded, 1.0) ** (-0.5)
    scaledV = V * sqrted[None, :]
    orthoX = _mm(X, scaledV)
    keep = ((w > tau) * (torch.diag(inner) > 0.0))[None, :]
    orthoX *= keep.type(orthoX.dtype)
    norms = torch.linalg.norm(orthoX, ord=2, dim=0, keepdims=True)
    keep *= (norms > 0.0).type(keep.dtype)
    orthoX /= torch.where(keep, norms, 1.0)
    return orthoX


def _extend_basis(X, m):
    n, k = X.shape
    Xupper, Xlower = X[:k], X[k:]
    u, s, vt = torch.linalg.svd(Xupper)
    y = torch.concatenate([Xupper + _mm(u, vt), Xlower], dim=0)
    other = torch.concatenate(
        [
            torch.eye(m, dtype=X.dtype, device=X.device),
            torch.zeros((n - k - m, m), dtype=X.dtype, device=X.device),
        ],
        dim=0,
    )
    w = _mm(y, vt.T * ((2 * (1 + s)) ** (-1 / 2))[None, :])
    h = -2 * torch.linalg.multi_dot([w, w[k:, :].T, other])
    h[k:] += other
    return h


def _maybe_orthonormalize(X, assume_ortho=False, tol=None):
    if not assume_ortho:
        return _orthonormalize(X)
    k = X.shape[1]
    if tol is None:
        # a safe default; loosen if you run fp32
        tol = 100 * k * torch.finfo(X.dtype).eps
    G = X.T @ X
    err = torch.linalg.norm(G - torch.eye(k, dtype=G.dtype, device=G.device), ord='fro')
    if err <= tol:
        # If the input is already orthonormal, return it directly
        print(f"Input is already orthonormal with error {err:.2e} <= tol {tol:.2e}.")
    return X if err <= tol else _orthonormalize(X)


# wrap the matvec function A with an error handler that, in the event
# of an OOM, returns the shape of the input tensor that triggered it
def create_safe_wrapper(A):
    @wraps(A)  # This preserves the metadata of the original function A
    def safe_A(tensor):
        try:
            return A(tensor)
        except torch.cuda.OutOfMemoryError as e:            
            raise torch.cuda.OutOfMemoryError(
                f"{str(e)}\nTensor shape that caused the OOM: {tuple(tensor.shape)}"
            )
    
    return safe_A





def torch_lobpcg(A, X, max_iter=100, tol=None):
    """
    MANUAL FROM JAX DOCSTRING:
    -------------
    Compute the top-k standard eigenvalues using the LOBPCG routine.

    LOBPCG [1] stands for Locally Optimal Block Preconditioned Conjugate Gradient.
    The method enables finding top-k eigenvectors in an accelerator-friendly
    manner.

    This initial experimental version has several caveats.

        - Only the standard eigenvalue problem `A U = lambda U` is supported,
        general eigenvalues are not.
        - Gradient code is not available.
        - f64 will only work where jnp.linalg.eigh is supported for that type.
        - Finding the smallest eigenvectors is not yet supported. As a result,
        we don't yet support preconditioning, which is mostly needed for this
        case.

    The implementation is based on [2] and [3]; however, we deviate from these
    sources in several ways to improve robustness or facilitate implementation:

        - Despite increased iteration cost, we always maintain an orthonormal basis
        for the block search directions.
        - We change the convergence criterion; see the `tol` argument.
        - Soft locking [4] is intentionally not implemented; it relies on
        choosing an appropriate problem-specific tolerance to prevent
        blow-up near convergence from catastrophic cancellation of
        near-0 residuals. Instead, the approach implemented favors
        truncating the iteration basis.

    [1]: http://ccm.ucdenver.edu/reports/rep149.pdf
    [2]: https://arxiv.org/abs/1704.07458
    [3]: https://arxiv.org/abs/0705.2626
    [4]: DOI 10.13140/RG.2.2.11794.48327

    Args:
        A : An `(n, n)` array representing a square Hermitian matrix or a
            callable with its action.
        X : An `(n, k)` array representing the initial search directions for the `k`
            desired top eigenvectors. This need not be orthogonal, but must be
            numerically linearly independent (`X` will be orthonormalized).
            Note that we must have `0 < k * 5 < n`.
        m : Maximum integer iteration count; LOBPCG will only ever explore (a
            subspace of) the Krylov basis `{X, A X, A^2 X, ..., A^m X}`.
        tol : A float convergence tolerance; an eigenpair `(lambda, v)` is converged
            when its residual L2 norm `r = |A v - lambda v|` is below
            `tol * 10 * n * (lambda + |A v|)`, which
            roughly estimates the worst-case floating point error for an ideal
            eigenvector. If all `k` eigenvectors satisfy the tolerance
            comparison, then LOBPCG exits early. If left as None, then this is set
            to the float epsilon of `A.dtype`.

    Returns:
        `theta, U, i`, where `theta` is a `(k,)` array
        of eigenvalues, `U` is a `(n, k)` array of eigenvectors, `i` is the
        number of iterations performed.

    Raises:
        ValueError : if `A,X` dtypes or `n` dimensions do not match, or `k` is too
                    large (only `k * 5 < n` supported), or `k == 0`.
    """
    n, k = X.shape
    
    A = create_safe_wrapper(A)

    if n < 4 * k:
        A_mat = A(torch.eye(n, dtype=X.dtype, device=X.device))
        evals, evecs = _eigh_ascending(A_mat)
        return evals[:k], evecs[:, :k], 1

    dt = X.dtype
    if tol is None:
        tol = torch.finfo(dt).eps

    X = _maybe_orthonormalize(X)
    P = _extend_basis(X, X.shape[1])
    AX = A(X)
    theta = torch.sum(X * AX, dim=0, keepdims=True)
    R = AX - theta * X
    converged = 0
    for i in range(max_iter):
        if converged >= k:
            break
        R = _project_out(torch.concatenate((X, P), dim=1), R)
        XPR = torch.concatenate((X, P, R), dim=1)
        theta, Q = _rayleigh_ritz_orth(A, XPR)
        theta = theta[:k]
        B = Q[:, :k]
        normB = torch.linalg.norm(B, ord=2, dim=0, keepdims=True)
        B /= normB
        X = _mm(XPR, B)
        normX = torch.linalg.norm(X, ord=2, dim=0, keepdims=True)
        X /= normX
        q, _ = torch.linalg.qr(Q[:k, k:].T)
        diff_rayleigh_ortho = _mm(Q[:, k:], q)
        P = _mm(XPR, diff_rayleigh_ortho)
        normP = torch.linalg.norm(P, ord=2, dim=0, keepdims=True)
        P /= torch.where(normP == 0, 1.0, normP)
        AX = A(X)
        R = AX - theta[None, :] * X
        resid_norms = torch.linalg.norm(R, ord=2, dim=0)
        reltol = 10 * n * (torch.linalg.norm(AX, ord=2, dim=0) + theta)
        res_converged = resid_norms < tol * reltol
        converged = torch.sum(res_converged)
    return theta, X, i


