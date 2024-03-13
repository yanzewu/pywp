
import numpy as np
import warnings
from typing import List
import scipy.linalg
import scipy.sparse


def make_banded(A, n=None):
    """ Transform a matrix to band matrix to meet the criteria of `scipy.linalg.banded_solve()`
    n:  number of off-diagonal on one side.
    """

    if n is None:
        for n in range(1, A.shape[0]):  # has to start from 1 to avoid diagonal 0
            if abs(A[0, n]) < 1e-12:
                break

        n -= 1

    B = np.zeros((2*n+1, A.shape[0]), dtype=A.dtype)
    for k in range(2*n+1):
        B[k, max(n-k,0):min(n-k+A.shape[0],A.shape[0])] = np.diag(A, n-k)

    return B


def itersolv(A:np.ndarray, b:np.ndarray, M:List[np.ndarray], atol:float=1e-8, rtol:float=1e-8, maxiter:int=500, x0:np.ndarray=None,
              adapt_scattering:bool=False, verbose:bool=False):
    """ Iterative solver of Ax=b, with reference matrices M representing A's diagonal band.

    M: List of matrices forming A's diagonal (assuming A is close to band diagonal). M must come in order.
    atol,rtol: The residual will be compared against |x|*rtol + atol
    maxiter: Maximum iteration.
    x0: Initial guess of x, defaults to M^{-1} b.
    adapt_scattering: Uses special optimizations to adapt inner call of `scatter1d()`. Typically increase performance by 3x.
        - `A` will be assumed to have a band-diagonal "body", starting from some offsets. Offset is determined by `M[0]`.
        - `M` will be assumed as real-valued except `M[0]`.

    Returns: x.
    """
    
    opt_band_diag = False        # fast inv of M
    opt_hermitian = False        # M is real
    opt_band_offdiag = False    # fast multiplication

    if adapt_scattering:
        opt_band_diag = True
        opt_hermitian = all((not np.any(m.imag) for m in M[1:]))
        opt_band_offdiag = opt_hermitian and A.shape[0] > 1000

    if verbose:
        print('opt_band_diag=%s, opt_hermitian=%s, opt_band_offdiag=%s'% (opt_band_diag, opt_hermitian, opt_band_offdiag))

    # check size

    assert A.shape[0] == A.shape[1] and A.ndim == 2
    assert A.shape[1] == b.shape[0] and b.ndim == 1

    sz = 0
    for m in M:
        assert m.shape[0] == m.shape[1] and m.ndim == 2
        sz += m.shape[0]

    assert sz == A.shape[0]

    # build M-1 frist

    segs = []
    start = 0
    for m in M:
        segs.append(slice(start,start+len(m)))
        start += len(m)

    if not opt_band_diag:
        invM = [np.linalg.inv(m) for m in M]
    else:
        invM = [np.linalg.inv(M[0])]
        bandM = [make_banded(m) for m in M[1:]]

    if opt_hermitian:
        offset = M[0].shape[0]
        A0 = A[:offset]
        A10 = A[offset:, :offset]

        if opt_band_offdiag:
            A11band = [
                [scipy.sparse.dia_array(A[s1, s2]) if np.any(A[s1, s2]) else None 
                    for s2 in segs[1:]]
                for s1 in segs[1:]
            ]
            def multiply_A11(y):
                z = np.zeros_like(y)
                for j in range(len(M)-1):
                    segj = slice(segs[j+1].start-offset, segs[j+1].stop-offset)
                    for k in range(len(M)-1):
                        a = A11band[j][k]
                        if a is None:
                            continue
                        z[segj] += a @ y[segs[k+1].start-offset:segs[k+1].stop-offset]
                return z


        else:
            multiply_A11 = lambda y: A[offset:, offset:] @ y



    # build x0
    if x0 is None:
        x = np.empty_like(b)
        if not opt_band_diag:
            for invm, seg in zip(invM, segs):
                x[seg] = invm @ b[seg]
        else:
            x[segs[0]] = invM[0] @ b[segs[0]]
            for bandm, seg in zip(bandM, segs[1:]):
                if opt_hermitian:
                    x[seg].real = scipy.linalg.solve_banded(((len(bandm)-1)//2, (len(bandm)-1)//2), bandm.real, b[seg].real)
                    x[seg].imag = scipy.linalg.solve_banded(((len(bandm)-1)//2, (len(bandm)-1)//2), bandm.real, b[seg].imag)
                else:
                    x[seg] = scipy.linalg.solve_banded(((len(bandm)-1)//2, (len(bandm)-1)//2), bandm, b[seg])
        
    else:
        assert x0.shape == b.shape
        x = x0.copy()

    s = np.empty_like(b)

    # shortcut 
    r = b - A @ x
    if np.linalg.norm(r) < atol + np.max(np.abs(x)) * rtol:
        return x

    Q = x[:,None]/ np.linalg.norm(x)

    AQ = A @ Q

    B = Q.T.conj() @ AQ
    p = Q.T.conj() @ b

    if verbose:
        print('chpt1')

    iter = 0
    while True:

        u = np.linalg.solve(B, p)
        x = Q @ u
        if not opt_band_offdiag:
            r = b - A @ x
        else:
            r = b - np.concatenate((
                A0 @ x, 
                A10 @ x[:offset] + multiply_A11(x[offset:])))

        if np.sqrt(np.vdot(r,r)) < atol + np.max(np.abs(x)) * rtol:
            if verbose:
                print('Total iteration', iter)
            break
        elif iter >= min(maxiter, A.shape[0]) - 1:
            warnings.warn('Warning: Maximum iteration (%s) reached' % (iter+1))
            break
        else:
            if not opt_band_diag:
                for invm, seg in zip(invM, segs):
                    s[seg] = invm @ r[seg]
            else:
                s[segs[0]] = invM[0] @ r[segs[0]]
                for bandm, seg in zip(bandM, segs[1:]):
                    if opt_hermitian:
                        s[seg].real = scipy.linalg.solve_banded(((len(bandm)-1)//2, (len(bandm)-1)//2), bandm.real, r[seg].real)
                        s[seg].imag = scipy.linalg.solve_banded(((len(bandm)-1)//2, (len(bandm)-1)//2), bandm.real, r[seg].imag)
                    else:
                        s[seg] = scipy.linalg.solve_banded(((len(bandm)-1)//2, (len(bandm)-1)//2), bandm, r[seg])

        if verbose:
            print('chpt2')

        Q_T_conj = Q.T.conj()
        q = s - Q @ (Q_T_conj @ s)  # orthogonalization
        q /= np.sqrt(np.vdot(q, q))

        n = len(B)
        if not opt_hermitian:
            Aq = A @ q
            ATqconj = A.T @ q.conj()
        
        else:   # Here we take advantage of the Hermicity of A11 -- so only 1 multiplication.
            A11q1 = multiply_A11(q[offset:])
            Aq = np.concatenate((A0 @ q, A10 @ q[:offset] + A11q1))
            ATqconj = A0.T @ q[:offset].conj()
            ATqconj[:offset] += A10.T @ q[offset:].conj()
            ATqconj[offset:] += A11q1.conj()    # since A11 is hermitian.
        
        B_new = np.empty_like(B, shape=(n+1, n+1))
        B_new[:n, :n] = B
        B_new[:n, n] = Q_T_conj @ Aq
        B_new[n, :n] = Q.T @ ATqconj
        B_new[n, n] = np.vdot(q, Aq)
        B = B_new
        p = np.concatenate((p, [np.vdot(q, b)]))
        Q = np.hstack((Q, q[:,None]))

        iter += 1
         

    return x
        
