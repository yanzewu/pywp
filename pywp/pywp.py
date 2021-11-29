
import numpy as np 
import itertools
from .potential import Potential

def abs2(x):
    return (x * x.conj()).real

def index_along(arr, ind, axis):
    indexer = [slice(None)] * len(arr.shape)
    indexer[axis] = ind
    return tuple(indexer)

def index_along2(arr, ind1, axis1, ind2, axis2):
    indexer = [slice(None)] * len(arr.shape)
    indexer[axis1] = ind1
    indexer[axis2] = ind2
    return tuple(indexer)


def preprocess(potential:Potential, N, L, sigma, R0, P0, n0, M, dt):
    """ Produce things needed for propatate().

    potential: class compatible with potential.Potential.
    N: grid. either a number, or list of numbers. len(N) must = potential->kdim()
    L: box length. number/list of numbers.
    sigma: array-like. initial wavepacket amplitude will be exp(-R^2/sigma^2)
    R0/P0: array-like.
    n0: int. the surface.
    M: mass
    """
    # NOTE currently we only support 2D

    nel = potential.get_dim()
    nk = potential.get_kdim()

    if isinstance(N, int):
        N = [N]*nk
    else:
        assert len(N) == nk
    if isinstance(L, (float, int)):
        L = [L]*nk
    else:
        assert len(L) == nk

    r = []
    k = []
    dA = 1.0
    dK = 1.0
    for j in range(nk):
        r.append(np.linspace(-L[j]/2, L[j]/2, N[j]))
        dr = L[j] / (N[j] - 1)
        k.append(np.fft.fftfreq(N[j], dr) * 2* np.pi)
        dA *= dr
        dK *= L[j] / (N[j] - 1) / N[j]

    R = np.meshgrid(*r, indexing='ij')
    K = np.meshgrid(*k, indexing='ij')
    
    Psi = np.zeros(list(N) + [nel], dtype=complex)
    Psi[index_along(Psi, n0, -1)] = np.exp(sum([1j*p*R_ - (R_-r)**2/s**2 for (r, p, R_, s) in zip(R0, P0, R, sigma)]))

    Psi /= np.sqrt(np.sum(abs2(Psi))* dA)
    H = potential.get_H(R)

    assert H.shape == tuple(list(N) + [nel, nel])

    KE = sum((K_**2 for K_ in K))/2/M
    TU = np.exp(-1j*dt*KE)

    # superfast eigenvalue for nel == 2

    if nel == 2:

        dE = np.real(H[index_along2(H, 1, -2, 1, -1)] - H[index_along2(H, 0, -2, 0, -1)])/2
        Eave = np.real(H[index_along2(H, 1, -2, 1, -1)] + H[index_along2(H, 0, -2, 0, -1)])/2
        delta = np.sqrt(dE**2 + abs2(H[index_along2(H, 0, -2, 1, -1)]))

        cos_half_theta = np.sqrt(0.5*(1 + dE/delta))
        sin_half_theta = np.sqrt(0.5*(1 - dE/delta))

        Phi = potential.get_phase(R) if potential.has_get_phase() else 1

        U = np.zeros(H.shape, dtype=complex)
        U[index_along2(U, 0, -2, 0, -1)] = cos_half_theta * Phi
        U[index_along2(U, 1, -2, 0, -1)] = -sin_half_theta
        U[index_along2(U, 0, -2, 1, -1)] = sin_half_theta * Phi
        U[index_along2(U, 1, -2, 1, -1)] = cos_half_theta

        E = np.zeros(H.shape[:-1])
        E[index_along(E, 0, -1)] = Eave - delta
        E[index_along(E, 1, -1)] = Eave + delta

    EU = np.zeros(H.shape, dtype=complex)
    EUhalf = np.zeros(H.shape, dtype=complex)

    for j in range(H.shape[-1]):
        EU[index_along2(EU, j, -2, j, -1)] = np.exp(-1j*dt*E[index_along(E, j, -1)])
        EUhalf[index_along2(EU, j, -2, j, -1)] = np.exp(-1j*dt/2*E[index_along(E, j, -1)])

    _tp_axes = list(range(len(U.shape)))
    _tp_axes[-1], _tp_axes[-2] = _tp_axes[-2], _tp_axes[-1]
    VU = np.matmul(U, np.matmul(EU, np.conj(np.transpose(U, axes=_tp_axes))))
    VUhalf = np.matmul(U, np.matmul(EUhalf, np.conj(np.transpose(U, axes=_tp_axes))))
    
    #VU = np.zeros(H.shape, dtype=complex)
    #VU[:,:,0,0] = np.exp(-1j*dt*H[:,:,0,0])
    #VUhalf = np.zeros(H.shape, dtype=complex)
    #VUhalf[:,:,0,0] = np.exp(-1j*dt/2*H[:,:,0,0])

    return Psi, H, KE, TU, VU, VUhalf, R, K, dA, dK, dt


def propagate(Psi, H, KE, TU, VU, VUhalf, R, K, dA, dK, dt, nstep, output_step, partitioner=None, partition_titles=None, trajfile=None, checkend=False, boundary=None, checkend_rtol=0.05, verbose=True, cuda_backend=False):
    """ The actual propagating function.
    partitioner: List of functions which will be called by p(R), where R is position meshgrid, list with potential.dim() element of size N x N ...
        it should return a boolean map (with size N x N ...). If None, a unit partitioner is used (just a number 1).
    partition_titles: List of string for verbose output.
    trajfile: If not None, will write wavefunction (on both position and momentum basis) to the file, ordered by electronic state. 
        All positions first, followed by momentums.
    checkend: If true, then call boundary(R) => bool map (where trajectory is in boundary).
        if the probability outside boundary is > checkend_rtol, then simulation is terminated.
    verbose: 
        If False, remains silent.
        If 1 or True, only print time, energy, kinetic energy, total population, and population of each partition region on each state.
        If 2, in addition, print position and momentum of each partition region on each state.

    Returns a list of [time, Population, Position, Momentum ], each
        is array with shape nel x m ( x nk ), where m is determined by partitioner (None -> 1). The collect timestep is same as output_step.
    """

    _tp_axes = list(range(len(VUhalf.shape)))
    _tp_axes[-1], _tp_axes[-2] = _tp_axes[-2], _tp_axes[-1]
    VUhalfinv = np.transpose(VUhalf.conjugate(), axes=_tp_axes)
    
    result = []

    partition_filter = [p(R) for p in partitioner] if partitioner else [1]
    boundary_filter = boundary(R) if boundary else 1
    nel = Psi.shape[-1]
    nk = len(Psi.shape) - 1
    
    # using_gpu: translate Psi, H, KE, TU, VU, VUHalf, VUHalfinv, R, K, partition_filter, boundary_filter
    
    if cuda_backend:
        import cupy as cp
        _backend = cp
        Psi = cp.asarray(Psi); H = cp.asarray(H); KE = cp.asarray(KE); TU = cp.asarray(TU)
        VU = cp.asarray(VU); VUhalf = cp.asarray(VUhalf); VUhalfinv = cp.asarray(VUhalfinv)
        R = [cp.asarray(R_) for R_ in R]
        K = [cp.asarray(K_) for K_ in K]
        partition_filter = [cp.asarray(p) if isinstance(p, np.ndarray) else p for p in partition_filter ]
        boundary_filter = cp.asarray(boundary_filter) if isinstance(boundary_filter, np.ndarray) else boundary_filter

        def dot_v(v, p):
            p1 = cp.zeros_like(p)
            for j in range(nel):
                p1[index_along(p1, j, -1)] = cp.sum(v[index_along(v, j, -2)] * p, axis=nk)
            return p1

    else:
        _backend = np

        dot_v = lambda v, p: _backend.einsum('ijkl,ijl->ijk', v, p)

    Psi = dot_v(VUhalf, Psi)

    result = []

    if verbose:
        pos_titles = 'XYZ'[:nk] # I don't think nk can be > 3
        mom_titles = 'xyz'[:nk]
        if not partition_titles:
            partition_titles = 'ABCDEFGHIJKLMNOPQ'[:len(partition_filter)]
        print('t\tE\tKE\tTotal', end='')
        print(''.join(('\t%s%d' % (x[1], x[0]) for x in itertools.product(range(nel), partition_titles))), end='')
        if verbose == 2:
            print(''.join(('\t%s%s%d' % (x[2], x[1], x[0]) for x in itertools.product(range(nel), partition_titles, pos_titles))) 
                + ''.join(('\tP%s%s%d' % (x[2], x[1], x[0]) for x in itertools.product(range(nel), partition_titles, mom_titles)))
            )
        else:
            print('')

    for i in range(nstep+1):
        
        if i % output_step == 0:
            if i != 0:
                Psi /= (_backend.sum(abs2(Psi))* dA)**0.5

            Psi_output = dot_v(VUhalfinv, Psi)
            Psip = [_backend.fft.fftn(Psi[index_along(Psi, j, -1)], axes=tuple(range(nk))) for j in range(nel)]

            Rhoave = []
            Rave = []
            Pave = []

            for j in range(nel):
                for pf in partition_filter:
                    abspsi = abs2(Psi[index_along(Psi, j, -1)]) * pf
                    
                    abspsip = abs2(_backend.fft.fftn(Psi[index_along(Psi, j, -1)] * pf, axes=tuple(range(nk))))
                    Rhoave.append(_backend.sum(abspsi).get() * dA)
                    Rave += [_backend.sum(abspsi*R_).get() * dA / Rhoave[-1] for R_ in R]
                    Pave += [_backend.sum(abspsip*K_).get() * dK / Rhoave[-1] for K_ in K]

            Rhotot = _backend.sum(abs2(Psi)) * dA
            KEave = sum((_backend.sum(abs2(Psip_) * KE) * dK for Psip_ in Psip))
            Eave = KEave + _backend.sum(_backend.real(_backend.conj(Psi_output) * dot_v(H, Psi_output))) * dA

            result.append((i*dt, 
                np.array(Rhoave).reshape((nel, len(partition_filter))), 
                np.array(Rave).reshape((nel, len(partition_filter), nk)), 
                np.array(Pave).reshape((nel, len(partition_filter), nk))))

            if trajfile:
                for j in range(nel):
                    Psi_output[index_along(Psi_output, j, -1)].tofile(trajfile)
                for psip in Psip:
                    _backend.fft.fftshift(psip).tofile(trajfile)

            if verbose:
                print('%.8g\t%.8g\t%.8g\t%.8g' % (i*dt, Eave, KEave, Rhotot), end='')
                print(''.join(('\t%.8g' % x for x in Rhoave)), end='')
                if verbose == 2:
                    print(''.join(('\t%.8g' % x for x in Rave)) + ''.join(('\t%.8g' % x for x in Pave)))
                else:
                    print('')

            if checkend and _backend.sum((boundary_filter * _backend.sum(abs2(Psi), axis=-1)))*dA < 1 - checkend_rtol:
                break

        for j in range(nel):
            Psi[index_along(Psi, j, -1)] = _backend.fft.ifftn(
                _backend.fft.fftn(Psi[index_along(Psi, j, -1)], axes=tuple(range(nk)))*TU, axes=tuple(range(nk)))
        Psi = dot_v(VU, Psi)

    return result
