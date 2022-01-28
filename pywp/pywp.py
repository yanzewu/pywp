
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


class PhysicalParameter:
    """ Represents the physical parameters that to use.
    """
    def __init__(self, Psi, H, KE, TU, VU, VUhalf, R, K, dA:float, dK:float, dt:float):
        self.Psi = Psi
        self.H = H
        self.KE = KE
        self.TU = TU
        self.VU = VU
        self.VUhalf = VUhalf
        self.R = R
        self.K = K
        self.dA = dA
        self.dK = dK
        self.dt = dt


def preprocess(potential:Potential, N, L, sigma, R0, P0, n0:int, M:float, dt:float) -> PhysicalParameter:
    """ Produce things needed for propatate(). Only two electronic states (potential.dim() == 2) are supported currently.

    potential: Class compatible with potential.Potential.
    N: grid. Either a number, or list of numbers, where len(N) must = potential.get_kdim().
    L: box length. Number/list of numbers. The allocated grid will be [-L/2, L/2].
    sigma: array-like. length equals to potential.get_kdim(). The initial wavepacket amplitude will be exp(-R^2/sigma^2)
    R0/P0: array-like. length equals to potential.get_kdim().
    n0: int. Initial surface.
    M: mass.
    dt: timestep.
    """
    # NOTE currently we only support 2 states

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

    if nel == 2 and potential.has_get_phase():

        dE = np.real(H[index_along2(H, 1, -2, 1, -1)] - H[index_along2(H, 0, -2, 0, -1)])/2
        Eave = np.real(H[index_along2(H, 1, -2, 1, -1)] + H[index_along2(H, 0, -2, 0, -1)])/2
        delta = np.sqrt(dE**2 + abs2(H[index_along2(H, 0, -2, 1, -1)]))

        cos_half_theta = np.sqrt(0.5*(1 + dE/delta))
        sin_half_theta = np.sqrt(0.5*(1 - dE/delta))

        Phi = potential.get_phase(R)

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

    else:   # fall back to expm
        from scipy.linalg import expm

        VU = np.zeros(H.shape, dtype=complex)
        VUhalf = np.zeros(H.shape, dtype=complex)

        for idx in np.ndindex(*N):
            VU[idx] = expm(-1j*dt*H[idx])
            VUhalf[idx] = expm(-1j*dt/2*H[idx])

    return PhysicalParameter(Psi, H, KE, TU, VU, VUhalf, R, K, dA, dK, dt)


def propagate(para:PhysicalParameter, nstep:int, output_step:int, partitioner=None, partition_titles=None, trajfile=None, checkend=False, boundary=None, checkend_rtol=0.05, verbose=True, cuda_backend=False) -> list:
    """ The actual propagating function.
    Args:
   
    - nstep: Maximum number of steps.
    - output_step: Step to output info and save result.
    - partitioner: None/list[list[array[float]]->array[bool]]. Each of the functions will be called by p(R), where R is position meshgrid, list with potential.dim() element of size N x N ...
        it should return a boolean map (with size N x N ...). If None, a unit partitioner is used (just a number 1).
    - partition_titles: list[str]. Titles for verbose output.
    - trajfile: None/file-like. Will write wavefunction (on both position and momentum basis) to the file, ordered by electronic state. 
        All positions first, followed by momentums.
    - checkend, checkend_rtol: If true, and sum(abs(boundary(R) * Psi)^2)) > 1 - checkend_rtol, then the simulation is terminated.
    - boundary: list[array[float]]->array[bool]. Return a bool map with shape equals to position grid where True means inside boundary. Invoked when checkend is true.
    - verbose: 
        - If False, remains silent.
        - If 1 or True, only print time (t), energy (Etot), kinetic energy (KE), total population (Pall), and population of each partition region on each state (P{title}n).
        - If 2, in addition to 1, print position and momentum of each partition region on each state.
    - cuda_backend: Uses cupy as backend instead of numpy.

    Returns:

    A list of [time, population, position, momentum ], each
        is array with shape nel x m ( x nk ), where m is determined by partitioner (None -> 1). The collection timestep is same as output_step.
    """

    Psi = para.Psi; H = para.H; KE = para.KE; TU = para.TU; VU = para.VU; VUhalf = para.VUhalf; R = para.R; K = para.K; dA = para.dA; dK = para.dK; dt = para.dt

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

        if nk == 1:
            dot_v = lambda v, p: _backend.einsum('ikl,il->ik', v, p)
        elif nk == 2:
            dot_v = lambda v, p: _backend.einsum('ijkl,ijl->ijk', v, p)
        elif nk == 3:
            dot_v = lambda v, p: _backend.einsum('ijzkl,ijzl->ijzk', v, p)
        else:
            def dot_v(v, p):
                p1 = np.zeros_like(p)
                for j in range(nel):
                    p1[index_along(p1, j, -1)] = np.sum(v[index_along(v, j, -2)] * p, axis=nk)
                return p1

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
                    if cuda_backend:
                        Rhoave.append(_backend.sum(abspsi).get() * dA)
                        Rave += [_backend.sum(abspsi*R_).get() * dA / Rhoave[-1] for R_ in R]
                        Pave += [_backend.sum(abspsip*K_).get() * dK / Rhoave[-1] for K_ in K]
                    else:
                        Rhoave.append(_backend.sum(abspsi) * dA)
                        Rave += [_backend.sum(abspsi*R_) * dA / Rhoave[-1] for R_ in R]
                        Pave += [_backend.sum(abspsip*K_) * dK / Rhoave[-1] for K_ in K]

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
