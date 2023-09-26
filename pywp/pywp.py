
import numpy as np 
import itertools
from .potential import Potential
from .util import expm_batch, Grid
from typing import Union, List

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


def preprocess(potential:Union[Potential, np.ndarray], grid:Grid, wavepacket, c0:Union[int,float,complex], M:float, dt:float) -> PhysicalParameter:
    """ Produce things needed for propatate(). Only two electronic states (potential.dim() == 2) are supported currently.

    potential: potential.Potential or a (N1 x N2 x ... x Nel) numpy array.
    grid: A grid instance specifying the actual grid.
    wavepacket: A (N1 x N2 x ... x Nel) numpy array, or a function that returns such array by taking position (with size 
        List[array(N1 x ...)]) as the argument. Normalization is not required.
    c0: int/List[float]. Initial surface or amplitude.
    M: mass.
    dt: timestep.
    """

    if isinstance(potential, np.ndarray):
        nel = potential.shape[-1]
        nk = potential.ndim - 2
        H = potential
        assert H.shape[-2] == H.shape[-1]
        assert H.shape[:-2] == grid.shape
    else:
        nel = potential.get_dim()
        nk = potential.get_kdim()
        assert nk == grid.ndim

    # build K grid
    r = grid.build_individual()
    k = []
    dA = 1.0
    dK = 1.0
    for j in range(nk):
        dr = r[j][1] - r[j][0]
        k.append(np.fft.fftfreq(grid.shape[j], dr) * 2* np.pi)
        dA *= dr
        dK *= (r[j][-1] - r[j][0]) / (grid.shape[j] - 1) / grid.shape[j]

    R = np.meshgrid(*r, indexing='ij')
    K = np.meshgrid(*k, indexing='ij')
    
    # build wavefunction
    Psi = np.zeros(grid.shape + (nel,), dtype=complex)
    
    if isinstance(wavepacket, np.ndarray):
        psi0 = wavepacket
    elif callable(wavepacket):
        psi0 = wavepacket(R)
    else:
        raise ValueError("wavepacket: Either an array or a callable return an array")

    if isinstance(c0, int):
        Psi[..., c0] = psi0
    else:
        c = np.array(c0)
        c /= np.linalg.norm(c)
        for j in range(len(c)):
            Psi[..., j] = psi0 * c[j]

    Psi /= np.sqrt(np.sum(abs2(Psi))* dA)

    # build H 
    if not isinstance(potential, np.ndarray):
        H = potential.get_H(R)

    assert H.shape == tuple(grid.shape + (nel, nel))
    assert Psi.shape == tuple(grid.shape + (nel,))

    # build exponential of operators
    KE = sum((K_**2 for K_ in K))/2/M
    TU = np.exp(-1j*dt*KE)
    VU = expm_batch(H, dt)
    VUhalf = expm_batch(H, dt/2)

    return PhysicalParameter(Psi, H, KE, TU, VU, VUhalf, R, K, dA, dK, dt)


def propagate(para:PhysicalParameter, nstep:int, output_step:int, partitioner=None, partition_titles=None, analyzer=None, trajfile=None, checkend=False, 
              boundary=None, checkend_rtol=0.05, verbose=True, cuda_backend=False, extra_normalize:bool=True) -> list:
    """ The actual propagating function.
    Args:
   
    - nstep: Maximum number of steps.
    - output_step: Step to output info and save result.
    - partitioner: None/list[list[array[float]]->array[bool]]. Each of the functions will be called by p(R), where R is position meshgrid, list with potential.dim() element of size N x N ...
        it should return a boolean map (with size N x N ...). If None, a unit partitioner is used (just a number 1).
    - partition_titles: list[str]. Titles for verbose output.
    - analyzer: None/list[(R:list[array], K:list[array], psi:array, cuda_backend:bool)->any]. The analyzer will be called every output_step, and result will be stored in the returned variable.
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

    A list of [time, population, position, momentum, extra ], each (except extra)
        is array with shape nel x m ( x nk ), where m is determined by partitioner (None -> 1). The collection timestep is same as output_step.
    extra is a list containing the return of each analyzer.
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
            if i != 0 and extra_normalize:
                Psi /= (_backend.sum(abs2(Psi))* dA)**0.5

            Psi_output = dot_v(VUhalfinv, Psi)
            Psip = [_backend.fft.fftn(Psi_output[index_along(Psi, j, -1)], axes=tuple(range(nk))) for j in range(nel)]

            Rhoave = []
            Rave = []
            Pave = []
            Extra = []

            for j in range(nel):
                for pf in partition_filter:
                    abspsi = abs2(Psi_output[index_along(Psi_output, j, -1)]) * pf
                    
                    abspsip = abs2(_backend.fft.fftn(Psi_output[index_along(Psi_output, j, -1)] * pf, axes=tuple(range(nk))))
                    if cuda_backend:
                        Rhoave.append(_backend.sum(abspsi).get() * dA)
                        Rave += [_backend.sum(abspsi*R_).get() * dA / Rhoave[-1] for R_ in R]
                        Pave += [_backend.sum(abspsip*K_).get() * dK / Rhoave[-1] for K_ in K]
                    else:
                        Rhoave.append(_backend.sum(abspsi) * dA)
                        Rave += [_backend.sum(abspsi*R_) * dA / Rhoave[-1] for R_ in R]
                        Pave += [_backend.sum(abspsip*K_) * dK / Rhoave[-1] for K_ in K]

            if analyzer:
                for a in analyzer:
                    Extra.append(a(R, K, Psi_output, cuda_backend))

            Rhotot = _backend.sum(abs2(Psi_output)) * dA
            KEave = sum((_backend.sum(abs2(Psip_) * KE) * dK for Psip_ in Psip))
            Eave = KEave + _backend.sum(_backend.real(_backend.conj(Psi_output) * dot_v(H, Psi_output))) * dA

            result.append((i*dt, 
                np.array(Rhoave).reshape((nel, len(partition_filter))), 
                np.array(Rave).reshape((nel, len(partition_filter), nk)), 
                np.array(Pave).reshape((nel, len(partition_filter), nk)),
                Extra,
                ))

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

            if checkend and _backend.sum((boundary_filter * _backend.sum(abs2(Psi), axis=-1))) < (1 - checkend_rtol)*_backend.sum(abs2(Psi)):
                break

        for j in range(nel):
            Psi[index_along(Psi, j, -1)] = _backend.fft.ifftn(
                _backend.fft.fftn(Psi[index_along(Psi, j, -1)], axes=tuple(range(nk)))*TU, axes=tuple(range(nk)))
        Psi = dot_v(VU, Psi)

    return result
