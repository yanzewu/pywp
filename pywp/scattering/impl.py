
import numpy as np

from typing import Union, Tuple, Callable
import dataclasses
import warnings

from ..util import Grid
from ..potential import Potential
from ..fd import drv_kernel, drv_matrix

from .itersolv import itersolv

@dataclasses.dataclass(eq=False)
class Channel:
    edge: int                                   # 0/-1
    edge_range: tuple                           # (start:stop:step) index of connector [inner, outer)
    offset: int = dataclasses.field(default=0)  # relative offset of the Hamiltonian
    size: int = dataclasses.field(default=0)    # number of available electronic states
    wavevector: np.ndarray = dataclasses.field(default=None)    # k of each electronic channel (in absolute value)
    basis: np.ndarray = dataclasses.field(default=None)     # basis of each electronic channel, projected to diabatic basis


def scatter1d(potential:Union[Potential,np.ndarray,Callable[[list[np.ndarray]],np.ndarray]], grid:Grid, M:float, KE:float, 
              incoming_state:Union[float,int,np.ndarray], incoming_side:str='left', 
              *, drvcoupling:Union[None,np.ndarray,Callable[[list[np.ndarray]],np.ndarray]]=None, side:str='both', KE_accuracy:int=2, 
              adiabatic_boundary:bool=False, grid_warn_fraction:int=4, itersolver=None, itersolve_kwargs={}) -> Tuple[dict, np.ndarray]:
    """ Solves scattering problem in 1D.

    potential: An (Ngrid x Nel x Nel) numpy array, or an `Potential` instance.
    grid: A grid instance.
    M: mass (au).
    KE: The incoming kinetic energy (au).
    incoming_state: An integer or an amplitude array with size = (Nel,). Always on DIABATIC basis.
    incoming_side: 'left'/'right'.
    side: 'both'/'left'/'right'. Specify the opening sides for the system.
    KE_accuracy: An positve integer. Specify the stencil accuracy (see `fd.drv_kernel()` for details).
    adiabatic_boundary: Treat the off-diagonal coupling as finite constant through infinity. If set, will use 
        $|n_A> \otimes e^{ik_A R} $ (where $|n_A>$ are adiabatic states) as basis, and will use off-diagonal elements 
        to calculate energies. Otherwise, only diagonal elements of the Hamiltonian is used.
    grid_warn_fraction: How many grids span a period of wave, before triggers the warning. The smaller, more frequent the warning.
    itersolver: Using iterative solver. By default applies to matrix size > 1000.
    itersolve_kwargs: Additional arguments passed to itersolver.
        - maxiter: Maximum iteration (defaults to 500);
        - atol, rtol: Tolerence (defaults to 1e-8);
        - x0: Initial guess of solution;
    
    Returns: {side: probability}, wavefunction, (optional) {side: basis}
        side: 'left' or 'right'.
        probability: (Nel,) array of probability appearing on the side. The sum over all probability and all sides equal to 1. If 
            adiabatic_boundary=True, will be probabilities on adiabatic basis.
        wavefunction: (N x Nel) numpy array of the wavefunction obtained, represented on diabatic basis.
        basis: (only when adiabatic_boundary=True) (Nel x Nel) array of <diabat|adiabat>. NOTE only energetic
            accessible parts are provided.
    """
    
    assert grid.ndim == 1, "Only 1D grid is allowed"
    assert KE_accuracy >= 1

    R = grid.build()[0]
    dx = R[1] - R[0]
    
    if isinstance(potential, Potential):
        assert potential.get_kdim() == grid.ndim, "Dimension of potential does not match dimension of grid"
        V = potential.get_H((R,))
    elif not callable(potential):
        V = potential.copy()
    else:
        V = potential([R])

    assert V.ndim == 3 and V.shape[0] == R.shape[0] and V.shape[1] == V.shape[2], "Invalid dimension of potential"

    if drvcoupling is not None:
        if not callable(drvcoupling):
            D = drvcoupling.copy()
        else:
            D = drvcoupling([R])

        assert D.ndim == 3 and D.shape[0] == D.shape[0] and D.shape[1] == D.shape[2] and D.shape[1] == V.shape[1], "Invalid dimension of drvcoupling"
    else:
        D = None

    real_valued = D is None and V.dtype == float

    nel = V.shape[1]
    N = len(R)

    # parse channel flags
    if side == 'both':
        channels = {'left': Channel(0, (KE_accuracy-1, None, -1) ),
                    'right': Channel(N-1, (N-KE_accuracy, N)),
                    }
        
    elif side == 'left':
        channels = {
            'left': Channel(0, (KE_accuracy-1, None, -1) )
        }
    elif side == 'right':
        channels = {
            'right': Channel(N-1, (N-KE_accuracy, N)),
        }
    else:
        raise ValueError('Invalid side', side)

    # initial state. len(ampl_inc) is always nel (i.e., unfiltered)
    if isinstance(incoming_state, int):
        ampl_inc = np.zeros(nel)
        ampl_inc[incoming_state] = 1
    else:
        assert len(incoming_state) == nel
        ampl_inc = np.array(incoming_state)

    incoming_channel = channels[incoming_side]
    
    if adiabatic_boundary:
        Etot = np.vdot(ampl_inc, V[incoming_channel.edge] @ ampl_inc) + KE
    else:
        Etot = np.vdot(ampl_inc, np.real(np.diag(V[incoming_channel.edge])) * ampl_inc) + KE

    # Set the filters
    offset = 0
    for c in channels.values():
        if adiabatic_boundary:
            E_edge, U_edge = np.linalg.eigh(V[c.edge])
            KE_local = Etot - E_edge
        else:
            KE_local = Etot - np.real(np.diag(V[c.edge]))
            U_edge = np.eye(nel, nel)

        c.offset = offset
        c.basis = U_edge[:, KE_local > 0]
        c.wavevector = np.sqrt(2*M*KE_local[KE_local > 0])  # absolute value!
        c.size = len(c.wavevector)

        if np.any(c.wavevector * dx > 2*np.pi / grid_warn_fraction):
            warnings.warn(f'Warning: Grid too big: Phase oscillation factor is {np.max(c.wavevector * dx / 2 / np.pi)}, requires {1/grid_warn_fraction}')

        offset += c.basis.shape[1]

    offset_wf = offset
    
    # fill H
    # ========== cols of H ==========
    # | [channel 0] nel0, nel1, ... , len(channel0.wavevector) | (@channel1.offset) [channel 1] nel0, ... | (@offset_wf) [x]


    H_bra_grid = np.zeros((offset_wf, nel, N), dtype=complex)
    H_grid_ket = np.zeros((nel, N, offset_wf), dtype=complex)
    H_bra_ket = np.zeros((offset_wf, offset_wf), dtype=complex)
    H_grid_grid = [[np.zeros((N, N), dtype=float if real_valued else complex) for k in range(nel)] for j in range(nel)]
    # H_grid_grid is special: in this way we can remove it easily from memory.
    
    T_mat = -drv_matrix(N, 2, accuracy=KE_accuracy) / (2*M*dx**2)
    T_kernel = -drv_kernel(2, KE_accuracy) / (2*M*dx**2)
    if D is not None:
        halfv_mat = -drv_matrix(N, 1, accuracy=KE_accuracy) / (M * dx * 2)
        D2 = D @ D / (2*M)

    for j in range(nel):
        for k in range(nel):
            H_grid_grid[j][k].flat[::N+1] = V[:,j,k] # equivalent to fill diagonal
            if D is not None:
                djk_v = D[:,j,k][:,None] * halfv_mat # equivalent to diag(D) @ v_mat
                H_grid_grid[j][k] += djk_v - djk_v.T
                H_grid_grid[j][k].flat[::N+1] -= D2[:,j,k]
                
        H_grid_grid[j][j] += T_mat

    # sides
    x_segment = np.arange(KE_accuracy+1) * dx   # the basis of incoming/outgoing wave. There is only one of them corresponding
                                                # to the right side: The basis of the left side is flipped.
    
    # T_mat_edge = -drv_matrix(2*KE_accuracy+1, 2, accuracy=KE_accuracy) / (2*M*dx**2)
    # For left channel, the grid basis are -a,..0,..a, so the actual bra begins at halfway [0,a] but the ket is [-a,0]
    # Right channel is similar.

    for c in channels.values():

        # potential
        Veff = V[c.edge] if adiabatic_boundary else np.diag(np.diag(V[c.edge]))
        H_bra_ket[c.offset:c.offset+c.size, c.offset:c.offset+c.size] += c.basis.T.conj() @ Veff @ c.basis

        # kinetic
        for j in range(c.size):
            ket_edge = np.exp(1j*c.wavevector[j]*x_segment)
            H_ket_edge = np.convolve(T_kernel, ket_edge, 'full')[:KE_accuracy+1]
            
            # This applies to *both* left and right edge:
            # For right edge, the edge_range is [N-a, N) so work as expected.
            # For left edge, the edge range is [a-1, -1) with stride -1.
            H_grid_ket[:, slice(*c.edge_range), c.offset+j] += np.outer(c.basis[:,j], H_ket_edge[:-1])
            H_bra_ket[c.offset+j, c.offset+j] += H_ket_edge[-1]
            H_bra_grid[c.offset+j, :, slice(*c.edge_range)] += np.outer(c.basis[:,j].conj(), T_kernel[:KE_accuracy])
            
    # rhs
    psi_inc_bra = np.zeros(offset_wf, dtype=complex)
    psi_inc_grid = np.zeros((nel, N), dtype=complex)

    c = incoming_channel
    incoming_wavevector = np.sqrt(2*M*KE)

    ket_edge = np.exp(-1j*incoming_wavevector*x_segment) # NOTE the direction is reversed. Also ket_edge is a constant here.
    H_ket_edge = np.convolve(T_kernel, ket_edge, 'full')[:KE_accuracy+1]
    
    ampl_inc_proj = c.basis.T.conj() @ ampl_inc
    Veff = V[c.edge] if adiabatic_boundary else np.diag(np.diag(V[c.edge]))
    psi_inc_bra[c.offset:c.offset+c.size] = c.basis.T.conj() @ Veff @ ampl_inc + H_ket_edge[-1] * ampl_inc_proj
    psi_inc_grid[:, slice(*c.edge_range)] =  np.outer(ampl_inc, H_ket_edge[:-1])
        
    # build the matrices
    H = np.empty((offset_wf + N*nel, offset_wf + N*nel), dtype=complex)
    
    # in this way we make use of COW. note we have to use reverse order because of del
    for j in range(nel-1, -1, -1):
        for k in range(nel-1, -1, -1):
            H[offset_wf+j*N:offset_wf+(j+1)*N, offset_wf+k*N:offset_wf+(k+1)*N] = H_grid_grid[j][k]
            del H_grid_grid[j][k]

    H[:offset_wf, :offset_wf] = H_bra_ket
    H[:offset_wf, offset_wf:] = H_bra_grid.reshape(offset_wf, N*nel)
    H[offset_wf:, :offset_wf] = H_grid_ket.reshape(N*nel, offset_wf)
    H.flat[np.arange(0, H.size, len(H)+1)] -= Etot

    psi_inc = np.concatenate((psi_inc_bra, psi_inc_grid.flatten()))
    psi_inc[incoming_channel.offset:incoming_channel.offset+incoming_channel.size] -= Etot * ampl_inc_proj

    if itersolver is None:
        itersolver = H.shape[0] > 1000

    if not itersolver:
        psi = -np.linalg.solve(H, psi_inc)

    else:
        H_kernel = [H[:offset_wf, :offset_wf]]
        for j in range(nel):
            H_kernel.append(H[offset_wf+j*N:offset_wf+(j+1)*N,offset_wf+j*N:offset_wf+(j+1)*N])
        
        psi = -itersolv(H, psi_inc, H_kernel, adapt_scattering=True, **itersolve_kwargs)

    prob = {}
    basis = {}
    for name, c in channels.items():
        if adiabatic_boundary:
            prob[name] = np.zeros(nel)
            prob[name][:c.size] = np.abs(psi[c.offset:c.offset+c.size])**2 * c.wavevector / incoming_wavevector
            basis[name] = np.zeros((nel, nel))
            basis[name][:, :c.size] = c.basis
        else:
            # basis is a permutation
            prob[name] = np.abs(c.basis)**2 @ (np.abs(psi[c.offset:c.offset+c.size])**2 * c.wavevector / incoming_wavevector)

    if adiabatic_boundary:
        return prob, psi[offset_wf:].reshape(nel, N).T, basis
    else:
        return prob, psi[offset_wf:].reshape(nel, N).T
    

