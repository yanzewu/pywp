
import numpy as np

class Grid:

    def __init__(self, box:list, ngrid):
        """ Construct a N-dimensional grid.
        box: List[Tuple[float]|float]. Specifying the box size in each dimension.
            If it's a number then the value is fixed.
        ngrid: List[int] or int. grid number in each dimension **where box[n] is a tuple**.
        """
        self.box = box
        self.is_griding_direction = [not isinstance(b, (float, int)) for b in self.box]
        if isinstance(ngrid, int):
            self.ngrid = [ngrid] * len(self.box)
        else:
            self.ngrid = ngrid

    def build(self, return_scalar=True):
        """ build the grid using np.linspace() and np.meshgrid(). The order will be "ij".
        If return_scalar is specified, only returns a number for nongriding directions.
        """
        inputs = []
        for j in range(len(self.box)):
            if self.is_griding_direction[j]:
                inputs.append(np.linspace(self.box[j][0], self.box[j][1], self.ngrid[j]))
            else:
                inputs.append(np.array([self.box[j]]))
        
        outputs = np.meshgrid(*inputs, indexing='ij')
        if return_scalar:
            return [outputs[j] if self.is_griding_direction[j] else self.box[j] for j in range(len(self.box))]
        else:
            return outputs

    def dx(self, return_all=False):
        """ Return the grid size. 
        If return_all is set, will return `None' for non-griding directions. Otherwise will only return griding directions.
        """
        if return_all:
            return [
                (self.box[j][1] - self.box[j][0])/self.ngrid[j] if self.is_griding_direction[j] else None
                for j in range(len(self.box))]
        else:
            return [
                (self.box[j][1] - self.box[j][0])/self.ngrid[j] for j in range(len(self.box)) if self.is_griding_direction[j]]

    def extent(self, return_all=False):
        """ Return the boundary.
        If return_all is set, will return all directions. Otherwise only griding directions are returned.
        """
        ext = []
        if return_all:
            for j in range(len(self.box)):
                ext += [self.box[j][0], self.box[j][1]] if self.is_griding_direction[j] else [self.box[j], self.box[j]]                    
        else:
            for j in range(len(self.box)):
                if self.is_griding_direction[j]:
                    ext += [self.box[j][0], self.box[j][1]]
        return ext


def adiabatic_surface(Hel:np.ndarray):
    """ Get adiabatic energies.
    Hel: (... x 2 x 2) array.
    Returns: (... x 2) array.
    """
    return np.linalg.eigvalsh(Hel)


def adiabat(Hel:np.ndarray):
    """ Get adiabatic energies and states.
    Hel: (... x 2 x 2) array.
    Returns: 
        E: (... x 2) array.
        U: (... x 2 x 2) array.
    """
    return np.linalg.eigh(Hel)


def gradient(Hel:np.ndarray, dx, direction='all'):
    """ A simple wrapper around np.gradient().
    Hel: (... x 2 x 2) array.
    dx: float/list[float]. The grid size on each direction.
    direction: 'all' or list[int] specifying the directions.

    Returns: list[ndarray], each being the gradient on one direction.
    """
    if direction == 'all':
        direction = tuple(range(Hel.ndim-2))

    if isinstance(dx, (int, float)):
        dx = [dx] * len(direction)

    deltaH = []

    for d, dx_ in zip(direction, dx):
        deltaH.append(
            np.gradient(Hel, dx_, axis=d)
        )

    return deltaH


def drv_coupling_hf(deltaH:list, E:np.ndarray, U:np.ndarray):
    """ Calculate derivative coupling by Hellmann-Feynmann theory.
    deltaH: list[ndarray]. Gradient of Hamiltonian (in one or more directions).
    E, U: Energy and adiabatic states.

    Returns: list[ndarray], each being the derivative coupling on one direction.
    """

    drv_coupling = []
    invE = np.empty(U.shape)
    for j in range(E.shape[-1]):
        for k in range(E.shape[-1]):
            if j == k:
                invE[..., j, k].fill(0)
            else:
                invE[..., j, k] = 1/(E[..., k] - E[..., j])

    for dh in deltaH:
        dc = np.empty(dh.shape, dtype=complex)
        for index in np.ndindex(U.shape[:-2]):
            dc[index] = (U[index].T @ dh[index] @ U[index])*invE[index]
        drv_coupling.append(dc)

    return drv_coupling


def expm_batch(M:np.ndarray, dt:float):
    """ Calculating exp(-1j*M*dt).
    """
    D, U = np.linalg.eigh(M)
    DD = np.zeros_like(U)
    for j in range(D.shape[-1]):
        DD[..., j, j] = np.exp(-1j*dt*D[..., j])

    nk = len(U.shape)-2
    tp_index = list(range(nk)) + [nk+1, nk]
    return U @ DD @ np.conj(np.transpose(U, tp_index))
    

def project_wavefunction(psi:np.ndarray, U:np.ndarray, inverse=False):
    if inverse:
        return (psi[..., None, :] @ np.conj(U))[..., 0, :]
    else:
        return (U @ psi[..., None])[..., 0]

