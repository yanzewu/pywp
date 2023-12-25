
import numpy as np
from . import util

class Snapshots:
    
    def __init__(self, data, box, dt):
        self.data = data
        self.box = box
        self.dt = dt

    def kdim(self) -> int:
        return self.data[0][0].ndim-1

    def eldim(self) -> int:
        return self.data[0][0].shape[0]

    def get_grid(self):
        """ Return a grid instance.
        """
        return util.Grid([(-b/2, b/2) for b in self.box], [self.data[0][0].shape[n+1] for n in range(self.kdim())])

    def grid(self) -> tuple:
        """ Return a list[int] specifing grids size on each kinetic dimension.
        """
        return self.data[0][0].shape[1:]

    def get_R_grid(self):
        """ Return a list[array] as position grids (created by meshgrid())
        """
        r = []
        for L, g in zip(self.box, self.data[0][0].shape[1:]):
            r.append(np.linspace(-L/2, L/2, g))
        return np.meshgrid(*r, indexing='ij')

    def get_P_grid(self):
        """ Return a list[array] as momentum grids (created by meshgrid())
        """
        p = []
        for L, g in zip(self.box, self.data[0][0].shape[1:]):
            p.append(np.fft.fftshift(np.fft.fftfreq(g, L/(g-1)))*2*np.pi)
        return np.meshgrid(*p, indexing='ij')

    def get_snapshot(self, index:int, order:str='k', momentum:bool=True, time:bool=False):
        """ Get a specific snapshot at index.
        index: integer less than len(Snapshots).
        order: e/k.
            k: Put kinetic dimensions at the beginning, shape will be like (nk1, nk2, ..., nel)
            e: Put electronic dimensions at the beginning, shape will be like (nel, nk1, nk2, ...)
        momentum: Whether momentum is returned.
        time: Whether time is returned.
        
        Returns:
            If momentum is set, will return (psiR, psiP), otherwise returns psiR only.
            If time is set, will return (..., index*dt)
        """
        psiR, psiP = self.data[index]

        if order == 'k':
            tp_index = list(range(1, self.kdim()+1)) + [0]
            psiR = np.transpose(psiR, tp_index)
            if momentum:
                psiP = np.transpose(psiP, tp_index)

        if momentum:
            if time:
                return psiR, psiP, self.dt * (index % len(self.data))
            else:
                return psiR, psiP
        else:
            if time:
                return psiR, self.dt*index * (index % len(self.data))
            else:
                return psiR

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        return self.get_snapshot(index, momentum=False)

class SnapshotWriter:

    def __init__(self, filename:str):
        self.filename = filename
        self.file = None
        self.record_count = 0

    def __call__(self, para, checkpoint):
        if not self.file:
            self.file = open(self.filename, 'wb')
            
        for j in range(checkpoint.psiR.shape[-1]):
            checkpoint.psiR[...,j].tofile(self.file)

        for psip in checkpoint.psiK:
            checkpoint.backend.fft.fftshift(psip).tofile(self.file)

        self.record_count += 1

        if self.record_count == 1:
            self.t_last = checkpoint.time
            self.dt = 1
            self.box = [(r.flat[-1] - r.flat[0])/2 for r in para.R]
            self.grid = para.R[0].shape
            self.nel = checkpoint.psiR.shape[-1]
        elif self.record_count == 2:
            self.dt = checkpoint.time - self.t_last

    def close(self):
        if self.file:
            self.file.close()
            with open(self.filename + '.meta', 'w') as f:
                f.write('-L %s -N %s -n %d -dt %f -step %d' % (
                    ','.join((str(x) for x in self.box)),
                    ','.join((str(x) for x in self.grid)),
                    self.nel * 2, self.dt, self.record_count))


def load_file(filename:str) -> Snapshots:
    """ Load a snapshot file with "filename.meta" as metadata.
    Returns a Snapshots instance.
    """
    with open(filename + '.meta', 'r') as metaf:
        metadata = metaf.readline().split()

    for j in range(len(metadata)//2):
        opt = metadata[2*j][1:]
        arg = metadata[2*j+1]

        if opt == 'L':
            box = [float(x) for x in arg.split(',')]
        elif opt == 'Ly':
            box.append(float(arg))
        elif opt == 'N':
            grid = [int(x) for x in arg.split(',')]
        elif opt == 'Ny':
            grid.append(int(arg))
        elif opt == 'n':
            nel = int(arg)//2   # n recorded is total number
        elif opt == 'dt':
            dt = float(arg)
        elif opt == 'step':
            nsnapshot = int(arg)
        
    data = load_file_raw(filename, grid, nel, nsnapshot)
    return Snapshots(data, box, dt)


def load_file_raw(filename:str, grid:list, nel:int, nsnapshot:int):
    """ Load a bindary trajectory file.
    grid: list of grid number in each dimension;
    nel: electronic state number;
    nsnapshot: number of snapshots in the file.

    Returns: list[array(nel x grid1 x ... gridn), array(nel x grid1 ...)], 
        each corresponding to a snapshot (in position space & k space).
    """
    sz_nu = np.prod(grid)
    rawf = np.fromfile(filename, complex, count=sz_nu*nel*2*nsnapshot)
    assert rawf.shape[0] >= sz_nu*nel*2*nsnapshot, 'Loading error: No enough snapshots stored'
    data = []
    for j in range(nsnapshot):
        data.append((
            np.reshape(rawf[2*j*nel*sz_nu:(2*j+1)*nel*sz_nu], [nel] + grid),
            np.reshape(rawf[(2*j+1)*nel*sz_nu:(2*j+2)*nel*sz_nu], [nel] + grid),
            ))

    return data


def transform_r_to_p(Psi:np.ndarray, has_electronic=True) -> np.ndarray:
    if has_electronic:
        return np.array([  
            np.fft.fftshift(np.fft.fftn(Psi[j]))
            for j in range(Psi.shape[0])])
    else:
        return np.fft.fftshift(np.fft.fftn(Psi))
