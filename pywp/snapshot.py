
import numpy as np

class Snapshots:
    
    def __init__(self, data, box, dt):
        self.data = data
        self.box = box
        self.dt = dt

    def kdim(self):
        return len(self.data[0][0].shape)-1

    def eldim(self):
        return self.data[0][0].shape[0]

    def grid(self):
        return self.data[0][0].shape[1:]

    def get_R_grid(self):
        r = []
        for L, g in zip(self.box, self.data[0][0].shape[1:]):
            r.append(np.linspace(-L/2, L/2, g))
        return np.meshgrid(*r, indexing='ij')

    def get_P_grid(self):
        p = []
        for L, g in zip(self.box, self.data[0][0].shape[1:]):
            p.append(np.fft.fftshift(np.fft.fftfreq(g, L/(g-1)))*2*np.pi)
        return np.meshgrid(*p, indexing='ij')


def load_file(filename:str) -> Snapshots:
    
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
