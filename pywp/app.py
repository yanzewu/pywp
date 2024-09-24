
import sys
import argparse
import numpy as np

from . import preprocess, propagate, potential
from .util import Grid

class Application:

    def __init__(self, pottype:type=None, potential:potential.Potential=None, partitioner=None, partition_titles=None, boundary=None, wp_generator=None, analyzer=None):
        """ pottype: class name of the potential.
            partitioner: list[list[array[float]]->array[bool]] boolean map of position meshgrids. 
            partition_titles: Names of each partition.
            boundary: list[array[float]]->array[bool] boolean map of position meshgrids.
            analyzer: list[(R:list[array], K:list[array], psi:array, cuda_backend:bool)->any]. 
                The analyzer will be called every output_step, and result will be stored in the returned variable.
        """
        if potential is not None:
            self.pot = potential
            self.pottype = type(potential)
        else:
            self.pottype = pottype
            self.pot = None

        if not partitioner:
            self.partitioner = [lambda _: 1, lambda x:x[0] >= 0, lambda x:x[0] < 0]
            self.partition_titles = ['P', 'T', 'R']
        else:
            self.partitioner = partitioner
            self.partition_titles = partition_titles

        self.wp_generator = wp_generator
        self.boundary = boundary
        self.analyzer = analyzer

    def parse_args(self, args=sys.argv[1:]):

        parsefloatlist = lambda s: [float(x) for x in s.split(',')]
        parseintlist = lambda s: [int(x) for x in s.split(',')]

        parser = argparse.ArgumentParser('Wavepacket simulator')
        parser.add_argument('-L', '--L', help='box size', type=float)
        parser.add_argument('--Ly', help='box size',  type=float)
        parser.add_argument('--box', help='box size multiple dimensional', type=parsefloatlist)
        parser.add_argument('-M', '--M', help='grid number', type=int)
        parser.add_argument('--My', help='grid number', type=int)
        parser.add_argument('--grid', help='grid number multiple dimensional', type=parseintlist)
        parser.add_argument('--mass', help='mass', type=int, required=True)
        parser.add_argument('--init_x', help='init x', type=float, required=True)
        parser.add_argument('--init_y', help='init y', type=float, required=True)
        parser.add_argument('--init_r', help='init position', type=parsefloatlist)
        parser.add_argument('--sigma_x', help='sigma x', type=float, required=True)
        parser.add_argument('--sigma_y', help='sigma y', type=float)
        parser.add_argument('--sigma', help='standard deviation of wavepacket', type=parsefloatlist)
        parser.add_argument('--init_px', help='init px', type=float, required=True)
        parser.add_argument('--init_py', help='init py', type=float)
        parser.add_argument('--init_p', help='init momentum', type=parsefloatlist)
        parser.add_argument('--init_s', help='init surface', type=int)
        parser.add_argument('--init_c', help='init amplitudes', type=parsefloatlist)
        parser.add_argument('--xwall_left', help='left boundary', type=float, default=-0.9)
        parser.add_argument('--xwall_right', help='right boundary', type=float, default=0.9)
        parser.add_argument('--potential_params', help='potential_params', default='')
        parser.add_argument('--Nstep', help='# step', type=int, required=True)
        parser.add_argument('--dt', help='time step', type=float, required=True)
        parser.add_argument('--output_step', help='output step', type=int, default=1000)
        parser.add_argument('--checkend', help='check end', default=True, type=lambda x:x.lower() == 'true', nargs='?', const=True)
        parser.add_argument('--rtol', help='checkend rtol', type=float, default=0.05)
        parser.add_argument('--traj', help='trajectory filename', default='')
        parser.add_argument('--output', help='output level', type=int, default=1)
        parser.add_argument('--gpu', help='using gpu', default=False, type=lambda x:x.lower() == 'true', nargs='?', const=True)

        if self.pottype is None:
            parser.add_argument('--potential', help='name of potential', default='test.Tully1')

        self.args = parser.parse_args(args)
        self.analyzer = None

    def set_args(self, box:list, grid:list, mass:float, init_r:list, init_p:list, sigma:list, init_c, Nstep:int, dt:float, potential_params:list=[],
        output_step:int=1000, checkend:bool=True, xwall_left:float=-0.9, xwall_right:float=0.9, rtol:float=0.05, output:int=1, traj:str='', gpu:bool=False):
        """ 
        Args:
            - box: `list[float]`. The simulation box. For example, \[L1,L2,L3] will make the actual box x=[-L1/2, L1/2], y=[-L2/2, L2/2], z=[-L3/3, L3/3].
            - grid: `list[int]`. The grid number in each dimension.
            - mass: `float`. Mass of the nuclei. Currently it's a scalar.
            - init_r: `list[float]`. Initial position of the wavepacket (the center of the gaussian).
            - init_p: `list[float]`. Initial momentum of the wavepacket.
            - sigma: `list[float]`. The initial standard deviation of the wavepacket. The wavefunction will be exp(-x^2/sigmax^2-y^2/sigmay^2...).
            - init_c: `list[float]` or `int`. Initial amplitude or surface label.
            - Nstep: `int`. Maximum number of step for simulation.
            - dt: `float`. Integration time interval.
            - potential_params: `list`. Arguments for the potential, will passed to the potential's `__init__()` as the optional argument list.
            - output_step: `int`. Interval between outputs, analysis and writing trajectory.
            - checkend: `int`. Whether using the checkend algorithm. Will terminate the simulation if the fraction of wavepacket that goes beyond xwall_left or xwall_right in the first dimension is greater than rtol. Default: True.
            - xwall_left: `float`. The relative value of the leftwall for checkend. The actual wall will be located at `box[0]/2*xwall_left`. Default: -0.9.
            - xwall_right: `float`. The relative value of the rightwall for checkend. The actual wall will be located at `box[0]/2*xwall_right`. Default: 0.9.
            - rtol: `float`. The relative tolerance for checkend. Default: 0.05.
            - output: `int`. The output level to the console. 0: no output; 1: output population; 2: output population, position and momentum.
            - traj: `str`. Filename of trajectory. If empty then will not write trajectory. Note the trajectory file size can grow dramatically when the nuclear degrees of freedom increases.
            - gpu: `bool`. True/False. PyWP depends on package cupy for GPU functions.
        """
        self.args = argparse.Namespace(box=box, grid=grid, mass=mass, init_r=init_r, init_p=init_p, sigma=sigma, init_c=init_c, Nstep=Nstep,
            dt=dt, potential_params=potential_params, output_step=output_step, checkend=checkend, xwall_left=xwall_left,
            xwall_right=xwall_right, rtol=rtol, output=output, traj=traj, gpu=gpu)

    def run(self):

        def get_multidim_arg(args, argname, argname1d, argname2d, dim):
            if getattr(args, argname) is not None:
                return getattr(args, argname)
            else:
                if dim == 1:
                    return [getattr(args, argname1d)]
                elif dim == 2:
                    arg2d = getattr(args, argname2d)
                    if arg2d:
                        return [getattr(args, argname1d), arg2d]
                    else:
                        return [getattr(args, argname1d), getattr(args, argname1d)]
                else:
                    raise RuntimeError('argument %s not found' % argname)

        args = self.args
        trajfile = open(args.traj, 'w') if args.traj else None


        if self.pot:
            pot = self.pot
        else:
            if self.pottype is None:
                self.pottype = potential.get_potential(args.potential)
            if isinstance(args.potential_params, str):
                if args.potential_params:
                    pot = self.pottype(*[float(v) for v in args.potential_params.split(',')])
                else:
                    pot = self.pottype()
            elif isinstance(args.potential_params, list):
                pot = self.pottype(*args.potential_params)

        box = get_multidim_arg(args, 'box', 'L', 'Ly', pot.get_kdim())
        grid = get_multidim_arg(args, 'grid', 'M', 'My', pot.get_kdim())
        sigma = get_multidim_arg(args, 'sigma', 'sigma_x', 'sigma_y', pot.get_kdim())
        init_r = get_multidim_arg(args, 'init_r', 'init_x', 'init_y', pot.get_kdim())
        init_p = get_multidim_arg(args, 'init_p', 'init_px', 'init_py', pot.get_kdim())
        init_c = getattr(args, 'init_c', None)
        if init_c is None:
            init_c = getattr(args, 'init_s', None)

        if not self.boundary:
            self.boundary = lambda x: np.logical_and(x[0] > args.xwall_left*box[0]/2, x[0] < args.xwall_right*box[0]/2)

        if self.wp_generator is None:
            self.wp_generator = lambda R: np.exp(sum([1j*p*R_ - (R_-r)**2/s**2 for (r, p, R_, s) in zip(init_r, init_p, R, sigma)]))

        if isinstance(grid, int):
            N = [grid]*pot.get_kdim()
        else:
            assert len(grid) == pot.get_kdim()
            N = grid

        if isinstance(box, (float, int)):
            L = [(-box/2, box/2)]*pot.get_kdim()
        else:
            assert len(box) == pot.get_kdim()
            L = [(-b/2, b/2) for b in box]

        
        preproc_args = preprocess(pot, Grid(L, N), self.wp_generator, init_c, args.mass, args.dt)

        result = propagate(
            preproc_args,
            nstep=args.Nstep,
            output_step=args.output_step,
            partitioner=self.partitioner,
            partition_titles=self.partition_titles,
            checkend=args.checkend,
            boundary=self.boundary,
            verbose=args.output,
            cuda_backend=args.gpu,
            checkend_rtol=args.rtol,
        )

        if trajfile:
            trajfile.close()

            if pot.get_kdim() == 2:
                with open(args.traj + '.meta', 'w') as f:
                    f.write('-L %f -Ly %f -N %d -Ny %d -n %d -dt %f -step %d' % (
                        box[0], box[1], grid[0], grid[1],
                        pot.get_dim()*2, args.dt*args.output_step,
                        len(result)))
            else:
                with open(args.traj + '.meta', 'w') as f:
                    f.write('-L %s -N %s -n %d -dt %f -step %d' % (
                        ','.join((str(x) for x in box)),
                        ','.join((str(x) for x in grid)),
                        pot.get_dim()*2, args.dt*args.output_step,
                        len(result)))

        return result
