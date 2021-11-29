import argparse
import sys
import numpy as np
from . import pywp
from . import potential

def main():


    parser = argparse.ArgumentParser('Wavepacket simulator')
    parser.add_argument('-L', '--L', help='box size', type=float, required=True)
    parser.add_argument('--Ly', help='box size',  type=float)
    parser.add_argument('-M', '--M', help='grid number', type=int, required=True)
    parser.add_argument('--My', help='grid number', type=int)
    parser.add_argument('--mass', help='mass', type=int, required=True)
    parser.add_argument('--init_x', help='init x', type=float, required=True)
    parser.add_argument('--init_y', help='init y', type=float, required=True)
    parser.add_argument('--sigma_x', help='sigma x', type=float, required=True)
    parser.add_argument('--sigma_y', help='sigma y', type=float)
    parser.add_argument('--init_px', help='init px', type=float, required=True)
    parser.add_argument('--init_py', help='init py', type=float)
    parser.add_argument('--init_s', help='init surface', type=int, required=True)
    parser.add_argument('--xwall_left', help='left boundary', type=float, default=-0.9)
    parser.add_argument('--xwall_right', help='right boundary', type=float, default=0.9)
    parser.add_argument('--potential', help='name of potential', default='test.Tully1')
    parser.add_argument('--potential_params', help='potential_params', default='')
    parser.add_argument('--Nstep', help='# step', type=int, required=True)
    parser.add_argument('--dt', help='time step', type=float, required=True)
    parser.add_argument('--output_step', help='output step', type=int, default=1000)
    parser.add_argument('--checkend', help='check end', default=True, type=lambda x:x.lower() == 'true', nargs='?', const=True)
    parser.add_argument('--rtol', help='checkend rtol', type=float, default=0.05)
    parser.add_argument('--traj', help='trajectory filename', default='')
    parser.add_argument('--output', help='output level', type=int, default=1)
    parser.add_argument('--gpu', help='using gpu', default=False, type=lambda x:x.lower() == 'true', nargs='?', const=True)

    args = parser.parse_args(sys.argv[1:])
    
    partitioner = [lambda x: 1, lambda x:x[0] >= 0, lambda x:x[0] < 0]
    boundary = lambda x: np.logical_and(x[0] > args.xwall_left*args.L/2, x[0] < args.xwall_right*args.L/2)
    trajfile = open(args.traj, 'w') if args.traj else None

    pottype = potential.get_potential(args.potential)
    if pottype:
        pot = pottype(*[float(v) for v in args.potential_params.split(',')]) if args.potential_params else pottype()
    else:
        print('Potential %s not found' % args.potential)

    preproc_args = pywp.preprocess(
        pot,
        [args.M, args.My] if args.My is not None else args.M,
        [args.L, args.Ly] if args.Ly is not None else args.L,
        [args.sigma_x, args.sigma_y] if args.sigma_y is not None else [args.sigma_x],
        [args.init_x, args.init_y] if args.init_y is not None else [args.init_x],
        [args.init_px, args.init_py] if args.init_py is not None else [args.init_px],
        args.init_s,
        args.mass,
        args.dt,
    )

    result = pywp.propagate(
        *preproc_args,
        nstep=args.Nstep,
        output_step=args.output_step,
        partitioner=partitioner,
        partition_titles=['P', 'T', 'R'],
        trajfile=trajfile,
        checkend=args.checkend,
        boundary=boundary,
        verbose=args.output,
        cuda_backend=args.gpu,
        checkend_rtol=args.rtol,
    )

    if trajfile:
        trajfile.close()

        # TODO Ly My
        with open(args.traj + '.meta', 'w') as f:
            f.write('-L %f -Ly %f -N %d -Ny %d -n %d -dt %f -step %d' % (
                args.L, args.Ly if args.Ly else args.L, 
                args.M, args.My if args.My else args.M,
                pot.get_dim()*2, args.dt*args.output_step,
                len(result)))

main()
