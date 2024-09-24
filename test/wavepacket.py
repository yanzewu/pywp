import numpy as np
import sys

sys.path.append('..')

import pywp

grid = pywp.mgrid[-10:10:0.05]
P0 = 20

# Tully1
def tully1(R):
    return pywp.build_pe_tensor(
        0.01*(1 - np.exp(-1.6*R[0])) * (R[0] >= 0) - 0.01*(1 - np.exp(1.6*R[0])) * (R[0] < 0),
        0.005*np.exp(-R[0]**2),
        symmetric=True
    )

# Run
pp = pywp.preprocess(tully1, grid, wavepacket=lambda R: np.exp(-(R[0]+5)**2 + 1j*P0*R[0]), c0=0, M=2000, dt=0.1)
displayer = pywp.visualize.WavepacketDisplayer1D()
writer = pywp.snapshot.SnapshotWriter('traj-tully1-p20.trj')
populations = []
pop_trans = pywp.expec.PositionFunc(lambda R: R[0] > 0, normalization='none')

pywp.propagate(pp, nstep=12000, output_step=1000, 
                         on_output=[
                             lambda para, cp: populations.append(pop_trans(para, cp)),
                             displayer,
                             writer
                         ],
                         checkend=True, 
                         boundary=lambda R: R[0] < 9,
                        )
writer.close()


print('Transmitted population: ', populations[-1])

# Post-processing
snapshots = pywp.load_file('traj-tully1-p20.trj')
R = snapshots.get_R_grid()
for j in range(len(snapshots)):
    s = snapshots[j]
    print('At snapshot {}, state populations = '.format(j), np.sum(np.abs(s)**2 * (R[0] > 0)[...,None], axis=0) / np.sum(np.abs(s)**2))


# pywp.visualize.visualize_snapshot_1d(snapshots)
