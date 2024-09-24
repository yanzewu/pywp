
PyWP
===

Efficient and customizable code to run Cartesian wavepacket dynamics using the split operator method. Supports arbitrary number of electronic states and nuclear degrees of freedom. Supports GPU.

Requires python >= 3.7, numpy >= 1.9 and (for GPU only) cupy >= 9.5.

## Tutorial

PyWP handles these parameters in two steps: First, `preprocess()` generate necessary caches and then `propagate()` does the actual propagation. A runnable script is found at [test/wavepacket.py](test/wavepacket.py).

Here is an example of running Tully1:

(1) Define the necessary ingredients. To run a wavepacket simulation, we need (a) A "canvas" or grid; (b) An initial condition; (c) A Hamiltonian;
 
    import pywp

    # Defining the grid
    grid = pywp.mgrid[-10:10:0.01]

    # Defining the wavepacket (normalization is not needed)
    wavepacket = lambda R: np.exp(-(R[0]+5)**2 + 1j*P0*R[0])
    
    # Defining the potential
    tully1 = lambda R: pywp.build_pe_tensor(
        0.01*(1 - np.exp(-1.6*R[0])) * (R[0] >= 0) 
            - 0.01*(1 - np.exp(1.6*R[0])) * (R[0] < 0), # V00
        0.005*np.exp(-R[0]**2), # V01
        symmetric=True  # copy V11 = -V00
     )
    
(2) Do the preprocessing:

    para = pywp.preprocess(
        tully1, grid, wavepacket, 
        c0=[1,0],   # initial amplitude
        M=2000,     # mass
        dt=0.1,     # timestep
    )


(3) Define the collectors we want to see. Here we want to collected the transmitted population on individual electronic states and save the snapshots.

    pop_trans = []

    define pop_trans_collector(para:pywp.PhysicalParameter, checkpoint:pywp.CheckPoint):
        populations.append(
            np.sum(np.abs(checkpoint.psiR)**2 * (para.R[0] > 0)[...,None], axis=0) * para.dA
        )

    snapshot_writer = pywp.snapshot.SnapshotWriter('tully1.snapshot')

There are also a few tools to assist the calculation: 
- `pywp.snapshot.SnapshotWriter()` writes the snapshots;
- `pywp.expec` calculates the expectation value of a few operators (population, position, momentum, etc.);
- `pywp.visualize.WavepacketDisplayer1D()` displays the wavepackets while running.

(4) Do the propagation:

    pywp.propagate(para, 
                    nstep=12000,        # Total step for propagation
                    output_step=1000,   # Step interval per output
                    on_output=[pop_trans_collector, snapshot_writer],   # Output functions
                    checkend=True,      # Checkend
                    boundary=lambda R: R[0] < 9,    # Checkend boundaries
                )

    snapshot_writer.close() # Don't forget to do this

    print(pop_trans)    # should be a list of array of 2 elements

(5) To view your running result:

    snapshot = pywp.snapshot.load_file('tully1.snapshot')
    pywp.visualize.visualize_snapshot_1d(snapshot)  # press '.' to forward, ',' to backward and 'q' to quit


## Snapshots

The wavefunction can be retrieved by `snapshot.get_snapshot()`. The position and momentum grids can be retrieved by `snapshot.get_R_grid()` and `snapshot.get_P_grid()`.

## Visualization

PyWP comes with a powerful visualizing support, which make it easy to test custom potentials. Most functions are in `pywp.visualize`.

- `visualize_pe_1d()`, `visualize_pe_2d()` and `visualize_pe_2d_surf()` plot the potential surface in one or two specific dimension(s).
- `visualize_snapshot_1d()`, `visualize_snapshot_2d()` plot the wavepacket in one or two specific dimension(s).
- `imshow()`, `imshow_multiple()`, `surf_multiple()` are wrappers around matplotlib functions, which can conveniently plot 2D data.

`python -m pywp.tools.show1d` and `python -m pywp.tools.show2d` are command line wrappers for visualizing snapshots.


## Scattering Calculation (Experimental)

An experimental implementation of 1D scattering calculation is in `pywp.scattering.scatter1d()`. The usage is almost straightforward.