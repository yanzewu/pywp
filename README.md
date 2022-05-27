
PyWP
===

Efficient and customizable code to run wavepacket dynamics using the split operator method. Supports arbitrary number of electronic states and nuclear degrees of freedom (*: the output is limited to 3D, but in the future I'll extend it). Supports GPU.

Currently pywp only supports diabatic basis.

Requires python >= 3.5, numpy >= 1.9 and (for GPU only) cupy >= 9.5.

## Usage

Run

    python3 -m pywp [args]

PyWP can also be used as a library. See below for more details.

Most command line arguments are self-explanatory. They are:

- box: The simulation box. `--box=L1,L2,L3` will make the actual box x=[-L1/2, L1/2], y=[-L2/2, L2/2], z=[-L3/3, L3/3].
- grid: The grid number in each dimension. Use like `--grid=N1,N2,N3`.
- mass: Mass of the nuclei. Currently it's a scalar.
- init_r: Initial position of the wavepacket. Use like `--init_r=x,y,z`.
- init_p: Initial momentum of the wavepacket. Use like `--init_p=px,py,pz`.
- sigma: The initial standard deviation of the wavepacket. The wavefunction will be exp(-x^2/sigmax^2-y^2/sigmay^2...). Use like `--sigma=sigmax,sigmay,sigmaz`.
- init_s: Initial diabatic surface label. Integer.
- Nstep: Maximum number of step for simulation.
- dt: Integration time interval.
- output: The output level to the console. 0: no output; 1: output population; 2: output population, position and momentum.
- output_step: Interval between two outputs and writing trajectory.
- traj: Filename of trajectory. If empty then will not write trajectory. Note the trajectory file size can grow dramatically when the nuclear degrees of freedom increases.
- gpu: True/False. PyWP depends on package cupy for GPU functions.
- potential: Name of the potential. Only used when it's a built-in potential class. The format is `--potential=my_file.MyClass`. See below for more details.
- potential_params: Arguments for the potential, will passed to the potential's `__init__()` as the optional argument list. Use like `--potential_params=a,b,c,d`.
- checkend: Whether using the checkend algorithm. Will terminate the simulation if the fraction of wavepacket that goes beyond xwall_left or xwall_right in the first dimension is greater than rtol.
- xwall_left: The relative value of the leftwall for checkend.
- xwall_right: The relative value of the rightwall for checkend.
- rtol: The relative tolerance for checkend.

## Customizing

### Way 1: Running like a command

Just put your potential file into pywp/potential folder. The potential class should inherit `pywp.Potential`. In command argument, select `--potential=yourfile.Yourclassname`. For example, `--potential=test.Tully1` selects class Tully1 defined in pywp/potential/test.py.

### Way 2: Running as a custom application

Put the following code into your script:

    import pywp

    class MyPotential(pywp.Potential):
        ...

    app = pywp.Application(MyPotential)
    app.parse_args()
    app.run()

Then your script can accept command line args just like PyWP itself. Note it cannot accept the `--potential` argument, as the potential is already specified when building the app.

### Way 3: Running as a script

Put the following code into your script:

    import pywp

    class MyPotential(pywp.Potential):
        ...

    app = pywp.Application(MyPotential)
    app.set_args(init_r=.., init_p=.., dt=.., potential_params=..., ...)
    app.run()

Alternatively,

    app = pywp.Application(potential=MyPotential(myargs...))
    app.set_args(init_r=.., init_p=.., dt=.., ...)
    app.run()

Then your script can be executed directly. Note that all the required arguments must be set.


## The Potential Class

The potential class has six functions, and you may override part of them, or all or them.
- The constructor: Receives the parameter from `--potential_params`. Must call `super().__init__()` before returning.
- get_H(R): R is a list with Ndof (= get_kdim()) elements. Each element is a Ngrid_1 x Ngrid_2 x ... x Ngrid_Ndof array, representing the value of first, second, ... nuclear coord in the grid. It is just like what you get in `meshgrid(..., indexing=ij)`. The returned Hamiltonian has shape Ngrid_1 x Ngrid_2 x ... x Ngrid_Ndof x Nel x Nel. The last two are electronic dimensions.
- get_kdim(), get_dim(): Return nuclear DOF and electronic DOF. You don't need to override them if you have already passed them in `super().__init__()`.
- has_get_phase(), get_phase(R): This is special for two-state Hamiltonian. If `has_get_phase()` returns True, will call `get_phase()` to return the phase (that is, x/abs(x)) of the off-diagonal element `H[...,0,1]`. This will preserve a gauge in adiabatic transformation (but does not affect the dynamics itself). The returned phase should have the same dimension as `R[0]`.


## Visualization

PyWP comes with a strong visualizing support, which make it easy to test custom potentials. Most functions are in `pywp.visualize`.

- visualize_pe_1d, visualize_pe_2d and visualize_pe_2d_surf plot the potential surface in one or two specific dimension(s).
- visualize_snapshot_1d, visualize_snapshot_2d plot the wavepacket in one or two specific dimension(s).
- imshow, imshow_multiple, surf_multiple are wrappers around matplotlib functions, which can quickly plot 2D data.
