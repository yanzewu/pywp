
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as anim
from . import Potential
from . import snapshot
from .pywp import abs2

def visualize_pe_1d(pot:Potential, box:float, grid:int, coord=None, vmin=None, vmax=None, show_off_diagonal=True, show_phase=False):
    """ Visualize the PE surface on a dimension.
    box, grid: the length and number of grid for plotting.
    coord: a list/None. If coord is None, will show the first dimension, and
        assume the remaining coordinates are all 0. Otherwise, coord should be 
        a list containing floats and `None`'s. It will show j'th dimension if 
        coord[j] is None, and fix other coordination values as coord[n].
    show_off_diagonal: Show off-diagonal elements.
    show_phase: Show phase of off-diagonal elements.
    """
    nel = pot.get_dim()

    # make the grids
    if coord is None:
        coord = [None] + [0] * (pot.get_kdim()-1)

    assert len(coord) == pot.get_kdim()

    shape_out = [1 if coord[i] != None else grid for i in range(pot.get_kdim())]
    R  = []
    for i in range(nel):
        if coord[i] is None:
            r = np.linspace(-box/2, box/2, grid)
            dim = i
            R.append(np.reshape(r, shape_out))
        else:
            R.append(np.reshape(coord[i]*np.ones(grid), shape_out))

    H = pot.get_H(R)[tuple((0 if i != dim else slice(None) for i in range(nel)))]
    E = np.zeros((grid, nel))
    for n in range(grid):
        E[n] = np.linalg.eigvalsh(H[n])
    
    colors1 = cm.get_cmap('Set1').colors
    colors2 = cm.get_cmap('Accent').colors

    if show_phase:
        _, (ax1, ax2) = plt.subplots(1, 2)
    else:
        _, ax1 = plt.subplots(1, 1)

    for i in range(nel):
        ax1.plot(r, np.real(H[:,i,i]), color=colors1[i], lw=1, label='H%d%d' % (i,i))
        ax1.plot(r, np.real(E[:,i]), color=colors1[i], lw=1, ls='--', label='E%d' % i)

    if show_off_diagonal:
        c = 0
        for i in range(nel):
            for j in range(i+1,nel):
                ax1.plot(r, np.real(H[:,i,j]), color=colors2[c], lw=1, label='H%d%d'%(i,j))
                c += 1

    if show_phase:
        c = 0
        for i in range(nel):
            for j in range(i+1,nel):
                ax2.plot(r, np.angle(H[:,i,j]), color=colors2[c], lw=1, label='H%d%d'%(i,j))
                c += 1
        ax2.set_xlabel('x')
        ax2.set_ylabel('phase')
        ax2.legend(frameon=False)

    ax1.set_ylim([vmin, vmax])
    ax1.legend(fancybox=False)
    ax1.set_xlabel('x')
    ax1.set_ylabel('E')
    plt.tight_layout()
    plt.show()
    

def visualize_pe_2d(pot:Potential, box, grid, coord=None, vmin=None, vmax=None, show_off_diagonal=True, show_phase=False):
    """ Visualize the PE surface on two dimensions.
    box, grid: the length and number of grid for plotting, each as a pair.
    coord: a list/None. If coord is None, will show the first two dimensions, and 
        assume the remaining coordinates are all 0. Otherwise, coord should be a list 
        containing floats and `None`'s. It will show i'th and j'th dimension if coord[i] 
        and coord[j] is None, and fix other coordination values as coord[n].
    show_off_diagonal: Show off-diagonal elements.
    show_phase: Show phase of off-diagonal elements.
    """
    if pot.get_kdim() < 2:
        raise ValueError('The potential must be at least 2 dimensional')

    nel = pot.get_dim()

    # make the grids
    if coord is None:
        coord = [None, None] + [0] * (pot.get_kdim()-2)

    assert len(coord) == pot.get_kdim()

    dims = [i for i in range(len(coord)) if coord[i] is None]
    assert len(dims) == 2

    shape_out = [1]*pot.get_kdim()
    shape_out[dims[0]] = grid[0]
    shape_out[dims[1]] = grid[1]
    R  = []
    x, y = np.meshgrid(np.linspace(-box[0]/2,box[0]/2,grid[0]), np.linspace(-box[1]/2,box[1]/2,grid[1]), indexing='ij')
    
    for i in range(nel):
        if i == dims[0]:
            R.append(np.reshape(x, shape_out))
        elif i == dims[1]:
            R.append(np.reshape(y, shape_out))
        else:
            R.append(np.reshape(coord[i]*np.ones_like(x), shape_out))

    H = pot.get_H(R)[tuple((0 if coord[i] is not None else slice(None) for i in range(nel)))]
    E = np.zeros((grid[0], grid[1], nel))

    for i in range(grid[0]):
        for j in range(grid[1]):
            E[i, j] = np.linalg.eigvalsh(H[i, j])

    if vmax is not None and vmin is not None:
        H[np.logical_and(H > vmax, H < vmin)] = np.nan
        E[np.logical_and(E > vmax, E < vmin)] = np.nan
    elif vmax is not None:
        H[H > vmax] = np.nan
        E[E > vmax] = np.nan

    ext = [-box[0]/2,box[0]/2,-box[1]/2,box[1]/2]

    if not show_off_diagonal:
        _, axs = plt.subplots(1, nel,sharex=True, sharey=True)
        axs = axs[None, :]
    elif not show_phase:
        _, axs = plt.subplots((nel+2)//2, nel, sharex=True, sharey=True)
    else:
        _, axs = plt.subplots(nel, nel)

    for i in range(nel):
        axs[0, i].imshow(np.flipud(np.real(H[:,:,i,i]).T), extent=ext)
        axs[0, i].set_title('H%d%d' % (i, i))

    c = nel
    if show_off_diagonal:
        for i in range(nel):
            for j in range(i+1, nel):
                ax = axs[c // nel, c % nel]
                ax.imshow(np.flipud(np.abs(H[:,:,i,j]).T), extent=ext)
                ax.set_title('|V%d%d|' % (i, j))
                c += 1

    if show_phase:
        for i in range(nel):
            for j in range(i+1, nel):
                ax = axs[c // nel, c % nel]
                ax.imshow(np.flipud(np.angle(H[:,:,i,j]).T), extent=ext)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_title('Phase%d%d' % (i, j))
                c += 1

    # trick for blanck
    for j in range(c, axs.size):
        #axs[j // nel, j % nel].imshow(np.flipud(np.real(H[:,:,0,0]).T), extent=ext)
        axs[j//nel, j % nel].set_visible(False)

    plt.tight_layout()

    _, axs2 = plt.subplots(1, nel, sharex=True, sharey=True)
    for i in range(nel):
        axs2[i].imshow(np.flipud(E[:,:,i].T), extent=ext)
        axs2[i].set_title('E%d' % i)

    plt.tight_layout()

    plt.show()



def visualize_pe_2d_surf(pot:Potential, box, grid, coord=None, vmin=None, vmax=None):
    """ Similar as visualize_pe_2d(), but only show the diabatic and adiabatic surfaces in two 3D plots.
    """
    from mpl_toolkits.mplot3d import Axes3D

    if pot.get_kdim() < 2:
        raise ValueError('The potential must be at least 2 dimensional')

    nel = pot.get_dim()

    # make the grids
    if coord is None:
        coord = [None, None] + [0] * (pot.get_kdim()-2)

    assert len(coord) == pot.get_kdim()

    dims = [i for i in range(len(coord)) if coord[i] is None]
    assert len(dims) == 2

    shape_out = [1]*pot.get_kdim()
    shape_out[dims[0]] = grid[0]
    shape_out[dims[1]] = grid[1]
    R  = []
    x, y = np.meshgrid(np.linspace(-box[0]/2,box[0]/2,grid[0]), np.linspace(-box[1]/2,box[1]/2,grid[1]), indexing='ij')
    
    for i in range(nel):
        if i == dims[0]:
            R.append(np.reshape(x, shape_out))
        elif i == dims[1]:
            R.append(np.reshape(y, shape_out))
        else:
            R.append(np.reshape(coord[i]*np.ones_like(x), shape_out))

    H = pot.get_H(R)[tuple((0 if coord[i] is not None else slice(None) for i in range(nel)))]
    E = np.zeros((grid[0], grid[1], nel))

    for i in range(grid[0]):
        for j in range(grid[1]):
            E[i, j] = np.linalg.eigvalsh(H[i, j])

    if vmax is not None and vmin is not None:
        H[np.logical_and(H > vmax, H < vmin)] = np.nan
        E[np.logical_and(E > vmax, E < vmin)] = np.nan
    elif vmax is not None:
        H[H > vmax] = np.nan
        E[E > vmax] = np.nan

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    colors1 = cm.get_cmap('Dark2').colors
    for i in range(nel):
        ax1.plot_surface(np.flipud(x.T), np.flipud(y.T), np.flipud(np.real(H[:,:,i,i]).T), color=colors1[i], shade=False)
        ax2.plot_surface(np.flipud(x.T), np.flipud(y.T), np.flipud(E[:,:,i].T), color=colors1[i], shade=False)

    ax1.set_title('Diabats')
    ax2.set_title('Adiabats')
    plt.tight_layout()
    plt.show()    


def visualize_snapshot_1d(snapshots:snapshot.Snapshots, axis=0, relative_vmax=2.0, show_momentum=False, interactive=True, output_callback=False, figsize=None):
    """ Visualize the snapshot file in one dimension.
    axis: The selected dimension;
    relative_vmax: Axis limit, relative to maximum value of data at all times;
    show_momentum: Display k-space data;
    interactive: Interactive plotting (press '.' to forward, ',' to backward and 'q' to quit)
    output_callback: Returns a tuple (figure, callback, n_frames) which could be passed directly to write_video(). Will not display anything.
    """
    fig = plt.figure(figsize=figsize)

    if output_callback:
        interactive = False

    index = [0]
    if interactive:
        pressed_flag = [0]
        exit_flag = []
        fig.canvas.mpl_connect('key_press_event', lambda e: _plt_key_press_callback(e, index, len(snapshots.data), pressed_flag))
        fig.canvas.mpl_connect('close_event', lambda e: exit_flag.append(1))

    grid = snapshots.data[0][0].shape[1+axis]
    r = np.linspace(-snapshots.box[axis]/2, snapshots.box[axis]/2, grid)
    p = np.fft.fftshift(np.fft.fftfreq(grid, snapshots.box[axis]/(grid-1)))*2*np.pi

    nk = snapshots.kdim()
    nel = snapshots.eldim()
    sum_index=tuple((j+1 for j in range(nk) if j != axis)) if nk > 1 else None
    colors1 = cm.get_cmap('Set1').colors

    plot_data = [(np.sum(abs2(d[0]), axis=sum_index), np.sum(abs2(d[1]), axis=sum_index)) for d in snapshots.data]

    vmax1 = max([np.max(d[0]) for d in plot_data]) * relative_vmax
    vmax2 = max([np.max(d[1]) for d in plot_data]) * relative_vmax

    def display_callback(frame):
        fig.clear()
        fig.text(0, 0, 'Frame=%d, Time=%.4g' % (frame, snapshots.dt * frame))
        if not show_momentum:
            ax1 = fig.add_subplot(1, 2, 1)
        else:
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

        for i in range(nel):
            ax1.plot(r, plot_data[frame][0][i], lw=1, color=colors1[i], label='State %d'%i)

        ax1.legend(fancybox=False)
        ax1.set_ylim([0, vmax1])
        ax1.set_xlabel('R')
        ax1.set_ylabel('population')

        if show_momentum:
            ax2.clear()
            for i in range(nel):
                ax2.plot(p, plot_data[frame][1][i], lw=1, color=colors1[i], label='State %d'%i)

            ax2.legend(fancybox=False)
            ax2.set_ylim([0, vmax2])
            ax2.set_xlabel('P')
            ax2.set_ylabel('population')
            return ax1, ax2
        else:
            return ax1

    if output_callback:
        return fig, display_callback, len(snapshots.data)

    while index[0] < len(snapshots.data):    
        display_callback(index[0])
        if interactive:
            try:
                pressed_flag[0] = 0
                while not pressed_flag[0] and not exit_flag:
                    plt.waitforbuttonpress(timeout=1)
                if exit_flag:
                    break
            except (Exception, KeyboardInterrupt):
                break
        else:
            index[0] += 1
        plt.draw()


def visualize_snapshot_2d(snapshots:snapshot.Snapshots, axis=(0, 1), relative_vmax=0.2, show_momentum=False, interactive=True, output_callback=False, figsize=None):
    """ Similar to visualize_snapshot_1d(), but using imshow() in two selected dimensions.
    """
    fig = plt.figure(figsize=figsize)

    if output_callback:
        interactive = False

    index = [0]
    if interactive:
        pressed_flag = [0]
        exit_flag = []
        fig.canvas.mpl_connect('key_press_event', lambda e: _plt_key_press_callback(e, index, len(snapshots.data), pressed_flag))
        fig.canvas.mpl_connect('close_event', lambda e: exit_flag.append(1))

    assert len(axis) == 2
    Lx, Ly = snapshots.box[axis[0]], snapshots.box[axis[1]]
    gx, gy = snapshots.grid()[axis[0]], snapshots.grid()[axis[1]]
    ext1 = [-Lx/2,Lx/2,-Ly/2,Ly/2]
    ext2 = [-(gx-1)/Lx*np.pi,(gx/2-1)*(gx-1)/(Lx*gx)*2*np.pi,-(gy-1)/Ly*np.pi,(gy/2-1)*(gy-1)/(Ly*gy)*2*np.pi]

    nk = snapshots.kdim()
    nel = snapshots.eldim()
    sum_index=tuple((j+1 for j in range(nk) if j not in axis)) if nk > 1 else None

    plot_data = [(np.sum(abs2(d[0]), axis=sum_index), np.sum(abs2(d[1]), axis=sum_index)) for d in snapshots.data]

    vmax1 = max([np.max(d[0]) for d in plot_data]) * relative_vmax
    vmax2 = max([np.max(d[1]) for d in plot_data]) * relative_vmax

    def display_callback(frame):
        fig.clear()
        fig.text(0, 0, 'Frame=%d, Time=%.4g' % (frame, snapshots.dt * frame))
        
        rows = 1 if not show_momentum else 2
        for i in range(nel):
            ax = fig.add_subplot(rows, nel, i+1)
            ax.imshow(np.flipud(plot_data[frame][0][i].T), vmin=0, vmax=vmax1, extent=ext1, aspect='auto')
            ax.set_title('State %d [R]'%i)

        if show_momentum:
            for i in range(nel):
                ax = fig.add_subplot(rows, nel, i+nel+1)
                ax.imshow(np.flipud(plot_data[frame][1][i].T), vmin=0, vmax=vmax2, extent=ext2, aspect='auto')
                ax.set_title('State %d [P]'%i)
        plt.tight_layout()

    if output_callback:
        return fig, display_callback, len(snapshots.data)

    while index[0] < len(snapshots.data):
        display_callback(index[0])
        if interactive:
            try:
                pressed_flag[0] = 0
                while not pressed_flag[0] and not exit_flag:
                    plt.waitforbuttonpress(timeout=1)
                if exit_flag:
                    break
            except (Exception, KeyboardInterrupt):
                break
        else:
            index[0] += 1
        plt.draw()

def _plt_key_press_callback(event, index, nstep, pressed_flag):
    pressed_flag[0] = 1
    if event.key == ',' and index[0] > -nstep+1:
        index[0] = (index[0] - 1) % nstep
    elif event.key == '.' and index[0] < nstep-1:
        index[0] += 1
    elif event.key == 'q':
        index[0] = nstep


def write_video(filename, figure, update_callback, n_frames, fps=5, dpi=192):
    mw = anim.FFMpegWriter(fps=fps)
    with mw.saving(figure, filename, dpi=dpi):
        for j in range(n_frames):
            update_callback(j)
            mw.grab_frame()


def write_gif(filename, figure, update_callback, n_frames, fps=5, dpi=192):
    mw = anim.ImageMagickWriter(fps=fps)
    with mw.saving(figure, filename, dpi=dpi):
        for j in range(n_frames):
            update_callback(j)
            mw.grab_frame()