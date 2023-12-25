
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as anim
from . import Potential
from . import snapshot
from .pywp import abs2
from . import util

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
    nk = pot.get_kdim()

    # make the grids
    if coord is None:
        coord = [None] + [0] * (pot.get_kdim()-1)

    assert len(coord) == pot.get_kdim()

    shape_out = [1 if coord[i] != None else grid for i in range(pot.get_kdim())]
    R  = []
    for i in range(nk):
        if coord[i] is None:
            r = np.linspace(-box/2, box/2, grid)
            dim = i
            R.append(np.reshape(r, shape_out))
        else:
            R.append(np.reshape(coord[i]*np.ones(grid), shape_out))

    H = pot.get_H(R)[tuple((0 if i != dim else slice(None) for i in range(nk)))]
    E = util.adiabatic_surface(H)
    
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
    nk = pot.get_kdim()

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
    
    for i in range(nk):
        if i == dims[0]:
            R.append(np.reshape(x, shape_out))
        elif i == dims[1]:
            R.append(np.reshape(y, shape_out))
        else:
            R.append(np.reshape(coord[i]*np.ones_like(x), shape_out))

    H = pot.get_H(R)[tuple((0 if coord[i] is not None else slice(None) for i in range(nk)))]
    E = util.adiabatic_surface(H)

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



def visualize_pe_2d_surf(pot:Potential, box, grid, coord=None, vmin=None, vmax=None, colormap='Dark2', **kwargs):
    """ Similar as visualize_pe_2d(), but only show the diabatic and adiabatic surfaces in two 3D plots.
    """
    from mpl_toolkits.mplot3d import Axes3D

    if pot.get_kdim() < 2:
        raise ValueError('The potential must be at least 2 dimensional')

    nel = pot.get_dim()
    nk = pot.get_kdim()

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
    
    for i in range(nk):
        if i == dims[0]:
            R.append(np.reshape(x, shape_out))
        elif i == dims[1]:
            R.append(np.reshape(y, shape_out))
        else:
            R.append(np.reshape(coord[i]*np.ones_like(x), shape_out))

    H = pot.get_H(R)[tuple((0 if coord[i] is not None else slice(None) for i in range(nk)))]
    E = util.adiabatic_surface(H)

    if vmax is not None and vmin is not None:
        H[np.logical_and(H > vmax, H < vmin)] = np.nan
        E[np.logical_and(E > vmax, E < vmin)] = np.nan
    elif vmax is not None:
        H[H > vmax] = np.nan
        E[E > vmax] = np.nan

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    colors1 = cm.get_cmap(colormap).colors
    if kwargs.get('occlusion', False):
        multi_surf(ax1, [(x.T, y.T, np.flipud(np.real(H[:,:,i,i]).T), colors1[i]) for i in range(nel)], **kwargs)
    else:
        for i in range(nel):
            ax1.plot_surface(np.flipud(x.T), np.flipud(y.T), np.flipud(np.real(H[:,:,i,i]).T), color=colors1[i], shade=False)

    for i in range(nel):    # adiabats are strictly ascending, so no occlusion needed
        ax2.plot_surface(np.flipud(x.T), np.flipud(y.T), np.flipud(E[:,:,i].T), color=colors1[i], shade=False)

    ax1.set_title('Diabats')
    ax2.set_title('Adiabats')
    plt.tight_layout()
    plt.show()    


def visualize_snapshot_1d(snapshots:snapshot.Snapshots, axis=0, relative_vmax=2.0, show_momentum=False, interactive=True, output_callback=False, figsize=None, colormap='Set1'):
    """ Visualize the snapshot file in one dimension.
    axis: The selected dimension;
    relative_vmax: Axis limit, relative to maximum value of data at all times;
    show_momentum: Display k-space data;
    interactive: Interactive plotting (press '.' to forward, ',' to backward and 'q' to quit)
    output_callback: Returns a tuple (figure, callback, n_frames) which could be passed directly to write_video(). Will not display anything.
    """
    fig = plt.figure(figsize=figsize)

    grid = snapshots.data[0][0].shape[1+axis]
    r = np.linspace(-snapshots.box[axis]/2, snapshots.box[axis]/2, grid)
    p = np.fft.fftshift(np.fft.fftfreq(grid, snapshots.box[axis]/(grid-1)))*2*np.pi

    nk = snapshots.kdim()
    nel = snapshots.eldim()
    sum_index=tuple((j+1 for j in range(nk) if j != axis)) if nk > 1 else None
    colors1 = cm.get_cmap(colormap).colors

    if nk > 1:
        plot_data = [(np.sum(abs2(d[0]), axis=sum_index), np.sum(abs2(d[1]), axis=sum_index)) for d in snapshots.data]
    else:
        plot_data = [(abs2(d[0]), abs2(d[1])) for d in snapshots.data]

    vmax1 = max([np.max(d[0]) for d in plot_data]) * relative_vmax
    vmax2 = max([np.max(d[1]) for d in plot_data]) * relative_vmax

    def display_callback(frame):
        fig.clear()
        fig.text(0, 0, 'Frame=%d, Time=%.4g' % (frame, snapshots.dt * frame))
        if not show_momentum:
            ax1 = fig.add_subplot(1, 1, 1)
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
    else:
        return _display_loop(fig, display_callback, len(snapshots.data), interactive)


def visualize_snapshot_2d(snapshots:snapshot.Snapshots, axis=(0, 1), relative_vmax=0.2, show_momentum=False, adiabatic=False, interactive=True, output_callback=False, figsize=None, colormap='viridis', **kwargs):
    """ Similar to visualize_snapshot_1d(), but using imshow() in two selected dimensions.
    """
    fig = plt.figure(figsize=figsize)

    assert len(axis) == 2
    Lx, Ly = snapshots.box[axis[0]], snapshots.box[axis[1]]
    gx, gy = snapshots.grid()[axis[0]], snapshots.grid()[axis[1]]
    ext1 = [-Lx/2,Lx/2,-Ly/2,Ly/2]
    ext2 = [-(gx-1)/Lx*np.pi,(gx/2-1)*(gx-1)/(Lx*gx)*2*np.pi,-(gy-1)/Ly*np.pi,(gy/2-1)*(gy-1)/(Ly*gy)*2*np.pi]

    nk = snapshots.kdim()
    nel = snapshots.eldim()
    sum_index=tuple((j for j in range(nk) if j not in axis)) if nk > 1 else None

    plot_data_r = []
    plot_data_p = []
    if adiabatic:
        H = kwargs['potential'].get_H(snapshots.get_grid().build())
        E, U = util.adiabat(H)

    for j in range(len(snapshots)):
        if adiabatic:
            psiR = snapshots.get_snapshot(j, momentum=False)
            psiRad = util.project_wavefunction(psiR, U, inverse=True)
            plot_data_r.append(np.flipud(np.transpose(np.sum(abs2(psiRad), axis=sum_index), (1,0,2))))
            if show_momentum:
                psiPad = np.stack([snapshot.transform_r_to_p(psiRad[..., k], False) for k in range(psiRad.shape[-1])], axis=-1)
                plot_data_p.append(np.flipud(np.transpose(np.sum(abs2(psiPad), axis=sum_index), (1,0,2))))

        elif show_momentum:
            psiR, psiP = snapshots.get_snapshot(j)
            plot_data_r.append(np.flipud(np.transpose(np.sum(abs2(psiR), axis=sum_index), (1,0,2))))
            plot_data_p.append(np.flipud(np.transpose(np.sum(abs2(psiP), axis=sum_index), (1,0,2))))
        else:
            psiR = snapshots.get_snapshot(j, momentum=False)
            plot_data_r.append(np.flipud(np.transpose(np.sum(abs2(psiR), axis=sum_index), (1,0,2))))

    vmax1 = max([np.max(d) for d in plot_data_r]) * relative_vmax
    if show_momentum:
        vmax2 = max([np.max(d) for d in plot_data_p]) * relative_vmax

    def display_callback(frame):
        fig.clear()
        fig.text(0, 0, 'Frame=%d, Time=%.4g' % (frame, snapshots.dt * frame))
        
        rows = 1 if not show_momentum else 2
        for i in range(nel):
            ax = fig.add_subplot(rows, nel, i+1)
            ax.imshow(plot_data_r[frame][..., i], vmin=0, vmax=vmax1, extent=ext1, aspect='auto', cmap=colormap)
            ax.set_title('State %d [R]'%i if show_momentum else 'State %d' % i)

        if show_momentum:
            for i in range(nel):
                ax = fig.add_subplot(rows, nel, i+nel+1)
                ax.imshow(plot_data_p[frame][..., i], vmin=0, vmax=vmax2, extent=ext2, aspect='auto', cmap=colormap)
                ax.set_title('State %d [P]'%i)
        plt.tight_layout()

    if output_callback:
        return fig, display_callback, len(snapshots.data)
    else:
        return _display_loop(fig, display_callback, len(snapshots.data), interactive)



def multi_surf(ax, data, disconnect_edge=True, **kwargs):

    if not disconnect_edge:
        vdata = [d[2] for d in data]
    else:
        vdata = []
        for d in data:
            p = d[2].copy()
            p[:, 0] = np.nan
            vdata.append(p)

    cdata = []
    for d in data:
        if isinstance(d[3], np.ndarray):
            cdata.append(d[3])
        else:
            cdata.append(np.tile(np.array(list(d[3])), (data[0][0].shape[0], data[0][0].shape[1] , 1)))

    ax.plot_surface(
        np.hstack([d[0] for d in data]),
        np.hstack([d[1] for d in data]),
        np.hstack(vdata),
            facecolors=np.hstack(cdata),
            shade=kwargs.get('shade', False),
            rcount=kwargs.get('rcount', 32)*len(data),
            ccount=kwargs.get('ccount', 32),
            edgecolors=kwargs.get('edgecolors', None),
    )


def _plt_key_press_callback(event, index, nstep, pressed_flag):
    pressed_flag[0] = 1
    if event.key == ',' and index[0] > 0:
        index[0] = (index[0] - 1) % nstep
    elif event.key == '.' and index[0] < nstep-1:
        index[0] += 1
    elif event.key == 'q':
        index[0] = nstep


def _display_loop(fig, display_callback, length, interactive):

    index = [0]
    pressed_flag = [0]
    exit_flag = []
    if interactive:
        fig.canvas.mpl_connect('key_press_event', lambda e: _plt_key_press_callback(e, index, length, pressed_flag))
        fig.canvas.mpl_connect('close_event', lambda e: exit_flag.append(1))

    while index[0] < length:
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
            plt.draw()
        else:
            index[0] += 1
            plt.draw()
            plt.pause(0.01)


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


def imshow(data:np.ndarray, extent:list, vmin=None, vmax=None, ax:plt.Axes=None, colormap='viridis', xlabel='x', ylabel='y', title=''):
    """ Show a 2D data in correct orientation.
    data: 2D array. The first dimension represents x and the second represents y.
    extent: The boundary coordination (xmin, xmax, ymin, ymax).
    vmin, vmax: The min/max data for colormap.
    ax: The axis.
    colormap: Colormap used for data.
    """
    if ax is None:
        ax = plt.axes()

    ax.imshow(np.flipud(data.T), extent=extent, aspect='auto', vmin=vmin, vmax=vmax, cmap=cm.get_cmap(colormap))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def imshow_multiple(*data, extent:list, vmin=None, vmax=None, fig=None, colormap='viridis', xlabel='x', ylabel='y', titles=[]):
    """ Show multiple 2D data in correct orientation.
    data: list of 2D array. The first dimension represents x and the second represents y.
    extent: The boundary coordination (xmin, xmax, ymin, ymax).
    vmin, vmax: The min/max data for colormap.
    fig: The figure. If None, will call `plt.subplots()`.
    colormap: Colormap used for data.
    """

    nrow = int(np.floor(len(data)**0.5))
    ncol = int(np.ceil(len(data)/nrow))
    if fig:
        axs = fig.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False)
    else:
        _, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False)

    for j in range(nrow):
        for k in range(ncol):
            if j*ncol + k > len(data):
                axs[j, k].set_visible(False)
            else:
                axs[j, k].imshow(np.flipud(data[j*ncol+k].T), extent=extent, vmin=vmin, vmax=vmax, cmap=cm.get_cmap(colormap))
                if j*ncol + k < len(titles) and titles[j*ncol + k]:
                    axs[j, k].set_title(titles[j*ncol + k])
            
                if j == nrow-1:
                    axs[j, k].set_xlabel(xlabel)
                if k == 0:
                    axs[j, k].set_ylabel(ylabel)
            
            
def surf_multiple(*data, grid:tuple, vmin=None, vmax=None, fig=None, colormap='Dark2', edgecolors=None, xlabel='x', ylabel='y', titles=[], occlusion=False, shade=False):
    """ Surf plot multiple 2D data in correct orientation.
    data: list of 2D array. The first dimension represents x and the second represents y.
    grid: the coordinate grid (X, Y), each being an 2D array with same oritention of data.
    vmin, vmax: The min/max of z axis.
    fig: The figure. If None, will call `plt.subplots()`.
    colormap: Colormap used for each data.
    """

    from mpl_toolkits.mplot3d import Axes3D

    if fig is None:
        fig = plt.figure()
        
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    colors1 = cm.get_cmap(colormap).colors
    X, Y = grid

    if occlusion:
        multi_surf(ax1, [(np.flipud(X.T), np.flipud(Y.T), np.flipud(d.T), c) for c, d in zip(colors1, data)], shade=False, edgecolors=edgecolors)
    else:
        for c, d in zip(colors1, data):
            s = ax1.plot_surface(np.flipud(X.T), np.flipud(Y.T), np.flipud(d.T), color=c, shade=shade, edgecolors=edgecolors)
            if titles:
                s._edgecolors2d = s._edgecolor3d
                s._facecolors2d = s._facecolor3d
    
    ax1.set_zlim([vmin, vmax])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if not occlusion:
        ax1.legend(titles)


class WavepacketDisplayer1D:
    
    def __init__(self, axis=0, fig=None, show_momentum=False):
        if not fig:
            self.fig = plt.figure()

        self.frame = 0
        self.r = None
        self.axis = axis
        self.sum_axis = None

        #plt.ion()

    def __call__(self, para, checkpoint):
        
        if self.r is None:
            r_indexer = [0]*(para.R[0].ndim)
            r_indexer[self.axis] = slice(None)
            self.r = para.R[self.axis][tuple(r_indexer)]
            if checkpoint.backend.__name__ == 'cupy':
                self.r = self.r.get()

        if self.sum_axis is None:
            self.sum_axis = tuple(k for k in range(para.R[0].ndim) if k != self.axis)

        r_data = np.sum(np.abs(checkpoint.psiR)**2, axis=self.sum_axis)

        self.fig.clear()
        self.fig.text(0, 0, 'Frame=%d' % (self.frame))

        self.ax1 = self.fig.add_subplot(1, 1, 1)
        if checkpoint.backend.__name__ == 'cupy':
            r_data = r_data.get()

        for l in range(r_data.shape[-1]):
            self.ax1.plot(self.r, r_data[..., l])

        plt.draw()
        plt.pause(0.01)


