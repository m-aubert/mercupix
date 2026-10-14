"""Plotting utilities shared across the package."""
import numpy as np
import matplotlib.pyplot as plt

def plot_frame(data, ax=None, clim=[1,99], **kwargs):
    """Display a 2D frame with percentile-based color limits.

    Parameters
    ----------
    data : ndarray
        2D image of shape ``(ny, nx)`` to display.
    ax : matplotlib.axes.Axes, optional
        Axes on which the frame is drawn. If both ``fig`` and ``ax`` are
        None, a new figure and axes are created. Default is None.
    clim : list of float, optional
        Lower and upper percentiles of ``data`` used as color limits.
        Default is ``[1, 99]``.
    **kwargs
        Passed to `matplotlib.axes.Axes.imshow`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes on which the frame is drawn.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    vmin, vmax = np.nanpercentile(data.ravel(), clim)

    dp = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)
    fig.colorbar(dp, ax=ax, shrink=0.4)

    return fig


def plot_ramp(data, x=None, ax=None, **kwargs):
    """Plot one or several pixel ramps.

    Parameters
    ----------
    data : ndarray
        Ramp(s) to plot, of shape ``(ngroups,)`` or ``(ngroups, npixels)``.
    x : ndarray, optional
        Abscissa of shape ``(ngroups,)``, e.g. the group times. If None,
        the group index is used. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes on which the ramps are drawn. If both ``fig`` and ``ax`` are
        None, a new figure and axes are created. Default is None.
    **kwargs
        Passed to `matplotlib.axes.Axes.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes on which the ramps are drawn.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if x is None:
        ax.plot(data, **kwargs)
    else:
        ax.plot(x, data, **kwargs)

    return fig
