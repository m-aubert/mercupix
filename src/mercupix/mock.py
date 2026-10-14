"""Mock up-the-ramp data with a known truth, for tests and examples."""
import numpy as np


def mock_ramps(flux=100., shape=(32, 32), ngroups=30, dt=2.8, offset=1_000.,
               readnoise=10., nl_loss=0., full_well=60_000., poisson=True, seed=None):
    """Simulate active-area ramps with a constant flux and classical non-linearity.

    The detector accumulates ``s(t) = flux * t`` electrons (gain 1, so ADU),
    with Poisson noise if ``poisson`` is True, and records the non-linear
    signal ``offset + s - beta * s**2 + read noise``, where
    ``beta = nl_loss / full_well``: the recorded signal is ``nl_loss`` (a
    fraction) below the linear one when ``s`` reaches ``full_well``.

    Since ``s(0) = 0``, the slope of the recorded ramp at ``t = 0`` is
    ``flux`` and its value at ``t = 0`` is ``offset``, whatever ``nl_loss``.

    Parameters
    ----------
    flux : float or array_like, optional
        Input flux in ADU/s, constant in time. A scalar, or an array
        broadcastable to ``shape`` (e.g. one value per row with shape
        ``(ny, 1)``). Default is 100.
    shape : tuple of int, optional
        Shape of the image (one ramp per pixel). Use ``()`` for a single
        ramp. Default is ``(32, 32)``.
    ngroups : int, optional
        Number of groups. Default is 30.
    dt : float, optional
        Time between groups, in seconds. The first group is read at ``dt``.
        Default is 2.8.
    offset : float, optional
        Value of the ramp at ``t = 0``, in ADU. Default is 1000.
    readnoise : float, optional
        Gaussian read noise per group, in ADU. Default is 10.
    nl_loss : float, optional
        Fractional non-linearity loss at ``full_well`` (0 for linear ramps,
        0.1 for a 10 % loss). Must be below 0.5 for the recorded signal to
        keep increasing up to ``full_well``. Default is 0.
    full_well : float, optional
        Signal, in ADU, at which the loss equals ``nl_loss``. Default is
        60000.
    poisson : bool, optional
        Whether to add Poisson noise to the accumulated signal. Default is
        True.
    seed : int or numpy.random.Generator, optional
        Random seed or generator. Default is None.

    Returns
    -------
    times : ndarray
        Time of each group, of shape ``(ngroups,)``, as float32.
    data : ndarray
        Ramps of shape ``(ngroups, *shape)``, as float32.

    Examples
    --------
    >>> times, data = mock_ramps(flux=[[20.], [700.]], shape=(2, 100), nl_loss=0.1, seed=0)
    >>> data.shape
    (30, 2, 100)
    """
    rng = np.random.default_rng(seed)
    times = np.arange(1, ngroups + 1) * dt
    flux = np.broadcast_to(np.asarray(flux, dtype="float64"), shape)

    increments = np.broadcast_to(flux * dt, (ngroups, *shape))
    if poisson:
        increments = rng.poisson(increments)
    signal = np.cumsum(increments, axis=0, dtype="float64")

    beta = nl_loss / full_well
    data = offset + signal - beta * signal**2 + rng.normal(0, readnoise, signal.shape)
    return times.astype("float32"), data.astype("float32")
