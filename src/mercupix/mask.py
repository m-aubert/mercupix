"""Data quality bitmasks and pixel flagging.

Each flag is a power of two, defined in ``_BITFLAGS_DICT``:

- ``GOOD`` (0): good pixel,
- ``DIS`` (1): disconnected (dead) pixel,
- ``CR`` (2): cosmic ray (jump),
- ``SATURATED`` (4): saturated pixel,
- ``HOT`` (8): hot pixel,
- ``WARM`` (16): warm pixel,
- ``CM`` (32): charge migration or snowball candidate,
- ``REF`` (64): reference pixel,
- ``RC`` (128): RC pixel,
- ``NL`` (256): non-linearity issue,
- ``RTN`` (512): random telegraph noise.
"""
import numpy as np
import warnings

from astropy.nddata.bitmask import bitfield_to_boolean_mask

_MASK_DTYPE = 'uint32'
_BITFLAGS_DICT = dict(GOOD=0,
                      DIS=2**0, # Disconnected but might change to DEAD.
                      CR=2**1, # Cosmic rays but might change to JUMP.
                      SATURATED=2**2,
                      HOT=2**3,
                      WARM=2**4,
                      CM=2**5, #Basically candidate for CM & Snowball.
                      REF=2**6,
                      RC=2**7,
                      NL=2**8,
                      RTN=2**9)


def get_boolmask(mask, bitflags=_BITFLAGS_DICT, **kwargs):
    """Convert a bitmask to a boolean mask.

    Parameters
    ----------
    mask : ndarray
        Integer bitmask.
    bitflags : dict, optional
        Mapping between flag names and bit values. Default is
        ``_BITFLAGS_DICT``.
    **kwargs
        Passed to `astropy.nddata.bitmask.bitfield_to_boolean_mask`, e.g.
        ``ignore_flags`` to select the flags to consider.

    Returns
    -------
    ndarray
        Boolean mask with the shape of ``mask``.
    """
    boolmask = bitfield_to_boolean_mask(mask, flag_name_map=bitflags, **kwargs)
    return boolmask

def get_saturated(data=None, sat_threshold=1):
    """Flag the saturated pixels.

    Parameters
    ----------
    data : ndarray
        Raw datacube in ADU (uint16).
    sat_threshold : float, optional
        Saturation threshold as a fraction of the digital saturation
        (65535 ADU), to flag saturation below the digital limit. Default
        is 1.

    Returns
    -------
    ndarray
        Bitmask with the shape of ``data``, set to ``SATURATED`` where
        ``data >= sat_threshold * 65535``.
    """
    saturated_pixels = np.zeros(data.shape, dtype=_MASK_DTYPE)
    mask = data >= int(sat_threshold*65535)
    saturated_pixels[mask] = _BITFLAGS_DICT['SATURATED']

    return saturated_pixels

def get_charge_migration(mask, kernel=None):
    # Assume that mask will be issued directly from saturated pixels.
    """Flag the neighbours of saturated pixels as charge migration candidates.

    The flagged pixels of ``mask`` are dilated with ``kernel``, group by
    group, and the pixels added by the dilation are set to ``CM``.

    Parameters
    ----------
    mask : ndarray
        Saturation bitmask of shape ``(ngroups, ny, nx)``, as returned by
        `get_saturated`. Any non-zero value is treated as saturated. Not
        modified.
    kernel : ndarray, optional
        2D structuring element of the dilation. If None, a 3x3 kernel with
        full connectivity (the 8 neighbours) is used. Default is None.

    Returns
    -------
    ndarray
        New bitmask with the shape of ``mask``, set to ``CM`` on the
        neighbours of the flagged pixels and 0 elsewhere (the flagged
        pixels themselves are not set). Combine it with the saturation mask
        using ``|``.

    See Also
    --------
    get_saturated
    scipy.ndimage.binary_dilation
    """


    from scipy.ndimage import binary_dilation, generate_binary_structure

    if kernel is None :
        kernel = generate_binary_structure(2, 2)

    # Old from Marie. Is that change ok ? if so rm commented lines.
    #new_mask = binary_dilation(mask, kernel, axes=(1,2))
    #new_mask[mask.astype(bool)] = False
    #mask[new_mask.astype(bool)] = _BITFLAGS_DICT['CM']

    saturated = (mask & _BITFLAGS_DICT["SATURATED"]).astype(bool)
    neighbours = binary_dilation(saturated, kernel, axes=(1,2)) & ~saturated
    cm_mask = np.zeros(mask.shape, dtype=_MASK_DTYPE)
    cm_mask[neighbours] = _BITFLAGS_DICT['CM']

    return cm_mask




def get_downpix(path=None):
    """Get the disconnected (dead) pixel map.

    Parameters
    ----------
    path : str, optional
        Path to the disconnected pixel map, from the metadata. If None, a
        warning is issued and no pixel is flagged. Default is None.

    Returns
    -------
    ndarray or int
        Disconnected pixel map of shape ``(ny, nx)``, or 0 if ``path`` is
        None.

    See Also
    --------
    mercupix.io.load_downpix
    """
    from .io import load_downpix

    if path is not None :
        mask = load_downpix(path=path)
    else:
        warnings.warn('No disconnected pixel map provided. No DIS Mask.')
        mask = 0

    return mask

def get_hotpix(path=None, **kwargs):
    """Get the known hot and warm pixel map.

    Parameters
    ----------
    path : str, optional
        Path to the hot pixel map, from the metadata. If None, a warning is
        issued and no pixel is flagged. Default is None.
    **kwargs
        Unused.

    Returns
    -------
    ndarray or int
        Hot and warm pixel bitmask of shape ``(ny, nx)``, or 0 if ``path``
        is None.

    See Also
    --------
    detect_hotpix
    mercupix.io.load_hotpix
    """
    # Boolean
    # Get noisy pixels as bitmap or as boolean ?

    if path is not None :
        from .io import load_hotpix
        mask = load_hotpix(path)
    else :
        warnings.warn("No hot pixel map provided. \n"+ \
                      "No HOT and WARM Mask.")
        mask = 0

    return mask

def get_cosmics(data, inpaint=True, cr_threshold_mad=7, window_length=5):
    """Detect, and optionally correct, the jumps (cosmic rays) in a ramp.

    A jump shows up as one group difference much larger than its
    neighbours. Each group difference is compared with a reference level,
    the median of the ``window_length`` differences on each side of it
    (the tested difference itself is excluded). The residuals are expressed
    in units of their robust scatter (normalized median absolute deviation,
    per pixel), and those above ``cr_threshold_mad`` are flagged ``CR`` on
    the group where the jump appears.

    Parameters
    ----------
    data : ndarray
        Datacube of shape ``(ngroups, ny, nx)``. Integer input (e.g. raw
        uint16) is converted to float32.
    inpaint : bool, optional
        Whether to replace the flagged differences by their reference level
        and rebuild the ramp from the first group. Default is True.
    cr_threshold_mad : float, optional
        Detection threshold, in units of the normalized median absolute
        deviation of the residuals. Default is 7.
    window_length : int, optional
        Number of differences taken on **each side** of the tested one to
        build its reference level, i.e. a window of ``2 * window_length + 1``
        differences. Larger values give a less noisy reference; smaller
        values follow a flux that changes during the exposure. It is capped
        at the number of differences. Default is 5.

    Returns
    -------
    mask : ndarray
        Bitmask of shape ``(ngroups, ny, nx)``, set to ``CR`` on the groups
        where a jump appears.
    data : ndarray
        Datacube as float32, with the jumps removed if ``inpaint`` is True,
        unchanged otherwise.

    Warns
    -----
    UserWarning
        If the ramp has fewer than 3 group differences. No jump is then
        searched for, and ``data`` is returned unchanged.

    Notes
    -----
    - Only positive jumps are flagged.
    - Group 0 is never flagged: there is no difference to test before the
      first read.
    - Pixels whose residuals have zero scatter (e.g. noise-free data) emit
      division RuntimeWarnings; any non-zero positive residual is then
      flagged.

    See Also
    --------
    scipy.ndimage.median_filter
    """
    from scipy.ndimage import median_filter
    from scipy.stats import median_abs_deviation

    data = np.asarray(data, dtype="float32")
    mask = np.zeros(data.shape, dtype=_MASK_DTYPE)

    # difference image.
    diff = np.diff(data, axis=0)

    if diff.shape[0] < 3:
        warnings.warn("Fewer than 3 group differ")
        return mask, data

    # Centred moving median of the neighbours, excluding the tested difference.
    # => using scipy median_filter
    window_length = min(window_length, diff.shape[0])
    footprint = np.ones(2 * window_length + 1, dtype=bool) # all true...
    footprint[window_length] = False                       # ... except when it is false
    footprint = footprint.reshape((-1,) + (1,) * (diff.ndim - 1))
    # note: (`np.pad(..., mode="reflect")` in NumPy is what SciPy calls `mode="mirror"`;
    reference = median_filter(diff, footprint=footprint, mode="mirror")

    # cosmic score based on residual vs nmad.
    residual = diff - reference
    scale = median_abs_deviation(residual, axis=0, scale="normal")
    zscore = residual / scale            # noise-free pixels: inf at a jump, nan elsewhere
    cr = zscore >= cr_threshold_mad      # n

    # [1:] because we did a diff
    mask[1:][cr] = _BITFLAGS_DICT["CR"]

    if inpaint and cr.any():
        diff = np.where(cr, reference, diff)
        data = np.cumsum(np.concatenate([data[:1], diff], axis=0), axis=0)

    return mask, data.astype("float32")


def detect_hotpix(dark_slope, warm_threshold=0.15, hot_threshold=0.5):
    """Detect hot and warm pixels from a dark slope.

    Thresholds are taken from Roman.

    Parameters
    ----------
    dark_slope : ndarray
        Dark slope of shape ``(ny_act, nx_act)``.
    warm_threshold : float, optional
        Lower dark slope threshold for warm pixels. Default is 0.15.
    hot_threshold : float, optional
        Lower dark slope threshold for hot pixels. Default is 0.5.

    Returns
    -------
    ndarray
        Bitmask with the shape of ``dark_slope``, set to ``WARM`` or
        ``HOT``.

    Notes
    -----
    Linearity tests are still needed.
    """

    mask = np.zeros(dark_slope.shape, _MASK_DTYPE)
    mask[(dark_slope >= warm_threshold) & (dark_slope < hot_threshold )] |= _BITFLAGS_DICT['WARM']
    mask[(dark_slope >= hot_threshold)] |= _BITFLAGS_DICT['HOT']

    return mask

def detect_rc(data):
    """Detect RC pixels in dark data.

    Parameters
    ----------
    data : ndarray
        Stack of dark ramps and slopes.

    Returns
    -------
    ndarray
        Bitmask set to ``RC`` for the RC pixels.

    Notes
    -----
    Not implemented yet: currently returns None. The intended strategy is
    a selection on the fit quality of a stack of dark ramps.
    """

    # Strategy :
    # From stack of ramp data & slope
    # QF based selection.
    # Will see how this goes.
    pass
