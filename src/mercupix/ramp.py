"""Processing of raw up-the-ramp data prior to ramp fitting."""
import numpy as np
import skimage as ski
from .baseramp import BaseRamp

class RawRamp(BaseRamp):
    """Raw ramp with the corrections applied before ramp fitting.

    Provides reference pixel, offset, dark, non-linearity and IPC
    corrections, as well as data quality masking. See `~mercupix.baseramp.BaseRamp` for the
    constructor parameters.

    See Also
    --------
    mercupix.baseramp.BaseRamp
    """
    def _from_header_to_metadata(header):
        """Extract the processing metadata from a file header.

        Parameters
        ----------
        header : astropy.io.fits.Header
            Header of the raw data file.

        Returns
        -------
        dict
            Metadata, including the paths to the reference files needed for
            the processing (e.g. characterization files and external data).

        Notes
        -----
        Not implemented yet: currently returns None.
        """
        pass

    def correct_refpix(self, data=None, **kwargs):
        """Apply the reference pixel correction.

        Parameters
        ----------
        data : ndarray, optional
            Datacube of shape ``(ngroups, ny, nx)`` to correct. If None,
            ``data`` is used. Default is None.
        **kwargs
            Passed to `get_refpix_map`.

        Returns
        -------
        ndarray
            Reference pixel subtracted datacube. Its shape is
            ``(ngroups, ny_act, nx_act)`` when ``data`` is None.

        See Also
        --------
        mercupix.baseramp.BaseRamp.get_refpix_map
        """
        if data is None :
            return  self.data[:, self.actslice, self.actslice] - self.get_refpix_map(**kwargs)
        else :
            return data - self.get_refpix_map(data=data, **kwargs)

    def subtract_offset(self,
                        data=None,
                        metadata={},
                        offset=0):
        """Subtract an offset (bias or superbias) frame or cube.

        Parameters
        ----------
        data : ndarray, optional
            Datacube to correct. If None, the output of `correct_refpix` is
            used. Default is None.
        metadata : dict, optional
            Metadata with the path to the external dark under the ``'dark'``
            key. Default is ``{}``.
        offset : float or ndarray or None, optional
            Offset to subtract. If None, it is loaded from ``metadata``, or
            set to the first group of ``data`` if ``metadata`` is empty.
            Default is 0.

        Returns
        -------
        ndarray
            Offset subtracted datacube.
        """
        if data is None :
            data = self.correct_refpix()

        if offset is None:
            if not metadata:
                offset = data[0,:,:]
            else :
                from .io import get_bias
                offset = get_bias(metadata['dark'])

        return data - offset

    def correct_dark(self,
                     data=None,
                     metadata={},
                     darkdata=None):
        """Subtract the dark current.

        Parameters
        ----------
        data : ndarray, optional
            Datacube to correct. If None, the output of `correct_refpix` is
            used. Default is None.
        metadata : dict, optional
            Metadata with the path to the external dark under the ``'dark'``
            key. Default is ``{}``.
        darkdata : ndarray, optional
            Dark to subtract. If 2D, it is a dark rate multiplied by `times`
            before subtraction. If 3D, it is a dark cube subtracted as is.
            Default is None.

        Returns
        -------
        ndarray
            Dark subtracted datacube.
        """

        if data is None :
            data = self.correct_refpix()

        if darkdata is None:
            darkdata = np.atleast_1d(0)

        elif bool(metadata):
            from .io import get_dark
            darkdata = get_dark(metadata['dark'])

        if darkdata.ndim == 2:
            return data - darkdata*self.times[:,None,None]
        else :
            return data - darkdata


    def get_mask(self, data=None,
                 disconnected=True,
                 cosmics=False,
                 saturated=False,
                 hot=False,
                 charge_migration=False,
                 sat_threshold=1.,
                 inpaint=False,
                 cr_threshold_mad=9):
        """Build the data quality bitmask of the ramp.

        Flag values are defined in ``mercupix.mask._BITFLAGS_DICT``.

        Parameters
        ----------
        data : ndarray, optional
            Processed datacube of shape ``(ngroups, ny_act, nx_act)`` used
            for the jump detection. Default is None.
        disconnected : bool, optional
            Whether to flag disconnected (dead) pixels. Default is True.
        cosmics : bool, optional
            Whether to flag jumps (cosmic rays) in the ramp. Default is
            False.
        saturated : bool, optional
            Whether to flag saturated pixels. Default is False.
        hot : bool, optional
            Whether to flag hot and warm pixels. Default is False.
        charge_migration : bool, optional
            Whether to flag the neighbors of saturated pixels as charge
            migration candidates. Requires ``saturated=True``. Default is
            False.
        sat_threshold : float, optional
            Saturation threshold as a fraction of the digital saturation
            (65535 ADU). Default is 1.
        inpaint : bool, optional
            Whether to correct the jumps detected in ``data``. If True, the
            ramp is rebuilt from the corrected group differences. Default is
            False.
        cr_threshold_mad : float, optional
            Jump detection threshold, in units of the normalized median
            absolute deviation. Default is 9.

        Returns
        -------
        mask : ndarray
            Bitmask of shape ``(ny_act, nx_act)``, or
            ``(ngroups, ny_act, nx_act)`` if a per-group flag is computed.
        data : ndarray
            Input datacube, with the jumps corrected if ``inpaint`` and
            ``cosmics`` are True.

        See Also
        --------
        mercupix.mask.get_cosmics
        mercupix.mask.get_saturated
        """
        # Initialize mask
        from .mask import _MASK_DTYPE

        mask = np.zeros(self.active_shape, dtype=_MASK_DTYPE)

        if disconnected :
            from .mask import get_downpix
            path = None if self.metadata is None else self.metadata['path']['DIS']
            mask += get_downpix(path=path)

        if hot :
            from .mask import get_hotpix
            # flag hot and warm
            path = None if self.metadata is None else self.metadata['path']['HOT']
            mask += get_hotpix(path=path)

        if saturated:
            from .mask import get_saturated
            # raw uint16 input, expands mask dims here.
            # Explicit sum cause inplace add does not broadcast.
            sat_mask = get_saturated(self.data, #Specifically use int raw data for saturation.
                                  sat_threshold=sat_threshold)[:, self.actslice, self.actslice]

            mask = mask + sat_mask

        if charge_migration :
            from .mask import get_charge_migration
            mask_cm = get_charge_migration(sat_mask)

            mask = mask + mask_cm

        if cosmics :
            from .mask import get_cosmics
            maskjump , data = get_cosmics(data=data,
                                cr_threshold_mad=cr_threshold_mad,
                                inpaint=inpaint)

            mask = mask + maskjump #For some dtype, inplace sum does not work. Document for next time.

        return mask, data

    def correct_ipc(self, data=None,
                          nl=False,
                          metadata={},
                          ipckernel=np.ones((1,5,5)), dtype='float32'):
        """Deconvolve the IPC kernel, frame by frame.

        The deconvolution uses 5 iterations of the Richardson-Lucy
        algorithm on zero-padded frames.

        Parameters
        ----------
        data : ndarray, optional
            Datacube of shape ``(ngroups, ny, nx)`` to correct. Default is
            None.
        nl : bool, optional
            Whether to also apply the non-linear IPC correction. Only used
            when the kernel is loaded from ``metadata``. Default is False.
        metadata : dict, optional
            Metadata with the path to the IPC coefficients under the
            ``'ipc'`` key. Default is ``{}``.
        ipckernel : ndarray or None, optional
            IPC kernel to deconvolve, of shape ``(1, k, k)``. If None, it is
            loaded from ``metadata``. Default is ``np.ones((1, 5, 5))``.
        dtype : str, optional
            Unused. Default is ``'float32'``.

        Returns
        -------
        ndarray
            IPC corrected datacube of shape ``(ngroups, ny, nx)``, as
            float32.

        See Also
        --------
        skimage.restoration.richardson_lucy

        Notes
        -----
        This is currently the main bottleneck of the pipeline; its cost
        increases with the number of groups.
        """

        if ipckernel is None :
            from .io import load_ipc
            from .calib import ipc_kernel

            # Will probably define a kernel model (3x3) or (5x5).
            ipckernel = ipc_kernel(load_ipc(metadata['ipc'], nl=nl))

        pad_ = int(np.ceil(ipckernel.shape[-1]/2))

        data = np.pad(data.astype('float64'), #float64 to avoid overflow during deconv.
                      {1:pad_, 2: pad_},  #Padding to avoid overflow on borders.
                      constant_values=0)
        data = ski.restoration.richardson_lucy(data,
                                               ipckernel,
                                               num_iter=5, #Low iter number otherwise degrade img quality.
                                               clip=False)

        return data[:, pad_:-pad_, pad_:-pad_].astype("float32")


    def correct_nl(self, data=None,
                         metadata={},
                         nl_coefficients=None):
        """Correct the classical non-linearity.

        The linearized data are ``data * (1 + sum_i c_i * data**i)``, with
        coefficients ``c_i`` estimated following Canipe et al. (2017).

        Parameters
        ----------
        data : ndarray, optional
            Datacube to correct. Default is None.
        metadata : dict, optional
            Metadata with the path to the coefficients under the ``'nl'``
            key. Default is ``{}``.
        nl_coefficients : array_like, optional
            Correction coefficients, in increasing polynomial order. Each
            coefficient is a scalar or a per-pixel map. If None, they are
            loaded from ``metadata``. Default is None.

        Returns
        -------
        ndarray
            Linearized datacube, as float32.

        See Also
        --------
        mercupix.carac.ClassicalNL.get_nlcorr
        """

        if nl_coefficients is None :
            from .io import load_nl_coefficients

            nl_coefficients = load_nl_coefficients(path=metadata['nl'])

        correction_term = 1
        for i, coef in enumerate(nl_coefficients):
            correction_term += coef*data**i

        return (data*correction_term).astype("float32") #Forcing type for memory management.


    def get_variance(self,data=None):
        """Estimate the per-pixel variance of the count rate.

        The variance is computed over the group differences divided by the
        time between groups.

        Parameters
        ----------
        data : ndarray, optional
            Datacube of shape ``(ngroups, ny, nx)``. If None, 0 is returned.
            Default is None.

        Returns
        -------
        ndarray or int
            Variance of the count rate, of shape ``(ny, nx)``.

        Notes
        -----
        Whether a temporal or per-pixel variance is needed is still to be
        decided.
        """

        #Clarifier ce qu'on veut. Variance temporelle ou variance par pixel ?
        var = 0
        if data is not None :
            var = np.diff(data, axis=0)
            var /=np.diff(self.times)[:,None,None]
            var = np.var(var,ddof=1, axis=0)

        return var


    def preprocess_ramp(self,
                        data= None,
                        corr_refpix=True,
                        corr_ipc=False,
                        corr_offset=False,
                        corr_dark=False,
                        corr_nl=False,
                        corr_cosmics=False,
                        mask=False,
                        comp_var=False,
                        mask_kwargs={},
                        metadata={},
                        nl_coefficients=None,
                        ipckernel=np.ones((5,5)),
                        darkdata=None,
                        offset=0):
        """Apply the corrections required before fitting the ramp.

        The corrections are applied in the following order: reference
        pixels, offset, non-linearity, IPC, dark, and finally masking and
        jump correction.

        Parameters
        ----------
        data : ndarray, optional
            Datacube to process. If None, ``data`` is used. Default is None.
        corr_refpix : bool, optional
            Whether to apply the reference pixel correction. Default is True.
        corr_ipc : bool, optional
            Whether to apply the IPC correction. Default is False.
        corr_offset : bool, optional
            Whether to subtract the offset (bias). Default is False.
        corr_dark : bool, optional
            Whether to subtract the dark current. Default is False.
        corr_nl : bool, optional
            Whether to correct the classical non-linearity. Default is False.
        corr_cosmics : bool, optional
            Whether to correct (inpaint) the detected jumps. Default is
            False.
        mask : bool, optional
            Whether to compute the data quality bitmask. Default is False.
        comp_var : bool, optional
            Whether to compute the variance. Default is False.
        mask_kwargs : dict, optional
            Passed to `get_mask`. Default is ``{}``.
        metadata : dict, optional
            Metadata with the paths to the external files, as returned by
            `mercupix.io.fetch_metadata`. Default is ``{}``.
        nl_coefficients : array_like, optional
            Non-linearity coefficients, used if ``corr_nl=True``. See
            `correct_nl`. Default is None.
        ipckernel : ndarray, optional
            IPC kernel, used if ``corr_ipc=True``. Default is
            ``np.ones((5, 5))``.
        darkdata : ndarray, optional
            Dark rate (2D) or dark cube (3D), used if ``corr_dark=True``. A
            2D dark is multiplied by `times` before subtraction. Default is
            None.
        offset : float or ndarray, optional
            Offset to subtract, used if ``corr_offset=True``. Default is 0.

        Returns
        -------
        datacube : ndarray
            Processed datacube of shape ``(ngroups, ny_act, nx_act)``.
        varcube : ndarray or int
            Variance from `get_variance`, or 0 if ``comp_var`` is False.
        mask : ndarray
            Data quality bitmask from `get_mask`. If ``mask`` is False,
            an array of zeros with the shape of ``datacube``.
        """
        if corr_refpix :
            data = self.correct_refpix(data=data)

        if data is None :
            data = self.data[:, self.actslice, self.actslice].astype("float32")

        if corr_offset :
            data = self.subtract_offset(data=data,
                                        metadata=metadata,
                                        offset=offset) #Basically bias here.
        if corr_nl :
            data= self.correct_nl(data=data,
                                       metadata=metadata,
                                       nl_coefficients=nl_coefficients)

        if corr_ipc :
            data= self.correct_ipc(data=data,
                                metadata=metadata,
                                ipckernel=ipckernel)

        if corr_dark :
            data= self.correct_dark(data=data,
                                    metadata=metadata,
                                    darkdata=darkdata)

        if comp_var :
            varcube = self.get_variance(data=data) # Unclear when and what
        else:
            varcube=0

        if mask:
            mask, data = self.get_mask(data=data, inpaint=corr_cosmics, **mask_kwargs)

        else :
            if corr_cosmics :
                from .mask import get_cosmics
                _, data = get_cosmics(data=data, inpaint=corr_cosmics,
                                      cr_threshold_mad=mask_kwargs.pop('cr_threshold_mad', 7))

            mask = np.zeros(data.shape, dtype=np.int8)

        return (data, varcube, mask)

    # ============== #
    # properties     #
    # ============== #
    @property
    def metadata(self):
        """Metadata of the ramp. Currently always None."""
        return None
