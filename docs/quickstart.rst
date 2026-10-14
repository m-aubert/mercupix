Getting started
===============

Every ``mercupix`` workflow has four steps:

1. **Load** a raw ramp.
2. **Preprocess** it: corrections and data quality mask.
3. **Fit** each pixel ramp to measure the flux.
4. **Build** the 2D image.

1. Load a raw ramp
------------------

`~mercupix.ramp.RawRamp` holds the raw datacube of shape
``(ngroups, ny, nx)``, reference pixels included, with the time of each group.
Pick the reader matching your data:

.. tab-set::
   :sync-group: format

   .. tab-item:: pyxim
      :sync: pyxim

      .. code-block:: python

         from mercupix.ramp import RawRamp

         ramp = RawRamp.from_filename("ramp.fits", datatype="pyxim")

   .. tab-item:: Roman
      :sync: roman

      .. code-block:: python

         from mercupix.ramp import RawRamp

         ramp = RawRamp.from_filename("ramp.fits", datatype="roman")

   .. tab-item:: IP2I (HDF5)
      :sync: ip2i

      .. code-block:: python

         from mercupix.ramp import RawRamp

         ramp = RawRamp.from_filename(
             "ramp.h5", datatype="ip2i",
             frame_path="/run/seq_1/asic_1/ramp_1/frames",
         )

   .. tab-item:: In memory
      :sync: array

      .. code-block:: python

         from mercupix.ramp import RawRamp

         # data: (ngroups, ny, nx) array, times: (ngroups,) array
         ramp = RawRamp(data, times, nrefpixels=4, nchannels=32)

2. Preprocess
-------------

`~mercupix.ramp.RawRamp.preprocess_ramp` applies the corrections you turn on
and crops the result to the active area:

.. code-block:: python

   data, var, mask = ramp.preprocess_ramp(
       corr_refpix=True,                 # reference pixel correction
       mask=True,                        # data quality bitmask
       mask_kwargs=dict(saturated=True, cosmics=True, disconnected=False),
   )
   data.shape    # (ngroups, ny_act, nx_act)

See :doc:`notebooks/howto_preprocess` for all the corrections.

3. Fit the ramps
----------------

`~mercupix.fitter.fit_ramp` preprocesses the ramp and fits it in one call.
The fitter is chosen by name:

.. tab-set::
   :sync-group: fitter

   .. tab-item:: Brandt
      :sync: brandt

      Optimal weighting of read and photon noise, with jump detection.

      .. code-block:: python

         from mercupix.fitter import fit_ramp

         result = fit_ramp(
             ramp, method="Brandt",
             raw_ramp_kwargs=dict(corr_refpix=True, mask=True,
                                  mask_kwargs=dict(saturated=True, disconnected=False)),
             fit_args=dict(jumps=True, jump_threshold=[20, 30]),
             readnoise=10,       # read noise of a single read [ADU]
         )

   .. tab-item:: LSQ
      :sync: lsq

      Polynomial least-squares fit, fast and simple. With ``degree=2`` the
      flux is not biased by a curved ramp (e.g. non-linearity), but it is
      about 2.5 times noisier than with ``degree=1``.

      .. code-block:: python

         from mercupix.fitter import fit_ramp

         result = fit_ramp(
             ramp, method="LSQ",
             raw_ramp_kwargs=dict(corr_refpix=True, corr_cosmics=True),
             degree=2,
         )

All fitters return a `~mercupix.fitter.RampFitResults`:

.. code-block:: python

   result.slope    # (ny_act, nx_act) flux, in ADU per unit of ramp.times
   result.var      # variance of the slope
   result.qof      # quality of fit

:doc:`methods/ramp_fitting` compares the fitters.

4. Build the image
------------------

.. code-block:: python

   from mercupix.image import Exposure

   exposure = Exposure(data=result.slope, mask=result.mask2d)
   fig = exposure.show()

Next steps
----------

- :doc:`concepts`: detector geometry, processing steps and bitmask flags.
- :doc:`tutorials`: step-by-step notebooks.
- :doc:`methods/index`: which method to pick for each step.
