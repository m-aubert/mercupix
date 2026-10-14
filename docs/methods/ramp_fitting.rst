Ramp fitting
============

A ramp fitter measures the flux of each pixel: the slope of its ramp at
``t = 0``, in ADU per unit of the group times. All fitters derive from
`~mercupix.fitter.Fitter`, are registered by name in
``mercupix.fitter.model_register``, and return a
`~mercupix.fitter.RampFitResults`. Swapping methods only changes ``method``
in `~mercupix.fitter.fit_ramp`.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 22 26 26 26

   * -
     - `~mercupix.fitter.LSQFit`
     - `~mercupix.fitter.Brandt2024`
     - `~mercupix.fitter.Kubik2015`
   * - ``method=``
     - ``"LSQ"``
     - ``"Brandt"``
     - ``"Kubik"``
   * - Principle
     - Unweighted least-squares fit of a polynomial in time (Legendre,
       Hermite or monomial).
     - Generalized least-squares on the group differences, with the exact
       read and photon noise covariance (Brandt 2024).
     - Moment-based estimator on the group differences (Kubik et al. 2016).
   * - Noise model
     - None (equal weights).
     - Read noise + photon noise, per pixel.
     - Read noise + photon noise.
   * - Ramp shape
     - Any polynomial degree. The flux is the slope of the full polynomial
       at ``t = 0``, so with ``degree >= 2`` curvature (e.g. classical
       non-linearity) does not bias it, at the cost of more noise.
     - Linear.
     - Linear.
   * - Grouped reads (MACC)
     - Uses group times only.
     - Yes, through ``nframes_list``.
     - Yes, through ``macc``.
   * - Cosmic rays
     - Must be corrected before (``corr_cosmics=True``).
     - Detected and dropped during the fit (``jumps=True``).
     - Not handled.
   * - Uses the mask
     - No.
     - Yes: masked groups are ignored.
     - No.
   * - Outputs
     - ``slope``, ``offset`` (value at ``t = 0``).
     - ``slope``, ``var``, ``qof`` (chi-square), ``mask2d``; ``offset``
       with ``intercept=True``.
     - ``slope``, ``var`` (overestimated when read noise matters).
   * - Speed :sup:`1`
     - 9 ms (degree 1), 12 ms (degree 2)
     - 29 ms, 0.5 s with ``jumps=True``
     - —
   * - Status
     - Stable.
     - Stable; the ``reset`` and ``intercept=True`` options currently fail.
     - Experimental: no offset, variance not reliable at low flux.

:sup:`1` 256 x 256 pixels, 30 groups, on a laptop. See
:doc:`../notebooks/compare_ramp_fitters`.

Usage
-----

.. tab-set::
   :sync-group: fitter

   .. tab-item:: LSQFit
      :sync: lsq

      .. code-block:: python

         from mercupix.fitter import fit_ramp, LSQFit

         # from a raw ramp
         result = fit_ramp(ramp, method="LSQ", degree=2, polytype="legendre",
                           raw_ramp_kwargs=dict(corr_refpix=True, corr_cosmics=True))

         # from an array, keeping the model to inspect residuals
         fitter = LSQFit(data, times=times, degree=2)
         result = fitter.fit(inplace=True)
         residuals = fitter.get_residuals()

      Main options: ``degree`` (polynomial degree, default 5) and
      ``polytype`` (``"legendre"``, ``"hermite"`` or ``"monomial"``).
      ``times`` must be given. `~mercupix.fitter.LSQFit.get_slope_and_offset`
      returns the flux and offset at ``t = 0`` from stored coefficients, and
      `~mercupix.fitter.LSQFit.scan_fit` fits the ramp truncated at an
      increasing number of groups.

      The flux at ``t = 0`` is an extrapolation of the polynomial: its scatter
      is about 2.5 times the degree-1 scatter at degree 2, and 10 times at
      degree 5. Use the lowest degree that follows your ramps.

   .. tab-item:: Brandt2024
      :sync: brandt

      .. code-block:: python

         from mercupix.fitter import fit_ramp, Brandt2024

         # from a raw ramp
         result = fit_ramp(ramp, method="Brandt", readnoise=10,
                           raw_ramp_kwargs=dict(corr_refpix=True, mask=True,
                                                mask_kwargs=dict(saturated=True, disconnected=False)),
                           fit_args=dict(jumps=True, jump_threshold=[20, 30]))

         # from an array
         result = Brandt2024(data, times=times, readnoise=10).fit(
             jumps=True, jump_threshold=[20, 30])

      Main options: ``readnoise`` (single read, in ADU; if not given, 11 ADU
      is used with a warning), ``nframes_list``
      (frames averaged per group), ``reset`` (first group measured from
      reset), and the ``fit`` arguments ``jumps`` and ``jump_threshold``
      (chi-square improvement needed to drop one or two differences).

   .. tab-item:: Kubik2015
      :sync: kubik

      .. code-block:: python

         from mercupix.fitter import Kubik2015

         from mercupix.fitter import fit_ramp, Kubik2015

         # from a raw ramp: readnoise is a fit() argument for Kubik2015
         result = fit_ramp(ramp, method="Kubik", fit_args=dict(readnoise=10))

         # from an array
         fitter = Kubik2015(data, times=times, macc=(ngroups, nframes, ndrops))
         result = fitter.fit(readnoise=10)      # ADU/s; per_second=False for ADU per group

      The flux is unbiased and, when photon noise dominates, as precise as
      `~mercupix.fitter.Brandt2024`.

      .. warning::

         The returned ``var`` overestimates the scatter of the flux when read
         noise matters (~1.7x at 100 ADU/s with 10 ADU of read noise, ~20x at
         1 ADU/s). Cosmic rays are not handled and the model has no offset.

See :doc:`ramp_fitting_validation` for how the fitters are tested, including
frame times from 0.5 to 15 s and skipped frames: `~mercupix.fitter.Kubik2015`
needs regular reads.

Which one should I use?
-----------------------

- **Science images**: `~mercupix.fitter.Brandt2024`. It is unbiased, has the
  lowest scatter (up to ~8 % lower than ``LSQ`` when photon noise dominates),
  ignores masked groups, and handles cosmic rays itself.
- **Ramps that are not linear** (no non-linearity correction available, or
  checking one): `~mercupix.fitter.LSQFit` with ``degree=2``. The flux at
  ``t = 0`` is then free of the curvature bias, but about 2.5 times noisier
  than a straight-line fit; `~mercupix.fitter.Fitter.get_residuals` shows
  what the polynomial does not capture. Linearizing the ramps first
  (`~mercupix.ramp.RawRamp.correct_nl`) and fitting them with
  `~mercupix.fitter.Brandt2024` keeps the lower noise.
- **Quick looks and large batches**: `~mercupix.fitter.LSQFit` with
  ``degree=1``, the fastest. Correct the cosmic rays first.

API
---

.. currentmodule:: mercupix.fitter

.. autosummary::
   :nosignatures:

   fit_ramp
   Fitter
   LSQFit
   Brandt2024
   Kubik2015
   RampFitResults
