Ramp fitting
============

A ramp fitter measures the slope of each pixel ramp. All fitters derive from
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
     - Any polynomial degree: absorbs curvature such as residual
       non-linearity.
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
     - ``slope``, ``offset``, ``nl``.
     - ``slope``, ``var``, ``qof`` (chi-square), ``mask2d``; ``offset``
       with ``intercept=True``.
     - ``slope``, ``var``.
   * - Speed :sup:`1`
     - 9 ms (degree 1), 12 ms (degree 2)
     - 29 ms, 0.5 s with ``jumps=True``
     - —
   * - Status
     - Stable.
     - Stable.
     - Experimental: ``fit`` currently fails.

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
      `~mercupix.fitter.LSQFit.scan_fit` fits the ramp truncated at an
      increasing number of groups.

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

      Main options: ``readnoise`` (single read, in ADU), ``nframes_list``
      (frames averaged per group), ``reset`` (first group measured from
      reset), and the ``fit`` arguments ``jumps`` and ``jump_threshold``
      (chi-square improvement needed to drop one or two differences).

   .. tab-item:: Kubik2015
      :sync: kubik

      .. code-block:: python

         from mercupix.fitter import Kubik2015

         fitter = Kubik2015(data, times=times, macc=(ngroups, nframes, ndrops))
         flux = fitter.estim_f(sigma_r=10)      # per group interval

      .. warning::

         ``Kubik2015.fit`` and ``fit_ramp(method="Kubik")`` currently fail,
         and ``estim_f`` returns the flux offset by ``2 * sigma_r**2``. See
         :doc:`../notebooks/compare_ramp_fitters`.

Which one should I use?
-----------------------

- **Science images**: `~mercupix.fitter.Brandt2024`. It is unbiased, has the
  lowest scatter (up to ~8 % lower than ``LSQ`` when photon noise dominates),
  ignores masked groups, and handles cosmic rays itself.
- **Ramps that are not linear** (e.g. checking a non-linearity correction):
  `~mercupix.fitter.LSQFit` with ``degree >= 2``. The higher-order
  coefficients are returned in ``result.nl`` and the residuals are available.
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
