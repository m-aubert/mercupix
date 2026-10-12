Cosmic rays
===========

A cosmic ray (or any sudden charge deposit) adds a *jump* to a ramp: all
groups after the hit are offset. Left untreated, it biases the fitted flux
high. ``mercupix`` handles jumps either **before** or **during** the fit.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 22 39 39

   * -
     - Before the fit: `~mercupix.mask.get_cosmics`
     - During the fit: `~mercupix.fitter.Brandt2024`
   * - How to enable
     - ``preprocess_ramp(corr_cosmics=True)`` or
       ``get_mask(cosmics=True, inpaint=True)``
     - ``Brandt2024.fit(jumps=True, jump_threshold=[20, 30])``
   * - Detection
     - Outliers in the group differences, relative to a moving median and in
       units of the robust scatter of the ramp.
     - Chi-square improvement when dropping one or two consecutive
       differences, with the read and photon noise model.
   * - Correction
     - Flagged differences are replaced by the moving median and the ramp is
       rebuilt (inpainting).
     - Flagged differences are dropped from the fit.
   * - Threshold
     - ``cr_threshold_mad`` (default 7).
     - ``jump_threshold`` (default ``[20, 30]``).
   * - Output
     - ``CR`` flag in the bitmask, per group.
     - ``result.mask2d``: number of differences kept, per pixel.
   * - Works with
     - Any fitter.
     - ``Brandt2024`` only.
   * - 50 % completeness :sup:`1`
     - ~150 ADU
     - ~70 ADU

:sup:`1` Read noise 10 ADU, 30 ADU/s, 30 groups of 2.8 s, default thresholds.
See :doc:`../notebooks/compare_jump_detection`.

Usage
-----

.. tab-set::

   .. tab-item:: Before the fit

      .. code-block:: python

         from mercupix.fitter import fit_ramp

         result = fit_ramp(
             ramp, method="LSQ", degree=1,
             raw_ramp_kwargs=dict(corr_refpix=True, corr_cosmics=True,
                                  mask=True, mask_kwargs=dict(cosmics=True,
                                                              cr_threshold_mad=7,
                                                              disconnected=False)),
         )

      Or directly on a datacube:

      .. code-block:: python

         from mercupix.mask import get_cosmics

         mask, corrected = get_cosmics(data, inpaint=True, cr_threshold_mad=7)

   .. tab-item:: During the fit

      .. code-block:: python

         from mercupix.fitter import fit_ramp

         result = fit_ramp(
             ramp, method="Brandt", readnoise=10,
             fit_args=dict(jumps=True, jump_threshold=[20, 30]),
         )
         hit = result.mask2d < ramp.ngroups - 1     # pixels with dropped differences

Which one should I use?
-----------------------

- With `~mercupix.fitter.Brandt2024`, use its jump detection: it detects
  fainter jumps for the same false detection rate, because it knows the
  expected noise of each difference.
- With any other fitter, or to get the cosmic rays in the data quality mask,
  use `~mercupix.mask.get_cosmics`.
- The first group has no difference: ``get_cosmics`` flags it by sigma
  clipping across the frame, which also catches bright sources. Ignore
  ``mask[0]`` when counting cosmic rays.

API
---

.. currentmodule:: mercupix

.. autosummary::
   :nosignatures:

   ~mask.get_cosmics
   ~mask.get_window
   ~fitter.Brandt2024
   ~mask.get_cosmics_dep
