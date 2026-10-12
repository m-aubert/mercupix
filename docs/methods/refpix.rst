Reference pixels
================

The readout electronics add offsets that drift from read to read and are
correlated across pixels. Reference pixels, insensitive to light, see the
same offsets and are used to remove them. ``mercupix`` provides two
corrections.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 22 39 39

   * -
     - Up-down / left-right
     - SIRS-like (Fourier)
   * - Functions
     - `~mercupix.ramp.RawRamp.correct_refpix`,
       `~mercupix.refpixels.get_refpix_ud`,
       `~mercupix.refpixels.get_refpix_lr`
     - `~mercupix.sirs.train_sirs`, `~mercupix.sirs.get_sirs_ref`
   * - Model
     - One offset per readout channel and group (top and bottom reference
       rows), plus one offset per row and group (left and right reference
       columns, smoothed with a Savitzky-Golay filter).
     - Frequency-dependent weights ``alpha`` and ``beta`` applied to the
       Fourier transforms of the left and right reference columns, for each
       channel.
   * - Calibration
     - None.
     - Trained on a set of dark ramps.
   * - Noise removed
     - Channel offsets and row-correlated (low frequency) noise.
     - Correlated noise up to higher frequencies, following the readout
       order.
   * - Status
     - Used by ``preprocess_ramp(corr_refpix=True)``.
     - Experimental: training tuned for IP2I data, not yet part of
       ``preprocess_ramp``.

Usage
-----

.. tab-set::

   .. tab-item:: Up-down / left-right

      .. code-block:: python

         corrected = ramp.correct_refpix()       # (ngroups, ny_act, nx_act)
         offsets = ramp.get_refpix_map()         # the subtracted correction

         # Within the full preprocessing:
         data, var, mask = ramp.preprocess_ramp(corr_refpix=True)

      The left-right smoothing is set by ``polyorder`` and ``red_fac``,
      passed as keyword arguments to ``correct_refpix``.

   .. tab-item:: SIRS-like

      .. code-block:: python

         from mercupix.sirs import train_sirs, save_sirs, get_sirs_ref

         # 1. Train on dark ramps: list of (filename, frame_path)
         alpha, beta, fft_basis, results, freqs = train_sirs(darks, nroh=12, njobs=5)
         n = ny * (npix_per_channel + 12)          # pixels in a channel readout sequence
         save_sirs(fft_basis, alpha, beta, freqs, n, results, filepath="sirs.h5")

         # 2. Correction of one channel, in readout order, from the
         #    reference columns of a full frame
         ref = get_sirs_ref(frame, fft_basis, alpha[channel], beta[channel], nroh=12)

Which one should I use?
-----------------------

Use the up-down / left-right correction, which needs no calibration and is
part of the standard preprocessing. The SIRS-like correction is for
characterizing and reducing correlated noise when dark ramps are available;
`~mercupix.carac.get_psd` measures the noise power spectrum to check its
effect.

API
---

.. currentmodule:: mercupix

.. autosummary::
   :nosignatures:

   ~ramp.RawRamp.correct_refpix
   ~baseramp.BaseRamp.get_refpix_map
   ~refpixels.get_refpix_ud
   ~refpixels.get_refpix_lr
   ~sirs.train_sirs
   ~sirs.get_sirs_ref
   ~carac.get_psd
