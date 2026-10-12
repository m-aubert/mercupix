Key concepts
============

Ramps, groups and resultants
----------------------------

An HxRG detector is read non-destructively while it integrates: each pixel
accumulates charge, and the detector samples it repeatedly *up the ramp*. A
**ramp** is the resulting datacube of shape ``(ngroups, ny, nx)``. Each
**group** (or *resultant*) is a read, or the average of several reads.
``ramp.times`` gives the time of each group.

The flux of a pixel is the **slope** of its ramp, in ADU per unit of
``times``.

Detector geometry
-----------------

.. list-table::
   :widths: 30 70

   * - **Reference pixels**
     - A border of ``nrefpixels`` pixels (4 by default) on each side of the
       frame. They are not sensitive to light, but they see the same
       electronic offsets as the active pixels.
   * - **Active area**
     - The light-sensitive pixels, of shape ``ramp.active_shape``.
       ``ramp.actslice`` selects it along each spatial axis. Most processed
       products are cropped to it.
   * - **Readout channels**
     - The columns are split in ``nchannels`` channels of
       ``npix_per_channel`` columns, read in parallel. Odd channels are read
       in the opposite direction.

Processing steps
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Step
     - What it does
     - Main objects
   * - **Load**
     - Read the datacube, the group times and the number of channels.
     - `~mercupix.baseramp.BaseRamp.from_filename`, `mercupix.io`
   * - **Reference pixels**
     - Subtract the offsets common to the reference and active pixels.
     - `~mercupix.ramp.RawRamp.correct_refpix`, `mercupix.sirs`
   * - **Offset, dark**
     - Subtract a bias frame and the dark current.
     - `~mercupix.ramp.RawRamp.subtract_offset`,
       `~mercupix.ramp.RawRamp.correct_dark`
   * - **Non-linearity**
     - Linearize the ramps with a polynomial correction.
     - `~mercupix.ramp.RawRamp.correct_nl`,
       `~mercupix.carac.ClassicalNL`
   * - **IPC**
     - Deconvolve the inter-pixel capacitance kernel.
     - `~mercupix.ramp.RawRamp.correct_ipc`
   * - **Masking**
     - Flag bad pixels and groups, detect and correct cosmic rays.
     - `~mercupix.ramp.RawRamp.get_mask`, `mercupix.mask`
   * - **Ramp fitting**
     - Measure the slope of each pixel ramp.
     - `~mercupix.fitter.fit_ramp`, `mercupix.fitter`
   * - **Image**
     - Store and post-process the flux map.
     - `~mercupix.image.Exposure`

`~mercupix.ramp.RawRamp.preprocess_ramp` chains all the steps before the
fit, and `~mercupix.fitter.fit_ramp` adds the fit.

Data quality flags
------------------

Masks are integer bitmasks of dtype ``uint32``. Each problem sets one bit, so
a pixel can carry several flags. `~mercupix.mask.get_boolmask` converts a
bitmask to a boolean mask, e.g. ``get_boolmask(mask, ignore_flags="~CR")``
keeps the cosmic rays only.

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Flag
     - Value
     - Meaning
   * - ``GOOD``
     - 0
     - Good pixel.
   * - ``DIS``
     - 1
     - Disconnected (dead) pixel.
   * - ``CR``
     - 2
     - Cosmic ray (jump).
   * - ``SATURATED``
     - 4
     - Saturated.
   * - ``HOT``
     - 8
     - Hot pixel.
   * - ``WARM``
     - 16
     - Warm pixel.
   * - ``CM``
     - 32
     - Charge migration or snowball candidate (neighbor of a saturated pixel).
   * - ``REF``
     - 64
     - Reference pixel.
   * - ``RC``
     - 128
     - RC pixel.
   * - ``NL``
     - 256
     - Non-linearity issue.
   * - ``RTN``
     - 512
     - Random telegraph noise.
