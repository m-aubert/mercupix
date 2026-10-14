Ramp fitting: validation
========================

The ramp fitters are checked against simulated ramps with a known flux, in
``tests/test_fitter.py``. This page summarizes what is tested and what the
tests show, in particular how the fitters behave with different frame times
and when some frames are skipped.

How the fitters are tested
--------------------------

The mock data come from `mercupix.mock`:

- `~mercupix.mock.mock_times` gives the read times on a frame clock of period
  ``dt`` (frame ``k`` read at ``k * dt``), optionally with **skipped frames**:
  frames that are not read, so that the time between two reads is a multiple
  of ``dt``.
- `~mercupix.mock.mock_ramps` simulates ramps read at those times: a constant
  flux accumulated with Poisson noise, an offset of 1000 ADU, 10 ADU of read
  noise, and an optional classical non-linearity.

.. code-block:: python

   from mercupix.mock import mock_ramps, mock_times

   times = mock_times(ngroups=30, dt=15., skip=10, seed=1)    # 30 reads, 10 frames skipped
   times, data = mock_ramps(flux=[[20.], [100.], [300.], [700.]], shape=(4, 1000),
                            times=times, nl_loss=0.2, seed=0)

Each test fits 4 fluxes (20, 100, 300 and 700 ADU/s), one per row, with 1000
pixels each. A fitter passes if, for every row, the mean fitted flux is within
5 standard errors of the true flux.

Frame times and skipped frames
------------------------------

Every fitter is run with 30 reads, for 6 frame times and 3 read patterns:

- frame times: 0.5, 1, 2.8, 5, 10 and 15 s;
- every frame read;
- every 3rd frame skipped (frames 3, 6, 9, ... not read);
- 10 frames skipped at random.

For linear ramps the fluxes are fixed in ADU/s, so the frame time changes the
noise regime: at 0.5 s the faint rows are dominated by read noise, at 15 s by
photon noise. For non-linear ramps the fluxes are scaled by
``2.8 s / frame time``, so that the ramps end at the same signal, below full
well, whatever the frame time.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 34 22 22 22

   * - Fitter (ramps)
     - Every frame read
     - Every 3rd frame skipped
     - 10 random frames skipped
   * - `~mercupix.fitter.LSQFit`, degree 1 (linear)
     - unbiased
     - unbiased
     - unbiased
   * - `~mercupix.fitter.LSQFit`, degree 2 (20 % non-linearity)
     - unbiased
     - unbiased
     - unbiased
   * - `~mercupix.fitter.Brandt2024` (linear)
     - unbiased
     - unbiased
     - unbiased
   * - `~mercupix.fitter.Kubik2015` (linear)
     - unbiased
     - **+5.2 %**
     - **+7.3 %**

The results are the same for every frame time from 0.5 to 15 s.

`~mercupix.fitter.LSQFit` and `~mercupix.fitter.Brandt2024` use the actual
read times, so uneven intervals between reads do not affect them.
`~mercupix.fitter.Kubik2015` estimates the flux from the second moment of the
group differences, assuming they all cover the same time. When frames are
skipped, some differences cover two or three frames, and the flux is
overestimated. Use it only with regular reads; MACC modes with a constant
number of dropped frames per group are regular.

Other checks
------------

The same file also checks:

- **non-linearity**: from degree 2, `~mercupix.fitter.LSQFit` returns an
  unbiased flux for non-linearity losses from 0 to 40 % at full well, with the
  Legendre, Hermite and monomial bases; at degree 1 the flux of bright pixels is
  biased low, as expected;
- **the formula itself**: pixel by pixel, the flux and offset of
  `~mercupix.fitter.LSQFit` match an independent polynomial fit
  (`numpy.polynomial.polynomial.polyfit`) at degrees 1 to 5;
- **input shapes**: 3D, 2D ``(ngroups, npix)`` and single-ramp inputs;
- **Kubik2015**: unbiased flux in ADU/s for 1 to 1000 ADU/s and 5 to 30 ADU of
  read noise, ``per_second`` conversion, precision within 5 % of
  `~mercupix.fitter.Brandt2024` when photon noise dominates, and
  `~mercupix.fitter.fit_ramp` support.

Known limitations, marked as expected failures in the tests:

- `~mercupix.fitter.Kubik2015` with skipped frames (above);
- the variance returned by `~mercupix.fitter.Kubik2015` overestimates the
  scatter of the flux when read noise matters (~1.7 times at 100 ADU/s with
  10 ADU of read noise, ~20 times at 1 ADU/s).

Running the tests
-----------------

.. code-block:: bash

   pytest tests/test_fitter.py                      # all fitter tests
   pytest tests/test_fitter.py -k frame_time        # frame times and skipped frames
