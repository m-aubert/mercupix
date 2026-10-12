mercupix
========

**From raw HxRG ramps to calibrated images.**

``mercupix`` processes up-the-ramp data from HxRG infrared detectors. It
corrects the raw datacube (reference pixels, non-linearity, IPC, dark, cosmic
rays), fits each pixel ramp to measure the incoming flux, and returns a
calibrated 2D image.

.. code-block:: python

   from mercupix.ramp import RawRamp
   from mercupix.fitter import fit_ramp

   ramp = RawRamp.from_filename("my_ramp.fits", datatype="pyxim")
   result = fit_ramp(ramp, method="Brandt",
                     fit_args=dict(jumps=True, jump_threshold=[20, 30]))
   result.slope        # flux map [ADU/s]

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: sd-mt-4

   .. grid-item-card:: :octicon:`stack;1.2em` Ramp preprocessing
      :class-card: sd-border-0 sd-shadow-sm

      Reference pixel, offset, dark, classical non-linearity and IPC
      corrections, chained in a single call.

   .. grid-item-card:: :octicon:`graph;1.2em` Several ramp fitters
      :class-card: sd-border-0 sd-shadow-sm

      Polynomial least-squares, Kubik et al. (2016) and the optimal
      Brandt (2024) estimator, all returning the same result object.

   .. grid-item-card:: :octicon:`zap;1.2em` Cosmic rays
      :class-card: sd-border-0 sd-shadow-sm

      Detect and correct jumps before the fit, or drop them during the fit
      with a likelihood-based test.

   .. grid-item-card:: :octicon:`shield-check;1.2em` Data quality
      :class-card: sd-border-0 sd-shadow-sm

      Bitmasks for saturation, charge migration, hot, warm and disconnected
      pixels, and cosmic rays.

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/m-aubert/mercupix.git

See :doc:`installation` for development installs.

Where to go next
----------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.5em` Getting started
      :link: quickstart
      :link-type: doc
      :class-card: sd-shadow-sm

      From a raw file to a flux map in a few lines.

   .. grid-item-card:: :octicon:`book;1.5em` How-to guides
      :link: tutorials
      :link-type: doc
      :class-card: sd-shadow-sm

      Notebooks covering each step of the processing.

   .. grid-item-card:: :octicon:`git-compare;1.5em` Choosing a method
      :link: methods/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Side-by-side comparisons of methods that do the same job.

   .. grid-item-card:: :octicon:`code-square;1.5em` API reference
      :link: api/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Every class and function, grouped by processing step.

.. toctree::
   :hidden:
   :caption: Getting started

   installation
   quickstart
   concepts

.. toctree::
   :hidden:
   :caption: User guide

   tutorials
   methods/index

.. toctree::
   :hidden:
   :caption: Reference

   api/index
