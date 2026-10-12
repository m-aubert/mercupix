How-to guides
=============

Each guide is a Jupyter notebook that runs on a small simulated ramp, so you
can reproduce it without data. Use the download button at the top of a page
to run it yourself.

Processing a ramp
-----------------

Follow the processing chain, one step per guide.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Load and inspect a ramp
      :link: notebooks/howto_load_ramp
      :link-type: doc

      Read a raw file, check the detector geometry, look at frames and
      pixel ramps.

   .. grid-item-card:: Preprocess a ramp
      :link: notebooks/howto_preprocess
      :link-type: doc

      Reference pixels, data quality mask and cosmic ray correction.

   .. grid-item-card:: Fit a ramp and build an image
      :link: notebooks/howto_fit_ramp
      :link-type: doc

      ``fit_ramp``, the fit result, residuals and the 2D exposure.

Comparing methods
-----------------

Quantitative comparisons on simulations with a known truth. See also
:doc:`methods/index`.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Ramp fitters
      :link: notebooks/compare_ramp_fitters
      :link-type: doc

      Bias, precision, speed and robustness to cosmic rays of
      ``LSQFit``, ``Brandt2024`` and ``Kubik2015``.

   .. grid-item-card:: Cosmic ray strategies
      :link: notebooks/compare_jump_detection
      :link-type: doc

      ``get_cosmics`` before the fit or ``Brandt2024`` during the fit:
      completeness, false detections and flux recovery.

.. toctree::
   :hidden:
   :caption: Processing a ramp

   notebooks/howto_load_ramp
   notebooks/howto_preprocess
   notebooks/howto_fit_ramp

.. toctree::
   :hidden:
   :caption: Comparing methods

   notebooks/compare_ramp_fitters
   notebooks/compare_jump_detection
