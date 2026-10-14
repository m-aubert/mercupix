Choosing a method
=================

Several steps of the processing can be done with different methods. Each
page below puts the alternatives side by side: what they assume, what they
return, how to call them and when to use them.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`graph;1.2em` Ramp fitting
      :link: ramp_fitting
      :link-type: doc

      ``LSQFit``, ``Brandt2024`` and ``Kubik2015``.

   .. grid-item-card:: :octicon:`zap;1.2em` Cosmic rays
      :link: jumps
      :link-type: doc

      Inpainting before the fit (``get_cosmics``) or jump detection during
      the fit (``Brandt2024``).

   .. grid-item-card:: :octicon:`columns;1.2em` Reference pixels
      :link: refpix
      :link-type: doc

      Up-down / left-right correction or SIRS-like Fourier correction.

   .. grid-item-card:: :octicon:`checklist;1.2em` Ramp fitting: validation
      :link: ramp_fitting_validation
      :link-type: doc

      How the fitters are tested: frame times from 0.5 to 15 s, skipped
      frames, non-linearity.

.. toctree::
   :hidden:

   ramp_fitting
   ramp_fitting_validation
   jumps
   refpix
