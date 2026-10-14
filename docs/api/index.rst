API reference
=============

Objects are grouped by processing step, in the order they are used. The
:doc:`modules` page lists every module with all its content.

.. currentmodule:: mercupix

Ramps
-----

.. autosummary::
   :nosignatures:

   ~ramp.RawRamp
   ~baseramp.BaseRamp

Reading data
------------

.. autosummary::
   :nosignatures:

   ~baseramp.BaseRamp.from_filename
   ~io.from_pyxim
   ~io.from_roman
   ~io.from_ip2i
   ~io.from_anyfits

Preprocessing
-------------

.. autosummary::
   :nosignatures:

   ~ramp.RawRamp.preprocess_ramp
   ~ramp.RawRamp.correct_refpix
   ~ramp.RawRamp.subtract_offset
   ~ramp.RawRamp.correct_dark
   ~ramp.RawRamp.correct_nl
   ~ramp.RawRamp.correct_ipc
   ~ramp.RawRamp.get_variance

Reference pixels
~~~~~~~~~~~~~~~~

See :doc:`../methods/refpix` for a comparison.

.. autosummary::
   :nosignatures:

   ~refpixels.get_refpix_ud
   ~refpixels.get_refpix_lr
   ~sirs.train_sirs
   ~sirs.save_sirs
   ~sirs.get_sirs_ref

Data quality and cosmic rays
----------------------------

See :doc:`../methods/jumps` for a comparison of the cosmic ray strategies.

.. autosummary::
   :nosignatures:

   ~ramp.RawRamp.get_mask
   ~mask.get_boolmask
   ~mask.get_saturated
   ~mask.get_charge_migration
   ~mask.get_downpix
   ~mask.get_hotpix
   ~mask.detect_hotpix
   ~mask.get_cosmics

Ramp fitting
------------

See :doc:`../methods/ramp_fitting` for a comparison of the fitters.

.. autosummary::
   :nosignatures:

   ~fitter.fit_ramp
   ~fitter.RampFitResults
   ~fitter.Fitter
   ~fitter.LSQFit
   ~fitter.Brandt2024
   ~fitter.Kubik2015

Images
------

.. autosummary::
   :nosignatures:

   ~image.Exposure
   ~image.MetaFrame

Characterization
----------------

.. autosummary::
   :nosignatures:

   ~carac.ClassicalNL
   ~carac.get_ptc
   ~carac.get_psd
   ~carac.get_readout_sequence
   ~calib.PersistenceModel

Simulation
----------

Mock ramps with a known truth, for tests and examples.

.. autosummary::
   :nosignatures:

   ~mock.mock_ramps
   ~mock.mock_times

.. toctree::
   :hidden:

   modules
