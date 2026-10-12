Installation
============

``mercupix`` requires Python 3.11 or later.

From GitHub
-----------

.. code-block:: bash

   pip install git+https://github.com/m-aubert/mercupix.git

For development
---------------

Clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/m-aubert/mercupix.git
   cd mercupix
   pip install -e ".[docs]"

Building the documentation
--------------------------

The ``docs`` extra installs Sphinx and the theme. Build the HTML pages with:

.. code-block:: bash

   cd docs
   make html

The pages are written to ``docs/_build/html``. The tutorial notebooks are
stored with their outputs and are not executed during the build.
