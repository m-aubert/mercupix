# Sphinx configuration for the mercupix documentation.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import datetime
import os
import sys

# Document the source tree (and not an older installed version).
sys.path.insert(0, os.path.abspath("../src"))

import mercupix  # noqa: E402

# -- Project information -----------------------------------------------------
project = "mercupix"
author = "Marie Aubert"
copyright = f"2025-{datetime.date.today().year}, {author}"
release = mercupix.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
]

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# `foo` in docstrings is a cross-reference to the Python object foo.
default_role = "py:obj"

# -- API (autodoc / autosummary / napoleon) ----------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "groupwise"
autoclass_content = "class"
autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
add_module_names = False
toc_object_entries_show_parents = "hide"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
# Link the usual numpydoc type names (ndarray, array_like, ..., optional).
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ndarray": "numpy.ndarray",
    "array_like": ":term:`numpy:array_like`",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
}

# -- Notebooks (myst-nb) -----------------------------------------------------
# Notebooks are stored with their outputs and are not re-executed.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath"]

# -- Copy button: strip prompts ----------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "mercupix"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "repository_url": "https://github.com/m-aubert/mercupix",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "show_toc_level": 2,
    "show_navbar_depth": 1,
    "home_page_in_toc": False,
    "navigation_with_keys": False,
}
