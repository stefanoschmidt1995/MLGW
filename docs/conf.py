# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# We are using the theme sphinxawesome.
#This file has some interesting configurations options:
#https://github.com/kai687/sphinxawesome-theme/blob/master/docs/conf.py

# -- Project information -----------------------------------------------------

project = "mlgw"
copyright = "2023, Stefano Schmidt"
author = "Stefano Schmidt"

# The full version, including alpha/beta/rc tags
release = "3.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon", #This is important to type properly the function parameters
    "myst_parser", #pip install myst-parser
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
#    "sphinxarg.ext",
	"sphinxcontrib.programoutput",
    "sphinx_sitemap", #pip install sphinx-sitemap
]

napoleon_custom_sections = [
    ("Class Attributes", "params_style"),
    ("Abstract Properties", "params_style"),
]

# Add any paths that contain templates here, relative to this directory.
#templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

#https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-toc_object_entries
toc_object_entries = False

	# This option is `True` by default
#html_awesome_code_headers = False

	#In case you want to add your logo
#html_logo = "assets/auto_awesome.svg"
#html_favicon = "assets/favicon-128x128.png"

html_theme_options = {
}



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ["_static"]

add_module_names = False

master_doc = "index"
autoclass_content = 'both'


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
