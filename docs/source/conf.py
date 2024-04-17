# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "graphix"
copyright = "2022, Team Graphix"
author = "Shinichi Sunami"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosectionlabel_prefix_document = True


sys.path.insert(0, os.path.abspath("../.."))


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_rtd_theme"
# html_theme = 'pydata_sphinx_theme'
html_static_path = ["_static"]
html_logo = "../logo/white_with_text.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": ["../../examples"],
    # path to where to save gallery generated output
    "gallery_dirs": ["gallery"],
    "filename_pattern": "/",
    "thumbnail_size": (800, 550),
}
html_context = {
    "css_files": [
        "_static/basic.css",
        "_static/pygments.css",
        "_static/sg_gallery-binder.css",
        "_static/sg_gallery-dataframe.css",
        "_static/sg_gallery-rendered-html.css",
        "_static/sg_gallery.css",
        "_static/css/badge_only.css",
        "_static/css/theme.css",
        "_static/css/my_theme.css",
        "_static/css/custom.css",
    ],
}
