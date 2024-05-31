# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from dataclasses import asdict
from sphinxawesome_theme import ThemeOptions

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


html_theme = "sphinxawesome_theme"
extensions += [
    "sphinxawesome_theme.highlighting",
]
# html_theme = 'pydata_sphinx_theme'
html_static_path = ["_static"]

html_context = {
    "mode": "production",
}

theme_options = ThemeOptions(
    show_breadcrumbs=True,
    logo_dark="../logo/white.png",
    logo_light="../logo/black_with_name.png",
)

html_theme_options = asdict(theme_options)

sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": ["../../examples"],
    # path to where to save gallery generated output
    "gallery_dirs": ["gallery"],
    "filename_pattern": "/",
    "thumbnail_size": (800, 550),
}

