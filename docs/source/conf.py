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
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "networkx": ("https://networkx.github.io/documentation/stable/", None),
}

sys.path.insert(0, os.path.abspath("../../"))


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_title = " "  # title for documentation (shown in sidebar, kept empty)

html_static_path = ["_static"]

html_context = {
    "mode": "production",
}

# code highlighting for light and dark themes
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# customizing theme options
html_theme_options = {
    "light_logo": "black_with_name.png",
    "dark_logo": "white_with_text.png",
}

sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": ["../../examples"],
    # path to where to save gallery generated output
    "gallery_dirs": ["gallery"],
    "filename_pattern": "/",
    "thumbnail_size": (800, 550),
}

suppress_warnings = ["config.cache"]
