# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations
from typing import Any, Literal

from sphinx.application import Sphinx

import os
import sys

import shutil
import subprocess
import tempfile
from pathlib import Path

from sphinx.errors import ExtensionError

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "graphix"
copyright = "2022 -- 2026, Team Graphix"
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
    "jupyter_sphinx",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    # "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.github.io/documentation/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

sys.path.insert(0, os.path.abspath("../../"))


def skip(
    app: Sphinx,
    what: Literal["module", "class", "exception", "function", "method", "attribute"],
    name: str,
    obj: Any,
    would_skip: bool,
    options: dict[str, bool],
) -> bool:
    if name == "__init__":
        return False
    return would_skip


CIRCUITS_DIR = Path(__file__).parent / "tutorials" / "plots" / "circuits"


def build_circuits(app: Sphinx) -> None:
    """Compile every circuits/*.tex into a sibling .svg (skipped if up to date)."""
    tex_files = sorted(CIRCUITS_DIR.glob("*.tex"))

    for tex in tex_files:
        svg = tex.with_suffix(".svg")
        if svg.exists() and svg.stat().st_mtime >= tex.stat().st_mtime:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp}", tex.name],
                    cwd=CIRCUITS_DIR,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise ExtensionError(f"Failed to build pdf. {tex.name}:\n{e.stdout}\n{e.stderr}") from e
            try:
                subprocess.run(
                    ["pdf2svg", str(Path(tmp) / f"{tex.stem}.pdf"), str(svg)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise ExtensionError(f"Failed to build svg {tex.name}:\n{e.stdout}\n{e.stderr}") from e


def setup(app: Sphinx) -> None:
    app.connect("autodoc-skip-member", skip)
    app.connect("builder-inited", build_circuits)


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

default_role = "any"

# sphinx_gallery_conf = {
#     # path to your example scripts
#     "examples_dirs": ["../../examples"],
#     # path to where to save gallery generated output
#     "gallery_dirs": ["gallery"],
#     "filename_pattern": "/",
#     "thumbnail_size": (800, 550),
#     "parallel": True,
# }

suppress_warnings = ["config.cache"]

mathjax3_config = {
    "loader": {"load": ["[tex]/braket"]},
    "tex": {"packages": {"[+]": ["braket"]}},
}
# For LaTeX/PDF output:
latex_elements = {
    "preamble": r"\usepackage{braket}",
}
bibtex_bibfiles = ["references.bib"]  # path relative to conf.py
bibtex_default_style = "alpha"

plot_formats = ["svg"]  # HTML only: one sharp vector output
plot_html_show_source_link = False  # hides the "Source code" link
plot_html_show_formats = False  # hides the "png", "hires.png", "pdf" links

