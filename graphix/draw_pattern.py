"""Helper module for drawing pattern."""

import io
import subprocess
import warnings
from pathlib import Path

import PIL

from graphix import Pattern


def latex_file_to_image(tmpdirname: Path, tmpfilename: Path) -> PIL.Image.Image:
    """Convert a latex file located in `tmpdirname/tmpfilename` to an image representation."""
    try:
        subprocess.run(
            [
                "pdflatex",
                "-halt-on-error",
                f"-output-directory={tmpdirname}",
                f"{tmpfilename}.tex",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except OSError as exc:
        # OSError should generally not occur, because it's usually only triggered if `pdflatex`
        # doesn't exist as a command, but we've already checked that.
        raise Exception("`pdflatex` command could not be run.") from exc
    except subprocess.CalledProcessError as exc:
        with Path("latex_error.log").open("wb") as error_file:
            error_file.write(exc.stdout)
        warnings.warn(
            "Unable to compile LaTeX. Perhaps you are missing the `qcircuit` package."
            " The output from the `pdflatex` command is in `latex_error.log`.",
            stacklevel=2,
        )
        raise Exception("`pdflatex` call did not succeed: see `latex_error.log`.") from exc
    base = Path(tmpdirname) / tmpfilename
    try:
        subprocess.run(
            ["pdftocairo", "-singlefile", "-png", "-q", base.with_suffix(".pdf"), base],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        message = "`pdftocairo` failed to produce an image."
        warnings.warn(message, stacklevel=2)
        raise Exception(message) from exc

    def trim(image) -> PIL.Image.Image:
        """Trim a PIL image and remove white space."""
        background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = PIL.ImageChops.difference(image, background)
        diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            image = image.crop(bbox)
        return image

    return trim(PIL.Image.open(base.with_suffix(".png")))


def pattern_to_latex_document(pattern: Pattern, left_to_right: bool) -> str:
    """Generate a latex document with the latex representation of the pattern written in it.

    Parameters
    ----------
    left_to_right: bool
        whether or not represent the pattern from left to right representation. Default is left to right, otherwise it's right to left
    """
    header_1 = r"\documentclass[border=2px]{standalone}" + "\n"

    header_2 = r"""
\usepackage{graphicx}

\begin{document}
"""

    output = io.StringIO()
    output.write(header_1)
    output.write(header_2)

    output.write(pattern.to_latex(left_to_right))

    output.write("\n\\end{document}")
    contents = output.getvalue()
    output.close()

    return contents
