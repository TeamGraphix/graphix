from __future__ import annotations
import numpy as np
import pytest
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphix.sim.statevec import Statevec
from graphix.states import BasicStates

def test_draw_text():
    sv = Statevec(data=[BasicStates.ZERO, BasicStates.ONE])
    # to_dict for 01 with MSB should be {'01': 1+0j}
    res = sv.draw(format=\"text\")
    assert \"01\" in res
    assert \"(1+0j)\" in res or \"(1.0+0j)\" in res

def test_draw_text_multiple():
    sv = Statevec(nqubit=1, data=BasicStates.PLUS)
    # to_dict for |+> should be {'0': 1/sqrt(2), '1': 1/sqrt(2)}
    res = sv.draw(format=\"text\")
    assert \"0.707\" in res
    assert \"|0>\" in res
    assert \"|1>\" in res
    assert \" + \" in res

def test_draw_latex():
    sv = Statevec(data=[BasicStates.ZERO, BasicStates.ONE])
    res = sv.draw(format=\"latex\")
    # We just check if it returns a Latex object and has the expected string
    try:
        from IPython.display import Latex
        assert isinstance(res, Latex)
        assert r\"\\lvert 01 \\rangle\" in res.data
    except ImportError:
        # If IPython is not available, we skip this test or just check the logic
        pass

def test_draw_invalid_format():
    sv = Statevec(nqubit=1)
    with pytest.raises(ValueError):
        sv.draw(format=\"invalid\")
