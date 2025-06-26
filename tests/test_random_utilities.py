from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import graphix.random_objects as randobj
from graphix import linalg_validations as lv
from graphix.channels import KrausChannel
from graphix.ops import Ops
from graphix.sim.density_matrix import DensityMatrix

if TYPE_CHECKING:
    from numpy.random import Generator


class TestUtilities:
    def test_rand_herm(self, fx_rng: Generator) -> None:
        tmp = randobj.rand_herm(fx_rng.integers(2, 20), fx_rng)
        assert np.allclose(tmp, tmp.conj().T)

    # TODO : work on that. Verify an a random vector and not at the operator level...

    def test_rand_unit(self, fx_rng: Generator) -> None:
        d = fx_rng.integers(2, 20)
        tmp = randobj.rand_unit(d, fx_rng)
        print(type(tmp), tmp.dtype)

        # different default values for testing.assert_allclose and all_close!
        assert np.allclose(tmp @ tmp.conj().T, np.eye(d), atol=1e-15)
        assert np.allclose(tmp.conj().T @ tmp, np.eye(d), atol=1e-15)

    def test_random_channel_success(self, fx_rng: Generator) -> None:
        nqb = int(fx_rng.integers(1, 5))
        dim = 2**nqb  # fx_rng.integers(2, 8)

        # no rank feature
        channel = randobj.rand_channel_kraus(dim=dim, rng=fx_rng)

        assert isinstance(channel, KrausChannel)
        assert channel[0].operator.shape == (dim, dim)
        assert channel.nqubit == nqb
        assert len(channel) == dim**2

        rk = int(fx_rng.integers(1, dim**2 + 1))
        channel = randobj.rand_channel_kraus(dim=dim, rank=rk, rng=fx_rng)

        assert isinstance(channel, KrausChannel)
        assert channel[0].operator.shape == (dim, dim)
        assert channel.nqubit == nqb
        assert len(channel) == rk

    def test_random_channel_fail(self, fx_rng: Generator) -> None:
        # incorrect rank type
        with pytest.raises(TypeError):
            _ = randobj.rand_channel_kraus(dim=2**2, rank=3.0, rng=fx_rng)

        # null rank
        with pytest.raises(ValueError):
            _ = randobj.rand_channel_kraus(dim=2**2, rank=0, rng=fx_rng)

    def test_rand_gauss_cpx(self, fx_rng: Generator) -> None:
        nsample = int(1e4)

        dim = fx_rng.integers(2, 20)
        tmp = [randobj.rand_gauss_cpx_mat(dim=dim, rng=fx_rng) for _ in range(nsample)]

        dimset = {i.shape for i in tmp}
        assert len(dimset) == 1
        assert next(iter(dimset)) == (dim, dim)

    def test_check_psd_success(self, fx_rng: Generator) -> None:
        # Generate a random mixed state from state vectors with same probability
        # We know this is PSD

        nqb = fx_rng.integers(2, 7)

        dim = 2**nqb
        m = fx_rng.integers(1, dim)

        dm = np.zeros((dim,) * 2, dtype=np.complex128)

        # TODO optimize that
        for _ in range(m):
            psi = fx_rng.uniform(size=dim) + 1j * fx_rng.uniform(size=dim)
            psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
            dm += np.outer(psi, psi.conj()) / m

        assert lv.is_psd(dm)

    def test_check_psd_fail(self, fx_rng: Generator) -> None:
        # not hermitian
        # don't use dim = 2, too easy to have a PSD matrix.
        # NOTE useless test since eigvalsh treats the matrix as hermitian and takes only the L or U part

        lst = fx_rng.integers(5, 20)

        mat = fx_rng.uniform(size=(lst, lst)) + 1j * fx_rng.uniform(size=(lst, lst))

        # eigvalsh doesn't raise a LinAlgError since just use upper or lower part of the matrix.
        # instead Value error
        assert not lv.is_psd(mat)

        # hermitian but not positive eigenvalues
        mat = randobj.rand_herm(lst, rng=fx_rng)

        assert not lv.is_psd(mat)

    def test_rand_dm(self, fx_rng: Generator) -> None:
        # needs to be power of 2 dimension since builds a DM object
        dm = randobj.rand_dm(2 ** fx_rng.integers(2, 5), rng=fx_rng)

        assert isinstance(dm, DensityMatrix)
        assert lv.is_qubitop(dm.rho)
        assert lv.is_hermitian(dm.rho)
        assert lv.is_psd(dm.rho)
        assert lv.is_unit_trace(dm.rho)

    # try with incorrect dimension
    def test_rand_dm_fail(self, fx_rng: Generator) -> None:
        with pytest.raises(ValueError):
            _ = randobj.rand_dm(2 ** fx_rng.integers(2, 5) + 1, rng=fx_rng)

    def test_rand_dm_rank(self, fx_rng: Generator) -> None:
        rk = 3
        dm = randobj.rand_dm(2 ** fx_rng.integers(2, 5), rank=rk, rng=fx_rng)

        assert isinstance(dm, DensityMatrix)
        assert lv.is_qubitop(dm.rho)
        assert lv.is_hermitian(dm.rho)
        assert lv.is_psd(dm.rho)
        assert lv.is_unit_trace(dm.rho)

        evals = np.linalg.eigvalsh(dm.rho)

        evals[np.abs(evals) < 1e-15] = 0

        assert rk == np.count_nonzero(evals)

    # TODO move that somewhere else?
    def test_pauli_tensor_ops(self, fx_rng: Generator) -> None:
        nqb = int(fx_rng.integers(2, 6))
        pauli_tensor_ops = Ops.build_tensor_pauli_ops(nqb)

        assert len(pauli_tensor_ops) == 4**nqb

        dims = np.array([i.shape for i in pauli_tensor_ops])
        # or np.apply_along_axis ?
        assert np.all(dims == (2**nqb, 2**nqb))

    def test_pauli_tensor_ops_fail(self, fx_rng: Generator) -> None:
        with pytest.raises(TypeError):
            _ = Ops.build_tensor_pauli_ops(fx_rng.integers(2, 6) + 0.5)

        with pytest.raises(ValueError):
            _ = Ops.build_tensor_pauli_ops(0)

    def test_random_pauli_channel_success(self, fx_rng: Generator) -> None:
        nqb = int(fx_rng.integers(2, 6))
        rk = int(fx_rng.integers(1, 2**nqb + 1))
        pauli_channel = randobj.rand_pauli_channel_kraus(dim=2**nqb, rank=rk, rng=fx_rng)  # default is full rank

        assert isinstance(pauli_channel, KrausChannel)
        assert pauli_channel.nqubit == nqb
        assert len(pauli_channel) == rk

    def test_random_pauli_channel_fail(self, fx_rng: Generator) -> None:
        nqb = 3
        rk = 2
        with pytest.raises(TypeError):
            randobj.rand_pauli_channel_kraus(dim=2**nqb, rank=rk + 0.5, rng=fx_rng)

        with pytest.raises(TypeError):
            randobj.rand_pauli_channel_kraus(dim=2**nqb + 0.5, rank=rk, rng=fx_rng)

        with pytest.raises(ValueError):
            randobj.rand_pauli_channel_kraus(dim=2**nqb, rank=-3, rng=fx_rng)

        with pytest.raises(ValueError):
            randobj.rand_pauli_channel_kraus(dim=2**nqb + 1, rank=rk, rng=fx_rng)
