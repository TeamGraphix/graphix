import unittest

import numpy as np

import tests.random_objects as randobj
from graphix.channels import KrausChannel
from graphix.linalg_validations import check_data_dims, check_hermitian, check_psd, check_square, check_unit_trace
from graphix.sim.density_matrix import DensityMatrix


class TestUtilities(unittest.TestCase):
    def test_rand_herm(self):
        tmp = randobj.rand_herm(np.random.randint(2, 20))
        np.testing.assert_allclose(tmp, tmp.conj().T)

    # TODO : work on that. Verify an a random vector and not at the operator level...

    def test_rand_unit(self):
        d = np.random.randint(2, 20)
        tmp = randobj.rand_unit(d)

        # check by applying to a random state
        # can compare both vectors directly since no global phase introduced in the computation.
        psi = np.random.rand(d) + 1j * np.random.rand(d)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        np.testing.assert_allclose(tmp @ tmp.conj().T @ psi, psi)
        np.testing.assert_allclose(tmp.conj().T @ tmp @ psi, psi)
        np.testing.assert_allclose(tmp.conj().T @ tmp @ psi, tmp @ tmp.conj().T @ psi)

        # direct assert equal identity doesn't seem to work. Precision issues?
        # np.testing.assert_allclose(tmp @ tmp.conj().T, np.eye(d))
        # np.testing.assert_allclose(tmp.conj().T @ tmp, np.eye(d))

    def test_random_channel_success(self):

        nqb = np.random.randint(1, 5)
        dim = 2**nqb  # np.random.randint(2, 8)

        # no rank feature
        channel = randobj.rand_channel_kraus(dim=dim)

        assert isinstance(channel, KrausChannel)
        assert check_data_dims(channel.kraus_ops)
        assert channel.kraus_ops[0]["operator"].shape == (dim, dim)
        assert channel.nqubit == nqb
        assert channel.size == dim**2
        assert channel.is_normalized

        rk = np.random.randint(1, dim**2 + 1)
        channel = randobj.rand_channel_kraus(dim=dim, rank=rk)

        assert isinstance(channel, KrausChannel)
        assert check_data_dims(channel.kraus_ops)
        assert channel.kraus_ops[0]["operator"].shape == (dim, dim)
        assert channel.nqubit == nqb
        assert channel.size == rk
        assert channel.is_normalized

    def test_random_channel_fail(self):

        # incorrect rank type
        with self.assertRaises(TypeError):
            mychannel = randobj.rand_channel_kraus(dim=2**2, rank=3.0)

        # null rank
        with self.assertRaises(ValueError):
            mychannel = randobj.rand_channel_kraus(dim=2**2, rank=0)

    def test_rand_gauss_cpx(self):

        nsample = int(1e4)

        dim = np.random.randint(2, 20)
        tmp = [randobj.rand_gauss_cpx_mat(dim=dim) for _ in range(nsample)]

        dimset = {i.shape for i in tmp}
        assert len(dimset) == 1
        assert list(dimset)[0] == (dim, dim)

    def test_check_psd_success(self):

        # Generate a random mixed state from state vectors with same probability
        # We know this is PSD

        nqb = np.random.randint(2, 7)

        dim = 2**nqb
        m = np.random.randint(1, dim)

        dm = np.zeros((dim,) * 2, dtype=np.complex128)

        # TODO optimize that
        for _ in range(m):
            psi = np.random.rand(dim) + 1j * np.random.rand(dim)
            psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
            dm += np.outer(psi, psi.conj()) / m

        assert check_psd(dm)

    def test_check_psd_fail(self):

        # not hermitian
        # don't use dim = 2, too easy to have a PSD matrix.
        # NOTE useless test since eigvalsh treats the matrix as hermitian and takes only the L or U part

        l = np.random.randint(5, 20)

        mat = np.random.rand(l, l) + 1j * np.random.rand(l, l)

        # eigvalsh doesn't raise a LinAlgError since just use upper or lower part of the matrix.
        # instead Value error
        with self.assertRaises(ValueError):
            check_psd(mat)

        # hermitian but not positive eigenvalues
        mat = randobj.rand_herm(l)

        with self.assertRaises(ValueError):
            check_psd(mat)

    def test_rand_dm(self):
        # needs to be power of 2 dimension since builds a DM object
        dm = randobj.rand_dm(2 ** np.random.randint(2, 5))

        assert isinstance(dm, DensityMatrix)
        assert check_square(dm.rho)
        assert check_hermitian(dm.rho)
        assert check_psd(dm.rho)
        assert check_unit_trace(dm.rho)

    # try with incorrect dimension
    def test_rand_dm_fail(self):
        with self.assertRaises(ValueError):
            dm = randobj.rand_dm(2 ** np.random.randint(2, 5) + 1)

    def test_rand_dm_rank(self):

        rk = 3
        dm = randobj.rand_dm(2 ** np.random.randint(2, 5), rank=rk)

        assert isinstance(dm, DensityMatrix)
        assert check_square(dm.rho)
        assert check_hermitian(dm.rho)
        assert check_psd(dm.rho)
        assert check_unit_trace(dm.rho)

        evals = np.linalg.eigvalsh(dm.rho)

        evals[np.abs(evals) < 1e-15] = 0

        assert rk == np.count_nonzero(evals)


if __name__ == "__main__":
    np.random.seed(2)
    unittest.main()
