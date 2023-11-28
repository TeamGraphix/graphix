import unittest
import numpy as np
import tests.random_objects as randobj
from graphix.kraus import Channel
from graphix.Checks.channel_checks import check_data_dims
from graphix.Checks.generic_checks import check_psd, check_unit_trace, check_hermitian, check_square
from graphix.sim.density_matrix import DensityMatrix

# from graphix.kraus import Channel, create_dephasing_channel, create_depolarising_channel


class TestUtilities(unittest.TestCase):

    # not 2**n as for QM but doesn't matter.
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

        assert isinstance(channel, Channel)
        assert check_data_dims(channel.kraus_ops)
        # just in case. Done in check_data_dims, check_square at instantiation.
        assert channel.kraus_ops[0]["operator"].shape == (dim, dim)
        assert channel.nqubit == nqb
        assert channel.size == dim**2
        assert channel.is_normalized

        # check rank feature. Eq (15) of [KNPPZ21]always satisfied with rk = M
        rk = np.random.randint(1, dim**2 + 1)
        channel = randobj.rand_channel_kraus(dim=dim, rank=rk)

        assert isinstance(channel, Channel)
        assert check_data_dims(channel.kraus_ops)
        # just in case. Done in check_data_dims, check_square at instantiation.
        assert channel.kraus_ops[0]["operator"].shape == (dim, dim)
        assert channel.nqubit == nqb
        assert channel.size == rk
        assert channel.is_normalized

        # NOTE test sigma feature??

    def test_random_channel_fail(self):
        # don't need to test for normalization.
        # If not normalized, the Channel can't be created! checks done there.

        # incorrect rank type
        with self.assertRaises(TypeError):
            mychannel = randobj.rand_channel_kraus(dim=2**2, rank=3.0)

        # null rank
        with self.assertRaises(ValueError):
            mychannel = randobj.rand_channel_kraus(dim=2**2, rank=0)

    def test_rand_gauss_cpx(self):

        nsample = int(1e5)

        # don't need to be qubit type
        dim = np.random.randint(2, 20)

        # default parameters test
        tmp = [randobj.rand_gauss_cpx_mat(dim=dim) for _ in range(nsample)]

        # set comprehension
        dimset = {i.shape for i in tmp}
        assert len(dimset) == 1
        assert list(dimset)[0] == (dim, dim)

        # guess this is useless since np.random.normal has been tested....

        # # variances real and imag add so if same, std takes a sqrt(2) factor.
        # np.testing.assert_allclose(np.std(tmp, axis = 0), np.full((dim,) * 2, 1.), rtol=0, atol=1e-2)

        # # TODO update and check test (nsample, atol)
        # sigm = 0.1 + np.random.rand()
        # tmp = [randobj.rand_gauss_cpx_mat(dim = dim, sig = sigm) for _ in range(nsample)]

        # # set comprehension
        # dimset = {i.shape for i in tmp}
        # assert len(dimset) == 1
        # assert list(dimset)[0] == (dim, dim)

        # # guess this is useless since np.random.normal has been tested....

        # # variances real and imag add so if same, std takes a sqrt(2) factor.
        # np.testing.assert_allclose(np.std(tmp, axis = 0), np.full((dim,) * 2, np.sqrt(2.)*sigm), rtol=0, atol=1e-2)

    # TODO add (complete) positivity test! Via Cholesky? Qitip mentions possible problems.

    # do it here or elsewhere?
    # tested out of graphix. Think since need a PSD matrix to test..

    def test_check_psd_success(self):

        # Generate a random mixed state from state vectors (equiprobable though)
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

        # assert check_square(dm)
        # assert check_hermitian(dm)
        assert check_psd(dm)
        # assert check_unit_trace(dm)

    def test_check_psd_fail(self):

        # not hermitian
        # don't use dim = 2, too easy to have a PSD matrix.
        # NOTE useless test since eigvalsh treats the matrix as hermitian and takes only the L or U part

        l = np.random.randint(5, 20)

        mat = np.random.rand(l, l) + 1j * np.random.rand(l, l)

        # eigvalsh doesn't raise a LinAlgError since just use upper or lower part of the matrix.
        # with self.assertRaises(np.linalg.LinAlgError):
        # instead Value error
        with self.assertRaises(ValueError):
            check_psd(mat)

        # hermitian but not positive eigenvalues
        mat = randobj.rand_herm(l)

        with self.assertRaises(ValueError):  # or LinAlgError?
            check_psd(mat)

    def test_rand_dm(self):
        # needs to be power of 2 dimension since builds a DM object
        dm = randobj.rand_dm(2 ** np.random.randint(2, 5))

        assert isinstance(dm, DensityMatrix)
        assert check_square(dm.rho)
        assert check_hermitian(dm.rho)
        assert check_psd(dm.rho)
        assert check_unit_trace(dm.rho)

    def test_rand_dm_rank(self):
        # check the rank feature

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
