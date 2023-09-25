import unittest
import numpy as np
import tests.random_objects as randobj
from graphix.kraus import Channel
from graphix.Checks.channel_checks import check_data_dims

# from graphix.kraus import Channel, create_dephasing_channel, create_depolarising_channel


class TestUtilities(unittest.TestCase):

    # not 2**n as for QM but doesn't matter.
    def test_rand_herm(self):
        tmp = randobj.rand_herm(np.random.randint(2, 20))
        np.testing.assert_allclose(tmp, tmp.conj().T)

    def test_rand_unit(self):
        d = np.random.randint(2, 20)
        tmp = randobj.rand_unit(d)

        # check by applying to a random state
        # can compare both vectors directly since no global phase introduced in the computation.
        psi = np.random.rand(d) + 1j * np.random.rand(d)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        np.testing.assert_allclose(tmp @ tmp.conj().T @ psi, psi)
        np.testing.assert_allclose(tmp.conj().T @ tmp @ psi, psi)

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


if __name__ == "__main__":
    np.random.seed(23)
    unittest.main()
