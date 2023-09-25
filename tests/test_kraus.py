import unittest

import numpy as np

from graphix.kraus import (
    Channel,
    create_2_qubit_dephasing_channel,
    create_dephasing_channel,
    create_depolarising_channel,
)
import tests.random_objects as randobj

from graphix.ops import Ops


class TestChannel(unittest.TestCase):
    """Tests for Channel class"""

    def test_init_with_data_success(self):
        "test for successful intialization"
        # TODO generate random data?
        prob = np.random.rand()
        mychannel = Channel(
            [
                {"parameter": np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                {"parameter": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
            ]
        )
        assert isinstance(mychannel.nqubit, int)
        assert mychannel.nqubit == 1
        assert mychannel.size == 2
        assert isinstance(mychannel.kraus_ops, (list, np.ndarray, tuple))
        assert mychannel.is_normalized

    def test_init_with_data_fail(self):
        "test for unsuccessful intialization"
        # TODO generate random data.

        prob = np.random.rand()

        # empty data
        with self.assertRaises(ValueError):
            mychannel = Channel([])

        # incorrect parameter type
        with self.assertRaises(TypeError):
            mychannel = Channel("a")

        # incorrect "parameter" key
        with self.assertRaises(KeyError):
            mychannel = Channel(
                [
                    {"parmer": np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"parameter": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )
        # incorrect "operator" key
        with self.assertRaises(KeyError):
            mychannel = Channel(
                [
                    {"parameter": np.sqrt(1 - prob), "oertor": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"parameter": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # incorrect parameter type
        with self.assertRaises(TypeError):
            mychannel = Channel(
                [
                    {"parameter": "a", "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"parameter": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # incorrect operator type
        with self.assertRaises(TypeError):
            mychannel = Channel(
                [
                    {"parameter": np.sqrt(1 - prob), "operator": "a"},
                    {"parameter": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # incorrect operator dimension
        with self.assertRaises(ValueError):
            mychannel = Channel(
                [
                    {"parameter": np.sqrt(1 - prob), "operator": np.array([1.0, 0.0])},
                    {"parameter": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # incorrect operator dimension: square but not qubits
        with self.assertRaises(ValueError):
            mychannel = Channel(
                [
                    {"parameter": np.sqrt(1 - prob), "operator": np.random.rand(3, 3)},
                    {"parameter": np.sqrt(prob), "operator": np.random.rand(3, 3)},
                ]
            )

        # doesn't square to 1. Not normalized. Parameter.
        with self.assertRaises(ValueError):
            mychannel = Channel(
                [
                    {"parameter": 2 * np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"parameter": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # doesn't square to 1. Not normalized. Operator.
        with self.assertRaises(ValueError):
            mychannel = Channel(
                [
                    {"parameter": np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"parameter": np.sqrt(prob), "operator": np.array([[1.0, 3.0], [0.0, -1.0]])},
                ]
            )

        # incorrect rank (number of kraus_operators)
        # use a random channel to do that.
        with self.assertRaises(ValueError):
            mychannel = randobj.rand_channel_kraus(dim=2**2, rank=20)

    def test_dephasing_channel(self):

        prob = np.random.rand()
        data = [
            {"parameter": np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
            {"parameter": np.sqrt(prob), "operator": Ops.z},
        ]
        dephase_channel = create_dephasing_channel(prob)
        assert isinstance(dephase_channel, Channel)
        assert dephase_channel.nqubit == 1
        assert dephase_channel.size == 2
        assert dephase_channel.is_normalized

        for i in range(len(dephase_channel.kraus_ops)):
            np.testing.assert_allclose(dephase_channel.kraus_ops[i]["parameter"], data[i]["parameter"])
            np.testing.assert_allclose(dephase_channel.kraus_ops[i]["operator"], data[i]["operator"])

    def test_depolarising_channel(self):

        prob = np.random.rand()
        data = [
            {"parameter": np.sqrt(1 - prob), "operator": np.eye(2)},
            {"parameter": np.sqrt(prob / 3.0), "operator": Ops.x},
            {"parameter": np.sqrt(prob / 3.0), "operator": Ops.y},
            {"parameter": np.sqrt(prob / 3.0), "operator": Ops.z},
        ]

        depol_channel = create_depolarising_channel(prob)

        assert isinstance(depol_channel, Channel)
        assert depol_channel.nqubit == 1
        assert depol_channel.size == 4
        assert depol_channel.is_normalized

        for i in range(len(depol_channel.kraus_ops)):
            np.testing.assert_allclose(depol_channel.kraus_ops[i]["parameter"], data[i]["parameter"])
            np.testing.assert_allclose(depol_channel.kraus_ops[i]["operator"], data[i]["operator"])

    def test_2_qubit_dephasing_channel(self):

        prob = np.random.rand()
        data = [
            {"parameter": np.sqrt(1 - prob), "operator": np.kron(np.eye(2), np.eye(2))},
            {"parameter": np.sqrt(prob / 3.0), "operator": np.kron(Ops.z, np.eye(2))},
            {"parameter": np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.z)},
            {"parameter": np.sqrt(prob / 3.0), "operator": np.kron(Ops.z, Ops.z)},
        ]

        dephase_channel_2_qubit = create_2_qubit_dephasing_channel(prob)

        assert isinstance(dephase_channel_2_qubit, Channel)
        assert dephase_channel_2_qubit.nqubit == 2
        assert dephase_channel_2_qubit.size == 4
        assert dephase_channel_2_qubit.is_normalized

        for i in range(len(dephase_channel_2_qubit.kraus_ops)):
            np.testing.assert_allclose(dephase_channel_2_qubit.kraus_ops[i]["parameter"], data[i]["parameter"])
            np.testing.assert_allclose(dephase_channel_2_qubit.kraus_ops[i]["operator"], data[i]["operator"])

    # def test_to_kraus_fail(self):
    #     A_wrong = [[0, 1, 2], [3, 4, 5]]
    #     A = [[0, 1], [2, 3]]

    #     # data type is invalid
    #     with self.assertRaises(TypeError):
    #         to_kraus(1)
    #     with self.assertRaises(TypeError):
    #         to_kraus("hello")
    #     with self.assertRaises(TypeError):
    #         to_kraus({})

    #     # (i) single unitary matrix A
    #     with self.assertRaises(ValueError):
    #         to_kraus([np.asarray(A_wrong, dtype=complex), 1])
    #     with self.assertRaises(ValueError):
    #         to_kraus([A_wrong, 1])
    #     with self.assertRaises(ValueError):
    #         to_kraus([A, 1j])

    #     # (ii) single Kraus set
    #     with self.assertRaises(ValueError):
    #         to_kraus([[np.asarray(A_wrong, dtype=complex)]])
    #     with self.assertRaises(ValueError):
    #         to_kraus([[A_wrong]])
    #     with self.assertRaises(ValueError):
    #         to_kraus([[]])
    #     with self.assertRaises(ValueError):
    #         to_kraus([A, A])
    #     with self.assertRaises(ValueError):
    #         to_kraus([[A, A], [A, A]])
    #     with self.assertRaises(AssertionError):
    #         to_kraus([[A, 1], [A, A]])

    # def test_to_kraus_success(self):
    #     # (i) single unitary matrix A
    #     A = [[0, 1], [2, 3]]
    #     kraus = to_kraus((A, 1))
    #     np.testing.assert_array_equal(kraus[0].data, np.asarray(A, dtype=complex))
    #     self.assertEqual(kraus[0].qarg, 1)

    #     kraus = to_kraus((np.asarray(A, dtype=complex), 1))
    #     np.testing.assert_array_equal(kraus[0].data, np.asarray(A, dtype=complex))
    #     self.assertEqual(kraus[0].qarg, 1)

    #     # (ii) single Kraus set
    #     B = [[4, 5], [6, 7]]
    #     kraus = to_kraus([(A, 1), (B, 2)])
    #     np.testing.assert_array_equal(kraus[0].data, np.asarray(A, dtype=complex))
    #     np.testing.assert_array_equal(kraus[1].data, np.asarray(B, dtype=complex))
    #     self.assertEqual(kraus[0].qarg, 1)
    #     self.assertEqual(kraus[1].qarg, 2)

    #     kraus = to_kraus([(np.asarray(A, dtype=complex), 1), (np.asarray(B, dtype=complex), 2)])
    #     np.testing.assert_array_equal(kraus[0].data, np.asarray(A, dtype=complex))
    #     np.testing.assert_array_equal(kraus[1].data, np.asarray(B, dtype=complex))
    #     self.assertEqual(kraus[0].qarg, 1)
    #     self.assertEqual(kraus[1].qarg, 2)

    # def test_generate_dephasing_kraus_fail(self):
    #     with self.assertRaises(AssertionError):
    #         generate_dephasing_kraus(2, 1)
    #     with self.assertRaises(AssertionError):
    #         generate_dephasing_kraus(0.5, "1")

    # def test_generate_dephasing_kraus_success(self):
    #     p = 0.5
    #     qarg = 1
    #     dephase_kraus = generate_dephasing_kraus(p, qarg)
    #     np.testing.assert_array_equal(dephase_kraus[0].data, np.asarray(np.sqrt(1 - p) * np.eye(2), dtype=complex))
    #     np.testing.assert_array_equal(dephase_kraus[1].data, np.asarray(np.sqrt(p) * np.diag([1, -1]), dtype=complex))
    #     self.assertEqual(dephase_kraus[0].qarg, qarg)
    #     self.assertEqual(dephase_kraus[1].qarg, qarg)

    # def test__is_kraus_op_fail(self):
    #     np.testing.assert_equal(_is_kraus_op(1), False)
    #     np.testing.assert_equal(_is_kraus_op("hello"), False)
    #     np.testing.assert_equal(_is_kraus_op([]), False)
    #     np.testing.assert_equal(_is_kraus_op([[], []]), False)
    #     np.testing.assert_equal(_is_kraus_op([[0, 1, 2], [3, 4, 5]]), False)

    # def test__is_kraus_op_success(self):
    #     A = [[0, 1], [2, 3]]
    #     np.testing.assert_equal(_is_kraus_op((A, 1)), True)
    #     np.testing.assert_equal(_is_kraus_op((np.asarray(A, dtype=complex), 1)), True)


if __name__ == "__main__":
    np.random.seed(2)
    unittest.main()
