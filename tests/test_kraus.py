import unittest

import numpy as np

import tests.random_objects as randobj
from graphix.channels import (
    Channel,
    create_2_qubit_dephasing_channel,
    create_2_qubit_depolarising_channel,
    create_dephasing_channel,
    create_depolarising_channel,
)
from graphix.ops import Ops


class TestChannel(unittest.TestCase):
    """Tests for Channel class"""

    def test_init_with_data_success(self):
        "test for successful intialization"

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
            randobj.rand_channel_kraus(dim=2**2, rank=20)

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

        depol_channel_2_qubit = create_2_qubit_dephasing_channel(prob)

        assert isinstance(depol_channel_2_qubit, Channel)
        assert depol_channel_2_qubit.nqubit == 2
        assert depol_channel_2_qubit.size == 4
        assert depol_channel_2_qubit.is_normalized

        for i in range(len(depol_channel_2_qubit.kraus_ops)):
            np.testing.assert_allclose(depol_channel_2_qubit.kraus_ops[i]["parameter"], data[i]["parameter"])
            np.testing.assert_allclose(depol_channel_2_qubit.kraus_ops[i]["operator"], data[i]["operator"])

    def test_2_qubit_depolarising_channel(self):

        prob = np.random.rand()
        data = [
            {"parameter": 1 - prob, "operator": np.kron(np.eye(2), np.eye(2))},
            {"parameter": prob / 3.0, "operator": np.kron(Ops.x, Ops.x)},
            {"parameter": prob / 3.0, "operator": np.kron(Ops.y, Ops.y)},
            {"parameter": prob / 3.0, "operator": np.kron(Ops.z, Ops.z)},
            {"parameter": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(Ops.x, np.eye(2))},
            {"parameter": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(Ops.y, np.eye(2))},
            {"parameter": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(Ops.z, np.eye(2))},
            {"parameter": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.x)},
            {"parameter": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.y)},
            {"parameter": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.z)},
            {"parameter": prob / 3.0, "operator": np.kron(Ops.x, Ops.y)},
            {"parameter": prob / 3.0, "operator": np.kron(Ops.x, Ops.z)},
            {"parameter": prob / 3.0, "operator": np.kron(Ops.y, Ops.x)},
            {"parameter": prob / 3.0, "operator": np.kron(Ops.y, Ops.z)},
            {"parameter": prob / 3.0, "operator": np.kron(Ops.z, Ops.x)},
            {"parameter": prob / 3.0, "operator": np.kron(Ops.z, Ops.y)},
        ]

        depol_channel_2_qubit = create_2_qubit_depolarising_channel(prob)

        assert isinstance(depol_channel_2_qubit, Channel)
        assert depol_channel_2_qubit.nqubit == 2
        assert depol_channel_2_qubit.size == 16
        assert depol_channel_2_qubit.is_normalized

        for i in range(len(depol_channel_2_qubit.kraus_ops)):
            np.testing.assert_allclose(depol_channel_2_qubit.kraus_ops[i]["parameter"], data[i]["parameter"])
            np.testing.assert_allclose(depol_channel_2_qubit.kraus_ops[i]["operator"], data[i]["operator"])


if __name__ == "__main__":
    np.random.seed(2)
    unittest.main()
