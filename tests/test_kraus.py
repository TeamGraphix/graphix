import unittest

import numpy as np

import tests.random_objects as randobj
from graphix.channels import (
    KrausChannel,
    two_qubit_depolarising_channel,
    dephasing_channel,
    depolarising_channel,
)
from graphix.ops import Ops


class TestChannel(unittest.TestCase):
    """Tests for Channel class"""

    def test_init_with_data_success(self):
        "test for successful intialization"

        prob = np.random.rand()
        mychannel = KrausChannel(
            [
                {"coef": np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                {"coef": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
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
            mychannel = KrausChannel([])

        # incorrect parameter type
        with self.assertRaises(TypeError):
            mychannel = KrausChannel("a")

        # incorrect "parameter" key
        with self.assertRaises(KeyError):
            mychannel = KrausChannel(
                [
                    {"coefficients": np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"coef": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # incorrect "operator" key
        with self.assertRaises(KeyError):
            mychannel = KrausChannel(
                [
                    {"coef": np.sqrt(1 - prob), "oertor": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"coef": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # incorrect parameter type
        with self.assertRaises(TypeError):
            mychannel = KrausChannel(
                [
                    {"coef": "a", "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"coef": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # incorrect operator type
        with self.assertRaises(TypeError):
            mychannel = KrausChannel(
                [
                    {"coef": np.sqrt(1 - prob), "operator": "a"},
                    {"coef": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # incorrect operator dimension
        with self.assertRaises(ValueError):
            mychannel = KrausChannel(
                [
                    {"coef": np.sqrt(1 - prob), "operator": np.array([1.0, 0.0])},
                    {"coef": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # incorrect operator dimension: square but not qubits
        with self.assertRaises(ValueError):
            mychannel = KrausChannel(
                [
                    {"coef": np.sqrt(1 - prob), "operator": np.random.rand(3, 3)},
                    {"coef": np.sqrt(prob), "operator": np.random.rand(3, 3)},
                ]
            )

        # doesn't square to 1. Not normalized. Parameter.
        with self.assertRaises(ValueError):
            mychannel = KrausChannel(
                [
                    {"coef": 2 * np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"coef": np.sqrt(prob), "operator": np.array([[1.0, 0.0], [0.0, -1.0]])},
                ]
            )

        # doesn't square to 1. Not normalized. Operator.
        with self.assertRaises(ValueError):
            mychannel = KrausChannel(
                [
                    {"coef": np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
                    {"coef": np.sqrt(prob), "operator": np.array([[1.0, 3.0], [0.0, -1.0]])},
                ]
            )

        # incorrect rank (number of kraus_operators)
        # use a random channel to do that.
        with self.assertRaises(ValueError):
            randobj.rand_channel_kraus(dim=2**2, rank=20)

    def test_dephasing_channel(self):

        prob = np.random.rand()
        data = [
            {"coef": np.sqrt(1 - prob), "operator": np.array([[1.0, 0.0], [0.0, 1.0]])},
            {"coef": np.sqrt(prob), "operator": Ops.z},
        ]
        dephase_channel = dephasing_channel(prob)
        assert isinstance(dephase_channel, KrausChannel)
        assert dephase_channel.nqubit == 1
        assert dephase_channel.size == 2
        assert dephase_channel.is_normalized

        for i in range(len(dephase_channel.kraus_ops)):
            np.testing.assert_allclose(dephase_channel.kraus_ops[i]["coef"], data[i]["coef"])
            np.testing.assert_allclose(dephase_channel.kraus_ops[i]["operator"], data[i]["operator"])

    def test_depolarising_channel(self):

        prob = np.random.rand()
        data = [
            {"coef": np.sqrt(1 - prob), "operator": np.eye(2)},
            {"coef": np.sqrt(prob / 3.0), "operator": Ops.x},
            {"coef": np.sqrt(prob / 3.0), "operator": Ops.y},
            {"coef": np.sqrt(prob / 3.0), "operator": Ops.z},
        ]

        depol_channel = depolarising_channel(prob)

        assert isinstance(depol_channel, KrausChannel)
        assert depol_channel.nqubit == 1
        assert depol_channel.size == 4
        assert depol_channel.is_normalized

        for i in range(len(depol_channel.kraus_ops)):
            np.testing.assert_allclose(depol_channel.kraus_ops[i]["coef"], data[i]["coef"])
            np.testing.assert_allclose(depol_channel.kraus_ops[i]["operator"], data[i]["operator"])

    def test_2_qubit_depolarising_channel(self):

        prob = np.random.rand()
        data = [
            {"coef": 1 - prob, "operator": np.kron(np.eye(2), np.eye(2))},
            {"coef": prob / 3.0, "operator": np.kron(Ops.x, Ops.x)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.y, Ops.y)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.z, Ops.z)},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(Ops.x, np.eye(2))},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(Ops.y, np.eye(2))},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(Ops.z, np.eye(2))},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.x)},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.y)},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.z)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.x, Ops.y)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.x, Ops.z)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.y, Ops.x)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.y, Ops.z)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.z, Ops.x)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.z, Ops.y)},
        ]

        depol_channel_2_qubit = two_qubit_depolarising_channel(prob)

        assert isinstance(depol_channel_2_qubit, KrausChannel)
        assert depol_channel_2_qubit.nqubit == 2
        assert depol_channel_2_qubit.size == 16
        assert depol_channel_2_qubit.is_normalized

        for i in range(len(depol_channel_2_qubit.kraus_ops)):
            np.testing.assert_allclose(depol_channel_2_qubit.kraus_ops[i]["coef"], data[i]["coef"])
            np.testing.assert_allclose(depol_channel_2_qubit.kraus_ops[i]["operator"], data[i]["operator"])


if __name__ == "__main__":
    np.random.seed(2)
    unittest.main()
