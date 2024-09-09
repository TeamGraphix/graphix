from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import graphix.random_objects as randobj
from graphix.channels import (
    KrausChannel,
    KrausData,
    dephasing_channel,
    depolarising_channel,
    two_qubit_depolarising_channel,
    two_qubit_depolarising_tensor_channel,
)
from graphix.ops import Ops

if TYPE_CHECKING:
    from numpy.random import Generator


class TestChannel:
    """Tests for Channel class."""

    def test_init_with_data_success(self, fx_rng: Generator) -> None:
        """Test for successful intialization."""
        prob = fx_rng.uniform()
        mychannel = KrausChannel(
            [
                KrausData(np.sqrt(1 - prob), np.array([[1.0, 0.0], [0.0, 1.0]])),
                KrausData(np.sqrt(prob), np.array([[1.0, 0.0], [0.0, -1.0]])),
            ],
        )
        assert isinstance(mychannel.nqubit, int)
        assert mychannel.nqubit == 1
        assert len(mychannel) == 2

    def test_init_with_data_fail(self, fx_rng: Generator) -> None:
        """Test for unsuccessful intialization."""
        prob = fx_rng.uniform()

        # empty data
        with pytest.raises(ValueError):
            _ = KrausChannel([])

        # incorrect operator dimension
        with pytest.raises(ValueError):
            _ = KrausChannel(
                [
                    KrausData(np.sqrt(1 - prob), np.array([1.0, 0.0])),
                    KrausData(np.sqrt(prob), np.array([[1.0, 0.0], [0.0, -1.0]])),
                ],
            )

        # incorrect operator dimension: square but not qubits
        with pytest.raises(ValueError):
            _ = KrausChannel(
                [
                    KrausData(np.sqrt(1 - prob), fx_rng.uniform(size=(3, 3))),
                    KrausData(np.sqrt(prob), fx_rng.uniform(size=(3, 3))),
                ],
            )

        # doesn't square to 1. Not normalized. Parameter.
        with pytest.raises(ValueError):
            _ = KrausChannel(
                [
                    KrausData(2 * np.sqrt(1 - prob), np.array([[1.0, 0.0], [0.0, 1.0]])),
                    KrausData(np.sqrt(prob), np.array([[1.0, 0.0], [0.0, -1.0]])),
                ],
            )

        # doesn't square to 1. Not normalized. Operator.
        with pytest.raises(ValueError):
            _ = KrausChannel(
                [
                    KrausData(np.sqrt(1 - prob), np.array([[1.0, 0.0], [0.0, 1.0]])),
                    KrausData(np.sqrt(prob), np.array([[1.0, 3.0], [0.0, -1.0]])),
                ],
            )

        # incorrect rank (number of kraus_operators)
        # use a random channel to do that.
        with pytest.raises(ValueError):
            randobj.rand_channel_kraus(dim=2**2, rank=20, rng=fx_rng)

    def test_dephasing_channel(self, fx_rng: Generator) -> None:
        prob = fx_rng.uniform()
        data = [
            KrausData(np.sqrt(1 - prob), np.array([[1.0, 0.0], [0.0, 1.0]])),
            KrausData(np.sqrt(prob), Ops.z),
        ]
        dephase_channel = dephasing_channel(prob)
        assert isinstance(dephase_channel, KrausChannel)
        assert dephase_channel.nqubit == 1
        assert len(dephase_channel) == 2

        for i in range(len(dephase_channel)):
            assert np.allclose(dephase_channel[i].coef, data[i].coef)
            assert np.allclose(dephase_channel[i].operator, data[i].operator)

    def test_depolarising_channel(self, fx_rng: Generator) -> None:
        prob = fx_rng.uniform()
        data = [
            KrausData(np.sqrt(1 - prob), np.eye(2)),
            KrausData(np.sqrt(prob / 3.0), Ops.x),
            KrausData(np.sqrt(prob / 3.0), Ops.y),
            KrausData(np.sqrt(prob / 3.0), Ops.z),
        ]

        depol_channel = depolarising_channel(prob)

        assert isinstance(depol_channel, KrausChannel)
        assert depol_channel.nqubit == 1
        assert len(depol_channel) == 4

        for i in range(len(depol_channel)):
            assert np.allclose(depol_channel[i].coef, data[i].coef)
            assert np.allclose(depol_channel[i].operator, data[i].operator)

    def test_2_qubit_depolarising_channel(self, fx_rng: Generator) -> None:
        prob = fx_rng.uniform()
        data = [
            KrausData(np.sqrt(1 - prob), np.kron(np.eye(2), np.eye(2))),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.x, Ops.x)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.z, Ops.z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.x, np.eye(2))),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.y, Ops.y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.y, np.eye(2))),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.z, np.eye(2))),
            KrausData(np.sqrt(prob / 15.0), np.kron(np.eye(2), Ops.x)),
            KrausData(np.sqrt(prob / 15.0), np.kron(np.eye(2), Ops.y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(np.eye(2), Ops.z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.x, Ops.y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.x, Ops.z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.y, Ops.x)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.y, Ops.z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.z, Ops.y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.z, Ops.x)),
        ]

        depol_channel_2_qubit = two_qubit_depolarising_channel(prob)

        assert isinstance(depol_channel_2_qubit, KrausChannel)
        assert depol_channel_2_qubit.nqubit == 2
        assert len(depol_channel_2_qubit) == 16

        for i in range(len(depol_channel_2_qubit)):
            assert np.allclose(depol_channel_2_qubit[i].coef, data[i].coef)
            assert np.allclose(depol_channel_2_qubit[i].operator, data[i].operator)

    def test_2_qubit_depolarising_tensor_channel(self, fx_rng: Generator) -> None:
        prob = fx_rng.uniform()
        data = [
            KrausData(1 - prob, np.kron(np.eye(2), np.eye(2))),
            KrausData(prob / 3.0, np.kron(Ops.x, Ops.x)),
            KrausData(prob / 3.0, np.kron(Ops.y, Ops.y)),
            KrausData(prob / 3.0, np.kron(Ops.z, Ops.z)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.x, np.eye(2))),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.y, np.eye(2))),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.z, np.eye(2))),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(np.eye(2), Ops.x)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(np.eye(2), Ops.y)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(np.eye(2), Ops.z)),
            KrausData(prob / 3.0, np.kron(Ops.x, Ops.y)),
            KrausData(prob / 3.0, np.kron(Ops.x, Ops.z)),
            KrausData(prob / 3.0, np.kron(Ops.y, Ops.x)),
            KrausData(prob / 3.0, np.kron(Ops.y, Ops.z)),
            KrausData(prob / 3.0, np.kron(Ops.z, Ops.x)),
            KrausData(prob / 3.0, np.kron(Ops.z, Ops.y)),
        ]

        depol_tensor_channel_2_qubit = two_qubit_depolarising_tensor_channel(prob)

        assert isinstance(depol_tensor_channel_2_qubit, KrausChannel)
        assert depol_tensor_channel_2_qubit.nqubit == 2
        assert len(depol_tensor_channel_2_qubit) == 16

        for i in range(len(depol_tensor_channel_2_qubit)):
            assert np.allclose(depol_tensor_channel_2_qubit[i].coef, data[i].coef)
            assert np.allclose(depol_tensor_channel_2_qubit[i].operator, data[i].operator)
