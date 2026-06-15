from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import graphix.random_objects as randobj
from graphix.channels import (
    KrausChannel,
    KrausData,
    amplitude_damping_channel,
    dephasing_channel,
    depolarising_channel,
    two_qubit_amplitude_damping_channel,
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
            KrausData(np.sqrt(prob), Ops.Z),
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
            KrausData(np.sqrt(1 - prob), Ops.I),
            KrausData(np.sqrt(prob / 3.0), Ops.X),
            KrausData(np.sqrt(prob / 3.0), Ops.Y),
            KrausData(np.sqrt(prob / 3.0), Ops.Z),
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
            KrausData(np.sqrt(1 - prob), np.kron(Ops.I, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.X)),
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
            KrausData(1 - prob, np.kron(Ops.I, Ops.I)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.Y)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.Z)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.X, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.Y, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.Z, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.X)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.Y)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.Y)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.Y)),
        ]

        depol_tensor_channel_2_qubit = two_qubit_depolarising_tensor_channel(prob)

        assert isinstance(depol_tensor_channel_2_qubit, KrausChannel)
        assert depol_tensor_channel_2_qubit.nqubit == 2
        assert len(depol_tensor_channel_2_qubit) == 16

        for i in range(len(depol_tensor_channel_2_qubit)):
            assert np.allclose(depol_tensor_channel_2_qubit[i].coef, data[i].coef)
            assert np.allclose(depol_tensor_channel_2_qubit[i].operator, data[i].operator)

    def test_amplitude_damping_channel(self, fx_rng: Generator) -> None:
        gamma = fx_rng.uniform(0, 0.5)
        sqrt_gamma = np.sqrt(gamma)
        sqrt_one_minus_gamma = np.sqrt(1 - gamma)
        data_k1 = np.array([[1, 0], [0, sqrt_one_minus_gamma]], dtype=np.complex128)
        data_k2 = np.array([[0, sqrt_gamma], [0, 0]], dtype=np.complex128)
        ad_channel = amplitude_damping_channel(gamma)
        assert isinstance(ad_channel, KrausChannel)
        assert ad_channel.nqubit == 1
        assert len(ad_channel) == 2
        assert np.allclose(ad_channel[0].operator, data_k1)
        assert np.allclose(ad_channel[0].coef, 1.0)
        assert np.allclose(ad_channel[1].operator, data_k2)
        assert np.allclose(ad_channel[1].coef, 1.0)

    def test_two_qubit_amplitude_damping_channel(self, fx_rng: Generator) -> None:
        gamma = fx_rng.uniform(0, 0.5)
        sqrt_gamma = np.sqrt(gamma)
        sqrt_one_minus_gamma = np.sqrt(1 - gamma)
        k1 = np.array([[1, 0], [0, sqrt_one_minus_gamma]], dtype=np.complex128)
        k2 = np.array([[0, sqrt_gamma], [0, 0]], dtype=np.complex128)
        ad2_channel = two_qubit_amplitude_damping_channel(gamma)
        assert isinstance(ad2_channel, KrausChannel)
        assert ad2_channel.nqubit == 2
        assert len(ad2_channel) == 4
        expected = [np.kron(k1, k1), np.kron(k1, k2), np.kron(k2, k1), np.kron(k2, k2)]
        for i, exp in enumerate(expected):
            assert np.allclose(ad2_channel[i].operator, exp)
            assert np.allclose(ad2_channel[i].coef, 1.0)

    def test_amplitude_damping_channel_edge_cases(self) -> None:
        # gamma=0 → only K1 (identity)
        c0 = amplitude_damping_channel(0.0)
        assert np.allclose(c0[0].operator, np.eye(2))
        assert np.allclose(c0[0].coef, 1.0)
        assert len(c0) == 2
        # gamma=1 → K1 = |0><0|, K2 = |0><1|
        c1 = amplitude_damping_channel(1.0)
        assert np.allclose(c1[0].operator, np.array([[1, 0], [0, 0]]))
        assert np.allclose(c1[1].operator, np.array([[0, 1], [0, 0]]))
        # invalid gamma
        with pytest.raises(ValueError):
            amplitude_damping_channel(-0.1)
        with pytest.raises(ValueError):
            amplitude_damping_channel(1.1)
