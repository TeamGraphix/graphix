"""MBQC simulator.

Simulates MBQC by executing the pattern.

"""

from __future__ import annotations

import abc
import logging
import warnings
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, overload

# assert_never introduced in Python 3.11
# override introduced in Python 3.12
from typing_extensions import assert_never, override

from graphix import command
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.clifford import Clifford
from graphix.command import BaseM, CommandKind, N
from graphix.sim import (
    Backend,
    DensityMatrixBackend,
    StatevectorBackend,
    TensorNetworkBackend,
)
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.random import Generator

    from graphix.command import BaseN
    from graphix.measurements import Measurement, Outcome
    from graphix.noise_models.noise_model import CommandOrNoise, NoiseModel
    from graphix.pattern import Pattern
    from graphix.sim import Data, DensityMatrix, MBQCTensorNet, Statevec

logger = logging.getLogger(__name__)

_BuiltinBackend = DensityMatrixBackend | StatevectorBackend | TensorNetworkBackend
_BackendLiteral = Literal["statevector", "densitymatrix", "tensornetwork", "mps"]

if TYPE_CHECKING:
    _BuiltinBackendState = DensityMatrix | MBQCTensorNet | Statevec

    _StateT = TypeVar("_StateT")

# This type variable should be defined outside TYPE_CHECKING block
# because it appears in the parameters of `PatternSimulator`.
_StateT_co = TypeVar("_StateT_co", covariant=True)


class PrepareMethod(abc.ABC):
    """Prepare method used by the simulator.

    See `DefaultPrepareMethod` for the default prepare method that implements MBQC.

    To be overwritten by custom preparation methods in the case of delegated QC protocols.

    Example: class `ClientPrepareMethod` in https://github.com/qat-inria/veriphix
    """

    @abc.abstractmethod
    def prepare(self, backend: Backend[_StateT_co], cmd: BaseN, rng: Generator | None = None) -> None:
        """Prepare a node."""


class DefaultPrepareMethod(PrepareMethod):
    """Default prepare method implementing standard preparation for MBQC."""

    @override
    def prepare(self, backend: Backend[_StateT_co], cmd: BaseN, rng: Generator | None = None) -> None:
        """Prepare a node."""
        if not isinstance(cmd, N):
            raise TypeError("The default prepare method requires all preparation commands to be of type `N`.")
        backend.add_nodes(nodes=[cmd.node], data=cmd.state)


class MeasureMethod(abc.ABC):
    """Measure method used by the simulator, with default measurement method that implements MBQC.

    To be overwritten by custom measurement methods in the case of delegated QC protocols.

    Example: class `ClientMeasureMethod` in https://github.com/qat-inria/veriphix
    """

    def measure(
        self,
        backend: Backend[_StateT_co],
        cmd: BaseM,
        noise_model: NoiseModel | None = None,
        rng: Generator | None = None,
    ) -> None:
        """Perform a measure."""
        description = self.describe_measurement(cmd)
        result = backend.measure(cmd.node, description, rng=rng)
        logger.debug("Measure: %s", result)
        if noise_model is not None:
            result = noise_model.confuse_result(cmd, result, rng=rng)
        self.store_measurement_outcome(cmd.node, result)

    @abc.abstractmethod
    def describe_measurement(self, cmd: BaseM) -> Measurement:
        """Return the description of the measurement performed by a command.

        Parameters
        ----------
        cmd : BaseM
            Measurement command whose description is required.

        Returns
        -------
        Measurement
            Plane and angle actually used by the backend.
        """
        ...

    @abc.abstractmethod
    def measurement_outcome(self, node: int) -> Outcome:
        """Return the result of a previous measurement.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.

        Returns
        -------
        bool
            Recorded measurement outcome.
        """
        ...

    @abc.abstractmethod
    def store_measurement_outcome(self, node: int, result: Outcome) -> None:
        """Store the result of a previous measurement.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.
        result : bool
            Measurement outcome to store.
        """
        ...

    def check_domain(self, domain: Iterable[int]) -> bool:
        """Check that the measurement outcomes match the domain condition.

        Parameters
        ----------
        domain : Iterable[int]
            domain on which to compute the condition for applying conditional commands.
        """
        return sum(self.measurement_outcome(j) for j in domain) % 2 == 1


class DefaultMeasureMethod(MeasureMethod):
    """Default measurement method implementing standard measurement plane/angle update for MBQC."""

    results: dict[int, Outcome]

    def __init__(self, results: Mapping[int, Outcome] | None = None):
        """Initialize with an optional result dictionary.

        Parameters
        ----------
        results : Mapping[int, Outcome] | None, optional
            Mapping of previously measured nodes to their results. If ``None``,
            an empty dictionary is created.

        Notes
        -----
        If a mapping is provided, it is treated as read-only. Measurements
        performed during simulation are stored in `self.results`, which is a copy
        of the given mapping. The original `results` mapping is not modified.
        """
        # results is coerced into dict, since `store_measurement_outcome` mutates it.
        self.results = {} if results is None else dict(results)

    @override
    def describe_measurement(self, cmd: BaseM) -> Measurement:
        """Return the description of the measurement performed by ``cmd``.

        Parameters
        ----------
        cmd : BaseM
            Measurement command whose plane and angle should be updated.

        Returns
        -------
        Measurement
            Updated measurement specification.
        """
        assert isinstance(cmd, command.M)
        # extract signals for adaptive angle
        s_signal = sum(self.results[j] for j in cmd.s_domain) % 2
        t_signal = sum(self.results[j] for j in cmd.t_domain) % 2
        measurement = cmd.measurement
        if s_signal:
            measurement = measurement.clifford(Clifford.X)
        if t_signal:
            measurement = measurement.clifford(Clifford.Z)
        return measurement

    @override
    def measurement_outcome(self, node: int) -> Outcome:
        """Return the result of a previous measurement.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.

        Returns
        -------
        Outcome
            Stored measurement outcome.
        """
        return self.results[node]

    @override
    def store_measurement_outcome(self, node: int, result: Outcome) -> None:
        """Store the result of a previous measurement.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.
        result : bool
            Measurement outcome to store.
        """
        self.results[node] = result


class PatternSimulator(Generic[_StateT_co]):
    """MBQC simulator.

    Executes the measurement pattern.
    """

    noise_model: NoiseModel | None
    backend: Backend[_StateT_co]

    @overload
    def __init__(
        self: PatternSimulator[Statevec],
        pattern: Pattern,
        backend: Literal["statevector"] = ...,
        prepare_method: PrepareMethod | None = None,
        measure_method: MeasureMethod | None = None,
        noise_model: NoiseModel | None = None,
        branch_selector: BranchSelector | None = None,
        graph_prep: str | None = None,
        symbolic: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self: PatternSimulator[DensityMatrix],
        pattern: Pattern,
        backend: Literal["densitymatrix"] = ...,
        prepare_method: PrepareMethod | None = None,
        measure_method: MeasureMethod | None = None,
        noise_model: NoiseModel | None = None,
        branch_selector: BranchSelector | None = None,
        graph_prep: str | None = None,
        symbolic: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self: PatternSimulator[MBQCTensorNet],
        pattern: Pattern,
        backend: Literal["tensornetwork", "mps"] = ...,
        prepare_method: PrepareMethod | None = None,
        measure_method: MeasureMethod | None = None,
        noise_model: NoiseModel | None = None,
        branch_selector: BranchSelector | None = None,
        graph_prep: str | None = None,
        symbolic: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self: PatternSimulator[_StateT],
        pattern: Pattern,
        backend: Backend[_StateT] = ...,
        prepare_method: PrepareMethod | None = None,
        measure_method: MeasureMethod | None = None,
        noise_model: NoiseModel | None = None,
        branch_selector: BranchSelector | None = None,
        graph_prep: str | None = None,
        symbolic: bool = False,
    ) -> None: ...

    def __init__(
        self: PatternSimulator[_StateT | _BuiltinBackendState],
        pattern: Pattern,
        backend: Backend[_StateT] | _BackendLiteral = "statevector",
        prepare_method: PrepareMethod | None = None,
        measure_method: MeasureMethod | None = None,
        noise_model: NoiseModel | None = None,
        branch_selector: BranchSelector | None = None,
        graph_prep: str | None = None,
        symbolic: bool = False,
    ) -> None:
        """
        Construct a pattern simulator.

        Parameters
        ----------
        pattern: :class:`Pattern` object
            MBQC pattern to be simulated.
        backend: :class:`Backend` object,
            or 'statevector', or 'densitymatrix', or 'tensornetwork'
            simulation backend (optional), default is 'statevector'.
        prepare_method: :class:`PrepareMethod`, optional
            Prepare method used by the simulator. Default is :class:`DefaultPrepareMethod`.
        measure_method: :class:`MeasureMethod`, optional
            Measure method used by the simulator. Default is :class:`DefaultMeasureMethod`.
        noise_model: :class:`NoiseModel`, optional
            [Density matrix backend only] Noise model used by the simulator.
        branch_selector: :class:`BranchSelector`, optional
            Branch selector used for measurements. Can only be specified if ``backend`` is not an already instantiated :class:`Backend` object.  Default is :class:`RandomBranchSelector`.
        graph_prep: str, optional
            [Tensor network backend only] Strategy for preparing the graph state.  See :class:`TensorNetworkBackend`.
        symbolic : bool, optional
            [State vector and density matrix backends only] If True, support arbitrary objects (typically, symbolic expressions) in measurement angles.

        .. seealso:: :class:`graphix.sim.statevec.StatevectorBackend`\
            :class:`graphix.sim.tensornet.TensorNetworkBackend`\
            :class:`graphix.sim.density_matrix.DensityMatrixBackend`\
        """
        self.backend = _initialize_backend(pattern, backend, noise_model, branch_selector, graph_prep, symbolic)
        self.noise_model = noise_model
        self.__pattern = pattern
        if prepare_method is None:
            prepare_method = DefaultPrepareMethod()
        self.__prepare_method = prepare_method
        if measure_method is None:
            measure_method = DefaultMeasureMethod(pattern.results)
        self.__measure_method = measure_method

    @property
    def pattern(self) -> Pattern:
        """Return the pattern."""
        return self.__pattern

    @property
    def measure_method(self) -> MeasureMethod:
        """Return the measure method."""
        return self.__measure_method

    def run(self, input_state: Data = BasicStates.PLUS, rng: Generator | None = None) -> None:
        """Perform the simulation.

        Returns
        -------
        input_state: Data, optional
            the output quantum state,
            in the representation depending on the backend used.
            Default: ``|+>``.
        rng: Generator, optional
            Random-number generator for measurements.
            This generator is used only in case of random branch selection
            (see :class:`RandomBranchSelector`).
        """
        if input_state is not None:
            self.backend.add_nodes(self.pattern.input_nodes, input_state)
        if self.noise_model is None:
            pattern: Iterable[CommandOrNoise] = self.pattern
        else:
            pattern = self.noise_model.input_nodes(self.pattern.input_nodes, rng=rng) if input_state is not None else []
            pattern.extend(self.noise_model.transpile(self.pattern, rng=rng))

        # We check runnability first to provide clearer error messages and
        # to catch these errors before starting the simulation.
        self.pattern.check_runnability()

        logger.debug("Initial state: %s", self.backend.state)

        for cmd in pattern:
            logger.debug("Command: %s", cmd)
            if cmd.kind == CommandKind.N:
                self.__prepare_method.prepare(self.backend, cmd, rng=rng)
            elif cmd.kind == CommandKind.E:
                self.backend.entangle_nodes(edge=cmd.nodes)
            elif cmd.kind == CommandKind.M:
                self.__measure_method.measure(self.backend, cmd, noise_model=self.noise_model, rng=rng)
            # Use of `==` here for mypy
            elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
                if self.__measure_method.check_domain(cmd.domain):
                    self.backend.correct_byproduct(cmd)
            elif cmd.kind == CommandKind.C:
                self.backend.apply_clifford(cmd.node, cmd.clifford)
            elif cmd.kind == CommandKind.T:
                # The T command is a flag for one clock cycle in a simulated
                # experiment, added via a hardware-agnostic
                # pattern modifier. Noise models can perform special
                # handling of ticks during noise transpilation.
                pass
            elif cmd.kind == CommandKind.ApplyNoise:
                if cmd.domain is None or self.__measure_method.check_domain(cmd.domain):
                    self.backend.apply_noise(cmd)
            elif cmd.kind == CommandKind.S:
                raise ValueError("S commands unexpected in simulated patterns.")
            else:
                assert_never(cmd.kind)
            logger.debug("State: %s", self.backend.state)
        self.backend.finalize(output_nodes=self.pattern.output_nodes)


@overload
def _initialize_backend(
    pattern: Pattern,
    backend: StatevectorBackend | Literal["statevector"],
    noise_model: NoiseModel | None,
    branch_selector: BranchSelector | None,
    graph_prep: str | None,
    symbolic: bool,
) -> StatevectorBackend: ...


@overload
def _initialize_backend(
    pattern: Pattern,
    backend: DensityMatrixBackend | Literal["densitymatrix"],
    noise_model: NoiseModel | None,
    branch_selector: BranchSelector | None,
    graph_prep: str | None,
    symbolic: bool,
) -> DensityMatrixBackend: ...


@overload
def _initialize_backend(
    pattern: Pattern,
    backend: TensorNetworkBackend | Literal["tensornetwork", "mps"],
    noise_model: NoiseModel | None,
    branch_selector: BranchSelector | None,
    graph_prep: str | None,
    symbolic: bool,
) -> TensorNetworkBackend: ...


@overload
def _initialize_backend(
    pattern: Pattern,
    backend: Backend[_StateT_co],
    noise_model: NoiseModel | None,
    branch_selector: BranchSelector | None,
    graph_prep: str | None,
    symbolic: bool,
) -> Backend[_StateT_co]: ...


def _initialize_backend(
    pattern: Pattern,
    backend: Backend[_StateT_co] | _BackendLiteral,
    noise_model: NoiseModel | None,
    branch_selector: BranchSelector | None,
    graph_prep: str | None,
    symbolic: bool,
) -> _BuiltinBackend | Backend[_StateT_co]:
    """
    Initialize the backend.

    Parameters
    ----------
    backend: :class:`Backend` object,
        'statevector', or 'densitymatrix', or 'tensornetwork'
        simulation backend (optional), default is 'statevector'.
    noise_model: :class:`NoiseModel`, optional
        [Density matrix backend only] Noise model used by the simulator.
    branch_selector: :class:`BranchSelector`, optional
        Branch selector used for measurements. Can only be specified if ``backend`` is not an already instantiated :class:`Backend` object.  Default is :class:`RandomBranchSelector`.
    graph_prep: str, optional
        [Tensor network backend only] Strategy for preparing the graph state.  See :class:`TensorNetworkBackend`.
    symbolic : bool, optional
        [State vector and density matrix backends only] If True, support arbitrary objects (typically, symbolic expressions) in measurement angles.

    Returns
    -------
    :class:`Backend`
        matching the appropriate backend
    """
    if isinstance(backend, Backend):
        if branch_selector is not None:
            raise ValueError("`branch_selector` cannot be specified if `backend` is already instantiated.")
        if graph_prep is not None:
            raise ValueError("`graph_prep` cannot be specified if `backend` is already instantiated.")
        if symbolic:
            raise ValueError("`symbolic` cannot be specified if `backend` is already instantiated.")
        return backend
    if branch_selector is None:
        branch_selector = RandomBranchSelector()
    if backend in {"tensornetwork", "mps"}:
        if noise_model is not None:
            raise ValueError("`noise_model` cannot be specified for tensor network backend.")
        if symbolic:
            raise ValueError("`symbolic` cannot be specified for tensor network backend.")
        if graph_prep is None:
            graph_prep = "auto"
        return TensorNetworkBackend(pattern, branch_selector=branch_selector, graph_prep=graph_prep)
    if graph_prep is not None:
        raise ValueError("`graph_prep` can only be specified for tensor network backend.")
    if backend == "statevector":
        if noise_model is not None:
            raise ValueError("`noise_model` cannot be specified for state vector backend.")
        return StatevectorBackend(branch_selector=branch_selector, symbolic=symbolic)
    if backend == "densitymatrix":
        if noise_model is None:
            warnings.warn(
                "Simulating using densitymatrix backend with no noise. To add noise to the simulation, give an object of `graphix.noise_models.Noisemodel` to `noise_model` keyword argument.",
                stacklevel=1,
            )
        return DensityMatrixBackend(branch_selector=branch_selector, symbolic=symbolic)
    raise ValueError(f"Unknown backend {backend}.")
