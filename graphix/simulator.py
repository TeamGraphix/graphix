"""MBQC simulator.

Simulates MBQC by executing the pattern.

"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypedDict, TypeVar, overload

# assert_never introduced in Python 3.11
# override introduced in Python 3.12
from typing_extensions import assert_never, override

from graphix import command
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.clifford import Clifford
from graphix.command import BaseM, CommandKind, N
from graphix.fundamentals import AbstractMeasurement
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

    # Unpack introduced in Python 3.12
    from typing_extensions import Unpack

    from graphix.command import BaseN
    from graphix.measurements import Measurement, Outcome
    from graphix.noise_models.noise_model import CommandOrNoise, NoiseModel
    from graphix.optimization import StandardizedPattern
    from graphix.parameter import ExpressionOrSupportsComplex
    from graphix.pattern import Pattern
    from graphix.sim import Data, DensityMatrix, MBQCTensorNet, Statevector
    from graphix.states import State

logger = logging.getLogger(__name__)

_BuiltinBackend = DensityMatrixBackend | StatevectorBackend | TensorNetworkBackend
_BackendLiteral = Literal["statevector", "densitymatrix", "tensornetwork", "mps"]

if TYPE_CHECKING:
    _BuiltinBackendState = DensityMatrix | MBQCTensorNet | Statevector

    _StateT = TypeVar("_StateT")

_AM_co = TypeVar("_AM_co", bound=AbstractMeasurement, covariant=True)

# This type variable should be defined outside TYPE_CHECKING block
# because it appears in the parameters of `PatternSimulator`.
_StateT_co = TypeVar("_StateT_co", covariant=True)


class PrepareMethod(ABC):
    """Prepare method used by the simulator.

    See :class:`DefaultPrepareMethod` for the default prepare method that implements MBQC.

    To be overwritten by custom preparation methods in the case of delegated QC protocols.

    Example: class ``ClientPrepareMethod`` in https://github.com/qat-inria/veriphix
    """

    @abstractmethod
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


class MeasureMethod(ABC):
    """Measure method used by the simulator, with default measurement method that implements MBQC.

    To be overwritten by custom measurement methods in the case of delegated QC protocols.

    Example: class ``ClientMeasureMethod`` in https://github.com/qat-inria/veriphix
    """

    def measure(
        self,
        backend: Backend[_StateT_co],
        cmd: BaseM,
        noise_model: NoiseModel | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> None:
        """Perform a measurement.

        Parameters
        ----------
        backend : :class:`Backend`
            The simulator backend to use.
        cmd: BaseM
            Measurement command.
        noise_model: :class:`NoiseModel`, optional
            Noise model used to confuse result.
        rng : Generator, optional
            Random number generator used to confuse result.
        stacklevel : int, optional
            Stack level to use for warnings. Defaults to 1, meaning that warnings
            are reported at this function's call site.
        """
        description = self.describe_measurement(cmd)
        result = backend.measure(cmd.node, description, rng=rng, stacklevel=stacklevel + 1)
        logger.debug("Measure: %s", result)
        if noise_model is not None:
            result = noise_model.confuse_result(cmd, result, rng=rng, stacklevel=stacklevel + 1)
        self.store_measurement_outcome(cmd.node, result)

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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
        performed during simulation are stored in ``self.results``, which is a copy
        of the given mapping. The original ``results`` mapping is not modified.
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


class SimulatorKwargs(TypedDict, total=False):
    """Common keyword arguments for simulator.

    The keys correspond to the fields of :class:`SimulatorOptions`.
    """

    prepare_method: PrepareMethod | None
    measure_method: MeasureMethod | None
    noise_model: NoiseModel | None
    branch_selector: BranchSelector | None
    graph_prep: str | None
    symbolic: bool


@dataclass(frozen=True)
class SimulatorOptions:
    """Options controlling simulator.

    Parameters
    ----------
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
        [Density matrix backend only] If True, support arbitrary objects (typically, symbolic expressions) in measurement angles.
    """

    prepare_method: PrepareMethod | None = None
    measure_method: MeasureMethod | None = None
    noise_model: NoiseModel | None = None
    branch_selector: BranchSelector | None = None
    graph_prep: str | None = None
    symbolic: bool = False


class PatternSimulator(Generic[_StateT_co]):
    """MBQC simulator.

    Executes the measurement pattern.
    """

    noise_model: NoiseModel | None
    backend: Backend[_StateT_co]

    @overload
    def __init__(
        self: PatternSimulator[Statevector],
        pattern: Pattern,
        backend: Literal["statevector"] = ...,
        *,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> None: ...

    @overload
    def __init__(
        self: PatternSimulator[DensityMatrix],
        pattern: Pattern,
        backend: Literal["densitymatrix"] = ...,
        *,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> None: ...

    @overload
    def __init__(
        self: PatternSimulator[MBQCTensorNet],
        pattern: Pattern,
        backend: Literal["tensornetwork", "mps"] = ...,
        *,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> None: ...

    @overload
    def __init__(
        self: PatternSimulator[_StateT],
        pattern: Pattern,
        backend: Backend[_StateT] = ...,
        *,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> None: ...

    def __init__(
        self: PatternSimulator[_StateT | _BuiltinBackendState],
        pattern: Pattern,
        backend: Backend[_StateT] | _BackendLiteral = "statevector",
        *,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> None:
        """
        Construct a pattern simulator.

        Parameters
        ----------
        pattern: :class:`Pattern` object
            MBQC pattern to be simulated.
        backend : :class:`Backend` or {'statevector', 'densitymatrix', 'tensornetwork'}, optional
            The simulator backend to use: either an instantiated backend or the
            name of a built-in backend. Default: ``'statevector'``.
        stacklevel : int, optional
            Stack level to use for warnings. Defaults to 1, meaning that warnings
            are reported at this function's call site.
        kwargs: Unpack[SimulatorKwargs]
            Options controlling simulator. See :class:`SimulatorOptions`.

        .. seealso:: :class:`graphix.sim.statevec.StatevectorBackend`\
            :class:`graphix.sim.tensornet.TensorNetworkBackend`\
            :class:`graphix.sim.density_matrix.DensityMatrixBackend`\
        """
        options = SimulatorOptions(**kwargs)
        self.backend = _initialize_backend(pattern, backend, stacklevel=stacklevel + 1, **kwargs)
        self.noise_model = options.noise_model
        self.__pattern = pattern
        self.__prepare_method = options.prepare_method or DefaultPrepareMethod()
        self.__measure_method = options.measure_method or DefaultMeasureMethod()

    @property
    def pattern(self) -> Pattern:
        """Return the pattern."""
        return self.__pattern

    @property
    def measure_method(self) -> MeasureMethod:
        """Return the measure method."""
        return self.__measure_method

    def run(
        self, input_state: Data | None = BasicStates.PLUS, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> None:
        """Perform the simulation.

        Parameters
        ----------
        input_state: Data | None, optional
            the output quantum state, in the representation depending on the backend used.
            Default: ``|+>``. If ``None``, no input nodes are added by the simulator to
            the backend: input nodes must have been prepared in the backend before
            running the simulation.
        rng: Generator, optional
            Random number generator for measurements.
            This generator is used only in case of random branch selection
            (see :class:`RandomBranchSelector`).
        stacklevel : int, optional
            Stack level to use for warnings. Defaults to 1, meaning that warnings
            are reported at this function's call site.
        """
        # Check whether the backend is properly initialized.  We
        # disable the check for TensorNetworkBackend because its
        # current behavior differs from other backends.
        if not isinstance(self.backend, TensorNetworkBackend):
            initial_nqubit = self.backend.nqubit
            if input_state is None:
                # No explicit state supplied: the backend must already contain the
                # required input qubits.
                input_nodes_len = len(self.pattern.input_nodes)
                if initial_nqubit != input_nodes_len:
                    raise ValueError(
                        f"`input_state` is `None`: the backend is expected to have {input_nodes_len} input nodes already prepared, but {initial_nqubit} were found."
                    )
            # An explicit state was supplied: the backend must start with a clean
            # state (no pre-allocated qubits).
            elif initial_nqubit != 0:
                raise ValueError(
                    f"`input_state` is not `None`: the backend is expected to have no pre-allocated qubits, but has {initial_nqubit} qubits."
                )
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
            match cmd.kind:
                case CommandKind.N:
                    self.__prepare_method.prepare(self.backend, cmd, rng=rng)
                case CommandKind.E:
                    self.backend.entangle_nodes(edge=cmd.nodes)
                case CommandKind.M:
                    self.__measure_method.measure(
                        self.backend, cmd, noise_model=self.noise_model, rng=rng, stacklevel=stacklevel + 1
                    )
                case CommandKind.X | CommandKind.Z:
                    if self.__measure_method.check_domain(cmd.domain):
                        self.backend.correct_byproduct(cmd)
                case CommandKind.C:
                    self.backend.apply_clifford(cmd.node, cmd.clifford)
                case CommandKind.T:
                    # The T command is a flag for one clock cycle in a simulated
                    # experiment, added via a hardware-agnostic
                    # pattern modifier. Noise models can perform special
                    # handling of ticks during noise transpilation.
                    pass
                case CommandKind.ApplyNoise:
                    if cmd.domain is None or self.__measure_method.check_domain(cmd.domain):
                        self.backend.apply_noise(cmd)
                case CommandKind.S:
                    raise ValueError("S commands unexpected in simulated patterns.")
                case _:
                    assert_never(cmd.kind)
            logger.debug("State: %s", self.backend.state)
        self.backend.finalize(output_nodes=self.pattern.output_nodes)


@overload
def _initialize_backend(
    pattern: Pattern,
    backend: StatevectorBackend | Literal["statevector"],
    *,
    stacklevel: int = 1,
    **kwargs: Unpack[SimulatorKwargs],
) -> StatevectorBackend: ...


@overload
def _initialize_backend(
    pattern: Pattern,
    backend: DensityMatrixBackend | Literal["densitymatrix"],
    *,
    stacklevel: int = 1,
    **kwargs: Unpack[SimulatorKwargs],
) -> DensityMatrixBackend: ...


@overload
def _initialize_backend(
    pattern: Pattern,
    backend: TensorNetworkBackend | Literal["tensornetwork", "mps"],
    *,
    stacklevel: int = 1,
    **kwargs: Unpack[SimulatorKwargs],
) -> TensorNetworkBackend: ...


@overload
def _initialize_backend(
    pattern: Pattern,
    backend: Backend[_StateT_co],
    *,
    stacklevel: int = 1,
    **kwargs: Unpack[SimulatorKwargs],
) -> Backend[_StateT_co]: ...


def _initialize_backend(
    pattern: Pattern,
    backend: Backend[_StateT_co] | _BackendLiteral,
    *,
    stacklevel: int = 1,
    **kwargs: Unpack[SimulatorKwargs],
) -> _BuiltinBackend | Backend[_StateT_co]:
    """
    Initialize the backend.

    Parameters
    ----------
    backend: :class:`Backend` object,
        'statevector', or 'densitymatrix', or 'tensornetwork'
        simulation backend (optional), default is 'statevector'.
    stacklevel : int, optional
        Stack level to use for warnings. Defaults to 1, meaning that warnings
        are reported at this function's call site.
    kwargs: Unpack[SimulatorKwargs]
        Options controlling simulator. See :class:`SimulatorOptions`.

    Returns
    -------
    :class:`Backend`
        matching the appropriate backend
    """
    options = SimulatorOptions(**kwargs)
    if isinstance(backend, Backend):
        if options.branch_selector is not None:
            raise ValueError("`branch_selector` cannot be specified if `backend` is already instantiated.")
        if options.graph_prep is not None:
            raise ValueError("`graph_prep` cannot be specified if `backend` is already instantiated.")
        if options.symbolic:
            raise ValueError("`symbolic` cannot be specified if `backend` is already instantiated.")
        return backend
    branch_selector = RandomBranchSelector() if options.branch_selector is None else options.branch_selector
    if backend in {"tensornetwork", "mps"}:
        if options.noise_model is not None:
            raise ValueError("`noise_model` cannot be specified for tensor network backend.")
        if options.symbolic:
            raise ValueError("`symbolic` cannot be specified for tensor network backend.")
        graph_prep = "auto" if options.graph_prep is None else options.graph_prep
        return TensorNetworkBackend(pattern, branch_selector=branch_selector, graph_prep=graph_prep)
    if options.graph_prep is not None:
        raise ValueError("`graph_prep` can only be specified for tensor network backend.")
    match backend:
        case "statevector":
            if options.noise_model is not None:
                raise ValueError("`noise_model` cannot be specified for state vector backend.")
            if options.symbolic:
                raise ValueError(
                    "Statevector backend does not support `symbolic` simulation. Consider using backend in `graphix-symbolic` plugin."
                )
            nqubits = pattern.max_space()
            return StatevectorBackend.with_capacity(nqubits, branch_selector=branch_selector)
        case "densitymatrix":
            if options.noise_model is None:
                warnings.warn(
                    "Simulating using densitymatrix backend with no noise. To add noise to the simulation, give an object of `graphix.noise_models.Noisemodel` to `noise_model` keyword argument.",
                    stacklevel=stacklevel + 1,
                )
            return DensityMatrixBackend(branch_selector=branch_selector, symbolic=options.symbolic)
        case _:
            raise ValueError(f"Unknown backend {backend}.")


class Simulable(ABC, Generic[_AM_co]):
    """Base class for simulable objects.

    This class is generic in the type of measurements (``_AM_co``),
    allowing generic classes to extend it while restricting simulable
    instances to concrete measurement types, i.e., when the type
    parameter is a subtype of :class:`Measurement`.

    Subclasses should implement at least one of the two methods
    ``to_pattern` or ``to_standardizedpattern``.
    """

    def __init_subclass__(cls) -> None:
        """Check for every subclass that at least one of `to_pattern` or `to_standardizedpattern` is implemented."""
        super().__init_subclass__()
        if cls.to_pattern is Simulable.to_pattern and cls.to_standardizedpattern is Simulable.to_standardizedpattern:
            raise TypeError(f"{cls.__name__} must implement at least one of `to_pattern` or `to_standardizedpattern`")

    def to_pattern(self: Simulable[_AM_co]) -> Pattern:
        "Return a representation as a pattern."
        return self.to_standardizedpattern().to_pattern()

    def to_standardizedpattern(self: Simulable[_AM_co]) -> StandardizedPattern:
        "Return a representation as a standardized pattern."
        # Circumvent import loop
        from graphix.optimization import StandardizedPattern  # noqa: PLC0415

        return StandardizedPattern.from_pattern(self.to_pattern())

    def to_optimized_pattern(self: Simulable[Measurement], *, stacklevel: int = 1) -> Pattern:
        """Return a representation as an optimized pattern.

        Optimized pattern is the form that is simulated when
        :meth:`simulate` is called with ``optimized=True`` (the
        default).

        Optimization passes are:
        - remove Pauli measurements,
        - minimize space.

        Note that Pauli measurements are not implicitly inferred.

        Parameters
        ----------
        stacklevel : int, optional
            Stack level to use for warnings. Defaults to 1, meaning that warnings
            are reported at this function's call site.

        Returns
        -------
        Pattern
            Optimized pattern.
        """
        standardized_pattern = self.to_standardizedpattern()
        standardized_pattern = standardized_pattern.minimize_space()
        standardized_pattern2 = standardized_pattern.infer_pauli_measurements().remove_pauli_measurements(
            stacklevel=stacklevel + 1
        )
        standardized_pattern2 = standardized_pattern.minimize_space()
        if standardized_pattern2.max_space() <= standardized_pattern.max_space():
            standardized_pattern = standardized_pattern2
        return standardized_pattern.to_space_optimal_pattern()

    @overload
    def simulate(
        self: Simulable[Measurement],
        backend: StatevectorBackend | Literal["statevector"] = "statevector",
        input_state: State
        | Statevector
        | Iterable[State]
        | Iterable[ExpressionOrSupportsComplex]
        | Iterable[Iterable[ExpressionOrSupportsComplex]]
        | None = ...,
        rng: Generator | None = ...,
        *,
        optimized: bool = True,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> Statevector: ...

    @overload
    def simulate(
        self: Simulable[Measurement],
        backend: DensityMatrixBackend | Literal["densitymatrix"],
        input_state: State
        | DensityMatrix
        | Iterable[State]
        | Iterable[ExpressionOrSupportsComplex]
        | Iterable[Iterable[ExpressionOrSupportsComplex]]
        | None = ...,
        rng: Generator | None = ...,
        *,
        optimized: bool = True,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> DensityMatrix: ...

    @overload
    def simulate(
        self: Simulable[Measurement],
        backend: TensorNetworkBackend | Literal["tensornetwork", "mps"],
        input_state: State
        | Iterable[State]
        | Iterable[ExpressionOrSupportsComplex]
        | Iterable[Iterable[ExpressionOrSupportsComplex]]
        | None = ...,
        rng: Generator | None = ...,
        *,
        optimized: bool = True,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> MBQCTensorNet: ...

    @overload
    def simulate(
        self: Simulable[Measurement],
        backend: Backend[_StateT_co],
        input_state: Data | None = ...,
        rng: Generator | None = ...,
        *,
        optimized: bool = True,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> _StateT_co: ...

    def simulate(
        self: Simulable[Measurement],
        backend: Backend[_StateT_co] | _BackendLiteral = "statevector",
        input_state: Data | None = BasicStates.PLUS,
        rng: Generator | None = None,
        *,
        optimized: bool = True,
        stacklevel: int = 1,
        **kwargs: Unpack[SimulatorKwargs],
    ) -> _StateT_co | _BuiltinBackendState:
        """Simulate the execution of the pattern by using :class:`graphix.simulator.PatternSimulator`.

        Parameters
        ----------
        backend : :class:`Backend` or {'statevector', 'densitymatrix', 'tensornetwork'}, optional
            The simulator backend to use: either an instantiated backend or the
            name of a built-in backend. Default: ``'statevector'``.
        input_state: Data or None, optional
            the output quantum state, in a representation compatible with the selected backend.
            Default: the ``|+>`` state (``BasicStates.PLUS``).
            If ``None``, no input nodes are added by the simulator; input nodes must have been prepared in the backend before running the simulation.
        rng: Generator, optional
            Random-number generator for measurements.
            This generator is used only in case of random branch selection
            (see :class:`RandomBranchSelector`).
        optimized : bool, optional
            Optimize the pattern before simulation. Defaults to ``True``.
        stacklevel : int, optional
            Stack level to use for warnings. Defaults to 1, meaning that warnings
            are reported at this function's call site.
        kwargs: Unpack[SimulatorKwargs]
            Options controlling simulator. See :class:`SimulatorOptions`.

        Returns
        -------
        state :
            quantum state representation for the selected backend.

        .. seealso:: :class:`graphix.simulator.PatternSimulator`
        """
        pattern = self.to_optimized_pattern(stacklevel=stacklevel + 1) if optimized else self.to_pattern()
        sim = PatternSimulator(pattern, backend=backend, stacklevel=stacklevel + 1, **kwargs)
        sim.run(input_state, rng=rng, stacklevel=stacklevel + 1)
        return sim.backend.state
