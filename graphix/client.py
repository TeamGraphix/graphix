from __future__ import annotations

import typing
from copy import deepcopy
from dataclasses import dataclass

import networkx as nx
import numpy as np

import graphix.command
import graphix.ops
import graphix.pattern
import graphix.pauli
import graphix.sim.base_backend
import graphix.sim.statevec
import graphix.simulator
from graphix.clifford import CLIFFORD, CLIFFORD_CONJ, CLIFFORD_MUL
from graphix.command import CommandKind
from graphix.pattern import Pattern
from graphix.pauli import Plane
from graphix.sim.base_backend import Backend
from graphix.sim.base_backend import State as BackendState
from graphix.simulator import MeasureMethod, PatternSimulator
from graphix.states import BasicStates, PlanarState, State

"""
Usage:

client = Client(pattern:Pattern, blind=False) ## For pure MBQC
sv_backend = StatevecBackend(client.pattern, meas_op = client.meas_op)

simulator = PatternSimulator(client.pattern, backend=sv_backend)
simulator.run()

"""


@dataclass
class TrappifiedRun:
    input_state: list
    tested_qubits: list[int]
    stabilizer: graphix.pauli


@dataclass
class Secret_a:
    a: dict[int, int]
    a_N: dict[int, int]


@dataclass
class Secrets:
    r: bool = False
    a: bool = False
    theta: bool = False


@dataclass
class SecretDatas:
    r: dict[int, int]
    a: Secret_a
    theta: dict[int, int]

    @staticmethod
    def from_secrets(secrets: Secrets, graph, input_nodes, output_nodes):
        node_list, edge_list = graph
        r = {}
        if secrets.r:
            # Need to generate the random bit for each measured qubit, 0 for the rest (output qubits)
            for node in node_list:
                r[node] = np.random.randint(0, 2) if node not in output_nodes else 0

        theta = {}
        if secrets.theta:
            # Create theta secret for all non-output nodes (measured qubits)
            for node in node_list:
                theta[node] = np.random.randint(0, 8) if node not in output_nodes else 0  # Expressed in pi/4 units

        a = {}
        a_N = {}
        if secrets.a:
            # Create `a` secret for all
            for node in node_list:
                a[node] = np.random.randint(0, 2)

            # After all the `a` secrets have been generated, the `a_N` value can be
            # computed from the graph topology
            for i in node_list:
                a_N_value = 0
                for j in node_list:
                    if (i, j) in edge_list or (j, i) in edge_list:
                        a_N_value ^= a[j]
                a_N[i] = a_N_value

        return SecretDatas(r, Secret_a(a, a_N), theta)


## TODO : extract somewhere else
import random


class Stabilizer:
    def __init__(self, graph: nx.Graph, nodes: list[int]) -> None:
        """
        Builds a multi-qubit Pauli operator from a list of nodes, computing the product of canonical stabilizer generators of the desired nodes

        graph : Instance of networkx.Graph, the underlying graph
        nodes : list of nodes, to take the product of their associated canonical stabilizers
        """
        self.graph = graph
        self.nodes = nodes
        self.chain: list[graphix.pauli.Pauli] = [graphix.pauli.I for _ in self.graph.nodes]
        self.init_chain(nodes)

    @property
    def size(self) -> int:
        return len(self.graph.nodes)

    @property
    def span(self) -> set[int]:
        """
        Return the nodes involved in the stabilizer product, and their neighbors, in the same set
        Equivalent to return the node indices where the Pauli isn't the identity
        """
        span = set(self.nodes)
        for node in self.nodes:
            for neighbor in self.graph.neighbors(node):
                span.add(neighbor)
        return span

    def init_chain(self, nodes):
        for node in nodes:
            self.compute_product(node)

    def compute_product(self, node):
        """
        Computes the product of the current stabilizer with the canonical generator of a given node
        A canonical generator for a node is : X on the node, Z on the neighbors
        The underlying graph structure helps to match the node indices
        """
        self.chain[node] @= graphix.pauli.X
        for neighbor in self.graph.neighbors(node):
            self.chain[neighbor] @= graphix.pauli.Z

    def __repr__(self) -> str:
        string = f"""
        Product stabilizer of nodes in {self.nodes}\n
        For graph with edges {self.graph.edges}\n
        """
        for node in sorted(self.graph.nodes):
            string += f"{node} {self.chain[node]}\n"

        return string


class TrappifiedCanvas:
    def __init__(self, graph: nx.Graph, traps_list: list[list[int]]) -> None:
        self.graph = graph
        self.traps_list = traps_list
        stabilizers = [Stabilizer(graph, trap_nodes) for trap_nodes in traps_list]
        self.stabilizer = self.merge_stabilizers(stabilizers)
        print(self.stabilizer)
        dummies_coins = self.generate_coins_dummies()
        self.coins = self.generate_coins_trap_qubits(coins=dummies_coins)
        self.spans = dict(zip(self.trap_qubits, [stabilizer.span for stabilizer in stabilizers]))
        self.states = self.generate_eigenstate()

    @property
    def trap_qubits(self):
        return [node for trap in self.traps_list for node in trap]

    @property
    def dummy_qubits(self):
        return [neighbor for trap in self.traps_list for node in trap for neighbor in list(self.graph.neighbors(node))]

    def merge_stabilizers(self, stabilizers: list[Stabilizer]):
        common_stabilizer = Stabilizer(self.graph, [])
        for stabilizer in stabilizers:
            for node in stabilizer.span:
                # If the Pauli was identity, just replace it by the upcoming Pauli
                if common_stabilizer.chain[node] == graphix.pauli.I:
                    common_stabilizer.chain[node] = stabilizer.chain[node]
                else:
                    # If the Pauli wasn't identity, the incoming Pauli must coincide with the previous one
                    # otherwise it's dramatic
                    if common_stabilizer.chain[node] != stabilizer.chain[node]:
                        print("Huge error")
                        return
                    else:
                        # (nothing to do as they already coincide)
                        pass
            common_stabilizer.nodes += stabilizer.nodes
        return common_stabilizer

    def generate_eigenstate(self) -> list[State]:
        states = []
        for node in sorted(self.stabilizer.graph.nodes):
            operator = self.stabilizer.chain[node]
            states.append(operator.get_eigenstate(eigenvalue=self.coins[node]))
        return states

    def generate_coins_dummies(self):
        coins = dict()
        for node in self.stabilizer.graph.nodes:
            if node not in self.trap_qubits:
                coins[node] = random.randint(0, 1)
            else:
                coins[node] = 0
        return coins

    def generate_coins_trap_qubits(self, coins):
        for node in self.trap_qubits:
            neighbors_coins = sum(coins[n] for n in self.stabilizer.graph.neighbors(node)) % 2
            coins[node] = neighbors_coins
        return coins

    def __repr__(self) -> str:
        text = ""
        for node in sorted(self.stabilizer.graph.nodes):
            trap_span = ""
            if node in self.trap_qubits:
                trap_span += f"{self.spans[node]}"
            text += f"{node} {self.stabilizer.chain[node]} ^ {self.coins[node]} {trap_span} -> {self.states[node]}\n"
        return text


@dataclass
class ByProduct:
    z_domain: list[int]
    x_domain: list[int]


def get_byproduct_db(pattern: Pattern) -> dict[int, ByProduct]:
    byproduct_db = dict()
    for node in pattern.output_nodes:
        byproduct_db[node] = ByProduct(z_domain=[], x_domain=[])

    for cmd in pattern.correction_commands():
        if cmd.node in pattern.output_nodes:
            if cmd.kind == CommandKind.Z:
                byproduct_db[cmd.node].z_domain = cmd.domain
            if cmd.kind == CommandKind.X:
                byproduct_db[cmd.node].x_domain = cmd.domain
    return byproduct_db


def remove_flow(pattern):
    clean_pattern = Pattern(pattern.input_nodes)
    for cmd in pattern:
        if cmd.kind in (CommandKind.X, CommandKind.Z):
            # If byproduct, remove it so it's not done by the server
            continue
        if cmd.kind == CommandKind.M:
            # If measure, remove measure parameters
            new_cmd = graphix.command.M(node=cmd.node)
        else:
            new_cmd = cmd
        clean_pattern.add(new_cmd)
    return clean_pattern


class Client:
    def __init__(self, pattern, input_state=None, measure_method_cls=None, secrets=None):
        self.initial_pattern = pattern

        self.input_nodes = self.initial_pattern.input_nodes.copy()
        self.output_nodes = self.initial_pattern.output_nodes.copy()
        self.graph = self.initial_pattern.get_graph()
        self.nodes_list = self.graph[0]

        # Copy the pauli-preprocessed nodes' measurement outcomes
        self.results = pattern.results.copy()
        if measure_method_cls is None:
            measure_method_cls = ClientMeasureMethod
        self.measure_method = measure_method_cls(self)

        self.measurement_db = {measure.node: measure for measure in pattern.get_measurement_commands()}
        self.byproduct_db = get_byproduct_db(pattern)

        if secrets is None:
            secrets = Secrets()
        self.secrets = SecretDatas.from_secrets(secrets, self.graph, self.input_nodes, self.output_nodes)

        pattern_without_flow = remove_flow(pattern)
        self.clean_pattern = pattern_without_flow.prepared_nodes_as_input_nodes()

        self.input_state = input_state if input_state != None else [BasicStates.PLUS for _ in self.input_nodes]

    def blind_qubits(self, backend: Backend) -> None:
        z_rotation = lambda theta: np.array([[1, 0], [0, np.exp(1j * theta * np.pi / 4)]])
        x_blind = lambda a: graphix.pauli.X if (a == 1) else graphix.pauli.I
        for node in self.nodes_list:
            theta = self.secrets.theta.get(node, 0)
            a = self.secrets.a.a.get(node, 0)
            backend.apply_single(node=node, op=x_blind(a).matrix)
            backend.apply_single(node=node, op=z_rotation(theta))

    def prepare_states(self, backend: Backend) -> None:
        # First prepare inputs
        backend.add_nodes(nodes=self.input_nodes, data=self.input_state)

        # Then iterate over auxiliaries required to blind
        aux_nodes = []
        for node in self.nodes_list:
            if node not in self.input_nodes and node not in self.output_nodes:
                aux_nodes.append(node)
        aux_data = [BasicStates.PLUS for _ in aux_nodes]
        backend.add_nodes(nodes=aux_nodes, data=aux_data)

        # Prepare outputs
        output_data = []
        for node in self.output_nodes:
            r_value = self.secrets.r.get(node, 0)
            a_N_value = self.secrets.a.a_N.get(node, 0)
            output_data.append(BasicStates.PLUS if r_value ^ a_N_value == 0 else BasicStates.MINUS)
        backend.add_nodes(nodes=self.output_nodes, data=output_data)

    def create_test_runs(self) -> tuple[list[TrappifiedCanvas], dict[int, int]]:
        graph = nx.Graph()
        nodes, edges = self.graph
        graph.add_edges_from(edges)
        graph.add_nodes_from(nodes)

        # Create the graph coloring
        coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
        colors = set(coloring.values())
        nodes_by_color = {c: [] for c in colors}
        for node in sorted(graph.nodes):
            color = coloring[node]
            nodes_by_color[color].append(node)

        # Create the test runs : one per color
        runs: list[TrappifiedCanvas] = []
        for color in colors:
            # 1 color = 1 test run = 1 set of traps = 1 stabilizer
            trap_qubits = nodes_by_color[color]
            isolated_traps = [[node] for node in trap_qubits]
            trappified_canvas = TrappifiedCanvas(graph, traps_list=isolated_traps)

            runs.append(trappified_canvas)
        return runs, coloring

    def delegate_test_run(self, backend: Backend, run: TrappifiedCanvas) -> list[int]:
        # The state is entirely prepared and blinded by the client before being sent to the server
        backend.add_nodes(nodes=sorted(self.graph[0]), data=run.states)
        self.blind_qubits(backend)

        # Modify the pattern to be all X-basis measurements, no shifts/signalling updates
        for node in self.measurement_db:
            self.measurement_db[node] = graphix.command.M(node=node)

        sim = PatternSimulator(backend=backend, pattern=self.clean_pattern, measure_method=self.measure_method)
        sim.run(input_state=None)

        trap_outcomes = []
        for trap in run.traps_list:
            outcomes = [self.results.get(component, 0) for component in trap]
            trap_outcome = sum(outcomes) % 2
            trap_outcomes.append(trap_outcome)

        return trap_outcomes

    def delegate_pattern(self, backend: Backend) -> PatternSimulator:
        self.prepare_states(backend)
        self.blind_qubits(backend)
        sim = PatternSimulator(backend=backend, pattern=self.clean_pattern, measure_method=self.measure_method)
        sim.run(input_state=None)
        self.decode_output_state(backend)
        # returns the final state
        return sim

    def decode_output_state(self, backend: Backend):
        for node in self.output_nodes:
            z_decoding, x_decoding = self.decode_output(node)
            if z_decoding:
                backend.apply_single(node=node, op=graphix.ops.Ops.z)
            if x_decoding:
                backend.apply_single(node=node, op=graphix.ops.Ops.x)

    def get_secrets_size(self):
        secrets_size = {}
        for secret in self.secrets:
            secrets_size[secret] = len(self.secrets[secret])
        return secrets_size

    def decode_output(self, node):
        z_decoding = sum(self.results[z_dep] for z_dep in self.byproduct_db[node].z_domain) % 2
        z_decoding ^= self.secrets.r.get(node, 0)
        x_decoding = sum(self.results[x_dep] for x_dep in self.byproduct_db[node].x_domain) % 2
        x_decoding ^= self.secrets.a.a.get(node, 0)
        return z_decoding, x_decoding


class ClientMeasureMethod(MeasureMethod):
    def __init__(self, client: Client):
        self.__client = client

    def get_measurement_description(self, cmd) -> graphix.simulator.MeasurementDescription:
        parameters = self.__client.measurement_db[cmd.node]

        r_value = self.__client.secrets.r.get(cmd.node, 0)
        theta_value = self.__client.secrets.theta.get(cmd.node, 0)
        a_value = self.__client.secrets.a.a.get(cmd.node, 0)
        a_N_value = self.__client.secrets.a.a_N.get(cmd.node, 0)

        # extract signals for adaptive angle
        s_signal = sum(self.__client.results[j] for j in parameters.s_domain)
        t_signal = sum(self.__client.results[j] for j in parameters.t_domain)
        measure_update = graphix.pauli.MeasureUpdate.compute(
            parameters.plane, s_signal % 2 == 1, t_signal % 2 == 1, graphix.clifford.TABLE[parameters.vop]
        )
        angle = parameters.angle * np.pi
        angle = angle * measure_update.coeff + measure_update.add_term
        angle = (-1) ** a_value * angle + theta_value * np.pi / 4 + np.pi * (r_value + a_N_value)
        # angle = angle * measure_update.coeff + measure_update.add_term
        return graphix.simulator.MeasurementDescription(measure_update.new_plane, angle)
        # return graphix.sim.base_backend.MeasurementDescription(measure_update.new_plane, angle)

    def get_measure_result(self, node: int) -> bool:
        raise ValueError("Server cannot have access to measurement results")

    def set_measure_result(self, node: int, result: bool) -> None:
        if self.__client.secrets.r:
            result ^= self.__client.secrets.r[node]
        self.__client.results[node] = result
