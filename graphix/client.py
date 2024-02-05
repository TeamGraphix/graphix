import dataclasses
import numpy as np

from graphix.clifford import CLIFFORD_CONJ, CLIFFORD, CLIFFORD_MUL
import graphix.sim.base_backend


"""
Usage:

client = Client(pattern:Pattern, blind=False) ## For pure MBQC
sv_backend = StatevecBackend(client.pattern, meas_op = client.meas_op)

simulator = PatternSimulator(client.pattern, backend=sv_backend)
simulator.run()

"""

@dataclasses.dataclass
class MeasureParameters:
    plane: graphix.pauli.Plane
    angle: float
    s_domain: list[int]
    t_domain: list[int]
    vop: int


class Client:
    def __init__(self, pattern, blind=False, secrets=None):
        self.pattern = pattern

        """
        Database containing the "measurement configuration"
        - Node
        - Measurement parameters : plane, angle, X and Z dependencies
        - Measurement outcome
        """
        self.measurement_db = {}
        self.results = pattern.results.copy()
        self.measure_method = ClientMeasureMethod(self)

    def init_measurement_db(self):
        for cmd in self.pattern:
            if cmd[0] == 'M':
                node = cmd[1]
                plane = graphix.pauli.Plane[cmd[2]]
                angle = cmd[3] * np.pi
                s_domain = cmd[4]
                t_domain = cmd[5]
                if len(cmd) == 7:
                    vop = cmd[6]
                else:
                    vop = 0
                self.measurement_db[node] = MeasureParameters(plane, angle, s_domain, t_domain, vop)
                # Erase the unnecessary items from the command to make sure they don't appear on the server's side
                del cmd[2:]


class ClientMeasureMethod(graphix.sim.base_backend.MeasureMethod):
    def __init__(self, client: Client):
        self.__client = client

    def get_measurement_description(self, cmd, results) -> graphix.sim.base_backend.MeasurementDescription:
        node = cmd[1]
        parameters = self.__client.measurement_db[node]
        # extract signals for adaptive angle
        s_signal = np.sum(self.__client.results[j] for j in parameters.s_domain)
        t_signal = np.sum(self.__client.results[j] for j in parameters.t_domain)
        measure_update = graphix.pauli.MeasureUpdate.compute(
            parameters.plane, s_signal % 2 == 1, t_signal % 2 == 1, graphix.clifford.TABLE[parameters.vop]
        )
        angle = parameters.angle * measure_update.coeff + measure_update.add_term
        return graphix.sim.base_backend.MeasurementDescription(measure_update.new_plane, angle)

    def set_measure_result(self, cmd, result: bool) -> None:
        node = cmd[1]
        self.__client.results[node] = result
