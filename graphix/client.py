import numpy as np

from clifford import CLIFFORD_CONJ, CLIFFORD, CLIFFORD_MUL
from graphix import Pattern
import graphix.sim.base_backend

class Client:
    def __init__(self, pattern: Pattern, blind=False, secrets=None):
        self.pattern = pattern

        """
        Database containing the "measurement configuration"
        - Node
        - Measurement parameters : plane, angle, X and Z dependencies
        - Measurement outcome
        """
        self.measurement_db = {}
        self.init_measurement_db()
        self.measure_method = ClientMeasureMethod(self)


    def init_measurement_db(self):
        for cmd in self.pattern.seq:
            if cmd[0] == 'M':
                node = cmd[1]
                plane = cmd[2]
                angle = cmd[3]
                s_domain = cmd[4]
                t_domain = cmd[5]
                if len(cmd) == 7:
                    vop = cmd[6]
                else:
                    vop = 0
                self.measurement_db[node] = {
                    "plane": plane,
                    "angle": angle,
                    "s_domain": s_domain,
                    "t_domain": t_domain,
                    "vop": vop
                    "result": 0
                }
                # Erase the unnecessary items from the command to make sure they don't appear on the server's side
                del cmd[2:]


class ClientMeasureMethod(graphix.sim.base_backend.MeasureMethod):
    def __init__(client: Client):
        self.__client = client

    def get_measurement_description(self, cmd, results) -> graphix.sim.base_backend.MeasurementDescription:
        node = cmd[1]
        info = self.__client.measurement_db[node]
        s_domain = info["s_domain"]
        t_domain = info["t_domain"]
        vop = info["vop"]
        angle = info[angle] * np.pi
        # extract signals for adaptive angle
        s_signal = np.sum(self.__client.measurement_db[j]["result"] for j in s_domain)
        t_signal = np.sum(self.__client.measurement_db[j]["result"] for j in t_domain)
        measure_update = graphix.pauli.MeasureUpdate.compute(
            graphix.pauli.Plane[cmd[2]], s_signal % 2 == 1, t_signal % 2 == 1, graphix.clifford.TABLE[vop]
        )
        angle = angle * measure_update.coeff + measure_update.add_term
        return graphix.sim.base_backend.MeasurementDescription(measure_update.new_plane, angle)

    def set_measure_result(self, cmd, result: bool) -> None:
        node = cmd[1]
        self.__client.measurement_db[node]["result"] = result
