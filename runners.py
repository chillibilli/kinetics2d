from typing import Callable, List, Dict, Tuple, Iterable

from mech import Mech
from parts import Joint, round_vector, Slider


class Runner1D:
    def __init__(self,
                 model: Mech,
                 motion: List[Tuple[str, Iterable[float]]] = None,
                 tracing_points: List[str] = None,
                 forces: Callable[[float], Dict[Tuple[str, str], Tuple[float, float]]] = None,
                 moments: Callable[[float], Dict[str, float]] = None,
                 draw_model: Callable = None,
                 draw_trace: Callable[[Mech, List[Dict], List[str]], None] = None):
        self.model = model
        self.motion = motion
        self.trace = []
        self.tracing_points = tracing_points
        self.draw_model = draw_model
        self.draw_trace = draw_trace

        self.forces = forces
        self.moments = moments

    def run(self):

        calc_forces = self.forces is not None or self.moments is not None

        starting = self.draw_model is not None
        self.trace = []

        for param_name, values in self.motion:
            for t in values:

                if calc_forces:
                    if self.forces is not None:
                        ext_forces = self.forces(t)
                    else:
                        ext_forces = {}

                    if self.moments is not None:
                        ext_moments = self.moments(t)
                    else:
                        ext_moments = {}

                self.model.input({param_name: t})
                self.model.solve()

                if calc_forces:
                    self.model.charge(ext_forces=ext_forces, ext_moments=ext_moments)

                if starting:
                    starting = False
                    self.draw_model(self.model)

                record = {'t': t}
                for part in self.model.parts.values():
                    if isinstance(part, Joint):
                        record[part.name] = part.is_connected()
                    else:
                        if isinstance(part, Slider):
                            record[f"{part.name}.length"] = part.length
                        for point in part.points.values():
                            record[f"{part.name}.{point.name}"] = round_vector(point.absolute)
                            if (calc_forces
                                    and part.name in self.model.force_points
                                    and point.name in self.model.force_points[part.name]):
                                record[f"{part.name}.{point.name}.F"] = round_vector(point.force)

                self.trace.append(record)

        if self.draw_trace is not None:
            self.draw_trace(self.model, self.trace, self.tracing_points)

        if self.draw_model is not None:
            self.draw_model(self.model, end_position=True)
