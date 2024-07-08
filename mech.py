from typing import Dict, List, Tuple
import yaml

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.linalg import solve, pinv

from parts import Param, Part, Base, Joint, Solid, Slider, Point


class Mech:
    def __init__(self):
        self.base = Base()
        self.parts: Dict[str, Part] = {self.base.name: self.base}
        self.free_params: List[Param] = []
        self.input_params: Dict[str, Param] = {}

        self.force_points = None
        self.force_points_index = None
        self.force_joints = None

        self.A: NDArray[2] | None = None
        self.b: NDArray[1] | None = None

    def add(self, p: Part):
        if p.name in self.parts:
            raise ValueError(f"Part '{p.name}' already in mech")
        self.parts[p.name] = p
        self.free_params += p.free_params()
        self.input_params.update(p.input_params())
        self.force_points = None  # invalidate force application points
        return self

    def adjust(self):
        for p in self.parts.values():
            p.adjust()

    def energy(self, param_values):
        for v, f in zip(param_values, self.free_params):
            f.setter(v)
        self.adjust()
        return sum(p.energy() for p in self.parts.values())

    def input(self, params: Dict[str, float]):
        for k, v in params.items():
            self.input_params[k].setter(v)
        self.adjust()

    def solve(self, global_params: Tuple[float] = ()):
        # TODO возвращать сошлись все соединения или нет
        initial = np.array([p.getter() for p in self.free_params])

        def e(p):
            res = self.energy(p)
            return res

        result = minimize(e, initial, args=global_params)
        optimum = result.x

        for p, x in zip(self.free_params, optimum):
            p.setter(x)
        self.adjust()

    def is_connected(self):
        connected = True
        for p in self.parts.values():
            if isinstance(p, Joint) and not p.is_connected():
                connected = False
        return connected

    @classmethod
    def from_yaml(cls, text: str):
        """
        :param text: YAML string, example:

        parts:
            lever1:
                type: solid
                points:
                    - [0, 0, 'fixed']
                    - [0, 1, 'j1']
            slider1:
                type: slider
                points:
                    - [0, 1, 'fixed']
                    - [1, 1, 'j1']
        run:
            - ['slider1', 0.5, 1.5]

        :return: data structure to create the model


        """
        data = yaml.load(text, yaml.Loader)

        model = Mech()
        base = model.base

        part_types = {
            'solid': Solid,
            'slider': Slider,
        }

        parts = data['parts']
        joints = {}
        runs = data['run']

        def join(joint_name: str, p: Point):
            if joint_name not in joints:
                joints[joint_name] = []
            joints[joint_name].append(p)

        base_joint_index = 0
        for part_name, part_data in parts.items():
            type_name = part_data['type']
            PT = part_types[type_name]
            part = PT(part_name)
            model.add(part)
            for i, point in enumerate(part_data['points']):
                x, y, joint_name = point
                if joint_name == 'fixed':
                    base_joint_name = 'base' + str(base_joint_index)
                    base_joint_index += 1
                    base_point_name = str(len(base.points))
                    base.point(base_point_name, x, y)
                    join(base_joint_name, base.points[base_point_name])
                    point_name = str(i)
                    part.point(point_name, x, y)
                    join(base_joint_name, part.points[point_name])
                elif joint_name == 'free':
                    pass
                else:
                    point_name = str(i)
                    part.point(point_name, x, y)
                    join(joint_name, part.points[point_name])

        for joint_name, points in joints.items():
            if len(points) >= 2:
                joint = Joint(joint_name, 2 if joint_name.startswith('base') else 1)
                model.add(joint)
                for point in points:
                    joint.join(point)

        motion = []
        for run in runs:
            part_name, start, end, n_steps = (run + [None])[:4]
            part = model.parts[part_name]
            if isinstance(part, Slider):
                motion.append([f"{part.name}.length", move(start, end, n_steps)])

        return model, motion

    def _collect_for_equations(self):
        if self.force_points is not None:
            return

        self.force_points = {}
        self.force_points_index = []
        self.force_joints = []

        for joint in self.parts.values():
            if isinstance(joint, Joint):
                n = 0
                for point in joint.points.values():
                    if point.part.name != 'base':
                        if point.part.name not in self.force_points:
                            self.force_points[point.part.name] = {}
                        self.force_points[point.part.name][point.name] = len(self.force_points_index)
                        self.force_points_index.append(point)
                        n += 1
                if n > 1:
                    self.force_joints.append(joint)

    def var_index(self, part_name, point_name, coord):
        return self.force_points[part_name][point_name] * 2 + coord

    def from_var_index(self, i):
        coord = i % 2
        point = self.force_points_index[i // 2]
        return point.part, point, coord

    def _static_force_equations(self, ext_forces, ext_moments) -> Tuple[NDArray[2], NDArray[1]]:

        self._collect_for_equations()

        # строим систему уравнений
        n = len(self.force_points_index) * 2
        A = np.zeros((n, n))
        b = np.zeros(n)

        i = 0  # номер уравнения по порядку

        # 3-й закон Ньютона

        for joint in self.force_joints:
            points = joint.points
            if len(points) >= 2:
                for point in points.values():
                    if point.part.name != 'base':
                        part = point.part
                        A[i, self.var_index(part.name, point.name, 0)] = 1
                        A[i + 1, self.var_index(part.name, point.name, 1)] = 1
                i += 2

        # равновесие сил

        for part_name, points in self.force_points.items():
            part = self.parts[part_name]
            for k in points.values():
                point = self.force_points_index[k]
                A[i, self.var_index(part.name, point.name, 0)] = 1
                A[i + 1, self.var_index(part.name, point.name, 1)] = 1
                if (part.name, point.name) in ext_forces:
                    f = ext_forces[(part.name, point.name)]
                    b[i] = -f[0]
                    b[i + 1] = -f[1]
            i += 2

        # равновесие моментов

        for part_name, points in self.force_points.items():
            part = self.parts[part_name]
            if part.name in ext_moments:
                b[i] = ext_moments[part.name]
            points = [self.force_points_index[k] for k in points.values()]
            p0 = points[0]
            for point in points[1:]:
                radius_vector = point.absolute - p0.absolute
                A[i, self.var_index(part.name, point.name, 0)] = -radius_vector[1]
                A[i, self.var_index(part.name, point.name, 1)] = radius_vector[0]
                if (part.name, point.name) in ext_forces:
                    f = ext_forces[(part.name, point.name)]
                    b[i] += (radius_vector[1] * f[0] - radius_vector[0] * f[1])
            i += 1

        return A, b

    def charge(self, ext_forces={}, ext_moments={}):
        """Calculate forces at points by solving system of linear equations"""
        self.A, self.b = self._static_force_equations(ext_forces, ext_moments)
        try:
            X = solve(self.A, self.b)
        except np.linalg.LinAlgError:
            X = pinv(self.A).dot(self.b)

        for i, x in enumerate(X):
            part, point, coord = self.from_var_index(i)
            point.force[coord] = x

    def __repr__(self):
        return '\n'.join([repr(p) for p in self.parts.values()])


def move(start, end, n=None):
    if n is None:
        n = 20
    step = (end - start) / n
    return np.arange(start, end, step)


if __name__ == '__main__':

    txt = """
parts:
    lever1:
        type: solid
        points:
            - [0, 0, 'fixed']
            - [0, 1, 'j1']
    slider1:
        type: slider
        points:
            - [0, 1, 'fixed']
            - [1, 1, 'j1']
run:
    - ['slider1', 0.5, 1.5]"""
