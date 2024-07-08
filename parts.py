from typing import List, Callable, Tuple, Dict

import numpy as np

PRECISION = 0.0001
DEC_PLACES = 4

FULL_CIRCLE = 2 * np.pi


def rotation_matrix(theta):
    x = np.cos(theta)
    y = np.sin(theta)
    return np.array([[x, -y], [y, x]])


def norm(v: np.array):
    return np.sqrt(v.dot(v))


def round_vector(v: np.array):
    return np.round(v, DEC_PLACES)


class Param:
    def __init__(self, setter: Callable[[float], None], getter: Callable[[], float]):
        self.setter = setter
        self.getter = getter


class Point:
    def __init__(self, part: 'Part', name: str, x: float, y: float):
        self.name: str = name
        self.part: Part = part
        self.relative = np.array([x, y])
        self.absolute = np.array([x, y])

        self.force = np.zeros(2)

    def same_as(self, p):
        return norm(self.absolute - p.absolute) < PRECISION

    def __repr__(self):
        return f"    point '{self.name}': {round_vector(self.relative)} -> {round_vector(self.absolute)}, F={round_vector(self.force)}"


def print_point(p: Point):
    return f"'{p.part.name}.{p.name}': {round_vector(p.relative)} -> {round_vector(p.relative)}, F={round_vector(p.force)}"


class Part:
    def __init__(self, name: str):
        self.name = name
        self.points: Dict[str, Point] = {}

    def free_params(self) -> List[Param]:
        return []

    def input_params(self) -> List[Tuple[str, Param]]:
        return []

    def adjust(self):
        pass

    def energy(self) -> float:
        return 0.0

    def __repr__(self):
        return f"Part '{self.name}'"


class Base(Part):  # неподвижная система координат
    def __init__(self, name='base'):
        super().__init__(name)
        self.points = {}

    def point(self, name: str, x: float, y: float):
        if name in self.points:
            raise ValueError(f"Point '{name}' already in part '{self.name}'")
        self.points[name] = Point(self, name, x, y)
        return self

    # TODO измерять длины отрезков между точками

    def __getitem__(self, i):
        return self.points[i]

    def __repr__(self):
        return f"Base '{self.name}':\n" + '\n'.join([str(p) for p in self.points.values()])


class Solid(Base):  # подвижная жесткая деталь
    def __init__(self, name, x=0, y=0, theta=None):
        super().__init__(name)
        self.v = np.zeros(2)
        self.v[0] = x
        self.v[1] = y
        if theta is None:
            L = norm(self.v)
            if L < PRECISION:
                theta = 0
            else:
                theta = np.arctan2(*(self.v / L))
        self.theta = theta

    def adjust(self):
        rotation = rotation_matrix(self.theta)
        for p in self.points.values():
            p.absolute = rotation.dot(p.relative) + self.v

    def point(self, name: str, x: float, y: float):
        super().point(name, x, y)
        self.adjust()
        return self

    def free_params(self) -> List[Param]:
        def set_x(x):
            self.v[0] = x

        def set_y(y):
            self.v[1] = y

        def set_theta(theta):
            self.theta = theta

        def get_x():
            return self.v[0]

        def get_y():
            return self.v[1]

        def get_theta():
            return self.theta

        return [
            Param(set_x, get_x),
            Param(set_y, get_y),
            Param(set_theta, get_theta),
        ]

    def _print_orientation(self):
        angle = round_vector(np.array([np.sin(self.theta), np.cos(self.theta)]))
        v = round_vector(self.v)
        return f"angle={angle}, v={v}"

    def __repr__(self):
        return f"Solid '{self.name}': {self._print_orientation()}\n" \
            + '\n'.join([str(p) for p in self.points.values()])


class Slider(Solid):

    def __init__(self, name: str):
        super().__init__(name)
        self.length = None
        self.relative_direction = None

    def adjust(self):
        if self.length is not None and len(self.points) == 2:
            points = list(self.points.values())
            points[1].relative = points[0].relative + self.relative_direction * self.length
            super().adjust()

    def input_params(self) -> Dict[str, Param]:
        def set_length(x):
            self.length = x

        def get_length():
            return self.length

        return {f"{self.name}.length": Param(set_length, get_length)}

    def point(self, name: str, x, y):
        if len(self.points) == 2:
            raise ValueError(f"Slider '{self.name}' can only have two points")
        super().point(name, x, y)
        if len(self.points) == 2:
            points = list(self.points.values())
            diff = points[1].relative - points[0].relative
            self.length = norm(diff)
            self.relative_direction = diff / norm(diff)
        return self

    def __repr__(self):
        return f"Slider '{self.name}': {self._print_orientation()}\n" \
            + '\n'.join([str(p) for p in self.points.values()])


class Joint(Part):  # Соединение, обеспечивающее свободное вращение деталей
    def __init__(self, name, k=1):
        super().__init__(name)
        self.k = k

    def join(self, p: Point):
        self.points[str(len(self.points))] = p
        return self

    def energy(self) -> float:
        points = list(self.points.values())
        e = 0
        for p1, p2 in zip(points[:-1], points[1:]):
            diff = p2.absolute - p1.absolute
            e += diff.dot(diff) * self.k
        return e

    def is_connected(self):
        points = list(self.points.values())
        connected = True
        for p1, p2 in zip(points[:-1], points[1:]):
            if not p1.same_as(p2):
                connected = False
        return connected

    def __repr__(self):
        return f"Joint '{self.name}': energy={round(self.energy(), DEC_PLACES)}\n" + \
            "\n".join(f"    {print_point(p)}" for p in self.points.values())


class SpiralSpring(Part):  # Спиральная пружина
    def __init__(self, name, joint, stiffness=0):
        super().__init__(name)
        self.joint = joint  # на какое соединение навешиваем пружину
        self.stiffness = stiffness  # На сколько единиц силы, деленных на единицу длины, увеличится момент пружины
        # при закручивании второй детали относительно первой на 1 оборот против часовой стрелки.
        # Если stiffness = 0, то пружина сопротивляется с постоянным моментом moment0 при любом повороте.
        self.angle0 = 0
        self.angle = 0
        self.moment0 = 0
        self.moment = 0

    def fasten(self, moment: float):
        # Закрепить угловое положение деталей относительно друг друга
        # и задать начальный момент пружины в этом положении.
        pass