import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yaml

from mech import Mech, move
from parts import Param, Part, Solid, Base, Joint, Slider, Point, round_vector
from runners import Runner1D

def draw_model(model, end_position = False):
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    cm = plt.get_cmap('tab20')
    colors = cm(np.linspace(0, 1, 20))
    for i, part in enumerate(model.parts.values()):
        if not isinstance(part, Joint) and not part.name == 'base':
            points = list(part.points.values())
            first_segment = True
            for p1, p2 in zip(points[:-1], points[1:]):
                x = [p1.absolute[0], p2.absolute[0]]
                y = [p1.absolute[1], p2.absolute[1]]
                label = part.name if first_segment and end_position else None
                plt.plot(x, y, color=colors[i], label= label)
                first_segment = False
    if end_position:
        plt.legend()

    plt.show()


def draw_trace(model, trace, columns):
    df = pd.DataFrame(trace)
    part_names = [k for k in model.parts.keys()]
    cm = plt.get_cmap('tab20')
    colors = cm(np.linspace(0, 1, 20))
    for col in columns:
        part_name = col.split('.')[0]
        part_index = part_names.index(part_name)
        tr = pd.DataFrame(df[col].tolist(), columns=['x', 'y'])
        plt.scatter(tr.x, tr.y, color=colors[part_index])

    plt.show()

def plot_forces(trace, cols=None):
    tr = pd.DataFrame(trace)
    forces = {'t': tr.t.values}

    if cols is None:
        cols = [c[:-2] for c in tr.columns if c.split('.')[-1] == 'F']

    for col in cols:
        cn = col.split('.')
        point_name = f"{cn[0]}.{cn[1]}"
        force = pd.DataFrame(tr[col + '.F'].tolist(), columns=['x', 'y'])
        forces[point_name] = np.sqrt(force.x ** 2 + force.y ** 2)

    forces = pd.DataFrame(forces).set_index('t')
    for col in forces.columns:
        plt.plot(forces.index.values, forces[col].values, label=col)
    plt.legend()
    plt.show()


def pt(point_name):
    point_loc = eval(point_name)
    if len(point_loc) < 3:
        point_loc += [point_name]
    yaml_array = f"[{point_loc[0]}, {point_loc[1]}, {point_loc[2]}]"
    return yaml_array


def distance(v1, v2):
    return np.sqrt((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)

def elongation(v1, v2, extra_length):
    diff = np.array([v2[0] - v1[0], v2[1] - v1[1]])
    norm = distance(v1, v2)
    direction = diff / norm
    end = np.array([v2[0], v2[1]]) + direction * extra_length
    return [end[0], end[1]]

L = 1 #np.sqrt(2) / 2

txt = f"""
parts:
    lever1:
        type: solid
        points:
            - [0, 0, 'fixed']
            - [0, {L}, 'j1']
    slider1:
        type: slider
        points:
            - [1, 0, 'fixed']
            - [1, {L}, 'j1']
run:
    - ['slider1', 0.5, 1.5]"""

yaml.load(txt, yaml.Loader)

model2, motion2 = Mech.from_yaml(txt)

model2.solve()
print(model2)

model2.charge(ext_moments = {'lever1': 1.0})

runner2 = Runner1D(model2,
                      motion=motion2,
                      tracing_points=['lever1.1'],
                      moments=lambda t: {'lever1': 1.0},
                      draw_model=draw_model,
                      draw_trace=draw_trace
                     )

runner2.run()
