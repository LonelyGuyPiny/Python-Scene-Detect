#
# Copyright (C) 2021 Brandon Castellano <http://www.bcastell.com>.
#



from typing import Tuple, Sequence
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Bar:
    x0: int
    x1: int
    colour: Tuple[float, float, float, float]
    height: float = 1.0
    offset: float = 0.0

@dataclass
class ChartRow:
    label: str
    values: Sequence[Bar]



def shift_scene_example_default():

    ROW_HEIGHT = 10
    BAR_HEIGHT = 9
    X_AXIS_BOUNDS = (0, 80, 10, 0)

    X_AXIS_BOUNDS_SMALL = (10, 70, 10, 4)
    blue = (44/255.0, 149/255.0, 216/255.0, 0.8)
    orange = (218/255.0, 158/255.0, 60/255.0, 0.8)


    font = {'fontname': 'JetBrains Mono'}

    CHARTS = [('SceneList: drop(min_len=L) and merge(min_len=L)', X_AXIS_BOUNDS_SMALL, "scene_list_drop_merge.png", [

        ChartRow(label='scene list', values=[
            Bar(x0 = 10, x1 = 20, colour = blue),
            Bar(x0 = 30, x1 = 40, colour = orange),
            Bar(x0 = 40, x1 = 60, colour = blue),
        ]),
        ChartRow(label='.drop(L=11)', values=[
            Bar(x0 = 40, x1 = 60, colour = blue),
        ]),
        ChartRow(label='.merge(L=11)', values=[
            Bar(x0 = 10, x1 = 20, colour = blue),
            Bar(x0 = 30, x1 = 60, colour = orange),
        ]),
        ChartRow(label='.merge(L=11)\n.drop(L=11)', values=[
            Bar(x0 = 30, x1 = 60, colour = orange),
        ]),
    ]),
    ('SceneList: merge(min_len=L, max_dist=D)', X_AXIS_BOUNDS_SMALL, "scene_list_merge.png", [

        ChartRow(label='scene list', values=[
            Bar(x0 = 10, x1 = 20, colour = blue),
            Bar(x0 = 30, x1 = 40, colour = orange),
            Bar(x0 = 40, x1 = 60, colour = blue),
        ]),
        ChartRow(label='.merge(\nL=11, D=9)', values=[
            Bar(x0 = 10, x1 = 20, colour = blue),
            Bar(x0 = 30, x1 = 60, colour = orange),
        ]),
        ChartRow(label='.merge(\nL=11, D=10)', values=[
            Bar(x0 = 10, x1 = 40, colour = blue),
            Bar(x0 = 40, x1 = 60, colour = orange),
        ]),
        ChartRow(label='.merge(\nL=21, D=9)', values=[
            Bar(x0 = 10, x1 = 20, colour = blue),
            Bar(x0 = 30, x1 = 60, colour = orange),
        ]),
        ChartRow(label='.merge(\nL=21, D=10)', values=[
            Bar(x0 = 10, x1 = 60, colour = blue),
        ]),
    ]),
    ('SceneList: contract(start=S, end=E)', X_AXIS_BOUNDS_SMALL, "scene_list_contract.png", [

        ChartRow(label='scene list', values=[
            Bar(x0 = 10, x1 = 40, colour = blue),
            Bar(x0 = 40, x1 = 50, colour = orange),
            Bar(x0 = 50, x1 = 60, colour = blue),
        ]),
        ChartRow(label='.contract(S=5)', values=[
            Bar(x0 = 15, x1 = 40, colour = blue),
            Bar(x0 = 45, x1 = 50, colour = orange),
            Bar(x0 = 55, x1 = 60, colour = blue),
        ]),
        ChartRow(label='.contract(E=5)', values=[
            Bar(x0 = 10, x1 = 35, colour = blue),
            Bar(x0 = 40, x1 = 45, colour = orange),
            Bar(x0 = 50, x1 = 55, colour = blue),
        ]),
        ChartRow(label='.contract(\nS=5, E=5)', values=[
            Bar(x0 = 15, x1 = 35, colour = blue),
        ]),
    ]),
    ('SceneList: expand(start=S, end=E, merge=True [default])', X_AXIS_BOUNDS, "scene_list_expand_default.png", [

        ChartRow(label='scene list', values=[
            Bar(x0 = 10, x1 = 20, colour = blue),
            Bar(x0 = 30, x1 = 40, colour = orange),
            Bar(x0 = 40, x1 = 60, colour = blue),
        ]),
        ChartRow(label='.expand(S=5)', values=[
            Bar(x0 = 5, x1 = 20, colour = blue),
            Bar(x0 = 25, x1 = 60, colour = orange),
        ]),
        ChartRow(label='.expand(E=5)', values=[
            Bar(x0 = 10, x1 = 25, colour = blue),
            Bar(x0 = 30, x1 = 65, colour = orange),
        ]),
        ChartRow(label='.expand(\nS=5, E=5)', values=[
            Bar(x0 = 5, x1 = 25, colour = blue),
            Bar(x0 = 25, x1 = 65, colour = orange),
        ]),
        ChartRow(label='.expand(\nS=10, E=10)', values=[
            Bar(x0 = 0, x1 = 20, colour = blue),
            Bar(x0 = 30, x1 = 70, colour = orange),
            Bar(x0 = 20, x1 = 30, colour = blue, height = 0.5, offset = 0.5),
            Bar(x0 = 20, x1 = 30, colour = orange, height = 0.5, offset = 0.0),
        ]),
        ChartRow(label='.expand(\nS=11, E=11)', values=[
            Bar(x0 = 0, x1 = 70, colour = blue),
        ]),


        ]),
    ('SceneList: expand(start=S, end=E, merge=False)', X_AXIS_BOUNDS, "scene_list_expand_no_merge.png", [

        ChartRow(label='scene list', values=[
            Bar(x0 = 10, x1 = 20, colour = blue),
            Bar(x0 = 30, x1 = 40, colour = orange),
            Bar(x0 = 40, x1 = 60, colour = blue),
        ]),
        ChartRow(label='.expand(\nS=5)', values=[
            Bar(x0 = 5, x1 = 20, colour = blue),
            Bar(x0 = 25, x1 = 35, colour = orange),
            Bar(x0 = 40, x1 = 60, colour = blue),
            Bar(x0 = 35, x1 = 40, colour = blue, height = 0.5, offset = 0.0),
            Bar(x0 = 35, x1 = 40, colour = orange, height = 0.5, offset = 0.5),
        ]),
        ChartRow(label='.expand(\nE=5)', values=[
            Bar(x0 = 10, x1 = 25, colour = blue),
            Bar(x0 = 30, x1 = 40, colour = orange),
            Bar(x0 = 45, x1 = 65, colour = blue),
            Bar(x0 = 40, x1 = 45, colour = blue, height = 0.5, offset = 0.0),
            Bar(x0 = 40, x1 = 45, colour = orange, height = 0.5, offset = 0.5),
        ]),
        ChartRow(label='.expand(\nS=5, E=10)', values=[
            Bar(x0 = 5, x1 = 25, colour = blue),
            Bar(x0 = 25, x1 = 30, colour = blue, height = 0.5, offset = 0.5),
            Bar(x0 = 25, x1 = 30, colour = orange, height = 0.5, offset = 0.0),
            Bar(x0 = 30, x1 = 35, colour = orange),
            Bar(x0 = 50, x1 = 70, colour = blue),
            Bar(x0 = 35, x1 = 50, colour = blue, height = 0.5, offset = 0.0),
            Bar(x0 = 35, x1 = 50, colour = orange, height = 0.5, offset = 0.5),
        ]),
        ChartRow(label='.expand(\nS=15, E=10)', values=[
            Bar(x0 = 0, x1 = 15, colour = blue),
            Bar(x0 = 15, x1 = 25, colour = blue, height = 0.3333, offset = 0.6666),
            Bar(x0 = 25, x1 = 30, colour = blue, height = 0.3333, offset = 0.6666),
            Bar(x0 = 15, x1 = 25, colour = orange, height = 0.6666, offset = 0.0),
            Bar(x0 = 50, x1 = 70, colour = blue),
            Bar(x0 = 25, x1 = 30, colour = blue, height = 0.3333, offset = 0.0),
            Bar(x0 = 30, x1 = 50, colour = blue, height = 0.3333, offset = 0.0),
            Bar(x0 = 25, x1 = 30, colour = orange, height = 0.3333, offset = 0.3333),
            Bar(x0 = 30, x1 = 50, colour = orange, height = 0.6666, offset = 0.3333),
        ]),


        ])]


    for i, (chart_title, x_bounds, filename, chart_rows) in enumerate(CHARTS):

        _, ax = plt.subplots()
        start_offset = ROW_HEIGHT + (ROW_HEIGHT / 2.0)
        y_offset = start_offset
        for chart_row in chart_rows[::-1]:
            [
                ax.broken_barh(
                    [(bar.x0, bar.x1 - bar.x0)],
                    (y_offset + bar.offset * BAR_HEIGHT, BAR_HEIGHT * bar.height),
                    facecolors=bar.colour,
                    edgecolor=(0, 0, 0, 0)) for bar in chart_row.values
            ]
            y_offset += ROW_HEIGHT


        ax.set_ylim(start_offset - 1, y_offset)
        ax.set_xlim(x_bounds[0] - x_bounds[3], x_bounds[3] + x_bounds[1] - x_bounds[2])
        ax.set_xlabel('frame number', family="monospace", **font)
        y_start = ROW_HEIGHT * (1 + len(chart_rows))
        y_end = ROW_HEIGHT
        ax.set_yticks(
            range(y_start, y_end, -ROW_HEIGHT),
            labels=[row.label for row in chart_rows],
            horizontalalignment='right',
            family="monospace",
             **font
        )
        ax.set_xticks(range(*x_bounds[0:3]), family="monospace", **font)
        ax.grid(True, alpha=0.8, axis='x')
        ax.set_axisbelow(True)

        plt.tight_layout(pad=1.5)
        plt.title(chart_title, family="monospace", **font)
        plt.savefig(fname=filename, dpi=300, bbox_inches='tight')


shift_scene_example_default()
