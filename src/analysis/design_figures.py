"""
Figures of Section 3 (model design), drawn with the current planner:

  design_city_basic.png    cell types and permitted directions of a two-block grid
  astar_walk_weights.png   pedestrian routes for beta = 1, 2, 3 (rho = 1)
  astar_drive_weights.png  driver routes for rho = 5, 1, 0.01 (beta = 1)

    cd src
    PYTHONPATH=. python -m analysis.design_figures [OUT_DIR]
"""
import os, sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from model.UrbanModelling import CityModel
from model.AgentImpl import AstarWalker, AstarDriver
from utils.AgentUtils import gen_city
from utils.AnimationUtils import plot_city, plot_route

DPI = 600


def model_for(city_grid, **extra):
    params = {'seed': 0, 'debug': False, 'display': False, 'steps': 1,
              'city_grid': city_grid, 'obstacles': [], 'potholes': []}
    params.update(extra)
    model = CityModel(params)
    model.setup()
    return model


def route_figure(model, cls, specs, colors, path):
    agents = []
    for spec in specs:
        agent = cls(model, **spec)
        model.city.add_agents([agent], positions=[spec['start']])
        agent.find_route()
        agent.initialize_agent()
        agents.append(agent)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_route(model.city, ax, agents, cmap=colors)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def city_figure(path):
    model = model_for(gen_city((15, 15), (1, 2), street_parking=False, two_way=True, nlanes=2, crop=0))
    fig, ax = plt.subplots(figsize=(10, 10))
    for color, label in (('#FFFFFF', 'zebra crossing'), ('#EDEAE3', 'sidewalk'),
                         ('#3B4250', 'roads'), ('#3FA76B', 'buildings')):
        ax.plot([0], [0], color=color, linewidth=5, label=label)
    plot_city(model.city, ax, alpha='AA')
    ax.xaxis.set_major_locator(MultipleLocator(1, offset=-0.53))
    ax.yaxis.set_major_locator(MultipleLocator(1, offset=-0.53))
    ax.grid(True, color='k', linewidth=.4)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.legend(loc='upper left', facecolor='white', framealpha=.7)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    city_figure(os.path.join(out_dir, 'design_city_basic.png'))

    walk = model_for(gen_city((15, 12), (1, 2), street_parking=False, nlanes=1, crop=0))
    route_figure(walk, AstarWalker,
                 [{'start': (3, 10), 'goal': (11, 13), 'speed': 5, 'beta': b, 'rho': 1, 'direction': (1, 0)} for b in (1, 2, 3)],
                 ['#1c6638', '#fd6a02', 'red'], os.path.join(out_dir, 'astar_walk_weights.png'))

    drive = model_for(gen_city((14, 16), (3, 3), street_parking=False, nlanes=2, crop=12))
    start, goal = (10, 5), (14, 10)
    route_figure(drive, AstarDriver,
                 [{'start': start, 'goal': goal, 'speed': 50, 'beta': 1, 'rho': r, 'direction': (-1, 0)} for r in (5, 1, .01)],
                 ['#006400', 'orange', 'r'], os.path.join(out_dir, 'astar_drive_weights.png'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(__file__), '../..', 'Experiments', 'analysis', 'design'))
