import random

from model.UrbanModelling import CityModel
from utils.Reporting import load_heatmaps, generate_heatmaps
from utils.AgentUtils import gen_city, gen_obstacles

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np
    # Custom risks

    random.seed(0)
    block_shape, city_shape = (15, 12), (1, 2)
    nlanes = 1
    city_grid = gen_city(block_shape, city_shape, street_parking=False, nlanes=nlanes, crop=0)
    obstacles, potholes = gen_obstacles(city_grid, 10, 10)

    # Custom risks
    max_block_size = max(block_shape) - 2 * nlanes
    road_risk = (max_block_size - 1) / nlanes
    walk_terrain = ['s', 'z', 'p', 'r', 't', 'l', 'b', 'o']
    walker_risk_map = dict(zip(walk_terrain, [0, 1, 2, 5, 10, np.inf, np.inf, np.inf]))
    # [0, 0, road_risk // 2,
    #              road_risk, 2*road_risk,
    #              4*road_risk, np.inf, np.inf]))

    parameters = {
        'walker_risk_map': walker_risk_map,
        'city_grid': city_grid,
        'walkers': [{'start': (8, 10), 'goal': (8, 13), 'speed': 10, 'weight': 1, 'risk_weight': 10}],
        'steps': 60,
        'display': False
    }
    fig = plt.figure()  # figsize=(8, 4))
    ax = fig.add_subplot(111)
    model = CityModel(parameters)
    model.run()

#     import numpy as np
#     random.seed(0)
#     block_shape, city_shape = (15, 12), (1, 2)
#     nlanes = 2
#     city_grid = gen_city(block_shape, city_shape, street_parking=False, nlanes=nlanes, crop=0)
#     obstacles, potholes = gen_obstacles(city_grid, 10, 10)
#
#     # Custom risks
#     # street, zcross, road, building, obstacle
#     max_block_size = max(block_shape) - 2 * nlanes
#     road_risk = 1 / max_block_size
#     walk_terrain = ['s', 'z', 'p', 'r', 't', 'l', 'b', 'o']
#     walker_risk_map = dict(zip(walk_terrain, [0, 0, road_risk,
#                                               max_block_size * road_risk, 2 * max_block_size * road_risk,
#                                               5 * max_block_size * road_risk, np.inf, np.inf]))
#
#     parameters = {
#         'walker_risk_map': walker_risk_map,
#         'city_grid': city_grid,
#         'walkers': [{'start': (5, 9), 'goal': (9, 14), 'max_speed': random.randint(5, 10), 'weight': i} for i in
#                     (1, 2, 3)],
#         'steps': 60,
#         'display': False
#     }
#
#     model = CityModel(parameters)
#     model.run()
    # heatmaps = load_heatmaps(model.metrics)
    # generate_heatmaps(model.city, heatmaps, figsize=(12, 10))
