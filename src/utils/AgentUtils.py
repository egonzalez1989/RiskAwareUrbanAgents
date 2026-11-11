import heapq, math
######################################################
# HEURISTICS
######################################################
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1])**2)

def weighted_astar(start, goal, grid, step_cost, step_risk,
                   heuristic=manhattan, beta=1, rho=1, start_dir=None,
                   cost_table=None, risk_cache=None):
    # With start_dir (the agent's heading at start), step_risk also receives the
    # heading with which the path reached each cell, and the search state is the
    # pair (cell, heading): two paths that reach a cell with different headings
    # face different maneuver risks ahead, so neither of them is discarded.
    # Optional speed-ups that do not change the result: cost_table maps every cell
    # to step_cost(cell), and risk_cache memoizes step_risk across searches (the
    # risk depends only on the action, the cell and the heading)
    actions = ['N', 'S', 'E', 'W']
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    moves = list(zip(actions, directions))
    rows, cols = grid.shape
    with_heading = start_dir is not None
    cost = cost_table.__getitem__ if cost_table is not None else step_cost
    risks = risk_cache if risk_cache is not None else {}
    h = (lambda v: abs(v[0] - goal[0]) + abs(v[1] - goal[1])) if heuristic is manhattan \
        else (lambda v: heuristic(v, goal))
    push, pop = heapq.heappush, heapq.heappop

    start_state = (start, start_dir)
    open_set = [(beta * h(start), 0, start_state)]
    cost_so_far = {start_state: 0}
    came_from = {}

    while open_set:
        _, current_g, state = pop(open_set)
        current, current_dir = state

        if current == goal:
            path = [current]
            while state in came_from:
                state = came_from[state]
                path.append(state[0])
            return path[::-1]

        if current_g > cost_so_far[state]:      # outdated entry
            continue

        for action, (dx, dy) in moves:
            nx, ny = current[0] + dx, current[1] + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            next_node = (nx, ny)

            # Accumulated path cost: traversal effort + weighted risk of entering next_node
            # r(a, next_node): risk of action a upon entering the target cell
            key = (action, current, current_dir)
            risk = risks.get(key)
            if risk is None:
                risk = risks[key] = step_risk(action, current, current_dir) if with_heading \
                    else step_risk(action, current)
            new_g = current_g + cost(next_node) + rho * risk

            # Evaluation function: f(v) = g(v) + rho*r(a,v) + beta*h(v)
            next_state = (next_node, (dx, dy) if with_heading else None)
            old = cost_so_far.get(next_state)
            if old is None or new_g < old:
                cost_so_far[next_state] = new_g
                came_from[next_state] = state
                push(open_set, (new_g + beta * h(next_node), new_g, next_state))
    return None

def is_right_turn(dir1, dir2):
    return dir2 == (dir1[1], -dir1[0])

def is_left_turn(dir1, dir2):
    return is_right_turn(dir2, dir1)
############################################################
# CITY AND BLOCK GENERATION (ENVIRONMENT)
############################################################
import random, numpy as np
"""
Function for the generation of a single city block with the following characteristics: 
- 4 cols and rows for lanes (2 for each direction)
- 2 cols and rows for parking places (if required) 
- 2 cols and rows for sidewalk
- 8 cells for zebra cross
- remaining inner cells are buildings 
:param block_shape: Shape of the required block in the form of tuple (int, int)
:param street_parking: Dictionary allowing parking on the directions with positive value. If boolean all values assume 0 or 1 
:return: A city block of the aforemention characteristics
"""
def gen_block(block_shape = (25, 25), nlanes = 1, street_parking = False):
    #block_shape = (block_shape[0]+2, block_shape[1]+2)
    # Create a block using measures of given elements:
    # - a sidewalk cell for each direction
    sw = dict(zip(['N', 'S', 'E', 'W'], [1] * 4))
    # - n lanes for each direction
    ln = dict(zip(['N', 'S', 'E', 'W'], [nlanes] * 4))
    shape = (block_shape[0], block_shape[1], 3)
    block = np.full(shape, [' ', ' ', ' '])
    # Parking
    val = 1 if street_parking else 0
    sp = dict(zip(['N', 'S', 'E', 'W'], [val] * 4))

    # Borders. It allows to add sidewalks and zebra crosses easily
    # Buildings. Road and sidewalk elements will overwrite accordingly
    block[:, :, 0] = 'r'
    block[ln['N'] + sp['N'] + sw['N']: -(ln['S'] + sp['S'] + sw['S']),
        ln['E'] + sp['E'] + sw['E']: -(ln['W'] + sp['W'] + sw['W']), 0] = 'b'

    # Parking places
    block[(ln['N'], -ln['S'] - 1), ln['W']+2:-ln['E']-1, 0] = block[ln['N']+2:-ln['S']-2, (ln['W'], -ln['E'] - 1), 0] = 'p'

    # Roads. Fill edge values with r
    #block[:ln['N'], :, 0] = block[-ln['S']:, :, 0] = block[:, :ln['E'], 0] = block[:, -ln['W']:, 0] = 'r'

    # Sidewalks
    block[ln['N']+sp['N']:-ln['S']-sp['S'], (ln['W']+sp['W'], -ln['E']-sp['E']-1), 0] = 's'
    block[(ln['N']+sp['N'], -ln['S']-sp['S']-1), ln['E']+sp['E']:-ln['W']-sp['W'], 0] = 's'
    blocksofar = np.array([[''.join(cell) for cell in row] for row in block])

    # Zebra crosses
    block[(ln['N']+sp['N'], -ln['S']-sp['S']-1), :ln['W']+sp['W'], 0] = block[(ln['N']+sp['N'], -ln['S']-sp['S']-1), -ln['E']-sp['E']:, 0] = 'z'
    block[:ln['N']+sp['N'], (ln['W']+sp['W'], -ln['E']-sp['E']-1), 0] = block[-ln['S']-sp['S']:, (ln['W']+sp['W'], -ln['E']-sp['E']-1), 0] = 'z'
    blocksofar = np.array([[''.join(cell) for cell in row] for row in block])

    # Directions
    block[:ln['N']+sp['N'], :, 2] = 'E'
    block[-ln['S']-sp['S']:, :, 2] = 'W'
    block[:, :ln['W']+sp['W'], 1] = 'N'
    block[:, -ln['E']-sp['E']:, 1] = 'S'

    #Turns
    block[ln['N']-1, ln['E']-1, 0] = block[ln['N']-1, -ln['W'], 0] =\
        block[-ln['S'], ln['E']-1, 0] = block[-ln['S'], -ln['W'], 0] = 't'
    if nlanes > 1:
        block[0, 0, 0] = block[0, -1, 0] = block[-1, 0, 0] = block[-1, -1, 0] = 'l'

    blocksofar = np.array([[''.join(cell) for cell in row] for row in block])


    block = np.array([[''.join(cell) for cell in row] for row in block])
    return block

"""
Function for the generation of a city of simple blocks.
:param block_shape: Shape of the required block in the form of tuple (int, int)
:param city_shape: Create an array of complete city blocks in the form (int, int)
:param obstacles: Number of obstacles in sidewalks
:param potholes: Number of obstacles in the road
:return: A city encodingtogether with the locations of obstacles, potholes, agent's sources and destinations
"""
def gen_city(block_shape = (25, 25), city_shape = (2, 2), street_parking = False, two_way=True, nlanes=1, crop = 0):
    city_block = gen_block(block_shape, street_parking= street_parking, nlanes=nlanes)
    if two_way:
        city_grid = np.tile(city_block, city_shape)
    else:
        city_grid = np.full((block_shape[0] * city_shape[0], block_shape[1] * city_shape[1]), '   ')
        city_grid[1:block_shape[0]+1, 1:block_shape[1]+1] = city_block
    if crop:
        city_grid = city_grid[crop:-crop, crop:-crop]
    return city_grid

def gen_closures(city_grid, n=8, length=3, seed=0):
    """Close one lane of n street segments (roadwork): `length` consecutive
    plain road cells of a single lane become obstacles. The opposite lane stays
    open, so a driver can pass the closure legally only by detouring around the
    block, or by driving the wrong way along the opposite lane."""
    rng = random.Random(seed)
    city_grid = np.array(city_grid)
    H, W = city_grid.shape
    candidates = []
    for i in range(H):
        for j in range(W):
            code = city_grid[i, j]
            if code[0] != 'r':
                continue
            # Lane direction: vertical lanes advance along rows, horizontal ones along columns
            step = (1, 0) if code[1] != ' ' else (0, 1)
            cells = [(i + k * step[0], j + k * step[1]) for k in range(length)]
            if all(0 <= x < H and 0 <= y < W and city_grid[x, y] == code for x, y in cells):
                candidates.append(cells)
    closures, used = [], set()
    rng.shuffle(candidates)
    for cells in candidates:
        # Keep closures apart (no shared street segment or block side)
        if all(abs(x - u) + abs(y - v) > 15 for x, y in cells for u, v in used):
            closures.append(cells)
            used.update(cells)
        if len(closures) == n:
            break
    return [c for cells in closures for c in cells]

def gen_obstacles(city_grid, obstacles = .05, potholes = .05):
    # If obstacles is int, generate that number of obstacles
    city_grid = np.array(city_grid)
    H, W = city_grid.shape
    sidewalk_cells = [(int(x[0]), int(x[1])) for x in np.argwhere(city_grid == 's  ')]
    road_cells = set(tuple(x) for x in np.argwhere(np.array([city_grid[i, j][0] in ('r', 't') for i in range(H) for j in range(W)]).reshape((H, W))))
    if type(obstacles) is not int:
        if obstacles >= 0 and obstacles <= 1:
            obstacles = int(obstacles * len(sidewalk_cells))
        else:
            raise ValueError('obtacles must be integer or float between 0 and 1')

    if type(potholes) is not int:
        if potholes >= 0 and potholes <= 1:
            potholes = int(potholes * len(road_cells))
        else:
            raise ValueError('potholes must be integer or float between 0 and 1')

    obstacle_cells = set(random.sample(list(sidewalk_cells), obstacles))
    potholes_cells = set(random.sample(list(road_cells), potholes))
    return obstacle_cells, potholes_cells


def get_walker_endpoints(city_grid):
    # Pedestrians can be spawned from any sidewalk cell
    H, W = city_grid.shape
    endpoints = [tuple(x) for x in np.argwhere(
        np.array([city_grid[i, j][0] == 's' for i in range(H) for j in range(W)]).reshape((H, W)))]
    return endpoints, endpoints[:]

def get_parking_spots(city_grid):
    H, W = city_grid.shape
    parking_spaces = [tuple(x) for x in np.argwhere(
        np.array([city_grid[i, j][0] == 'p' for i in range(H) for j in range(W)]).reshape((H, W)))]
    return parking_spaces

def get_driver_endpoints(city_grid):
    # Car will be spawned from street edges or empty sidewalks (simulating leaving abuilding)
    H, W = city_grid.shape
    driver_sources, driver_goals = [], []
    for j in range(W):
        if 'S' in city_grid[0, j]:
            driver_sources.append((0, j))
            driver_goals.append((H-1, j))
        if 'N' in city_grid[H-1, j]:
            driver_sources.append((H-1, j))
            driver_goals.append((0, j))
    for i in range(H):
        if 'E' in city_grid[i, 0]:
            driver_sources.append((i, 0))
            driver_goals.append((i, W-1))
        if 'W' in city_grid[i, W-1]:
            driver_sources.append((i, W-1))
            driver_goals.append((i, 0))

    building_entry = [tuple(x) for x in np.argwhere(city_grid == 's  ')]
    parking_spots = get_parking_spots(city_grid)
    driver_sources.extend(building_entry)
    driver_goals.extend(get_parking_spots(city_grid))
    if len(parking_spots) == 0:
        driver_goals.extend(building_entry)
    return driver_sources, driver_goals

def is_behind(dir, pos1, pos2):
    dir2 = (pos1[0] - pos2[0], pos1[1] - pos2[1])
    dir2 = (dir2[0] // max(1, abs(dir2[0])), dir2[1] // max(1, abs(dir2[1])))
    return dir2 == dir


if __name__ == '__main__':
    gen_block((10, 10), street_parking=True, nlanes=2)