import math
import numpy as np
from collections import deque
from model.AgentBase import AstarAgent, MobileAgent
from utils.AgentUtils import euclidean, is_right_turn, is_left_turn, manhattan, is_behind


#######################################
# CLASS FOR CARS
#######################################
class AstarDriver(AstarAgent):
    PLAN_WITH_HEADING = True
    # Physical scale: a cell is 5 m and a step 0.3 s, so one cell per step (60 speed units)
    # is 60 km/h and one speed unit is 1 km/h
    CELL_M, STEP_S = 5.0, 0.3
    DESIRED_SPEED = 50          # default desired speed: the urban speed limit, 50 km/h
    REACTION_STEPS = 4          # perception-reaction time: 4 steps = 1.2 s
    SAFE_DECEL = 3.4            # m/s^2, deceleration assumed when choosing a safe speed
    MAX_DECEL = 6.0             # m/s^2, maximum deceleration (emergency braking, about 0.6 g)
    ACCEL = 2.5                 # m/s^2, acceleration up to the desired speed
    GAP = 1.5                   # distance (cells, between centers) kept to a conflict point or to the vehicle ahead
    CROSSING_WINDOW = 3.0       # s: two drivers whose routes share a cell conflict when they would reach it
                                # within this time of each other
    MIN_SPEED = 5               # a target speed below this (km/h) means stopping
    DEADLOCK_WAIT = 10          # steps stopped after which waiting time decides priority
    MAX_WAIT_S = 300.0          # s stopped after which the trip is aborted (SUMO's time-to-teleport)

    #Initializing Car agent elements:
    def setup(self, start, goal, direction, speed, visibility = 10, awareness = 1, beta = 1, rho=1):
        # Dimensions of agent
        self.width = .7
        self.length = .7
        super().setup(start, goal, direction, speed, visibility, awareness, beta, rho)
        self.agent_type = 'driver'
        self.cost_map = self.model.drive_cost
        self.risk_map = self.model.drive_risk
        # Waiting times before reroute
        self.max_wait = 10
        self.wait = 0
        self.stopped = 0        # consecutive steps at speed 0 (not reset by replanning)
        self.seen = {}          # other agent id -> (first step, last step) of the current conflict episode
        self.priority_over = set()  # drivers over which self was given priority to resolve a standstill
        self.future_steps = 15  # predicted steps (4.5 s), within the visibility range
        self.reaction_steps = self.model.p.get('reaction_steps', self.REACTION_STEPS)
        # Trip record: maneuvers of the planned route and of the executed one,
        # replans, cells traversed and active steps (stored in spawned_agents)
        self.trip = {'planned': {}, 'executed': {}, 'replans': 0, 'cells': 0, 'steps': 0}
        # Speed changes per step, in speed units (km/h)
        self.brake_step, self.accel_step = (3.6 * self.STEP_S * a for a in (self.MAX_DECEL, self.ACCEL))

    # Highest speed (units) from which the driver can stop within dist cells, minus the gap, after
    # its perception-reaction time and braking at SAFE_DECEL. A vehicle ahead that drives away at
    # lead_speed needs its own stopping distance, at the maximum deceleration, to stop (Gipps-type
    # safe speed)
    def safe_speed(self, dist, lead_speed=0):
        b, tau = self.SAFE_DECEL, self.reaction_steps * self.STEP_S
        room = max(0., dist - self.GAP) * self.CELL_M + (lead_speed / 3.6) ** 2 / (2 * self.MAX_DECEL)
        v = 3.6 * b * (-tau + math.sqrt(tau ** 2 + 2 * room / b))
        return v if v >= self.MIN_SPEED else 0

    # Move the speed towards a target: brake at most at the maximum deceleration and accelerate at
    # most at ACCEL, never above the desired speed
    def adjust_speed(self, target):
        if target < self.speed:
            self.speed = max(self.speed - self.brake_step, target)
            if self.speed < 1:
                self.speed = 0
        else:
            self.speed = min(self.speed + self.accel_step, target, self.max_speed)

    # Cells of the planned route ahead of the vehicle, within its visibility (cached per cell and route)
    def path_ahead(self):
        key = (self.grid_position(), id(self.route))
        if getattr(self, '_ahead_key', None) != key:
            cell = key[0]
            i = self.route.index(cell) if cell in self.route else -1
            self._ahead = self.route[i + 1:i + 1 + self.visibility] if i >= 0 else \
                [c for c in dict.fromkeys(self.next_positions) if c != cell]
            self._ahead_key = key
        return self._ahead

    # Drivers enter the simulation at rest
    def initialize_agent(self):
        self.speed = 0
        self.set_next_positions()
        self.sense()

    # A vehicle in a driveway or parking space (not on a lane, crossing or intersection) is off the road
    def off_road(self):
        return self.city.city_grid[self.grid_position()][0] not in 'rztl'

    def step_cost(self, position):
        if position in self.city.obstacles or position in self.obstacles:
            cost = self.cost_map['o']
        elif position in self.city.potholes:
            cost = self.cost_map['ph']
        else:
            cost = super().step_cost(position)
        return cost

    #Simulates the risk of a step according to allowed movements
    def step_risk(self, action, position, heading=None):
        cell_code = self.city.city_grid[position]
        # Check type of movement: forward, backward, turning left, right, lane change
        move_type = self.move_type(action, cell_code, heading)
        return self.risk_map[move_type]

    # Maneuver of leaving a cell with the given action, for a vehicle arriving with
    # the given heading (default: its current direction)
    def move_type(self, action, cell_code, heading=None):
        heading = self.direction if heading is None else heading
        # Last move for parking.
        if action == 'V':
            result = 'fw'
        # Backward moves
        elif ((action, cell_code[1]) in [('N', 'S'), ('S', 'N')]) or ((action, cell_code[2]) in [('W', 'E'), ('E', 'W')]):
            result = 'bw'
        # Forward move or turns
        elif ((action, cell_code[1]) in [('N', 'N'), ('S', 'S')]) or ((action, cell_code[2]) in [('W', 'W'), ('E', 'E')]):
            new_dir = self.ACTION_MAP[action]
            # Forward movement (maintains direction).
            if heading == new_dir and cell_code[0]:
                result = 'fw'
            # If a right turn is performed and it is allowed
            elif is_right_turn(heading, new_dir) and cell_code[0] == 't':
                result = 'rt'
            # If a left turn is performed where it is allowed: at the left-turn cells of
            # multi-lane streets, or at any intersection cell when streets have one lane
            # per direction and intersections serve both turns
            elif is_left_turn(heading, new_dir) and cell_code[0] == self.city.left_turn_cell:
                result = 'lt'
            # Otherwise, consider a turn on the direction and not in corners as lane incorporation
            elif ' ' in cell_code:
                result = 'fw'
            else:
                result = 'ft'
        # Any other option is a turn in the wrong direction is considered a lane change
        else:
            result = 'ch'
        return result

    # Speed up towards the desired speed (used by MobileAgent.react when no conflict remains)
    def speedup(self):
        self.speed = min(self.max_speed, self.speed + self.accel_step)
        self.set_next_positions()

    # Count the maneuvers of a route (list of cells) starting with the given heading
    def route_maneuvers(self, route, heading):
        counts = {}
        for u, v in zip(route, route[1:]):
            action = self.DIRECTION_MAP[(v[0] - u[0], v[1] - u[1])]
            m = self.move_type(action, self.city.city_grid[u], heading)
            counts[m] = counts.get(m, 0) + 1
            heading = self.ACTION_MAP[action]
        return counts

    def find_route(self, grid=None):
        super().find_route(grid)
        if not self.trip['planned']:        # first plan: the route chosen at spawn
            self.trip['planned'] = self.route_maneuvers(self.route, self.direction)

    def reroute(self, is_obstacle=lambda x: False):
        self.trip['replans'] += 1
        super().reroute(is_obstacle)

    def move(self):
        # Heading with which the vehicle entered its cell: a move that does not leave the
        # cell already sets the direction of the next action
        cell, action = self.grid_position(), self.next_action
        heading = getattr(self, 'entry_heading', self.direction)
        self.trip['steps'] += 1
        super().move()
        # Executed maneuver: classified when the vehicle enters a new cell
        if action in 'NSEW' and self.grid_position() != cell:
            m = self.move_type(action, self.city.city_grid[cell], heading)
            self.trip['executed'][m] = self.trip['executed'].get(m, 0) + 1
            self.trip['cells'] += 1
            self.entry_heading = self.direction
            if m in ('ft', 'bw'):
                self.model.illegal_moves.append((cell, m))
        if self.grid_position() == self.goal and self.goal in self.city.parking_spaces:
            street_dir = self.city.city_grid[self.grid_position()][1:]
            new_dir = self.ACTION_MAP[street_dir[0] if street_dir[0] != ' ' else street_dir[1]]
            # Align with the street, as long as the car remains in its parking cell
            if self.grid_position(dir=new_dir) == self.goal:
                self.direction = new_dir
            self.removal_time = math.inf

    # A car is parking if inactive and in a parking spot (without removal scheduled)
    def is_parking(self):
        return (not self.active) and self.removal_time == math.inf

    # Sense for possible collisions. Consider agent's route
    def sense(self):
        # Parked or crashed cars do not perceive
        if not self.active:
            self.conflicts = []
            return self.conflicts
        if self.goal in self.next_positions:
            agents = self.city.get_agents_at(self.goal)
            if any(a is not self and a.agent_type == 'driver' and a.is_parking() for a in agents):
                self.find_parking()

        # Default conflict sensing
        self.conflicts = super().sense()
        return self.conflicts


    # This method is used when the parking space is occupied by another driver.
    def find_parking(self):
        pos = self.grid_position()
        occupied = {a.grid_position() for a in self.city.agents
                    if a is not self and a.agent_type == 'driver' and a.is_parking()}
        free = [p for p in self.city.parking_spaces if p != self.goal and p not in occupied]
        if not free:
            return
        # Check if there is a free parking space ahead (in visibility space), nearest first
        near = [p for p in free
                if max(abs(p[0] - pos[0]), abs(p[1] - pos[1])) <= self.visibility
                and not is_behind(self.direction, pos, p)]
        # If no parking is found, move randomly to another parking cell. A new route will be traced
        if near:
            self.goal = min(near, key=lambda p: manhattan(p, pos))
        else:
            self.goal = self.model.random.choice(free)
        self.reroute(lambda x: not x.active)

    # A conflict is acted upon once it has been perceived for the perception-reaction time.
    # A conflict episode continues while the conflict reappears within that time
    def perceived(self, conflicts):
        t, tau = self.model.t, self.reaction_steps
        for _, other in conflicts:
            first, last = self.seen.get(other.id, (t, t))
            self.seen[other.id] = (first if t - last <= tau else t, t)
        self.seen = {k: v for k, v in self.seen.items() if t - v[1] <= tau}
        self.priority_over &= self.seen.keys()     # kept until the conflict ends
        return [c for c in conflicts if t - self.seen[c[1].id][0] >= tau]

    def react(self, conflicts = None):
        # If a list of conflicts is received, use it, else pre-stored conflicts
        conflicts = list(self.conflicts if conflicts is None else conflicts)
        if not self.active:
            return
        self.stopped = self.stopped + 1 if self.speed == 0 else 0
        if self.speed == 0:
            if self.wait >= self.max_wait:
                self.wait = 0
                self.reroute(is_obstacle=lambda x: not x.active or x.speed == 0 or x in self.city.neighbors(self))
            else:
                self.wait += 1
        else:
            self.wait = 0
        # Conflicts not yet perceived do not affect a moving driver, who keeps its course meanwhile;
        # a stationary driver looks before moving off, so it acts on every conflict
        acted = self.perceived(conflicts)
        acted = acted if self.speed > 0 else conflicts
        ignored = [c[1] for c in conflicts if c not in acted]
        # Each perceived conflict sets a target speed (None when self has priority); the driver
        # adjusts its speed to the lowest target
        targets = []
        for location, other in acted:
            target = self.react_driver(other, location) if other.agent_type == 'driver' \
                else self.react_walker(other, location)
            if target is None:
                ignored.append(other)
            else:
                targets.append(target)
        if targets:
            self.adjust_speed(min(targets))
        # No conflict requires slowing down: proceed as default, without letting skipped conflicts cancel the speedup
        else:
            super().react([], ignore=ignored)
        self.set_next_positions()

    # Distance (cells) from the vehicle to the center of a cell
    def distance_to(self, cell):
        return euclidean(self.position, (cell[0] + .5, cell[1] + .5))

    # Target speed for a conflict with a walker, or None if it can be ignored
    def react_walker(self, walker, location):
        pos = walker.grid_position()
        # Pedestrians have priority on zebra crossings, and any pedestrian on the path is avoided
        on_path = pos in self.path_ahead()
        if self.city.city_grid[location][0] == 'z' or on_path:
            return self.safe_speed(self.distance_to(pos if on_path else location))
        return None

    # Target speed for a conflict with another driver, or None if self has priority
    def react_driver(self, other, location):
        pos = other.grid_position()
        # A vehicle on the path is followed: stop in time if it brakes (it moves away at its speed
        # when it drives in the same direction)
        if pos in self.path_ahead():
            lead = other.speed if other.active and other.direction == self.direction else 0
            return self.safe_speed(self.distance_to(pos), lead)
        # Otherwise yield at the conflict point unless self has priority
        if self.has_priority(other, location):
            return None
        return self.safe_speed(self.distance_to(location))

    # Priority between two drivers in conflict. At intersections the right-of-way rules apply, and
    # elsewhere a driver changing lanes yields. When both drivers are stopped and one of them has
    # waited DEADLOCK_WAIT steps, the one that has waited longer goes first (ties by id): this
    # order is total, so it resolves mutual yielding and cycles of drivers yielding to each other.
    # The priority so granted holds while its holder moves, until the conflict ends
    def has_priority(self, other, location):
        for a, b in ((self, other), (other, self)):
            if b.id in a.priority_over:
                if a.speed > 0:
                    return a is self
                a.priority_over.discard(b.id)       # the holder stopped again: order anew
        if self.speed == 0 and other.speed == 0 and max(self.stopped, other.stopped) >= self.DEADLOCK_WAIT:
            first = (self.stopped, -self.id) > (other.stopped, -other.id)
            (self.priority_over if first else other.priority_over).add(other.id if first else self.id)
            return first
        # A driver entering the road from a driveway or parking space yields to traffic on it
        if self.off_road() != other.off_road():
            return other.off_road()
        if self.city.is_intersection(location):
            return self.right_of_way(other)
        return other.has_lane_change()

    def has_lane_change(self):
        nxt, n = self.next_positions, len(self.next_positions)
        next_directions = set([(nxt[i][0] - nxt[i+1][0], nxt[i][1] - nxt[i+1][1]) for i in range(n-1)])
        if (0, 0) in next_directions:
            next_directions.remove((0, 0))
        return len(next_directions) > 1

    # Any agent on the route ahead is a conflict (car following, stopped or crashed vehicles,
    # pedestrians in the lane). Another driver whose route ahead crosses or joins this one is a
    # conflict when both would reach the shared cell within CROSSING_WINDOW of each other, unless
    # it drives behind self. Otherwise, a conflict is a predicted encounter (AstarAgent)
    def determine_conflict(self, other):
        pos = other.grid_position()
        if pos in self.path_ahead() and (other.active or other.agent_type == 'driver'):
            return pos, other
        if other.agent_type == 'driver' and other.active and self.grid_position() not in other.path_ahead():
            theirs = set(other.path_ahead())
            cell = next((c for c in self.path_ahead() if c in theirs), None)
            if cell is not None and abs(self.arrival_time(cell) - other.arrival_time(cell)) < self.CROSSING_WINDOW:
                return cell, other
        return super().determine_conflict(other)

    # Time (s) to reach the center of a cell at the current speed, or accelerating from rest
    def arrival_time(self, cell):
        d = self.distance_to(cell) * self.CELL_M
        return d / max(self.speed / 3.6, math.sqrt(self.ACCEL * d / 2), 1e-6)

    # Checks if a right, left, or no turn is expected from the agent.
    def turn_type(self):
        turn_side, turn_cell = 'straight', None
        curr_dir = self.direction
        nextpos = self.next_positions
        # Get all next directions to observe a change
        for i in range(len(self.next_positions) - 1):
            next_dir = (nextpos[i+1][0] - nextpos[i][0],  nextpos[i+1][1] - nextpos[i][1])
            # If direction does not change, cotinue
            if next_dir == (0, 0) or next_dir == curr_dir:
                continue
            # else, check possible outcomes
            if curr_dir == (0, 1):      # East
                turn_side, turn_cell = 'right' if next_dir == (1, 0) else 'left', nextpos[i]
                break
            elif curr_dir  == (0, -1):    # West
                turn_side, turn_cell = 'right' if next_dir == (-1, 0) else 'left', nextpos[i]
                break
            elif curr_dir == (1, 0):     # South
                turn_side, turn_cell = 'right' if next_dir == (0, -1) else 'left', nextpos[i]
                break
            elif curr_dir == (-1, 0):    # North
                turn_side, turn_cell = 'right' if next_dir == (0, 1) else 'left', nextpos[i]
                break
        return turn_side, turn_cell

    # Determines if self has right of way over agent other at an intersection.
    def right_of_way(self, other):
        d1, d2 = self.direction, other.direction
        tt1, tc1 = self.turn_type()     # Turn type and turn cell
        tt2, tc2 = other.turn_type()

        # Rule 1: Right agent has priority
        if d1 == (-d2[1], d2[0]):
            return True
#        if d2 == (-d1[1], d1[0]):
#            return False

        # Rule 2: Right turn has priority (uncommon)
        if tt1 == 'right' and tt2 != 'right':
            return True
#        if t2 == 'right' and t1 != 'right':
#            return False

        # Rule 3: Straight has priority over left turn
        if tt1 == 'straight' and tt2 == 'left':
            return True
#        if t2 == 'straight' and t1 == 'left':
#            return False

        # Rule 4: if both agents turn left, nearest has right of way:
        if tt1 == tt2 == 'left' and euclidean(self.grid_position(), tc1) < euclidean(other.grid_position(), tc2):
            return True
        # If both go straight, right hand agent has priority
        #if t1 == 'straight' and t2 == 'straight' and d1 == (-d2[1], d2[0]):
        #    return True

        # Default: no clear priority
        return False

#######################################
# CLASS FOR PEDESTRIAN
#######################################
class AstarWalker(AstarAgent):
    DESIRED_SPEED = 5       # walking speed in speed units (km/h): about 1.4 m/s
    '''
    Initializing agent elements:
    '''
    def setup(self, start, goal, direction, speed, visibility = 3, awareness = 1, beta = 1, rho=1):
        self.width = .1
        self.length = .1
        super().setup(start, goal, direction, speed, visibility, awareness, beta, rho)
        self.cost_map = self.model.walk_cost
        self.risk_map = self.model.walk_risk
        self.agent_type = 'walker'
        # As drivers, a walker stopped for max_wait steps replans around stopped vehicles
        self.max_wait = 10
        self.wait = 0
        self.trip = {'replans': 0}

    def step_cost(self, position):
        if position in self.city.obstacles or position in self.obstacles:
            return self.cost_map['o']
        return super().step_cost(position)

    def determine_conflict(self, other):
        # Walkers watch moving vehicles only: a stationary vehicle cannot hit them, and they
        # walk past it (it is the vehicle that waits until they have left its path)
        if other.agent_type != 'driver' or other.speed == 0:
            return False
        for i in range(min(len(self.next_positions), len(other.next_positions))):
            if self.next_positions[i] == other.next_positions[i]:
                return self.next_positions[i], other
        return False

    #Simulates the risk of a step according to allowed movements
    def step_risk(self, action, position, heading=None):
        # Walker risk depends only on the type of the cell entered
        dir = self.ACTION_MAP[action]
        position = (position[0] + dir[0], position[1] + dir[1])
        cell_code = self.city.city_grid[position]
        return self.risk_map[cell_code[0]]

    def sense(self):
        if not self.active:
            self.conflicts = []
            return self.conflicts
        conflicts = super().sense()
        self.conflicts = list(filter(lambda x: x[1].agent_type == 'driver', conflicts))
        return self.conflicts

    def react(self, conflicts = None):
        conflicts = self.conflicts if conflicts is None else conflicts
        if not self.active:
            return
        if self.speed == 0:
            if self.wait >= self.max_wait:
                # Blocked too long (e.g. by a vehicle queued on a crossing): replan
                # treating stopped or immobilized vehicles as obstacles
                self.wait = 0
                self.trip['replans'] += 1
                self.reroute(is_obstacle=lambda x: x.agent_type == 'driver' and (not x.active or x.speed == 0))
                conflicts = self.sense()
            else:
                self.wait += 1
        else:
            self.wait = 0
        if conflicts:
            nearest_collision, nearest_agent = conflicts[0]
            t1 = self.city.city_grid[self.grid_position()][0]
            t2 = self.city.city_grid[nearest_collision][0]
            # Pedestrians have preference on zebra cross, continue walking unless an obstacle is recognized.
            if (t1 == 'z' or t2 == 'z') and all(x[1].grid_position() not in self.next_positions for x in conflicts):
                # Resume walking (a previous slowdown may have stopped the walker), ignoring yielding drivers
                super().react([], ignore=[x[1] for x in conflicts])
            else:
                super().react(conflicts)
        else:
            super().react()

