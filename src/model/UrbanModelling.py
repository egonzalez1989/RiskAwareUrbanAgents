from model.AgentImpl import *
from utils.AgentUtils import *
from collections import defaultdict
import agentpy as ap, numpy as np

#######################################
# CLASS FOR THE SYSTEM (Ag, Env)
#######################################
class CityModel(ap.Model):

    def setup(self):
        self.p.city_grid = np.array(self.p.city_grid)
        self.city = City(self, shape=self.p.city_grid.shape)

        # Default removal times after incidents
        self.removal_times = {AstarDriver: 20, AstarWalker: 5}
        self.scheduled_removals = []

        # Initialise environment labels and costs
        self.setup_grid_labels()
        self.metrics = {}
        self.spawned_agents = {}
        self.arrived = []       # agents that entered their goal cell during the last step
        self.illegal_moves = [] # (cell, maneuver) of forbidden turns and wrong-way moves in the last step
        self.repopulate = False if 'repopulate' not in self.p else self.p.repopulate

        # Initialise agents
        self.setup_initial_agents()

        # Heatmap statistics
        h, w = self.p.city_grid.shape
        self.walker_speed_hm = np.zeros((h, w))
        self.driver_speed_hm = np.zeros((h, w))
        self.walker_count_hm = [[set() for j in range(w)] for i in range(h)]
        self.driver_count_hm = [[set() for j in range(w)] for i in range(h)]

    # ── Parameter-backed generator methods ───────────────────────────────────
    # Each method reads the corresponding callable from self.p, falling back
    # to a sensible default so the model works even when the parameter is
    # absent.  The callables are supplied by Experiments.py or
    # agent_generators.py and map to the paper's notation as follows:
    #   random_weight_walker  -> beta_w  (heuristic greediness, walker)
    #   random_weight_driver  -> beta_d  (heuristic greediness, driver)
    #   random_rho_walker     -> rho_w   (risk sensitivity, walker)
    #   random_rho_driver     -> rho_d   (risk sensitivity, driver)
    #   f_walker_speed        -> walker free-flow speed (sim units)
    #   f_driver_speed        -> driver free-flow speed (sim units)

    def walker_weight(self):
        """beta_w: heuristic greediness for walkers."""
        fn = getattr(self.p, 'random_weight_walker', None)
        return fn() if callable(fn) else 1

    def driver_weight(self):
        """beta_d: heuristic greediness for drivers."""
        fn = getattr(self.p, 'random_weight_driver', None)
        return fn() if callable(fn) else 1

    def walker_rho(self):
        """rho_w: risk sensitivity for walkers."""
        fn = getattr(self.p, 'random_rho_walker', None)
        return fn() if callable(fn) else 1

    def driver_rho(self):
        """rho_d: risk sensitivity for drivers."""
        fn = getattr(self.p, 'random_rho_driver', None)
        return fn() if callable(fn) else 1

    def walker_maxspeed(self):
        """Free-flow speed for walker agents (sim units)."""
        fn = getattr(self.p, 'f_walker_speed', None)
        return fn() if callable(fn) else AstarWalker.DESIRED_SPEED

    def driver_maxspeed(self):
        """Free-flow speed for driver agents (sim units)."""
        fn = getattr(self.p, 'f_driver_speed', None)
        return fn() if callable(fn) else AstarDriver.DESIRED_SPEED

    # ── Grid label and cost setup ─────────────────────────────────────────────
    def setup_grid_labels(self):
        walk_terrain = ['s', 'z', 'p', 'r', 't', 'l', 'f', 'b', 'o']
        self.walk_cost = dict(zip(walk_terrain,
                                  [1, 1, 1, 1, 1, 1, 2, 1000, 1000]))
        self.walk_risk = dict(zip(walk_terrain, [0, 1, 2, 5, 10, np.inf, np.inf, np.inf]))
        if 'walker_risk_map' in self.p:  # calibration override of road and intersection risk
            self.walk_risk.update({k: self.p.walker_risk_map[k] for k in ('r', 't')})
        drive_terrain  = ['s', 'z', 'p', 'r', 't', 'l', 'b', 'o', 'ph']
        self.drive_cost = dict(zip(drive_terrain,
                                   [1000, 2, 3, 1, 1, 1, 1000, 1001, 20]))
        if 'ph_cost' in self.p:          # per-experiment pothole cost override
            self.drive_cost['ph'] = self.p.ph_cost
        drive_move = ['fw', 'rt', 'lt', 'ch', 'ft', 'bw']
        self.drive_risk = dict(zip(drive_move, [0, 1, 3, 3, 5, 15]))
        if 'risk_override' in self.p:    # sensitivity analysis of the risk values (E7)
            self.walk_risk.update(self.p.risk_override.get('walk', {}))
            self.drive_risk.update(self.p.risk_override.get('drive', {}))

    # ── Agent initialisation ──────────────────────────────────────────────────
    def setup_initial_agents(self):
        # ── Walkers ──
        if 'walkers' not in self.p:
            self.nwalkers = self.p.initial_walker_count \
                if 'initial_walker_count' in self.p else 0
            self.spawn_walkers(self.nwalkers)
        else:
            agents = []
            self.nwalkers = len(self.p.walkers)
            default_vals = {
                'speed':      AstarWalker.DESIRED_SPEED,
                'visibility': 3,
                'awareness':  1,
                'beta':     1,      # beta
                'rho':1,      # rho
                'direction':  self.random.choice(
                    [(0, 1), (0, -1), (1, 0), (-1, 0)]),
            }
            for walker in self.p.walkers:
                for key in default_vals:
                    walker[key] = default_vals[key] \
                        if key not in walker else walker[key]
                start = walker['start']
                walker['start'] = tuple(
                    start[i] + self.random.uniform(.1, .9) for i in (0, 1))
                agent = AstarWalker(self, **walker)
                agents.append(agent)
                self.city.add_agents([agent], positions=[start])
                agent.find_route()
                self.spawned_agents[agent.id] = {
                    'start': agent.start, 'goal': agent.goal,
                    'type':  agent.type,  'spawn_time': self.t,
                    'max_speed': agent.max_speed,
                    'beta': walker['beta'],
                    'rho': walker['rho'],
                }
            [agent.initialize_agent() for agent in agents]

        # ── Drivers ──
        if 'drivers' not in self.p:
            self.ndrivers = self.p.initial_driver_count \
                if 'initial_driver_count' in self.p else 0
            self.spawn_drivers(self.ndrivers,
                               sources=self.city.road_cells)
        else:
            agents = []
            self.ndrivers = len(self.p.drivers)
            default_vals = {
                'speed':      60,
                'visibility': 5,
                'awareness':  1,
                'beta':     1,      # beta
                'rho':1,      # rho
            }
            for driver in self.p.drivers:
                for key in default_vals:
                    driver[key] = default_vals[key] \
                        if key not in driver else driver[key]
                start = driver['start']
                if 'direction' not in driver:
                    ways = self.city.city_grid[start][1:].strip()
                    d    = MobileAgent.ACTION_MAP
                    driver['direction'] = d[self.random.choice('NSEW')] \
                        if ways == '' else d[self.random.choice(ways)]
                position = tuple(
                    start[i] + self.random.uniform(.4, .6) for i in (0, 1))
                agent = AstarDriver(self, **driver)
                agents.append(agent)
                self.city.add_agents([agent], positions=[start])
                agent.find_route()
                self.spawned_agents[agent.id] = {
                    'start': agent.start, 'goal': agent.goal,
                    'type':  agent.type,  'spawn_time': self.t,
                    'max_speed': agent.max_speed,
                    'beta': driver['beta'],
                    'rho': driver['rho'],
                }
            [agent.initialize_agent() for agent in agents]

    # ── Spawn helpers ─────────────────────────────────────────────────────────
    def spawn_walkers(self, n):
        """Spawn n walkers from random sidewalk sources.
        beta  (weight)      drawn from random_weight_walker()
        rho   (rho) drawn from random_rho_walker()
        speed               drawn from f_walker_speed()
        """
        agents, pos = [], []
        for _ in range(n):
            start     = list(self.random.choice(self.city.walker_sources))
            position  = (start[0] + self.random.uniform(.1, .9),
                         start[1] + self.random.uniform(.1, .9))
            goal      = self.random.choice(self.city.walker_goals)
            direction = self.random.choice(
                [(0, 1), (0, -1), (1, 0), (-1, 0)])
            beta  = self.walker_weight()
            rho   = self.walker_rho()
            speed = self.walker_maxspeed()
            agent = AstarWalker(self, position, goal,
                                direction=direction, speed=speed,
                                beta=beta, rho=rho)
            pos.append(start)
            agents.append(agent)
        self.city.add_agents(agents, positions=pos)
        [agent.find_route() for agent in agents]
        [agent.initialize_agent() for agent in agents]
        for agent in agents:
            self.spawned_agents[agent.id] = {
                'start': agent.start, 'goal': agent.goal,
                'type':  agent.type,  'spawn_time': self.t,
                'max_speed': agent.max_speed,
                'beta': agent.beta,
                'rho': agent.rho,
                'trip': agent.trip,     # walker replans, updated in place
            }

    def spawn_drivers(self, n, sources=None):
        """Spawn n drivers from road-edge sources.
        beta  (weight)      drawn from random_weight_driver()
        rho   (rho) drawn from random_rho_driver()
        speed               drawn from f_driver_speed()
        """
        agents, pos = [], []
        rnd, city   = self.random, self.city
        sources = city.driver_sources[:] if sources is None else sources
        sources = list(set(sources).difference(city.positions.values()))
        for _ in range(n):
            start    = rnd.choice(sources)
            goal     = rnd.choice(city.driver_goals)
            while manhattan(goal, start) < 10:
                goal = rnd.choice(city.driver_goals)
            pos.append(start)
            position = (start[0] + .5, start[1] + .5)
            beta     = self.driver_weight()
            rho      = self.driver_rho()
            speed    = self.driver_maxspeed()
            ways = city.city_grid[start][1:].strip()
            d    = MobileAgent.ACTION_MAP
            direction = d[rnd.choice('NSEW')] if ways == '' \
                else d[rnd.choice(ways)]
            agent = AstarDriver(self, position, goal,
                                direction=direction, speed=speed,
                                beta=beta, rho=rho)
            agents.append(agent)
            sources.remove(start)
        city.add_agents(agents, positions=pos)
        [agent.find_route() for agent in agents]
        [agent.initialize_agent() for agent in agents]
        for agent in agents:
            self.spawned_agents[agent.id] = {
                'start': agent.start, 'goal': agent.goal,
                'type':  agent.type,  'spawn_time': self.t,
                'max_speed': agent.max_speed,
                'beta': agent.beta,
                'rho': agent.rho,
                'trip': agent.trip,     # updated in place during the run (AstarDriver.trip)
            }

    # ── Simulation loop ───────────────────────────────────────────────────────
    def update(self):
        self.metrics[self.t] = {}

        if self.scheduled_removals:
            self.city.remove_agents(self.scheduled_removals)
            self.scheduled_removals = []

        if self.repopulate:
            active_w = [x for x in self.city.agents
                        if x.agent_type == 'walker' and x.active]
            self.spawn_walkers(self.nwalkers - len(active_w))
            active_d = [x for x in self.city.agents
                        if x.agent_type == 'driver' and x.active]
            self.spawn_drivers(self.ndrivers - len(active_d))

        self.goal_agents()
        self.city.agents.sense()
        self.city.agents.react()
        self.city.agents.choose_action()
        self.report_metrics()

    def goal_agents(self):
        # Agents that reached their goal cell while moving (MobileAgent.move)
        arrivals = {agent.id: self.t for agent in self.arrived}
        self.arrived = []
        for agent in self.city.agents:
            if agent.active and euclidean(agent.position, agent.goal) < .5:
                arrivals[agent.id] = self.t
                if agent.goal not in self.city.parking_spaces:
                    agent.deactivate(removal_time=0)
                else:
                    agent.deactivate(removal_time=np.inf)
        self.metrics[self.t]['arrivals'] = arrivals
        # A vehicle stopped for longer than MAX_WAIT_S is removed and its trip recorded as aborted,
        # as SUMO does with vehicles that wait longer than its time-to-teleport: this resolves the
        # residual gridlocks that no priority rule can, e.g. a vehicle leaving a driveway cell that
        # another vehicle is about to enter
        limit = round(AstarDriver.MAX_WAIT_S / AstarDriver.STEP_S)
        aborted = [a for a in self.city.agents if a.active and a.agent_type == 'driver' and a.stopped >= limit]
        for agent in aborted:
            agent.trip['aborted'] = True
            agent.deactivate(removal_time=0)
        self.metrics[self.t]['aborted'] = [a.id for a in aborted]

    def step(self):
        self.city.agents.execute()
        # Stop when all agents finish (no step limit set)
        if len(self.city.agents) == 0 and 'steps' not in self.p:
            self.end()
            self.stop()

    def end(self):
        self.report('metrics',        self.metrics)
        self.report('spawned_agents', self.spawned_agents)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def report_metrics(self):
        collisions = self.get_collisions()
        self.manage_collisions(collisions)
        self.manage_jaywalking()
        self.manage_active_agents()
        self.compute_average_speed()
        self.update_heatmaps()

    def update_heatmaps(self):
        for w in [x for x in self.city.agents
                  if x.agent_type == 'walker' and x.active]:
            pos = self.city.positions[w]
            self.walker_count_hm[pos[0]][pos[1]].add(w.id)
            n = len(self.walker_count_hm[pos[0]][pos[1]])
            self.walker_speed_hm[pos] = (
                (n - 1) * self.walker_speed_hm[pos] + w.speed) / n
        for d in [x for x in self.city.agents
                  if x.agent_type == 'driver' and x.active]:
            pos = self.city.positions[d]
            self.driver_count_hm[pos[0]][pos[1]].add(d.id)
            n = len(self.driver_count_hm[pos[0]][pos[1]])
            self.driver_speed_hm[pos] = (
                (n - 1) * self.driver_speed_hm[pos] + d.speed) / n

    def manage_collisions(self, collisions):
        self.metrics[self.t]['collisions'] = []
        self.metrics[self.t]['runovers']   = []
        for key, agents in collisions.items():
            if any(x.active for x in agents):
                if all(x.agent_type == 'driver' for x in agents):
                    self.metrics[self.t]['collisions'].append(key)
                else:
                    self.metrics[self.t]['runovers'].append(key)
            for agent in agents:
                if agent.active:
                    # Crash type and impact speed (length of the last move, in speed units) recorded in the trip
                    agent.trip['crash'] = 'collision' if all(x.agent_type == 'driver' for x in agents) else 'runover'
                    moved = agent.track_t == self.t
                    agent.trip['crash_speed'] = euclidean(agent.track[0], agent.track[-1]) * MobileAgent.SPEED_LIMIT if moved else 0
                    agent.deactivate(self.removal_times[type(agent)])
                    agent.speed = 0     # a collided agent is immobilized until removed

    # Contacts between a driver and another agent. Agents are compared along the straight moves they
    # made during the last step (both at constant speed), not only at their final
    # positions, so two agents that cross each other between two steps also collide.
    # A contact counts as a crash only when a vehicle involved moved during the step: a walker
    # passing a stationary vehicle, or two queued vehicles, do not crash
    COLLISION_SAMPLES = 20      # time samples per step along the paths

    def get_collisions(self):
        collisions = defaultdict(list)
        agents = list(self.city.agents)
        tracks = {a: self.timed_track(a) for a in agents}
        reach = 3       # agents move at most one cell per step, so they approach by at most two cells
        contacts = {}       # agent -> earliest contact sample
        for agent in agents:
            for neigh in self.city.neighbors(agent, distance=reach):
                if not (self.moved_driver(agent) or self.moved_driver(neigh)):
                    continue
                gap = (agent.width + neigh.width) / 2
                ta, tb = tracks[agent], tracks[neigh]
                k = next((k for k in range(len(ta)) if euclidean(ta[k], tb[k]) < gap), None)
                if k is None:
                    continue
                for x in (agent, neigh):
                    contacts[x] = min(k, contacts.get(x, k))
                pos = agent.grid_position(ta[k])
                collisions[(int(pos[0]), int(pos[1]))].extend([agent, neigh])
        # A moving agent is stopped where its first contact happened
        for x, k in contacts.items():
            if x.active and k < len(tracks[x]) - 1:
                x.position = tracks[x][k]
                self.city.move_to(x, x.grid_position())
        return collisions

    def moved_driver(self, agent):
        return agent.agent_type == 'driver' and agent.track_t == self.t and agent.track[0] != agent.track[-1]

    # Positions of an agent at COLLISION_SAMPLES + 1 equally spaced times of the last step
    def timed_track(self, agent):
        n = self.COLLISION_SAMPLES
        pts = np.array(agent.track if agent.track_t == self.t else [agent.position], dtype=float)
        if len(pts) == 1:
            return [tuple(pts[0])] * (n + 1)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        if cum[-1] == 0:
            return [tuple(pts[0])] * (n + 1)
        d = np.linspace(0, cum[-1], n + 1)
        return list(zip(np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])))

    def manage_jaywalking(self):
        jw = defaultdict(list)
        road_only = self.city.road_cells - self.city.zebras
        for agent in self.city.agents:
            pos = agent.grid_position()
            if (agent.agent_type == 'walker'
                    and pos in road_only
                    and pos not in jw[agent.id]):
                jw[agent.id].append(tuple(pos))
        self.metrics[self.t]['jaywalking'] = jw
        # Forbidden turns ('ft') and wrong-way moves ('bw') executed by drivers in the last step
        self.metrics[self.t]['illegal'] = [[list(c), m] for c, m in self.illegal_moves]
        self.illegal_moves = []

    def compute_average_speed(self):
        # Only active agents count: stopped ones contribute speed 0, while
        # immobilized (collided) or arrived agents are left out
        walkers = [x for x in self.city.agents if x.agent_type == 'walker' and x.active]
        drivers = [x for x in self.city.agents if x.agent_type == 'driver' and x.active]
        self.metrics[self.t]['avg_speed_walkers'] = (
            np.mean([x.speed for x in walkers]) if walkers else 0)
        self.metrics[self.t]['avg_speed_drivers'] = (
            np.mean([x.speed for x in drivers]) if drivers else 0)
        # Active agents standing still at this step (stationary time)
        self.metrics[self.t]['stopped_walkers'] = int(sum(x.speed == 0 for x in walkers))
        self.metrics[self.t]['stopped_drivers'] = int(sum(x.speed == 0 for x in drivers))

    def manage_active_agents(self):
        walkers  = [x for x in self.city.agents if x.agent_type == 'walker']
        drivers  = [x for x in self.city.agents if x.agent_type == 'driver']
        act_w    = [x for x in walkers if x.active]
        act_d    = [x for x in drivers if x.active]
        parked_d = [x for x in drivers
                    if not x.active
                    and x.grid_position() in self.city.parking_set]
        t = self.t
        self.metrics[t]['env_walkers']    = len(walkers)
        self.metrics[t]['active_walkers'] = len(act_w)
        self.metrics[t]['env_drivers']    = len(drivers)
        self.metrics[t]['active_drivers'] = len(act_d)
        self.metrics[t]['parked_drivers'] = len(parked_d)

    def schedule_removal(self, agent):
        self.scheduled_removals.append(agent)


####################################################################
# ANIMATION / PLOTTING HELPERS (kept importable from here; single source in AnimationUtils)
####################################################################
from utils.AnimationUtils import animation_plot, plot_collisions, plot_agents, plot_city  # noqa: E402,F401


####################################################################
# POISSON SPAWNING MODEL
####################################################################
class PoisonCityModel(CityModel):
    """
    Variant that spawns agents each step according to a Poisson process
    targeting the initial population counts.
    """
    def update(self):
        self.metrics[self.t] = {}

        if self.scheduled_removals:
            self.city.remove_agents(self.scheduled_removals)
            self.scheduled_removals = []

        cur_w = len([x for x in self.city.agents if x.agent_type == 'walker'])
        cur_d = len([x for x in self.city.agents if x.agent_type == 'driver'])

        target_w = getattr(self.p, 'initial_walker_count', 0)
        target_d = getattr(self.p, 'initial_driver_count', 0)

        n_w = max(0, np.random.poisson(max(0, target_w - cur_w)))
        n_d = max(0, np.random.poisson(max(0, target_d - cur_d)))
        self.spawn_walkers(n_w)
        self.spawn_drivers(n_d)

        self.goal_agents()
        self.city.agents.sense()
        self.city.agents.react()
        self.city.agents.choose_action()
        self.report_metrics()


####################################################################
# ENVIRONMENT
####################################################################
class City(ap.Grid):
    def setup(self):
        self.city_grid  = self.p.city_grid
        grid, (h, w)    = self.p.city_grid, self.p.city_grid.shape
        self.parking_spaces  = self.parking_cells()
        self.parking_set     = set(self.parking_spaces)     # for membership tests
        self.obstacles  = set(self.p.obstacles)  if 'obstacles'  in self.p else set()
        self.potholes   = set(self.p.potholes)   if 'potholes'   in self.p else set()
        # Cells where a left turn is legal: designated 'l' cells on multi-lane streets,
        # otherwise the intersection cells ('t'), which then serve both turns
        self.left_turn_cell = 'l' if any(c[0] == 'l' for c in grid.flat) else 't'
        if 'driver_endpoints' in self.p:
            self.driver_sources, self.driver_goals = self.p.driver_endpoints
            self.walker_sources, self.walker_goals = self.p.walker_endpoints
        else:
            self.driver_sources, self.driver_goals = self.driver_endpoints()
            self.walker_sources, self.walker_goals = self.walker_endpoints()
        self.road_cells = set(
            tuple(x) for x in np.argwhere(
                np.array([grid[i,j][0] in ('r','z','p')
                          for i in range(h) for j in range(w)]
                         ).reshape((h, w))))
        self.zebras = set(
            tuple(x) for x in np.argwhere(
                np.array([grid[i,j][0] == 'z'
                          for i in range(h) for j in range(w)]
                         ).reshape((h, w))))

    # agentpy keeps the agents of each cell in a set ordered by memory address, so the
    # neighbors are sorted by id: otherwise the same seed gives different runs
    def neighbors(self, agent, distance=1):
        return sorted(super().neighbors(agent, distance), key=lambda x: x.id)

    def is_intersection(self, cell):
        return ' ' not in self.city_grid[cell]

    def get_agents_at(self, position):
        return [x for x in self.agents if x.grid_position() == position]

    # Obstacle cells are impassable, so no agent starts or ends its trip on one
    def free_cells(self, cells):
        return [c for c in cells if tuple(c) not in self.obstacles]

    def walker_endpoints(self):
        return tuple(map(self.free_cells, get_walker_endpoints(self.city_grid)))

    def parking_cells(self):
        return get_parking_spots(self.city_grid)

    def driver_endpoints(self):
        return tuple(map(self.free_cells, get_driver_endpoints(self.city_grid)))