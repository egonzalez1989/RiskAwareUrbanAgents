"""
Tests of the risk-aware weighted A* (utils.AgentUtils.weighted_astar).

Run from src/:  python -m unittest discover tests
"""
import heapq, itertools, random, unittest
import numpy as np
from utils.AgentUtils import weighted_astar

MOVES = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}


def path_cost(path, cost, risk, rho, start_dir=None):
    """Accumulated cost g of a path: sum of c(v) + rho * r(a, v | heading)."""
    g, heading = 0, start_dir
    for u, v in zip(path, path[1:]):
        a = next(k for k, d in MOVES.items() if d == (v[0] - u[0], v[1] - u[1]))
        g += cost(v) + rho * (risk(a, u, heading) if start_dir is not None else risk(a, u))
        heading = MOVES[a]
    return g


def dijkstra(start, goal, shape, cost, risk, rho, start_dir):
    """Exhaustive least-cost search over (cell, heading) states, for reference."""
    best, queue = {(start, start_dir): 0}, [(0, start, start_dir)]
    while queue:
        g, u, d = heapq.heappop(queue)
        if u == goal:
            return g
        if g > best[(u, d)]:
            continue
        for a, (dx, dy) in MOVES.items():
            v = (u[0] + dx, u[1] + dy)
            if 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]:
                ng = g + cost(v) + rho * risk(a, u, d)
                if ng < best.get((v, (dx, dy)), np.inf):
                    best[(v, (dx, dy))] = ng
                    heapq.heappush(queue, (ng, v, (dx, dy)))
    return None


class WorkedExample(unittest.TestCase):
    """3 x 3 grid of unit-cost cells; entering the middle cell of the top row, (0, 1),
    has risk 5 (a walker entering a road cell). From (0, 0) to (0, 2):
      straight path (0,0) (0,1) (0,2):              g = 2 + 5 rho
      detour (0,0) (1,0) (1,1) (1,2) (0,2):         g = 4
    so the agent crosses the risky cell when rho < 0.4 and detours when rho > 0.4."""
    grid = np.full((3, 3), 'r  ')
    cost = staticmethod(lambda v: 1)

    @staticmethod
    def risk(a, u):
        d = MOVES[a]
        return 5 if (u[0] + d[0], u[1] + d[1]) == (0, 1) else 0

    def plan(self, rho):
        return weighted_astar((0, 0), (0, 2), self.grid, self.cost, self.risk, beta=1, rho=rho)

    def test_low_risk_sensitivity_takes_risky_shortcut(self):
        path = self.plan(rho=.25)
        self.assertEqual(path, [(0, 0), (0, 1), (0, 2)])
        self.assertAlmostEqual(path_cost(path, self.cost, self.risk, .25), 3.25)

    def test_high_risk_sensitivity_detours(self):
        path = self.plan(rho=1)
        self.assertEqual(path, [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)])
        self.assertEqual(path_cost(path, self.cost, self.risk, 1), 4)


class HeadingStates(unittest.TestCase):
    """Driver-like risks that depend on the heading: with beta = 1 the search over
    (cell, heading) states must return a least-cost path (Manhattan distance is a
    consistent heuristic because every step costs at least 1)."""

    def random_case(self, rng, shape=(7, 7)):
        cells = {(i, j): rng.choice([1, 1, 1, 2, 5]) for i in range(shape[0]) for j in range(shape[1])}
        turn = {(i, j): rng.choice([0, 1, 3, 15]) for i in range(shape[0]) for j in range(shape[1])}
        cost = lambda v: cells[v]
        # Turning (changing heading) at cell u costs turn[u]; going straight is free
        risk = lambda a, u, d: 0 if d == MOVES[a] else turn[u]
        return cost, risk, shape

    def test_optimal_cost_matches_exhaustive_search(self):
        rng = random.Random(1)
        for _ in range(200):
            cost, risk, shape = self.random_case(rng)
            s, t = (rng.randrange(7), rng.randrange(7)), (rng.randrange(7), rng.randrange(7))
            d0, rho = rng.choice(list(MOVES.values())), rng.choice([0, .25, 1, 5])
            grid = np.full(shape, 'r  ')
            path = weighted_astar(s, t, grid, cost, risk, beta=1, rho=rho, start_dir=d0)
            self.assertAlmostEqual(path_cost(path, cost, risk, rho, d0),
                                   dijkstra(s, t, shape, cost, risk, rho, d0))

    def test_greedy_search_within_beta_bound(self):
        rng = random.Random(2)
        for beta in (2, 3, 5):
            for _ in range(50):
                cost, risk, shape = self.random_case(rng)
                s, t = (0, 0), (rng.randrange(7), rng.randrange(7))
                d0 = (0, 1)
                path = weighted_astar(s, t, np.full(shape, 'r  '), cost, risk, beta=beta, rho=1, start_dir=d0)
                self.assertLessEqual(path_cost(path, cost, risk, 1, d0),
                                     beta * dijkstra(s, t, shape, cost, risk, 1, d0) + 1e-9)

    def test_cost_table_and_risk_cache_do_not_change_routes(self):
        rng = random.Random(3)
        cache = {}
        for _ in range(200):
            cost, risk, shape = self.random_case(rng)
            cache.clear()       # risks differ between cases
            table = {(i, j): cost((i, j)) for i, j in itertools.product(range(shape[0]), range(shape[1]))}
            s, t = (rng.randrange(7), rng.randrange(7)), (rng.randrange(7), rng.randrange(7))
            kw = dict(beta=rng.choice([1, 2, 5]), rho=rng.choice([0, .5, 2]), start_dir=(1, 0))
            grid = np.full(shape, 'r  ')
            self.assertEqual(weighted_astar(s, t, grid, cost, risk, **kw),
                             weighted_astar(s, t, grid, cost, risk, cost_table=table, risk_cache=cache, **kw))


if __name__ == '__main__':
    unittest.main()
