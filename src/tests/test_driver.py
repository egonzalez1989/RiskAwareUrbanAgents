"""
Tests of the driver kinematics and conflict rules (model.AgentImpl).

Run from src/:  python -m unittest discover tests
"""
import itertools, unittest
from model.AgentImpl import AstarDriver, AstarWalker


def driver(id_, speed=0., stopped=0):
    """A driver with the attributes the tested methods use, without a model."""
    d = AstarDriver.__new__(AstarDriver)
    d.__dict__.update(id=id_, speed=speed, stopped=stopped, priority_over=set(), max_speed=50.,
                      reaction_steps=AstarDriver.REACTION_STEPS, next_positions=[])
    d.brake_step, d.accel_step = (3.6 * d.STEP_S * a for a in (d.MAX_DECEL, d.ACCEL))
    d.set_next_positions = lambda: None
    d.off_road = lambda: False
    return d


class TestKinematics(unittest.TestCase):
    def test_safe_speed_stops_within_the_available_distance(self):
        d, tau = driver(1), AstarDriver.REACTION_STEPS * AstarDriver.STEP_S
        for cells in (3, 5, 8, 12):
            v = d.safe_speed(cells) / 3.6       # m/s
            room = (cells - d.GAP) * d.CELL_M
            self.assertAlmostEqual(v * tau + v ** 2 / (2 * d.SAFE_DECEL), room, places=6)

    def test_safe_speed_is_zero_inside_the_gap_and_grows_with_distance(self):
        d = driver(1)
        self.assertEqual(d.safe_speed(d.GAP), 0)
        speeds = [d.safe_speed(x) for x in range(2, 12)]
        self.assertEqual(speeds, sorted(speeds))

    def test_moving_vehicle_ahead_allows_a_higher_speed(self):
        d = driver(1)
        self.assertGreater(d.safe_speed(4, lead_speed=40), d.safe_speed(4))

    def test_speed_changes_are_bounded(self):
        d = driver(1, speed=50)
        d.adjust_speed(0)
        self.assertAlmostEqual(d.speed, 50 - 3.6 * d.STEP_S * d.MAX_DECEL)
        d = driver(1, speed=0)
        d.adjust_speed(50)
        self.assertAlmostEqual(d.speed, 3.6 * d.STEP_S * d.ACCEL)
        d.speed = 49
        d.adjust_speed(80)
        self.assertEqual(d.speed, d.max_speed)


class TestStandstillResolution(unittest.TestCase):
    def test_waiting_time_gives_a_total_order(self):
        # Three stopped drivers yielding to each other in a cycle: exactly one of each pair goes
        # first, and the order has no cycle, so the driver that waited longest can always move
        drivers = [driver(i, stopped=w) for i, w in ((1, 30), (2, 12), (3, 30))]
        wins = {d.id: 0 for d in drivers}
        for a, b in itertools.combinations(drivers, 2):
            ab, ba = a.has_priority(b, None), b.has_priority(a, None)
            self.assertNotEqual(ab, ba)
            wins[a.id if ab else b.id] += 1
        self.assertEqual(sorted(wins.values()), [0, 1, 2])
        self.assertEqual(max(wins, key=wins.get), 1)        # longest wait, ties by lower id

    def test_granted_priority_holds_while_the_holder_moves(self):
        a, b = driver(1, stopped=20), driver(2, stopped=15)
        self.assertTrue(a.has_priority(b, None))
        a.speed, a.stopped = 3, 0           # a moves off; b has now waited longer
        self.assertFalse(b.has_priority(a, None))
        a.speed = 0                         # a stopped again: the order is computed anew
        b.stopped = 16
        self.assertTrue(b.has_priority(a, None))

    def test_waiting_time_overrides_yielding_from_a_driveway(self):
        a, b = driver(1, stopped=40), driver(2, stopped=12)
        a.off_road = lambda: True           # a waits to leave a driveway; b is stopped on the road
        self.assertTrue(a.has_priority(b, None))
        self.assertFalse(b.has_priority(a, None))


class TestWalkers(unittest.TestCase):
    def test_walkers_ignore_stationary_vehicles(self):
        w = AstarWalker.__new__(AstarWalker)
        w.next_positions = [(3, 3)] * 5
        car = driver(9, speed=0)
        car.agent_type, car.next_positions = 'driver', [(3, 3)] * 5
        self.assertFalse(w.determine_conflict(car))
        car.speed = 10
        self.assertEqual(w.determine_conflict(car), ((3, 3), car))


if __name__ == '__main__':
    unittest.main()
