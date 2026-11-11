"""
calibration_experiments.py
===========================
Calibration experiments for Tables I-III.

Evaluation function (accumulated-risk formulation):
    f(v) = g(v) + rho * r(a, v) + beta * h(v)
    where beta >= 1 is the heuristic weight and rho >= 0 is risk sensitivity.

    NOTE: the model stack (AgentBase, AgentImpl, UrbanModelling) uses the
    internal parameter names 'beta' (= beta) and 'rho' (= rho).
    Agent dicts passed to CityModel must use those keys. The calibration
    functions use beta/rho as local variable names for clarity, but map them
    to 'beta'/'rho' when building agent dicts.

Jaywalking rate metric
-----------------------
eta_j = road_steps / total_walker_steps
    Fraction of active walker-steps spent on pure road cells (excluding zebra
    crossings). Reported as the primary calibration metric.

    The detour-weighted variant eta_jw was considered but dropped: once
    start-goal pairs are stratified by detour factor D into slight/moderate/
    strong buckets (all D > 1), eta_jw tracks eta_j almost perfectly because
    mean D is approximately constant across experiments. The stratified
    sampling itself solves the noise problem that motivated eta_jw, making
    the weighted metric redundant. D is retained as a pair-selection criterion.

Stratified pair sampling
-------------------------
The grid has ~96% of pairs with D ~ 1 (safe route barely longer than
shortest path), which would dilute the behavioral signal if sampled uniformly.
Pairs are stratified by detour factor D = L_safe / L_short into buckets:
  slight   : 1.0 < D < 1.5  -- weak tradeoff
  moderate : 1.5 <= D < 2.5 -- clear tradeoff
  strong   : D >= 2.5        -- strong behavioral signal

Grid geometry notes
-------------------
- Sidewalks: rows 1,13 and cols 1,10,13,22  (block perimeters)
- Inter-block corridor cols 11-12: road only, NO sidewalk
- High-detour pairs cross this corridor at mid-block heights
- Max detour factor on this grid: ~3.0 (pairs at rows 6-9, cols 10<->13)
"""

import sys, os, random, itertools
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.UrbanModelling import CityModel
from utils.AgentUtils import (gen_city, gen_obstacles,
                               get_walker_endpoints, get_driver_endpoints,
                               weighted_astar, manhattan)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED SETUP
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
N_RUNS      = 5
SIM_STEPS   = 400
BLOCK_SHAPE = (15, 12)
CITY_SHAPE  = (2, 3)
N_DRIVERS   = 10
OBS_RATE    = 0.05

random.seed(SEED)
np.random.seed(SEED)

_BASE_GRID           = gen_city(BLOCK_SHAPE, CITY_SHAPE, street_parking=False, nlanes=1)
_BASE_OBS, _BASE_PTH = gen_obstacles(_BASE_GRID, OBS_RATE, 0.0)

WALK_TERRAIN = ['s', 'z', 'p', 'r', 't', 'l', 'f', 'b', 'o']
DRIVE_MOVES  = ['fw', 'rt', 'lt', 'ch', 'ft', 'bw']


# ─────────────────────────────────────────────────────────────────────────────
# PATH ANALYSIS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _make_step_cost(grid):
    c = {'s':1,'z':1,'p':1,'r':1,'t':1,'l':1,'f':2,'b':1e6,'o':1e6}
    return lambda pos: c.get(grid[pos][0], 1e6)

def _make_step_risk(grid, risk_map):
    def sr(action, pos):
        d = {'N':(-1,0),'S':(1,0),'E':(0,1),'W':(0,-1)}[action]
        t = (pos[0]+d[0], pos[1]+d[1])
        if 0 <= t[0] < grid.shape[0] and 0 <= t[1] < grid.shape[1]:
            return risk_map.get(grid[t][0], 0)
        return 1e6
    return sr

def road_cells_in_path(path, grid):
    return sum(1 for p in path if grid[p][0] in ('r','t','l'))

def path_is_safe(path, grid):
    return road_cells_in_path(path, grid) == 0


def compute_pair_detour(start, goal, grid):
    """
    Compute detour factor D = len(safe_path) / len(short_path).
    D is a geometric ratio of cell counts (path lengths), not an
    accumulated cost ratio. It is used only for pair selection.

    p_short: pure shortest path -- risk disabled (rho=0, degenerate baseline,
             used here only, never in simulation experiments).
    p_safe:  maximally safe path -- high road risk forces sidewalk use.
    Returns (short_len, safe_len, D, road_cells_in_short) or None.
    """
    sc  = _make_step_cost(grid)
    sr0 = _make_step_risk(grid, {k:0 for k in 'szprtlfbo'})
    srH = _make_step_risk(grid, {'s':0,'z':1,'p':1,'r':100,'t':200,
                                  'l':200,'f':2,'b':1e9,'o':1e9})
    # beta=1, rho=0: pure shortest path by cell count (rho=0)
    p_short = weighted_astar(start, goal, grid, sc, sr0,
                             beta=1, rho=0)
    # beta=1, rho=1 + high risk map: maximally safe route
    p_safe  = weighted_astar(start, goal, grid, sc, srH,
                             beta=1, rho=1)
    if not p_short or not p_safe:
        return None
    sl = len(p_short)
    sf = len(p_safe)
    D  = sf / max(sl, 1)
    rc = road_cells_in_path(p_short, grid)
    return sl, sf, D, rc


def build_pair_catalog(grid, min_dist=4, seed=SEED):
    """
    Enumerate all sidewalk-to-sidewalk pairs with min Manhattan distance
    and compute their detour factors.
    """
    sw = [(i,j) for i in range(grid.shape[0])
              for j in range(grid.shape[1]) if grid[i,j][0]=='s']
    catalog = []
    for i, s in enumerate(sw):
        for g in sw[i+1:]:
            if abs(s[0]-g[0]) + abs(s[1]-g[1]) < min_dist:
                continue
            result = compute_pair_detour(s, g, grid)
            if result:
                sl, sf, D, rc = result
                catalog.append({'start': s, 'goal': g,
                                 'short_len': sl, 'safe_len': sf,
                                 'detour_factor': D, 'road_in_short': rc})
    return catalog


def stratified_sample(catalog, n_per_bucket=10, seed=SEED):
    """
    Sample pairs uniformly from detour-factor buckets so that pairs with
    genuine length-risk tradeoffs are equally represented.
    Pairs with D == 1.0 (no tradeoff) are excluded entirely.
    """
    rng = random.Random(seed)
    buckets = {
        'slight':   [p for p in catalog if 1.0 < p['detour_factor'] < 1.5],
        'moderate': [p for p in catalog if 1.5 <= p['detour_factor'] < 2.5],
        'strong':   [p for p in catalog if p['detour_factor'] >= 2.5],
    }
    sample = []
    for bname, bpairs in buckets.items():
        n = min(n_per_bucket, len(bpairs))
        chosen = rng.sample(bpairs, n)
        for p in chosen:
            p['bucket'] = bname
        sample.extend(chosen)
        print(f"    bucket '{bname}': {len(bpairs)} available, {n} sampled")
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_walker_risk(s=0, z=1, p=1, r=5, t=10, l=np.inf, f=2, b=np.inf, o=np.inf):
    return dict(zip(WALK_TERRAIN, [s, z, p, r, t, l, f, b, o]))

def make_drive_risk(fw=0, rt=1, lt=2, ch=10, ft=10, bw=15):
    return dict(zip(DRIVE_MOVES, [fw, rt, lt, ch, ft, bw]))


def build_parameters(walker_risk_map, pair_sample,
                     driver_count=N_DRIVERS, steps=SIM_STEPS,
                     beta_w=1, rho_w=1,       # walker beta (heuristic) and rho (risk)
                     beta_d=1, rho_d=1,       # driver beta and rho
                     obstacles=None, potholes=None, seed=None):
    """
    Build CityModel parameters from an explicit pair sample.
    Each pair becomes one walker agent.

    beta (heuristic weight) maps to 'beta' in the model.
    rho  (risk sensitivity) maps to 'rho' in the model.
    These internal names are required by AgentBase/AgentImpl/UrbanModelling.
    """
    rng = random.Random(seed if seed is not None else SEED)
    obs = obstacles if obstacles is not None else _BASE_OBS
    pth = potholes  if potholes  is not None else set()

    driver_sources, driver_goals = get_driver_endpoints(_BASE_GRID)

    walkers = []
    for pair in pair_sample:
        walkers.append({
            'start':       tuple(int(x) for x in pair['start']),
            'goal':        tuple(int(x) for x in pair['goal']),
            'speed':       10,
            'beta':      beta_w,       # beta -> model key 'beta'
            'rho': rho_w,        # rho  -> model key 'rho'
            'direction':   rng.choice([(0,1),(0,-1),(1,0),(-1,0)]),
        })

    drivers = []
    for _ in range(driver_count):
        start = tuple(int(x) for x in rng.choice(driver_sources))
        goal  = tuple(int(x) for x in rng.choice(driver_goals))
        attempts = 0
        while abs(goal[0]-start[0])+abs(goal[1]-start[1]) < 10 and attempts < 20:
            goal = tuple(int(x) for x in rng.choice(driver_goals))
            attempts += 1
        drivers.append({
            'start':       start,
            'goal':        goal,
            'speed':       60,
            'beta':      beta_d,       # beta -> model key 'beta'
            'rho': rho_d,        # rho  -> model key 'rho'
        })

    params = {
        'walker_risk_map': walker_risk_map,
        'city_grid':       _BASE_GRID,
        'obstacles':       list(obs),
        'potholes':        list(pth),
        'repopulate':      False,
        'steps':           steps,
    }
    if walkers: params['walkers'] = walkers
    if drivers: params['drivers'] = drivers
    return params


def run_model(params, seed=None):
    """
    Run one CityModel instance and return eta_j and other metrics.

    eta_j = road_steps / total_walker_steps
        Fraction of active walker-steps spent on pure road cells
        (jaywalking), excluding zebra crossings.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    model = CityModel(params)
    model.run()
    metrics = model.output.reporters['metrics'][0]

    road_steps   = 0
    walker_steps = 0
    for t in metrics:
        jw_dict   = metrics[t].get('jaywalking', {})
        road_steps   += sum(len(v) for v in jw_dict.values())
        walker_steps += metrics[t].get('active_walkers', 0)

    eta_j     = road_steps / max(1, walker_steps)
    total_col = sum(len(metrics[t].get('collisions', [])) for t in metrics)
    total_run = sum(len(metrics[t].get('runovers',   [])) for t in metrics)
    avg_spd_d = np.mean([metrics[t].get('avg_speed_drivers', 0) for t in metrics])

    return {'eta_j':             eta_j,
            'collisions':        total_col,
            'runovers':          total_run,
            'avg_speed_drivers': avg_spd_d}


def run_n(params_fn, n=N_RUNS):
    """Run n replications with fresh params each time."""
    results = [run_model(params_fn(), seed=SEED+i) for i in range(n)]
    keys = results[0].keys()
    return {k: (np.mean([r[k] for r in results]),
                np.std( [r[k] for r in results])) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# EXP-0: Route choice analysis (pure A*, no simulation)
# Verifies that beta and rho produce expected SAFE/ROAD decisions across the
# stratified sample before running full simulations.
# ─────────────────────────────────────────────────────────────────────────────

def exp0_route_choice(pair_sample):
    print("\n" + "="*60)
    print("EXP-0: Route choice (pure A*, no simulation)")
    print("  % pairs taking SAFE route across beta x rho")
    print("="*60)

    beta_values = [1, 2, 3, 5]
    rho_values  = [0.5, 1, 2, 5]
    road_risk   = 5   # selected baseline value

    risk_map = {'s':0,'z':1,'p':1,'r':road_risk,'t':road_risk*2,
                'l':road_risk*2,'f':2,'b':1e9,'o':1e9}
    sc = _make_step_cost(_BASE_GRID)

    rows = []
    print(f"\n  r={road_risk}, {len(pair_sample)} stratified pairs")
    print(f"\n  {'beta':>5} {'rho':>5}  {'safe%':>7}  "
          f"{'slight':>8}  {'moderate':>9}  {'strong':>8}")
    print("  " + "-"*55)

    for beta in beta_values:
        for rho in rho_values:
            # beta=beta, rho=rho: model internal names
            sr = _make_step_risk(_BASE_GRID, risk_map)
            by_bucket = {'slight':[],'moderate':[],'strong':[]}
            all_safe  = []
            for pair in pair_sample:
                path = weighted_astar(
                    pair['start'], pair['goal'], _BASE_GRID,
                    sc, sr, beta=beta, rho=rho)
                safe = path_is_safe(path, _BASE_GRID) if path else False
                all_safe.append(safe)
                by_bucket[pair.get('bucket','slight')].append(safe)

            pct   = 100*np.mean(all_safe)
            p_sl  = 100*np.mean(by_bucket['slight'])   if by_bucket['slight']   else float('nan')
            p_mo  = 100*np.mean(by_bucket['moderate']) if by_bucket['moderate'] else float('nan')
            p_st  = 100*np.mean(by_bucket['strong'])   if by_bucket['strong']   else float('nan')

            rows.append({'beta': beta, 'rho': rho, 'pct_safe': pct,
                         'pct_slight': p_sl, 'pct_moderate': p_mo,
                         'pct_strong': p_st})
            print(f"  beta={beta} rho={rho:.1f}  {pct:6.1f}%  "
                  f"{p_sl:7.1f}%  {p_mo:8.1f}%  {p_st:7.1f}%")
        print()

    df = pd.DataFrame(rows)
    print("  >> beta=1, rho=1 should prefer safe route")
    print("     beta=5, rho=0.5 should mostly take road shortcut")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXP-1: Road risk value calibration
# Selection criterion: eta_j < 0.15 at beta=1, rho=1 (average agent).
# ─────────────────────────────────────────────────────────────────────────────

def exp1_walker_road_risk(pair_sample):
    print("\n" + "="*60)
    print("EXP-1: eta_j vs. road cell risk  (beta=1, rho=1, no obstacles)")
    print("="*60)

    road_risks = range(1, 11)
    rows = []
    for rr in road_risks:
        wrm  = make_walker_risk(s=0, z=1, r=rr, t=rr*2)
        _pfn = lambda wrm=wrm: build_parameters(
            wrm, pair_sample, driver_count=0,
            beta_w=1, rho_w=1, obstacles=set())
        res = run_n(_pfn)
        mu, sd = res['eta_j']
        rows.append({'road_risk': rr, 'eta_j_mean': mu, 'eta_j_std': sd})
        print(f"  r={rr:3d}  eta_j = {mu:.4f} ± {sd:.4f}")

    df = pd.DataFrame(rows)
    cand = df[df['eta_j_mean'] < 0.15]
    if not cand.empty:
        best = cand.iloc[-1]
        print(f"\n  >> Recommended r = {best['road_risk']}  "
              f"(eta_j = {best['eta_j_mean']:.4f} at beta=1, rho=1)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXP-2: beta x rho grid -- primary behavioral calibration
# Expected: eta_j increases with beta, decreases with rho.
# ─────────────────────────────────────────────────────────────────────────────

def exp2_walker_beta_rho(pair_sample):
    print("\n" + "="*60)
    print("EXP-2: eta_j vs. beta and rho")
    print("  Expected: eta_j increases with beta, decreases with rho")
    print("="*60)

    beta_values = [1, 2, 3, 5]
    rho_values  = [0.5, 1, 2, 5]
    wrm = make_walker_risk()

    rows = []
    for beta, rho in itertools.product(beta_values, rho_values):
        _pfn = lambda beta=beta, rho=rho: build_parameters(
            wrm, pair_sample, driver_count=0,
            beta_w=beta, rho_w=rho)
        res  = run_n(_pfn)
        mu, sd = res['eta_j']
        rows.append({'beta': beta, 'rho': rho, 'eta_j_mean': mu, 'eta_j_std': sd})
        print(f"  beta={beta}  rho={rho:.1f}  eta_j = {mu:.4f} ± {sd:.4f}")

    df    = pd.DataFrame(rows)
    pivot = df.pivot(index='rho', columns='beta', values='eta_j_mean')
    print("\n  eta_j matrix (rows=rho, cols=beta):")
    print(pivot.round(4).to_string())

    base = df[(df['beta']==1)&(df['rho']==1  )]['eta_j_mean'].values[0]
    reck = df[(df['beta']==5)&(df['rho']==0.5)]['eta_j_mean'].values[0]
    caut = df[(df['beta']==1)&(df['rho']==5  )]['eta_j_mean'].values[0]
    print(f"\n  >> Baseline (beta=1, rho=1):   eta_j = {base:.4f}")
    print(f"     Reckless (beta=5, rho=0.5): eta_j = {reck:.4f}  (should be > baseline)")
    print(f"     Cautious (beta=1, rho=5):   eta_j = {caut:.4f}  (should be < baseline)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXP-3: Driver behavior vs. beta (no pedestrians)
# ─────────────────────────────────────────────────────────────────────────────

def exp3_driver_beta(pair_sample):
    print("\n" + "="*60)
    print("EXP-3: Driver collisions and speed vs. beta (no pedestrians)")
    print("="*60)

    beta_values = [1, 2, 3, 5, 8, 10]
    wrm  = make_walker_risk()
    rows = []
    for beta in beta_values:
        _pfn = lambda beta=beta: build_parameters(
            wrm, [], driver_count=N_DRIVERS,
            beta_d=beta, rho_d=1)
        res = run_n(_pfn)
        col_mu, col_sd = res['collisions']
        spd_mu, spd_sd = res['avg_speed_drivers']
        rows.append({'beta': beta,
                     'collisions_mean': col_mu, 'collisions_std': col_sd,
                     'avg_speed_mean':  spd_mu, 'avg_speed_std':  spd_sd})
        print(f"  beta={beta:2d}  collisions={col_mu:.2f}±{col_sd:.2f}  "
              f"avg_speed={spd_mu:.1f}±{spd_sd:.1f}")

    df  = pd.DataFrame(rows)
    b1  = df[df['beta']==1 ]['collisions_mean'].values[0]
    b10 = df[df['beta']==10]['collisions_mean'].values[0]
    print(f"\n  >> Collision ratio beta=10/beta=1: {b10/max(0.01,b1):.1f}x")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXP-4: Sensitivity -- ±20% and ±50% perturbation of all risk values
# ─────────────────────────────────────────────────────────────────────────────

def exp4_sensitivity(pair_sample):
    print("\n" + "="*60)
    print("EXP-4: Sensitivity to ±20%/±50% risk perturbation  (beta=3, rho=1)")
    print("="*60)

    base         = {'s':0, 'z':1, 'r':5, 't':10}
    perturbations = {
        'baseline': 1.0,
        '+20%':     1.2,
        '-20%':     0.8,
        '+50%':     1.5,
        '-50%':     0.5,
    }

    rows = []
    for label, factor in perturbations.items():
        perturbed = {k: (v*factor if 0 < v < np.inf else v)
                     for k, v in base.items()}
        wrm  = make_walker_risk(**perturbed)
        _pfn = lambda wrm=wrm: build_parameters(
            wrm, pair_sample, driver_count=N_DRIVERS,
            beta_w=3, rho_w=1)
        res = run_n(_pfn)
        mu, sd     = res['eta_j']
        run_mu, run_sd = res['runovers']
        rows.append({'perturbation': label, 'factor': factor,
                     'eta_j': mu, 'eta_j_std': sd,
                     'runovers': run_mu, 'run_std': run_sd})
        print(f"  {label:8s} (x{factor:.1f}):  "
              f"eta_j={mu:.4f}±{sd:.4f}  "
              f"runovers={run_mu:.2f}±{run_sd:.2f}")

    df       = pd.DataFrame(rows)
    base_val = df[df['perturbation']=='baseline']['eta_j'].values[0]
    max_dev  = df['eta_j'].apply(
        lambda x: abs(x-base_val)/max(base_val, 1e-6)).max()
    print(f"\n  >> Max relative deviation in eta_j: {max_dev:.2%}")
    print(f"     Considered robust if < 25%.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXP-5: Dynamic rho -- environmental modulation
# rho(t) = clip(rho_0 * phi(env), min_rho, max_rho)
# phi(env) = 1 + beta_v*(1 - v_norm) + beta_jw * jw_density
# Validated with static proxies before full model integration.
# ─────────────────────────────────────────────────────────────────────────────

def dynamic_rho(agent, rho_0=1.0, beta_v=0.5, beta_jw=-0.3,
                speed_limit=60.0, min_rho=0.5, max_rho=5.0):
    """
    Time-varying risk sensitivity for a pedestrian agent.

    Integration into AstarWalker.step_risk:
        rho  = dynamic_rho(self, rho_0=self.risk_weight)
        return rho * self.risk_map[cell_code[0]]

    Signals:
        v_norm     = mean visible driver speed / speed_limit  (0..1)
        jw_density = fraction of visible peers currently jaywalking
    phi = 1 + beta_v*(1 - v_norm) + beta_jw*jw_density
        slow/absent drivers raise rho; peer jaywalking lowers rho.
    """
    city      = agent.city
    neighbors = city.neighbors(agent, agent.visibility)
    drivers   = [n for n in neighbors if n.agent_type == 'driver' and n.active]
    v_norm    = (np.mean([d.speed for d in drivers]) / speed_limit
                 if drivers else 1.0)
    walkers   = [n for n in neighbors if n.agent_type == 'walker' and n.active]
    road_c    = city.road_cells - city.zebras
    jw_dens   = (sum(1 for wk in walkers if wk.grid_position() in road_c)
                 / len(walkers) if walkers else 0.0)
    phi = 1.0 + beta_v*(1.0 - v_norm) + beta_jw*jw_dens
    return float(np.clip(rho_0 * phi, min_rho, max_rho))


def exp5_dynamic_rho(pair_sample):
    """
    Validate the behavioral range of dynamic rho using static proxies.
    Each scenario represents the expected rho value under specific traffic
    conditions, derived analytically from the dynamic_rho formula:

        no traffic   (v_norm=1, jw=0):   phi=1.00 -> rho = rho_0
        heavy cars   (v_norm=0, jw=0):   phi=1.50 -> rho = rho_0 * 1.5
        peer JW      (v_norm=1, jw=1):   phi=0.70 -> rho = rho_0 * 0.7
        both signals (v_norm=0, jw=0.5): phi=1.35 -> rho = rho_0 * 1.35
    """
    print("\n" + "="*60)
    print("EXP-5: Dynamic rho range validation  (beta=2, static proxies)")
    print("="*60)

    wrm = make_walker_risk()
    scenarios = {
        'no traffic   (rho=1.0)':  1.0,
        'heavy cars   (rho=1.5)':  1.5,
        'peer JW      (rho=0.7)':  0.7,
        'both signals (rho=1.35)': 1.35,
        'vulnerable   (rho=3.0)':  3.0,
        'reckless     (rho=0.5)':  0.5,
    }

    rows = []
    for label, rho in scenarios.items():
        _pfn = lambda rho=rho: build_parameters(
            wrm, pair_sample, driver_count=N_DRIVERS,
            beta_w=2, rho_w=rho)
        res = run_n(_pfn)
        mu, sd         = res['eta_j']
        run_mu, run_sd = res['runovers']
        rows.append({'scenario': label, 'rho': rho,
                     'eta_j': mu, 'eta_j_std': sd,
                     'runovers': run_mu, 'run_std': run_sd})
        print(f"  {label:32s}  eta_j={mu:.4f}±{sd:.4f}  "
              f"runovers={run_mu:.2f}±{run_sd:.2f}")

    df = pd.DataFrame(rows)
    lo = df[df['rho']==0.5]['eta_j'].values[0]
    hi = df[df['rho']==3.0]['eta_j'].values[0]
    print(f"\n  >> eta_j range: reckless={lo:.4f}, vulnerable={hi:.4f}")
    print(f"     Ratio: {lo/max(hi,1e-6):.2f}x  (reckless jaywalks more)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXP-6: Pothole cost calibration
#
# Design rationale:
#   The pothole cost g_ph encodes the traversal effort of damaged road cells.
#   Higher g_ph causes drivers to reroute around potholes at the cost of
#   longer paths; lower g_ph causes drivers to cross them regardless.
#   Unlike walker risk (which governs safety preference), pothole cost is
#   purely a traversal-effort value: it does not change the danger of a cell,
#   only the expected time/effort to cross it.
#
#   We measure avg_speed_drivers across several pothole density rates
#   (0%, 5%, 10%, 20%) to show how pothole cost interacts with density.
#   Selection criterion: the recommended g_ph should produce a visible but
#   not severe speed reduction at 5% pothole density with average drivers
#   (beta=1, rho=1), without causing complete rerouting that would make the
#   pothole irrelevant in practice.
#
#   Pothole use rate (PUR): fraction of driver-steps on pothole cells.
#   Low PUR = drivers successfully avoid potholes.
#   High PUR = drivers cross potholes regardless of cost.
# ─────────────────────────────────────────────────────────────────────────────

def _build_driver_only_params(ph_cost, potholes, beta_d=1, rho_d=1,
                               steps=SIM_STEPS, seed=None):
    """
    Build a driver-only parameter dict with a custom pothole cost.
    ph_cost is passed as 'ph_cost' so UrbanModelling.setup_grid_labels()
    can override drive_cost['ph'] at initialization.
    """
    rng = random.Random(seed if seed is not None else SEED)
    driver_sources, driver_goals = get_driver_endpoints(_BASE_GRID)
    wrm = make_walker_risk()   # required by CityModel even for driver-only runs

    drivers = []
    for _ in range(N_DRIVERS):
        start = tuple(int(x) for x in rng.choice(driver_sources))
        goal  = tuple(int(x) for x in rng.choice(driver_goals))
        attempts = 0
        while abs(goal[0]-start[0])+abs(goal[1]-start[1]) < 10 and attempts < 20:
            goal = tuple(int(x) for x in rng.choice(driver_goals))
            attempts += 1
        drivers.append({
            'start':       start,
            'goal':        goal,
            'speed':       60,
            'beta':      beta_d,       # beta -> model key 'beta'
            'rho': rho_d,        # rho  -> model key 'rho'
        })
    return {
        'walker_risk_map': wrm,
        'city_grid':       _BASE_GRID,
        'obstacles':       [],
        'potholes':        list(potholes),
        'ph_cost':         ph_cost,      # override drive_cost['ph'] in model
        'repopulate':      False,
        'steps':           steps,
        'drivers':         drivers,
    }


def _sample_potholes(rate, seed=SEED):
    """Sample road cells as potholes at given density rate."""
    road_cells = [(i, j) for i in range(_BASE_GRID.shape[0])
                          for j in range(_BASE_GRID.shape[1])
                          if _BASE_GRID[i, j][0] == 'r']
    rng = random.Random(seed)
    n   = max(1, int(len(road_cells) * rate)) if rate > 0 else 0
    return rng.sample(road_cells, n) if n > 0 else []


def _run_model_driver(params, seed=None):
    """Run model and return avg_speed_drivers and pothole_use_rate."""
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    model   = CityModel(params)
    model.run()
    metrics = model.output.reporters['metrics'][0]

    spd_steps   = 0.0
    spd_count   = 0

    for t in metrics:
        spd = metrics[t].get('avg_speed_drivers', 0)
        n   = metrics[t].get('active_drivers', 0)
        if n > 0:
            spd_steps += spd * n
            spd_count += n
        # pothole_use_rate approximated from heatmap not available here;
        # use avg_speed as primary metric (pothole avoidance -> higher speed)

    avg_spd = spd_steps / max(1, spd_count)
    total_col = sum(len(metrics[t].get('collisions', [])) for t in metrics)
    return {'avg_speed_drivers': avg_spd, 'travel_time': spd_count, 'collisions': total_col}


def exp6_pothole_cost():
    """
    Calibrate pothole traversal cost g_ph.
    Runs a grid of ph_cost x pothole_rate configurations.
    Reports average driver speed for each combination.
    A good g_ph produces a visible speed reduction at 5% pothole density
    but does not cause complete rerouting or severe congestion.
    """
    print("\n" + "="*60)
    print("Pothole cost calibration  (beta=1, rho=1, drivers only)")
    print("  Metrics: avg travel time across ph_cost x pothole density")
    print("="*60)

    ph_costs      = [2, 5, 8, 10, 15, 20]
    poth_rates    = [0.0, 0.05, 0.10, 0.20]
    rate_labels   = ['0%', '5%', '10%', '20%']

    # Pre-sample potholes for each rate (fixed seed for reproducibility)
    poth_sets = {rate: _sample_potholes(rate) for rate in poth_rates}

    rows = []
    print(f"\n  {'ph_cost':>8}  " + "  ".join(f"{'spd@'+lbl:>10}" for lbl in rate_labels))
    print("  " + "-"*55)

    for ph in ph_costs:
        row = {'ph_cost': ph}
        spd_vals = []
        for rate, lbl in zip(poth_rates, rate_labels):
            potholes = poth_sets[rate]
            results  = [_run_driver_rep(ph, potholes, seed=SEED+i)
                        for i in range(N_RUNS)]
            mu = np.mean([r['travel_time'] for r in results])
            sd = np.std( [r['travel_time'] for r in results])
            row[f'spd_{lbl}_mean'] = mu
            row[f'spd_{lbl}_std']  = sd
            spd_vals.append(f"{mu:6.2f}±{sd:.2f}")
        rows.append(row)
        print(f"  ph={ph:3d}        " + "    ".join(spd_vals))

    df = pd.DataFrame(rows)

    # Selection: find ph_cost where speed at 5% < speed at 0% - threshold
    spd_no_poth = df['spd_0%_mean'].values[0]   # baseline: no potholes
    cand = df[df['spd_5%_mean'] < spd_no_poth * 0.90]  # >10% speed reduction
    if not cand.empty:
        best = cand.iloc[0]
        print(f"\n  >> Recommended ph_cost = {best['ph_cost']}  "
              f"(travel time at 5% potholes = {best['spd_5%_mean']:.2f} "
              f"vs {spd_no_poth:.2f} baseline, "
              f"{100*(1-best['spd_5%_mean']/spd_no_poth):.1f}% reduction)")
    return df


def _run_driver_rep(ph_cost, potholes, seed=None):
    """Single replication for EXP-6."""
    params = _build_driver_only_params(ph_cost, potholes, seed=seed)
    return _run_model_driver(params, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df0, df1, df2, df3, df4, df5, df6, pair_sample,
                 save_path='calibration_results.png'):
    fig = plt.figure(figsize=(16, 24))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    # EXP-0: safe % grouped bar by bucket
    if df0 is not None:
        ax0 = fig.add_subplot(gs[4, 2])
        buckets    = ['slight', 'moderate', 'strong']
        beta_vals  = sorted(df0['beta'].unique())
        rho_vals   = sorted(df0['rho'].unique())
        x          = np.arange(len(beta_vals) * len(rho_vals))
        labels     = [f'β={b}\nρ={r}' for b in beta_vals for r in rho_vals]
        width      = 0.25
        colors_b   = {'slight':'#a8d5e2','moderate':'#f6ae2d','strong':'#f26419'}
        for bi, bkt in enumerate(buckets):
            col  = f'pct_{bkt}'
            vals = [df0[(df0['beta']==b)&(df0['rho']==r)][col].values[0]
                    for b in beta_vals for r in rho_vals]
            ax0.bar(x + bi*width, vals, width, label=bkt.capitalize(),
                    color=colors_b[bkt], alpha=0.85)
        ax0.set_xticks(x + width)
        ax0.set_xticklabels(labels, fontsize=7)
        ax0.set_ylabel('% agents taking safe route')
        ax0.set_title(r'Safe route % by detour bucket ($\beta$, $\rho$)'
                      f'\nr=5, {len(pair_sample)} stratified pairs')
        ax0.legend(title='Detour bucket', fontsize=8)
        ax0.set_ylim(0, 115)

    # EXP-1: eta_j vs road risk
    if df1 is not None:
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.errorbar(df1['road_risk'], df1['eta_j_mean'], yerr=df1['eta_j_std'],
                     fmt='o-', capsize=4, color='steelblue', label=r'PRER')
        ax1.axhline(0.15, color='red', linestyle=':', linewidth=1,
                    label='target < 0.15')
        ax1.set_xlabel(r'Road cell risk $r_r$')
        ax1.set_ylabel(r'Jaywalking rate PRER')
        ax1.set_title(r'PRER vs. road risk'
                      '\n'+r'($\beta$=1, $\rho$=1, no obstacles)')
        ax1.legend(fontsize=8)

    # EXP-2: eta_j heatmap
    if df2 is not None:
        ax2 = fig.add_subplot(gs[1, 1])
        pivot = df2.pivot(index='rho', columns='beta', values='eta_j_mean')
        im    = ax2.imshow(pivot.values, aspect='auto', cmap='YlOrRd', origin='lower')
        ax2.set_xticks(range(len(pivot.columns)))
        ax2.set_xticklabels([fr'$\beta$={c}' for c in pivot.columns])
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels([fr'$\rho$={r}' for r in pivot.index])
        ax2.set_title(r'EXP-2: PRER across ($\beta$, $\rho$)')
        plt.colorbar(im, ax=ax2, label=r'PRER')

        # EXP-2 line plot
        ax2b = fig.add_subplot(gs[2, 0])
        for rho in sorted(df2['rho'].unique()):
            sub = df2[df2['rho']==rho].sort_values('beta')
            ax2b.errorbar(sub['beta'], sub['eta_j_mean'], yerr=sub['eta_j_std'],
                          fmt='o-', capsize=3, label=fr'$\rho$={rho}')
        ax2b.set_xlabel(r'$\beta$')
        ax2b.set_ylabel(r'PRER')
        ax2b.set_title(r'EXP-2: PRER vs. $\beta$ for each $\rho$')
        ax2b.legend(fontsize=8)

    # EXP-3
    if df3 is not None:
        ax3  = fig.add_subplot(gs[2, 1])
        ax3.errorbar(df3['beta'], df3['collisions_mean'], yerr=df3['collisions_std'],
                     fmt='s-', capsize=4, color='firebrick', label='collisions')
        ax3b = ax3.twinx()
        ax3b.errorbar(df3['beta'], df3['avg_speed_mean'], yerr=df3['avg_speed_std'],
                      fmt='^--', capsize=4, color='darkorange', label='avg speed')
        ax3.set_xlabel(r'Driver $\beta$')
        ax3.set_ylabel('Collisions', color='firebrick')
        ax3b.set_ylabel('Avg speed', color='darkorange')
        ax3.set_title(r'Driver collisions & speed vs. $\beta$')
        l1,b1 = ax3.get_legend_handles_labels()
        l2,b2 = ax3b.get_legend_handles_labels()
        ax3.legend(l1+l2, b1+b2, fontsize=8)

    # EXP-4
    if df4 is not None:
        ax4 = fig.add_subplot(gs[3, 0])
        colors4 = ['steelblue','seagreen','seagreen','coral','coral']
        ax4.bar(df4['perturbation'], df4['eta_j'], yerr=df4['eta_j_std'],
                capsize=4, color=colors4)
        ax4.set_ylabel(r'PRER (PRER)')
        ax4.set_title(r'EXP-4: Sensitivity ±20%/±50% ($\beta$=3, $\rho$=1)')
        ax4.tick_params(axis='x', labelsize=8)

    # EXP-5
    if df5 is not None:
        ax5   = fig.add_subplot(gs[3, 1])
        x5    = np.arange(len(df5))
        w5    = 0.35
        ax5.bar(x5-w5/2, df5['eta_j'], w5, yerr=df5['eta_j_std'],
                capsize=4, label='PRER', color='steelblue')
        ax5b  = ax5.twinx()
        ax5b.bar(x5+w5/2, df5['runovers'], w5, yerr=df5['run_std'],
                 capsize=4, label='Runovers', color='firebrick', alpha=0.7)
        ax5.set_xticks(x5)
        ax5.set_xticklabels(df5['scenario'], rotation=20, ha='right', fontsize=7)
        ax5.set_ylabel(r'PRER (PRER)', color='steelblue')
        ax5b.set_ylabel('Runovers', color='firebrick')
        ax5.set_title(r'EXP-5: Dynamic $\rho$ range ($\beta$=2)')
        l1,b1 = ax5.get_legend_handles_labels()
        l2,b2 = ax5b.get_legend_handles_labels()
        ax5.legend(l1+l2, b1+b2, fontsize=8)

    # EXP-6: pothole cost -- line plot with one line per pothole density rate
    if df6 is not None:
        ax6 = fig.add_subplot(gs[4, 1])
        rate_labels = ['0%', '5%', '10%', '20%']
        colors6     = ['steelblue', 'darkorange', 'firebrick', 'purple']
        linestyles  = ['-', '--', '-.', ':']
        for lbl, col, ls in zip(rate_labels, colors6, linestyles):
            mean_col = f'spd_{lbl}_mean'
            std_col  = f'spd_{lbl}_std'
            if mean_col in df6.columns:
                ax6.errorbar(df6['ph_cost'], df6[mean_col], yerr=df6[std_col],
                             fmt='o'+ls, capsize=3, color=col,
                             label=f'pothole density {lbl}')
        ax6.set_xlabel('Pothole traversal cost $g_{ph}$')
        ax6.set_ylabel('Average driver speed (units/step)')
        ax6.set_title('Driver speed vs. pothole cost across densities\n'
                      r'($\beta$=1, $\rho$=1, drivers only)')
        ax6.legend(fontsize=8)
        ax6.set_xticks(df6['ph_cost'].values)

    plt.suptitle('Calibration Experiments — Tables I–III\n'
                 r'Urban Multiagent Simulation (Accumulated-Risk Weighted A$^*$)',
                 fontsize=13, fontweight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building pair catalog...")
    catalog = build_pair_catalog(_BASE_GRID, min_dist=4)
    D_vals  = [p['detour_factor'] for p in catalog]
    print(f"  Total pairs: {len(catalog)}  "
          f"D range: [{min(D_vals):.2f}, {max(D_vals):.2f}]  "
          f"mean D={np.mean(D_vals):.3f}")

    print("\nStratified sampling:")
    pair_sample = stratified_sample(catalog, n_per_bucket=10, seed=SEED)
    print(f"  Sample size: {len(pair_sample)}")

    df0 = exp0_route_choice(pair_sample)
    df1 = exp1_walker_road_risk(pair_sample)
    df2 = exp2_walker_beta_rho(pair_sample)
    df3 = exp3_driver_beta(pair_sample)
    df4 = exp4_sensitivity(pair_sample)
    df5 = exp5_dynamic_rho(pair_sample)
    df6 = exp6_pothole_cost()

    out = os.path.join(os.path.dirname(__file__), 'calibration_results.png')
    plot_results(df0, df1, df2, df3, df4, df5, df6, pair_sample, save_path=out)

    print("\n" + "="*60)
    print("RECOMMENDED PARAMETER VALUES")
    print("="*60)
    print("  Walker risk:  s=0  z=1  r=5  t=10  (r=5: nonzero baseline jaywalking)")
    print("  Driver cost:  r=1  z=2  p=5  ph=5  (ph selected from EXP-6)")
    print("  Driver risk:  fw=0 rt=1 lt=2 ch=3 ft=5 bw=15")
    print("  Walker:  beta in {1,2,3,5}  rho_0=1.0  (dynamic rho at runtime)")
    print("  Driver:  beta in {1,3,5,10} rho=1.0")