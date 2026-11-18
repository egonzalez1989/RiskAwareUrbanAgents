"""
Runs one job of the experiments reported in the paper.

Usage (from src/):  python run_experiment.py JOB SEED [--out DIR] [--steps N]

  E1_<beta>             pedestrians only, sidewalk obstruction 0-25% (beta 1, 3, 5), 2000 steps
  E2_<beta>             one beta slice of the (beta, rho) grid (beta 1, 2, 3, 5)
  E3_<beta>             (beta, rho) grid with eight one-lane closures (beta 1, 3, 5)
  E4_<drivers>          traffic load: 20 or 100 drivers, 200 walkers, 10% obstruction, 10000 steps
  E5                    street-market scenario, 10000 steps
  E6_<variant>          sensitivity to the assumptions: E2 setting, beta 3, rho 0.25, 1, 5
                        variant: base   risk values of Tables 2 and 4
                                 pert   every nonzero risk value multiplied by an independent
                                        log-uniform factor in [1/2, 2], drawn from the seed
                                 tau2   perception-reaction time 0.6 s (2 steps) instead of 1.2 s
                                 tau6   perception-reaction time 1.8 s (6 steps)
  R_<risk>              choice of the road risk of walkers: pedestrians only, beta = rho = 1,
                        road risk <risk> (1-10), sidewalk obstruction 0, 5 and 10%, 2000 steps

Seed protocol: SEED initializes Python's and NumPy's global generators (spawn
counts, agent parameters, obstacles, market stalls and lane closures) and the
model's own generators (parameter 'seed'). The same JOB and SEED give the same
run. Configurations compared within an experiment use the same seeds, so they
share the obstacle layout and can be compared seed by seed.

Results go to <DIR>/exp_<name>_seed_<SEED>/<job> (default ../Experiments); an
existing job directory is not overwritten. Run several jobs in parallel, e.g.
    for s in $(seq 0 19); do echo "E4_100 $s"; done | xargs -P 4 -L 1 python run_experiment.py
"""
import argparse, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis.Experiments as Experiments
from analysis.Experiments import (gen_city, gen_obstacles, execute_job, walker_obstacles,
                                  exp_beta_rho_grid, special_usecase, walker_road_risk)

# E6: nonzero risk values that are perturbed (walker cell types, driver maneuvers)
WALK_RISK = {'z': 1, 'r': 5, 't': 10}
DRIVE_RISK = {'rt': 1, 'lt': 3, 'ch': 3, 'ft': 5, 'bw': 15}


def risk_perturbation(seed):
    """Risk values multiplied by independent log-uniform factors in [1/2, 2]. A separate
    generator keeps the global streams, and so the obstacle layout, equal to those of E2."""
    rng = np.random.default_rng(10_000 + seed)
    factor = lambda: float(2 ** rng.uniform(-1, 1))
    return {'walk': {k: v * factor() for k, v in WALK_RISK.items()},
            'drive': {k: v * factor() for k, v in DRIVE_RISK.items()}}


def grid_city():
    """5 x 5 blocks of 15 x 15 cells, one lane per direction (E4)."""
    return gen_city(block_shape=(15, 15), city_shape=(5, 5), street_parking=False, two_way=True, crop=0)


def traffic_parameters(seed, steps, nd, nw, density=.10, **extra):
    """Parameters of E4: Poisson-replenished populations on grid_city."""
    spawn = lambda n: max(0, np.random.poisson(n))
    city_grid = grid_city()
    p = {'seed': seed, 'debug': False, 'steps': steps, 'display': False,
         'random_weight_walker': lambda: random.randint(1, 3),
         'random_weight_driver': lambda: random.randint(1, 5),
         'city_grid': city_grid, 'potholes': [], 'initial_walker_count': nw, 'initial_driver_count': nd,
         'walker_spawn_function': lambda: spawn(nw), 'driver_spawn_function': lambda: spawn(nd)}
    p.update(extra)
    random.seed(seed)
    p['obstacles'] = gen_obstacles(city_grid, obstacles=density)[0]
    return p


def run(job, seed, out, steps=None):
    Experiments.OUT_ROOT = out
    random.seed(seed)
    np.random.seed(seed)
    kind, _, arg = job.partition('_')
    if kind == 'R':
        walker_road_risk(int(arg), seed=seed)
    elif kind == 'E1':
        walker_obstacles(beta=lambda b=int(arg): b, seed=seed, exp_name=f'nodrivers_seed_{seed}_beta_{arg}')
    elif kind == 'E2':
        exp_beta_rho_grid(seed=seed, exp_name=f'beta_rho_grid_seed_{seed}', beta_values=(int(arg),))
    elif kind == 'E3':
        exp_beta_rho_grid(seed=seed, exp_name=f'lane_closures_seed_{seed}', beta_values=(int(arg),), closures=8)
    elif kind == 'E6':
        override = risk_perturbation(seed) if arg == 'pert' else None
        extra = {'reaction_steps': int(arg[3:])} if arg.startswith('tau') else None
        exp_beta_rho_grid(seed=seed, exp_name=f'sensitivity_seed_{seed}/{arg}', beta_values=(3,),
                          rho_values=(0.25, 1, 5), risk_override=override, extra=extra)
    elif kind == 'E4':
        nd = int(arg)
        execute_job(f'{out}/exp_interaction_seed_{seed}/obstacles_10_nw_200_nd_{nd:03d}',
                    traffic_parameters(seed, steps or 10000, nd=nd, nw=200))
    elif kind == 'E5':
        special_usecase(beta_walker=lambda: random.randint(1, 3), beta_driver=lambda: random.randint(1, 5),
                        walker_generator=lambda n: max(0, np.random.poisson(n)),
                        driver_generator=lambda n: max(0, np.random.poisson(n)),
                        seed=seed, exp_name=f'usecase_seed_{seed}')
    else:
        raise ValueError(f'unknown job {job}')
    print('DONE', job, seed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('job'); ap.add_argument('seed', type=int)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Experiments'))
    ap.add_argument('--steps', type=int, default=None, help='override the number of steps (E4)')
    a = ap.parse_args()
    run(a.job, a.seed, a.out, a.steps)
