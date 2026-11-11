import json, dill, random, os, numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from model.AgentBase import UrbanAgent, MobileAgent
from model.AgentImpl import AstarDriver, AstarWalker
from model.UrbanModelling import CityModel, City, PoisonCityModel
from utils.Reporting import load_heatmaps, generate_heatmaps
from utils.AgentUtils import gen_city, gen_obstacles, gen_closures

# ─────────────────────────────────────────────────────────────────────────────
# AUXILIARY FUNCTIONS  (unchanged from original except generate_summary)
# ─────────────────────────────────────────────────────────────────────────────
default_functions = {
    'walker_spawn_function': lambda: 0,
    'driver_spawn_function': lambda: 0,
    'random_weight_walker':  lambda: 1,
    'random_weight_driver':  lambda: 1,
    'f_walker_speed':        lambda: AstarWalker.DESIRED_SPEED,
    'f_driver_speed':        lambda: AstarDriver.DESIRED_SPEED,
}
set_parameters = ['obstacles', 'potholes']
# Root directory of the simulation output (run_experiment.py --out changes it)
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Experiments')


def execute_jobs(exp_dir, parameters, values_grid,
                 job_names=None, overwrite=False):
    if isinstance(parameters, str):
        parameters = load_parameters(parameters)
    param, values = values_grid
    job_names = job_names if job_names else \
        [param + '_' + str(val) for val in values]
    for (val, job_name) in sorted(zip(values, job_names)):
        parameters[param] = val
        execute_job(os.path.join(exp_dir, job_name), parameters, overwrite)


def execute_job(job_dir, parameters, overwrite=False):
    if os.path.exists(job_dir) and not overwrite:
        return
    os.makedirs(job_dir, exist_ok=True)
    if isinstance(parameters, str):
        parameters = load_parameters(parameters)
    model = PoisonCityModel(parameters)
    model.run()
    save_parameters(job_dir, parameters)
    metrics_file = os.path.join(job_dir, 'metrics.txt')
    with open(metrics_file, 'w+') as f:
        f.write(json.dumps(model.metrics))
    agents_file = os.path.join(job_dir, 'agents.txt')
    with open(agents_file, 'w+') as f:
        f.write(json.dumps(model.spawned_agents))
    np.save(os.path.join(job_dir, 'walker_speed_hm.npy'),
            model.walker_speed_hm)
    np.save(os.path.join(job_dir, 'driver_speed_hm.npy'),
            model.driver_speed_hm)
    h, w = model.city.city_grid.shape
    walker_count_hm = np.array(
        [[len(model.walker_count_hm[i][j]) for j in range(w)]
         for i in range(h)])
    np.save(os.path.join(job_dir, 'walker_count_hm.npy'), walker_count_hm)
    driver_count_hm = np.array(
        [[len(model.driver_count_hm[i][j]) for j in range(w)]
         for i in range(h)])
    np.save(os.path.join(job_dir, 'driver_count_hm.npy'), driver_count_hm)


def save_parameters(dir, parameters):
    import inspect, copy
    params_file = os.path.join(dir, 'parameters.txt')
    p = deepcopy(parameters)
    p['city_grid'] = [list(str(cell) for cell in row) for row in p['city_grid']]
    for key, val in p.items():
        if inspect.isfunction(val):
            p[key] = str(dill.dumps(val))
        if isinstance(val, set):
            p[key] = list(val)
    with open(params_file, 'w+') as f:
        f.write(json.dumps(p))


def load_parameters(parameters_file):
    p = json.load(open(parameters_file))
    functions  = set(default_functions.keys())
    sets       = set(set_parameters)
    for f in functions.intersection(set(p.keys())):
        p[f] = dill.loads(eval(p[f]))
    for s in sets.intersection(set(p.keys())):
        p[s] = set([tuple(x) for x in p[s]])
    return p


def generate_summary(outdir, exp_dir=None, job_dirs=None,
                     heatmaps=True, figsize=None):
    """
    Compute per-job summary metrics and save as .npy files.

    Metrics computed:
      collisions.npy  — collisions per driver
      runovers.npy    — runovers per walker
      prer.npy        — Pedestrian Road Exposure Rate (PRER = eta_j):
                        total road-cell steps / total walker steps,
                        matching the definition in Section 4 (eq:prer).
                        This replaces the old jaywalking.npy which counted
                        unique jaywalking agents rather than step fractions.
      jaywalking.npy  — fraction of walkers who jaywalked at least once
                        (kept for backward compatibility with existing figures).
    """
    import glob
    if exp_dir is not None:
        job_dirs = sorted(list(
            filter(lambda x: os.path.isdir(x),
                   glob.glob(os.path.join(exp_dir, '*')))))

    if heatmaps:
        for i in range(len(job_dirs)):
            plot_dir = os.path.join(job_dirs[i], 'heatmaps')
            os.makedirs(plot_dir, exist_ok=True)
            params = load_parameters(os.path.join(job_dirs[i], 'parameters.txt'))
            params['city_grid'] = np.array(params['city_grid'])
            model  = CityModel(params)
            city   = City(model, shape=params['city_grid'].shape)
            city.setup()
            metrics = load_heatmaps(os.path.join(job_dirs[i], 'metrics.txt'))
            generate_heatmaps(city, metrics, plot_dir, figsize=figsize)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    nc, nr, prer_list, jw_frac = [], [], [], []

    for dir in sorted(job_dirs):
        metrics = json.load(open(os.path.join(dir, 'metrics.txt'), 'r'))
        agents  = json.load(open(os.path.join(dir, 'agents.txt'),  'r'))
        num_drivers = len([k for k in agents if agents[k]['type'] == 'AstarDriver'])
        num_walkers = len([k for k in agents if agents[k]['type'] == 'AstarWalker'])

        # collisions and runovers
        nc.append(0 if num_drivers == 0 else
                  sum(len(metrics[t]['collisions']) for t in metrics) / num_drivers)
        nr.append(0 if num_walkers == 0 else
                  sum(len(metrics[t]['runovers']) for t in metrics) / num_walkers)

        # PRER: road_steps / total_walker_steps  (paper eq:prer)
        road_steps   = sum(
            sum(len(v) for v in metrics[t]['jaywalking'].values())
            for t in metrics)
        walker_steps = sum(
            metrics[t].get('active_walkers', 0) for t in metrics)
        prer_list.append(0.0 if walker_steps == 0 else
                         road_steps / walker_steps)

        # Fraction of walkers who jaywalked at least once (legacy metric)
        jw_agents = set()
        for t in metrics:
            jw_agents.update(metrics[t]['jaywalking'].keys())
        jw_frac.append(0 if num_walkers == 0 else
                       len(jw_agents) / num_walkers)

    np.save(os.path.join(outdir, 'collisions.npy'), nc)
    np.save(os.path.join(outdir, 'runovers.npy'),   nr)
    np.save(os.path.join(outdir, 'prer.npy'),        prer_list)
    np.save(os.path.join(outdir, 'jaywalking.npy'),  jw_frac)  # legacy


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

OBSTRUCTION_LEVELS = (0, 2, 5, 10, 15, 20, 25)      # % of sidewalk cells obstructed (E1)


def walker_obstacles(beta=lambda: 1, seed=0, overwrite=False, exp_name=None):
    """
    Pedestrian-only experiment: vary sidewalk obstacle density.
    Parameter renamed: weight -> beta  (heuristic greediness, paper notation).
    """
    parameters = {
        'seed':                  seed,
        'debug':                 False,
        'random_weight_walker':  beta,   # beta -> model key 'weight'
        'initial_walker_count':  500,
        'walker_spawn_function': lambda: 0,
        'display':               False,
        'steps':                 2000       # long enough for most walkers to finish their trip
    }
    random.seed(parameters['seed'])
    city_grid = gen_city(block_shape=(15, 15), city_shape=(5, 5),
                         street_parking=False, crop=0)
    parameters['city_grid'] = city_grid
    parameters['potholes']  = []
    obs_grid, job_names = [], []
    for i in OBSTRUCTION_LEVELS:
        density = .01 * i
        obs_grid.append(gen_obstacles(city_grid, obstacles=density)[0])
        job_names.append('obstacles_{}'.format(int(100 * density)))
    exp_name = exp_name or 'nodrivers_seed_{}'.format(seed)
    exp_dir  = OUT_ROOT + '/exp_{}/'.format(exp_name)
    execute_jobs(exp_dir, parameters, ('obstacles', obs_grid),
                 job_names, overwrite=overwrite)


def driver_walker_interaction(beta_walker=lambda: 1, beta_driver=lambda: 1,
                              walker_generator=lambda: 0,
                              driver_generator=lambda: 0,
                              seed=0, overwrite=False, exp_name=None):
    """
    Full interaction experiment: varies obstacle density, walker count,
    driver count. beta_walker and beta_driver map to model key 'weight'.
    rho (risk sensitivity) is left at the model default (1) here;
    use exp_beta_rho_grid for the rho sweep.
    """
    parameters = {
        'seed':                  seed,
        'debug':                 False,
        'random_weight_walker':  beta_walker,
        'random_weight_driver':  beta_driver,
        'steps':                 10000,
        'display':               False,
    }
    city_grid = gen_city(block_shape=(15, 15), city_shape=(5, 5),
                         street_parking=False, two_way=True, crop=0)
    parameters['city_grid'] = city_grid
    parameters['potholes']  = []
    for density in (0.10,):
        for nwalkers in (200,):
            for ndrivers in (20, 100):
                random.seed(seed)
                print('*** {} - {} - {} ***'.format(density, nwalkers, ndrivers))
                parameters['initial_walker_count'] = nwalkers
                parameters['initial_driver_count'] = ndrivers
                parameters['walker_spawn_function'] = \
                    lambda: walker_generator(nwalkers)
                parameters['driver_spawn_function'] = \
                    lambda: driver_generator(ndrivers)
                parameters['obstacles'] = \
                    gen_obstacles(city_grid, obstacles=density)[0]
                job_name = 'obstacles_{}_nw_{}_nd_{}'.format(
                    int(100 * density), '{:03d}'.format(nwalkers),
                    '{:03d}'.format(ndrivers))
                job_dir = OUT_ROOT + '/exp_{}/{}'.format(exp_name, job_name)
                execute_job(job_dir, parameters, overwrite=overwrite)


def exp_beta_rho_grid(seed=0, overwrite=False, exp_name='beta_rho_grid',
                      beta_values=(1, 2, 3, 5), rho_values=(0, 0.25, 0.5, 1, 2, 5),
                      closures=0, risk_override=None, extra=None):
    """
    NEW: Sweep (beta, rho) grid on a fixed population (200 walkers, 50 drivers,
    5% sidewalk obstacles) to produce the figure showing how behavioral weights
    jointly determine PRER and runover rate.

    This directly supports the claim in Section 4 that the model reproduces
    the full behavioral spectrum from cautious (low beta, high rho) to reckless
    (high beta, low rho), and provides the missing figure for Section 5.

    beta values: [1, 2, 3, 5]   (heuristic greediness)
    rho  values: [0.5, 1, 2, 5] (risk sensitivity)

    Each cell of the grid is one simulation run.  After all runs, the function
    generates:
      - prer_heatmap.png       : PRER across (beta x rho)
      - runovers_heatmap.png   : runover rate across (beta x rho)
      - prer_lines.png         : PRER vs beta, one line per rho value

    NOTE: the model uses internal parameter names 'weight' (=beta) and
    'risk_weight' (=rho), so these are set in the agent dict fields via the
    random_weight_walker / random_weight_driver lambda functions.
    """
    beta_values, rho_values = list(beta_values), list(rho_values)
    obstacle_density = 0.05
    nwalkers, ndrivers = 200, 50

    city_grid = gen_city(block_shape=(15, 15), city_shape=(5, 5),
                         street_parking=False, crop=0)
    obstacles = gen_obstacles(city_grid, obstacles=obstacle_density)[0]
    # Lane closures (roadwork) for the wrong-way driving scenario
    if closures:
        obstacles = set(obstacles) | set(gen_closures(city_grid, n=closures, seed=seed))

    exp_dir  = OUT_ROOT + '/exp_{}/'.format(exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    results = {}   # (beta, rho) -> {'prer': float, 'runovers': float}

    for beta in beta_values:
        for rho in rho_values:
            random.seed(seed)
            np.random.seed(seed)
            job_name = 'beta_{}_rho_{}'.format(beta, str(rho).replace('.', 'p'))
            job_dir  = os.path.join(exp_dir, job_name)
            parameters = {
                'seed':                  seed,
                'debug':                 False,
                'random_weight_walker':  lambda b=beta:  b,   # beta -> 'weight'
                'random_weight_driver':  lambda b=beta:  b,
                # CityModel.walker_rho()/driver_rho() call these to set rho
                'random_rho_walker':     lambda r=rho: r,
                'random_rho_driver':     lambda r=rho: r,
                'initial_walker_count':  nwalkers,
                'initial_driver_count':  ndrivers,
                'walker_spawn_function': lambda n=nwalkers: max(0, np.random.poisson(n)),
                'driver_spawn_function': lambda n=ndrivers: max(0, np.random.poisson(n)),
                'steps':                 1000,
                'city_grid':             city_grid,
                'obstacles':             obstacles,
                'potholes':              [],
                'display':               False,
            }
            if risk_override:
                parameters['risk_override'] = risk_override
            parameters.update(extra or {})      # e.g. reaction_steps (E7)
            execute_job(job_dir, parameters, overwrite=overwrite)

            # Read results
            metrics     = json.load(open(os.path.join(job_dir, 'metrics.txt')))
            agents      = json.load(open(os.path.join(job_dir, 'agents.txt')))
            num_walkers = len([k for k in agents
                               if agents[k]['type'] == 'AstarWalker'])

            road_steps   = sum(
                sum(len(v) for v in metrics[t]['jaywalking'].values())
                for t in metrics)
            walker_steps = sum(
                metrics[t].get('active_walkers', 0) for t in metrics)
            prer = road_steps / max(1, walker_steps)

            runovers = sum(len(metrics[t]['runovers']) for t in metrics)
            runover_rate = runovers / max(1, num_walkers)

            results[(beta, rho)] = {'prer': prer, 'runover_rate': runover_rate}
            print('  beta={} rho={:.1f}  PRER={:.4f}  runovers/walker={:.4f}'
                  .format(beta, rho, prer, runover_rate))

    # ── Plots ──────────────────────────────────────────────────────────────

    # 1. PRER heatmap
    prer_mat = np.array([[results[(b, r)]['prer'] for b in beta_values]
                          for r in rho_values])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(prer_mat, origin='lower', cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(beta_values)))
    ax.set_xticklabels([r'$\beta$={}'.format(b) for b in beta_values])
    ax.set_yticks(range(len(rho_values)))
    ax.set_yticklabels([r'$\rho$={}'.format(r) for r in rho_values])
    ax.set_xlabel(r'Heuristic weight $\beta$')
    ax.set_ylabel(r'Risk sensitivity $\rho$')
    ax.set_title(r'PRER across ($\beta$, $\rho$)')
    plt.colorbar(im, ax=ax, label=r'PRER')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'prer_heatmap.png'), dpi=150)
    plt.close()

    # 2. Runover rate heatmap
    run_mat = np.array([[results[(b, r)]['runover_rate'] for b in beta_values]
                         for r in rho_values])
    fig, ax = plt.subplots(figsize=(6, 4))
    im2 = ax.imshow(run_mat, origin='lower', cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(beta_values)))
    ax.set_xticklabels([r'$\beta$={}'.format(b) for b in beta_values])
    ax.set_yticks(range(len(rho_values)))
    ax.set_yticklabels([r'$\rho$={}'.format(r) for r in rho_values])
    ax.set_xlabel(r'Heuristic weight $\beta$')
    ax.set_ylabel(r'Risk sensitivity $\rho$')
    ax.set_title(r'Runover rate across ($\beta$, $\rho$)')
    plt.colorbar(im2, ax=ax, label='Runovers per walker')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'runovers_heatmap.png'), dpi=150)
    plt.close()

    # 3. PRER line plot by rho
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    for ri, rho in enumerate(rho_values):
        prer_vals = [results[(b, rho)]['prer'] for b in beta_values]
        ax.plot(beta_values, prer_vals, marker=markers[ri % len(markers)],
                label=r'$\rho$={}'.format(rho))
    ax.set_xlabel(r'Heuristic weight $\beta$')
    ax.set_ylabel(r'PRER')
    ax.set_title(r'PRER vs $\beta$ for each $\rho$ (5\% obstacles)')
    ax.legend()
    ax.set_xticks(beta_values)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'prer_lines.png'), dpi=150)
    plt.close()

    print('Figures saved to', exp_dir)
    return results


def special_usecase(beta_walker=lambda: 1, beta_driver=lambda: 1,
                    walker_generator=lambda: 0,
                    driver_generator=lambda: 0,
                    seed=0, overwrite=False, exp_name=None):
    """
    Use-case scenario: partial sidewalk obstruction (street market).
    Bug fix: density was undefined in original; fixed to explicit obstacle list.
    Parameter renamed: weight -> beta_walker / beta_driver.
    """
    parameters = {
        'seed':                  seed,
        'debug':                 False,
        'random_weight_walker':  beta_walker,
        'random_weight_driver':  beta_driver,
        'steps':                 10000,
        'display':               False,
    }
    city_grid = gen_city(block_shape=(15, 15), city_shape=(5, 5),
                         street_parking=False, two_way=True, crop=0)
    parameters['city_grid'] = city_grid
    parameters['potholes']  = []

    # Fixed obstacle placement representing a street-market scenario
    rng = random.Random(seed)
    obstacles = [(16 + i, 28) for i in rng.sample(range(13), 8)]
    obstacles += [(31 + i, 28) for i in rng.sample(range(13), 7)]
    obstacles += [(16 + i, 31) for i in rng.sample(range(13), 5)]
    obstacles += [(31 + i, 31) for i in rng.sample(range(13), 8)]
    parameters['obstacles'] = obstacles

    nwalkers, ndrivers = 100, 50
    random.seed(seed)
    print('*** special_usecase  nw={} nd={} ***'.format(nwalkers, ndrivers))
    parameters['initial_walker_count'] = nwalkers
    parameters['initial_driver_count'] = ndrivers
    parameters['walker_spawn_function'] = lambda: walker_generator(nwalkers)
    parameters['driver_spawn_function'] = lambda: driver_generator(ndrivers)

    job_name = 'nw_{}_nd_{}'.format('{:03d}'.format(nwalkers),
                                     '{:03d}'.format(ndrivers))
    job_dir  = OUT_ROOT + '/exp_{}/{}'.format(exp_name, job_name)
    execute_job(job_dir, parameters, overwrite=overwrite)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    seed = 0
    random.seed(seed)

    wspawn = lambda n: max(0, np.random.poisson(n))
    dspawn = lambda n: max(0, np.random.poisson(n))

    # ── 1. Pedestrian-only obstacle sweep (replicates existing Fig. 1d) ──
    for beta_val in (1, 3, 5):
        walker_obstacles(
            beta    = lambda b=beta_val: b,
            seed    = seed,
            exp_name='nodrivers_seed_{}_beta_{}'.format(seed, beta_val))

    # ── 2. Full interaction grid (replicates existing Figs. 2, 3, 4) ─────
    exp_name_interact = 'interaction_seed_{}'.format(seed)
    driver_walker_interaction(
        beta_walker      = lambda: random.randint(1, 3),
        beta_driver      = lambda: random.randint(1, 5),
        walker_generator = wspawn,
        driver_generator = dspawn,
        seed             = seed,
        exp_name         = exp_name_interact)

    # ── 3. NEW: beta x rho grid (produces missing Section 5 figure) ──────
    exp_beta_rho_grid(seed=seed, overwrite=False)

    # ── 3b. Lane closures: same grid with one lane closed on 8 street
    #        segments, where wrong-way driving shortens the detour ────────
    exp_beta_rho_grid(seed=seed, exp_name='lane_closures', beta_values=(1, 3, 5),
                      closures=8)

    # ── 4. Use-case scenario (replicates Fig. 5) ─────────────────────────
    exp_name_uc = 'usecase_seed_{}'.format(seed)
    special_usecase(
        beta_walker      = lambda: random.randint(1, 3),
        beta_driver      = lambda: random.randint(1, 5),
        walker_generator = wspawn,
        driver_generator = dspawn,
        seed             = seed,
        exp_name         = exp_name_uc,
        overwrite        = True)