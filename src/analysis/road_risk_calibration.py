"""
Choice of the road risk of walkers (Section 4, Fig. calibration_road_risk).

Walkers only, on a 2 x 3 grid of 15 x 12 blocks, travelling between start-goal
pairs stratified by detour factor D = L_safe / L_short (10 pairs in each of
slight 1 < D < 1.5, moderate 1.5 <= D < 2.5 and strong D >= 2.5; pairs with
D = 1 offer no trade-off between length and risk and are left out), so that
every walker faces a real choice between a shorter road route and a longer
legal one.

  (a) PRER against the road risk r (intersection risk 2r), beta = rho = 1,
      no obstacles.
  (b) PRER over the (beta, rho) grid at r = 5 with 5% of the sidewalk cells
      obstructed.

Each configuration is run for the seeds given; the pair sample and the
obstacle layout are fixed.

    cd src
    PYTHONPATH=. python -m analysis.road_risk_calibration run   [OUT_ROOT] [SEEDS]
    PYTHONPATH=. python -m analysis.road_risk_calibration plot  [OUT_ROOT] [FIG_DIR]
"""
import os, sys, glob, re, random, itertools
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.AgentUtils import gen_city, gen_obstacles
from utils.Calibration import build_pair_catalog, stratified_sample
from analysis.Experiments import execute_job

PAIR_SEED   = 42
BLOCK_SHAPE = (15, 12)
CITY_SHAPE  = (2, 3)
OBS_RATE    = 0.05
STEPS       = 1500          # long enough for every walker to arrive
ROAD_RISKS  = range(1, 11)
BETAS       = (1, 2, 3, 5)
RHOS        = (0.5, 1, 2, 5)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), '../..', 'Experiments', 'exp_road_risk_detour')


def setting():
    """Grid, stratified pair sample and 5% obstacle layout (fixed by PAIR_SEED)."""
    random.seed(PAIR_SEED); np.random.seed(PAIR_SEED)
    grid = gen_city(BLOCK_SHAPE, CITY_SHAPE, street_parking=False, nlanes=1)
    obstacles, _ = gen_obstacles(grid, OBS_RATE, 0.0)
    pairs = stratified_sample(build_pair_catalog(grid, min_dist=4, seed=PAIR_SEED),
                              n_per_bucket=10, seed=PAIR_SEED)
    return grid, pairs, sorted(obstacles)


def job(args):
    job_dir, grid, pairs, obstacles, r, beta, rho, seed = args
    walkers = [{'start': tuple(int(x) for x in p['start']),
                'goal':  tuple(int(x) for x in p['goal']),
                'beta': beta, 'rho': rho} for p in pairs]
    params = {'seed': seed, 'debug': False, 'display': False, 'steps': STEPS,
              'city_grid': grid, 'obstacles': list(obstacles), 'potholes': [],
              'walkers': walkers, 'repopulate': False,
              'walker_risk_map': {'r': r, 't': 2 * r}}
    execute_job(job_dir, params)
    return job_dir


def run(out_root=DEFAULT_OUT, seeds=range(5), processes=7):
    grid, pairs, obstacles = setting()
    print('pairs by bucket:', {b: sum(p['bucket'] == b for p in pairs) for b in ('slight', 'moderate', 'strong')})
    jobs = []
    for seed in seeds:
        for r in ROAD_RISKS:
            jobs.append((f'{out_root}/a_seed_{seed}/r_{r}', grid, pairs, [], r, 1, 1, seed))
        for beta, rho in itertools.product(BETAS, RHOS):
            jobs.append((f'{out_root}/b_seed_{seed}/beta_{beta}_rho_{rho}', grid, pairs, obstacles, 5, beta, rho, seed))
    with Pool(processes) as pool:
        for d in pool.imap_unordered(job, jobs):
            print('done', d, flush=True)


def plot(out_root=DEFAULT_OUT, fig_dir=None):
    from analysis.analysis import load_job, PlotStyle, mean_band
    fig_dir = fig_dir or os.path.join(out_root, '..', 'analysis')
    os.makedirs(fig_dir, exist_ok=True)
    a = {}
    for d in glob.glob(f'{out_root}/a_seed_*/r_*'):
        if os.path.exists(f'{d}/metrics.txt'):
            seed, r = map(int, re.search(r'a_seed_(\d+)/r_(\d+)$', d).groups())
            a.setdefault(seed, {})[r] = load_job(d)['prer']
    b = {}
    for d in glob.glob(f'{out_root}/b_seed_*/beta_*_rho_*'):
        if os.path.exists(f'{d}/metrics.txt'):
            m = re.search(r'beta_([\d.]+)_rho_([\d.]+)$', d)
            b.setdefault((float(m[1]), float(m[2])), []).append(load_job(d)['prer'])

    seeds = sorted(s for s in a if len(a[s]) == len(ROAD_RISKS))
    risks = list(ROAD_RISKS)
    fig, axes = plt.subplots(1, 2, figsize=(PlotStyle.TEXT_W, 2.6), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    mean_band(ax, risks, [[a[s][r] for r in risks] for s in seeds], PlotStyle.ordinal_colors(1, 'viridis', .3, .3)[0], None)
    ax.axvline(5, color='gray', lw=.8, ls=':')
    ax.set_xlabel('Road risk of walkers')
    ax.set_ylabel('PRER')
    ax.set_ylim(bottom=0)
    ax.set_xticks(risks)
    ax.grid(axis='x', visible=False)
    ax.set_title('(a)', fontsize=8)

    ax = axes[1]
    mat = np.array([[np.mean(b.get((float(beta), float(rho)), [np.nan])) for beta in BETAS] for rho in RHOS])
    im = ax.imshow(mat, cmap=PlotStyle.CMAP_RATE, origin='lower', aspect='auto', vmin=0)
    for i, j in itertools.product(range(len(RHOS)), range(len(BETAS))):
        v = mat[i, j]
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7,
                color='white' if v > .6 * np.nanmax(mat) else 'black')
    ax.set_xticks(range(len(BETAS)), [str(x) for x in BETAS])
    ax.set_yticks(range(len(RHOS)), [f'{x:g}' for x in RHOS])
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\rho$')
    ax.grid(False)
    ax.set_title('(b)', fontsize=8)
    fig.colorbar(im, ax=ax, label='PRER', shrink=.9)
    PlotStyle.save(fig, os.path.join(fig_dir, 'prer_road_risk_detour.png'))

    print(f'(a) n={len(seeds)} seeds: ' + ' '.join(
        f'r{r}={np.mean([a[s][r] for s in seeds]):.3f}' for r in risks))
    print('(a) sd at r=5: %.4f' % np.std([a[s][5] for s in seeds]))
    print('(b) rows rho, cols beta', BETAS)
    for rho, row in zip(RHOS, mat):
        print(f'   rho={rho:<4}', ' '.join(f'{v:.3f}' for v in row),
              ' n=', [len(b.get((float(beta), float(rho)), [])) for beta in BETAS])


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    out_root = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.abspath(DEFAULT_OUT)
    if cmd == 'run':
        seeds = range(int(sys.argv[3])) if len(sys.argv) > 3 else range(5)
        run(out_root, seeds)
    else:
        plot(out_root, sys.argv[3] if len(sys.argv) > 3 else None)
