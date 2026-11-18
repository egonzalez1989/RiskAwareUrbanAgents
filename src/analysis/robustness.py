"""
robustness.py
Statistics and figures that test the robustness of the results. Run from src/analysis:

    python robustness.py [EXPERIMENTS_DIR] [OUT_DIR]

  Stationarity             Stationary share of driver time in the first and last 2000
                           steps of the long runs (E4, E5)
  Market runovers          E5: runovers per road cell in the market area / elsewhere
  sensitivity.png          E6: indicators with the risk values of Tables 2 and 4, with
                           log-uniform perturbations of every nonzero value, and with a
                           perception-reaction time of 0.6 s and 1.8 s instead of 1.2 s
"""
import os, re, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import PlotStyle

BASE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                       os.path.join(os.path.dirname(__file__), '..', '..', 'Experiments'))
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, 'analysis'))
os.makedirs(OUT, exist_ok=True)
PlotStyle.apply()

WINDOW = 250
VARIANTS = ['base', 'pert', 'tau2', 'tau6']        # E6 variants, reference first
# Street-market area of E5 (as in replication_stats.py)
in_market = lambda x, y: 14 <= x <= 46 and 26 <= y <= 33


def load(d, name):
    return json.load(open(os.path.join(d, name)))


def seed_of(d):
    return int(re.search(r'_seed_(\d+)', d).group(1))


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    lo, hi = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x)) if x.std() else (x.mean(), x.mean())
    return x.mean(), lo, hi


def paired(diff, label):
    m, lo, hi = mean_ci(diff)
    p = stats.wilcoxon(diff).pvalue if np.any(diff) else np.nan
    print(f'    {label:<44} {m:+7.2f}  95% CI [{lo:+6.2f}, {hi:+6.2f}]  Wilcoxon p={p:.3g}  runs={len(diff)}')


# ──────────────────────────────────────────────────────────────────────────────
# Intersection standstills over time (E4, E5)
# ──────────────────────────────────────────────────────────────────────────────
def stationary_series(d):
    m = load(d, 'metrics.txt')
    steps = sorted(m, key=int)
    stopped = np.array([m[t].get('stopped_drivers', 0) for t in steps], float)
    active = np.array([m[t].get('active_drivers', 0) for t in steps], float)
    n = len(steps) // WINDOW
    return np.array([stopped[i * WINDOW:(i + 1) * WINDOW].sum() / max(1, active[i * WINDOW:(i + 1) * WINDOW].sum())
                     for i in range(n)])


def standstills():
    series = {'E4, 20 drivers': glob.glob(f'{BASE}/exp_interaction_seed_*/obstacles_10_nw_200_nd_020'),
              'E4, 100 drivers': glob.glob(f'{BASE}/exp_interaction_seed_*/obstacles_10_nw_200_nd_100'),
              'E5, street market': glob.glob(f'{BASE}/exp_usecase_seed_*/*')}
    print('\n── Stationary share of driver time: first and last 2000 steps ──')
    for label, dirs in series.items():
        runs = [stationary_series(d) for d in sorted(dirs) if os.path.isfile(d + '/metrics.txt')]
        if not runs:
            continue
        runs = np.array(runs)
        if runs.shape[1] < 2 * 2000 // WINDOW:
            continue
        first, last = runs[:, :2000 // WINDOW].mean(1), runs[:, -2000 // WINDOW:].mean(1)
        m1, m2 = mean_ci(first), mean_ci(last)
        p = stats.wilcoxon(last - first).pvalue if np.any(last != first) else np.nan
        print(f'  {label:<18} first {m1[0]:.3f} [{m1[1]:.3f}, {m1[2]:.3f}]  last {m2[0]:.3f} [{m2[1]:.3f}, {m2[2]:.3f}]'
              f'  last - first {np.mean(last - first):+.3f} (Wilcoxon p={p:.3g}); max window {runs.max():.3f}; n={len(runs)}')


# ──────────────────────────────────────────────────────────────────────────────
# E5: runovers around the street market
# ──────────────────────────────────────────────────────────────────────────────
def market_runovers():
    print('\n── E5: runovers and collisions per road cell, market area / elsewhere ──')
    if not glob.glob(f'{BASE}/exp_usecase_seed_*/*/metrics.txt'):
        return
    for horizon in (2000, 10000):
        ratios, ev = [], np.zeros(4)
        for d in sorted(glob.glob(f'{BASE}/exp_usecase_seed_*/*')):
            grid = load(d, 'parameters.txt')['city_grid']
            road = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j][0] in 'rtlzp']
            nm = sum(in_market(*c) for c in road)
            ne = len(road) - nm
            e = np.zeros(4)     # runovers in / out, collisions in / out
            for t, v in load(d, 'metrics.txt').items():
                if int(t) < horizon:
                    for c in v['runovers']:
                        e[0 if in_market(*c) else 1] += 1
                    for c in v['collisions']:
                        e[2 if in_market(*c) else 3] += 1
            ratios.append(np.log(((e[0] + .5) / nm) / ((e[1] + .5) / ne)))
            ev += e
        m, lo, hi = mean_ci(ratios)
        print(f'  first {horizon} steps: {len(ratios)} runs, runovers in the market area {ev[0]:.0f} of {ev[:2].sum():.0f}'
              f' ({ev[0] / ev[:2].sum():.3f}; {nm / (nm + ne):.3f} of the road cells); runover rate ratio'
              f' {np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}], above 1 in {sum(r > 0 for r in ratios)} runs;'
              f' collision share {ev[2] / ev[2:].sum():.3f}')


# ──────────────────────────────────────────────────────────────────────────────
# E6: sensitivity to the risk values and to the perception-reaction time
# ──────────────────────────────────────────────────────────────────────────────
def grid_indicators(d):
    """PRER, collisions per driver, runovers per 1000 steps and wrong-way cells per 100 (planned, driven) of one run."""
    m, a = load(d, 'metrics.txt'), load(d, 'agents.txt')
    road = sum(sum(len(v) for v in m[t]['jaywalking'].values()) for t in m)
    walker_steps = sum(m[t].get('active_walkers', 0) for t in m)
    drivers = [x for x in a.values() if x['type'] == 'AstarDriver']
    coll = sum(len(m[t]['collisions']) for t in m) / max(1, len(drivers))
    trips = [x['trip'] for x in drivers if 'planned' in x['trip']]
    per100 = lambda k: 100 * sum(t[k].get('bw', 0) for t in trips) / max(1, sum(sum(t[k].values()) for t in trips))
    runovers = 1000 * sum(len(m[t]['runovers']) for t in m) / len(m)
    return {'PRER': road / max(1, walker_steps), 'collisions per driver': coll, 'runovers per 1000 steps': runovers,
            'wrong-way, planned': per100('planned'), 'wrong-way, driven': per100('executed')}


def sensitivity():
    rows = defaultdict(dict)         # (variant, rho) -> seed -> indicators
    overrides = {}
    for d in glob.glob(f'{BASE}/exp_sensitivity_seed_*/*/beta_3_rho_*'):
        if not os.path.isfile(os.path.join(d, 'agents.txt')):
            continue
        variant = d.split(os.sep)[-2]
        rho = float(d.rsplit('rho_', 1)[1].replace('p', '.'))
        rows[variant, rho][seed_of(d)] = grid_indicators(d)
        if variant == 'pert':
            overrides[seed_of(d)] = load(d, 'parameters.txt').get('risk_override')
    if not rows:
        return
    rhos = sorted({r for _, r in rows})
    keys = ['PRER', 'wrong-way, driven', 'collisions per driver', 'runovers per 1000 steps', 'wrong-way, planned']
    plotted = keys[:4]
    print('\n── E6: sensitivity (beta = 3) ──')
    for key in keys:
        print(f'  {key}')
        for rho in rhos:
            b = rows.get(('base', rho), {})
            for variant in VARIANTS[1:]:
                p = rows.get((variant, rho), {})
                seeds = sorted(set(b) & set(p))
                if not seeds:
                    continue
                bv = np.array([b[s][key] for s in seeds])
                pv = np.array([p[s][key] for s in seeds])
                mb, mp = mean_ci(bv), mean_ci(pv)
                d = pv - bv
                pw = stats.wilcoxon(d).pvalue if np.any(d) else np.nan
                print(f'    rho={rho:<5} base {mb[0]:.3f} [{mb[1]:.3f}, {mb[2]:.3f}]   {variant} {mp[0]:.3f}'
                      f' [{mp[1]:.3f}, {mp[2]:.3f}]   paired diff {d.mean():+.4f} (Wilcoxon p={pw:.3g})  n={len(seeds)}')
        # Is the ordering in rho preserved within every seed?
        for variant in VARIANTS:
            seeds = sorted(set.intersection(*[set(rows.get((variant, r), {})) for r in rhos]))
            if seeds:
                v = np.array([[rows[variant, r][s][key] for r in rhos] for s in seeds])
                mono = np.mean([np.all(np.diff(x) < 0) for x in v])
                print(f'    {variant}: decreasing in rho in {mono:.0%} of {len(seeds)} seeds;'
                      f' rho 0.25 -> 5 ratio {np.mean(v[:, -1]) / np.mean(v[:, 0]):.3f}')
    # Which perturbed values matter: rank correlation of each log factor with the indicator
    base = {'walk': {'z': 1, 'r': 5, 't': 10}, 'drive': {'rt': 1, 'lt': 3, 'ch': 3, 'ft': 5, 'bw': 15}}
    names = [(k, c) for k in base for c in base[k]]
    print('  Spearman correlation of log perturbation factors with the indicator (perturbed runs)')
    for key in plotted:
        for rho in rhos:
            p = rows.get(('pert', rho), {})
            seeds = sorted(s for s in p if overrides.get(s))
            if len(seeds) < 5:
                continue
            y = [p[s][key] for s in seeds]
            cors = []
            for k, c in names:
                f = [np.log2(overrides[s][k][c] / base[k][c]) for s in seeds]
                r, pv = stats.spearmanr(f, y)
                cors.append(f'{c}:{r:+.2f}{"*" if pv < .05 else ""}')
            print(f'    {key:<22} rho={rho:<5} ' + ' '.join(cors))
    # Figure: base and perturbed indicators by rho (mean, 95% interval and single runs)
    fig, axes = plt.subplots(2, 2, figsize=(PlotStyle.TEXT_W, 4.4), constrained_layout=True)
    axes = axes.ravel()
    labels = {'base': 'Reference', 'pert': 'Perturbed risk values',
              'tau2': 'Reaction time 0.6 s', 'tau6': 'Reaction time 1.8 s'}
    width = .6 / len(VARIANTS)
    for ax, key in zip(axes, plotted):
        for i, variant in enumerate(VARIANTS):
            label, color = labels[variant], PlotStyle.CATEGORICAL[i]
            for j, rho in enumerate(rhos):
                v = np.array(list(rows.get((variant, rho), {}).values()))
                if not len(v):
                    continue
                y = np.array([x[key] for x in v])
                x = j + (i - (len(VARIANTS) - 1) / 2) * width
                ax.scatter(np.full(len(y), x) + np.random.default_rng(j).uniform(-.06, .06, len(y)), y,
                           s=5, color=color, alpha=.35, lw=0)
                m, lo, hi = mean_ci(y)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='o', color=color, ms=4, capsize=2,
                            label=label if j == 0 else None)
        ax.set_xticks(range(len(rhos)), [f'{r:g}' for r in rhos])
        ax.set_xlabel(r'Risk sensitivity $\rho$')
        ax.set_title(key[0].upper() + key[1:] + (' (per 100 cells)' if 'wrong' in key else ''), loc='left')
        ax.set_ylim(bottom=0)
        ax.grid(axis='x', visible=False)
    axes[0].legend(fontsize=7, loc='upper right')
    PlotStyle.save(fig, os.path.join(OUT, 'sensitivity.png'))


if __name__ == '__main__':
    print(f'Experiments: {BASE}\nFigures: {OUT}')
    standstills()
    market_runovers()
    sensitivity()
