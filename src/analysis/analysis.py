"""
analysis.py
Discovers all experiment job directories under ../../Experiments/ (or a path
given as argv[1]), loads per-job metrics, and produces:
  - A printed findings report (statistics for each experiment group)
  - The metric figures used in the paper, saved to <base>/analysis/
    (or to a directory given as argv[2])

Figures (one per paper panel, named after the files in Documentation/images):
  prer_vs_obstacles.png        PRER vs sidewalk obstruction, one line per beta
                               (exp_nodrivers_seed_S_beta_B; mean and seed range)
  jaywalking_map_oNN.png       Road cells used by walkers at NN% obstruction,
                               one panel per beta, shared color scale
  interaction_speed.png        Mean driver speed vs number of drivers,
  interaction_collisions.png   collisions per driver and runovers per walker;
  interaction_runovers.png     one panel per obstruction level, one line per
                               walker population (exp_interaction_seed_S)
  prer_heatmap.png             PRER, runovers per walker and collisions per
  runovers_heatmap.png         driver across the (beta, rho) grid, with values
  collisions_heatmap.png       written in each cell (exp_beta_rho_grid)
  prer_lines.png               PRER vs beta, one line per rho
  driver_speed_heatmap.png,    Driver indicators across the (beta, rho) grid:
  moving_speed_heatmap.png,    speed, moving speed, stationary share, replans
  wrongway_*_heatmap.png, ...  and wrong-way cells (rho > 0); lc_* for the
                               lane-closure grid (exp_lane_closures_seed_S)
  lc_wrongway_vs_rho.png       Wrong-way cells vs rho with and without closures
  prer_road_risk.png           PRER vs road risk of walkers (choice of the value)
  uc_heatmap_*.png             Use-case spatial maps: walker and driver counts,
                               mean driver speed and jaywalking (exp_usecase_seed_S)
"""
import os, re, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import PlotStyle
from utils.Reporting import urban_heatmap, heatmap_panels

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '../..', 'Experiments')
BASE_DIR    = os.path.abspath(BASE_DIR)
OUT_DIR     = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

MAP_DENSITY = 5     # obstruction level (%) shown in the jaywalking maps
MIN_SPEED_VISITS = 3    # drivers a cell needs before its mean speed is drawn

PlotStyle.apply()

# ──────────────────────────────────────────────────────────────────────────────
# Helper: load one job directory
# ──────────────────────────────────────────────────────────────────────────────
def load_job(job_dir):
    """Return a dict with pre-computed scalars for one job, or None on error."""
    mfile = os.path.join(job_dir, 'metrics.txt')
    afile = os.path.join(job_dir, 'agents.txt')
    pfile = os.path.join(job_dir, 'parameters.txt')
    if not (os.path.isfile(mfile) and os.path.isfile(afile)):
        return None
    try:
        metrics = json.load(open(mfile))
        agents  = json.load(open(afile))
    except Exception as e:
        print(f'  [WARN] Could not load {job_dir}: {e}')
        return None

    # ── agent counts ──────────────────────────────────────────────────────
    num_drivers = sum(1 for v in agents.values() if v.get('type') == 'AstarDriver')
    num_walkers = sum(1 for v in agents.values() if v.get('type') == 'AstarWalker')

    # ── collision / runover rates ──────────────────────────────────────────
    total_collisions = sum(len(metrics[t].get('collisions', [])) for t in metrics)
    total_runovers   = sum(len(metrics[t].get('runovers',   [])) for t in metrics)
    collision_rate   = total_collisions / max(num_drivers, 1)
    runover_rate     = total_runovers   / max(num_walkers, 1)

    # ── PRER (eq:prer): walker-steps on pure road cells / active walker-steps
    # metrics[t]['jaywalking'] maps each walker on a road cell at step t to its cell
    jw_union = set()
    road_steps = 0
    for t in metrics:
        jw = metrics[t].get('jaywalking', {})
        jw_union |= set(jw.keys())
        road_steps += sum(len(v) for v in jw.values())
    walker_steps = sum(metrics[t].get('active_walkers', 0) for t in metrics)
    prer    = road_steps / max(walker_steps, 1)
    jw_rate = len(jw_union) / max(num_walkers, 1)     # walkers that jaywalked at least once

    # ── average driver speed (only steps with drivers) ───────────────────
    spd_vals = [metrics[t].get('avg_speed_drivers') for t in metrics
                if metrics[t].get('active_drivers', 0) > 0]
    spd_vals = [v for v in spd_vals if v is not None and not np.isnan(v)]
    avg_spd  = float(np.mean(spd_vals)) if spd_vals else np.nan

    # ── stationary time: share of active driver-steps at speed 0 (runs made
    # before 'stopped_drivers' was recorded give nan) ──────────────────────
    active_d = sum(metrics[t].get('active_drivers', 0) for t in metrics)
    if active_d and all('stopped_drivers' in metrics[t] for t in metrics):
        stopped_share = sum(metrics[t]['stopped_drivers'] for t in metrics) / active_d
    else:
        stopped_share = np.nan

    # ── trips completed per driver spawned ──────────────────────────────
    driver_ids = {k for k, v in agents.items() if v.get('type') == 'AstarDriver'}
    driver_arrivals = sum(1 for t in metrics for aid in metrics[t].get('arrivals', {})
                          if aid in driver_ids)
    trips_done = driver_arrivals / num_drivers if num_drivers else np.nan

    # ── driver maneuvers (runs that store the trip record): left turns ('lt'),
    # invalid turns ('ft') and wrong-way moves ('bw') per 100 cells, in the route
    # planned at spawn and in the cells actually driven; replans per driver ───
    trips = [v['trip'] for v in agents.values()
             if v.get('type') == 'AstarDriver' and 'trip' in v]
    def per100(kind, moves):
        cells = sum(sum(t[kind].values()) for t in trips)
        return 100 * sum(t[kind].get(m, 0) for t in trips for m in moves) / cells if cells else np.nan
    ft_plan, ww_plan = per100('planned', ['ft']), per100('planned', ['bw'])
    ft_exec, ww_exec = per100('executed', ['ft']), per100('executed', ['bw'])
    lt_plan, lt_exec = per100('planned', ['lt']), per100('executed', ['lt'])
    replans = np.mean([t['replans'] for t in trips]) if trips else np.nan
    # Mean speed of the drivers that are moving (stationary ones left out)
    mv = [(metrics[t]['avg_speed_drivers'] * metrics[t]['active_drivers'],
           metrics[t]['active_drivers'] - metrics[t]['stopped_drivers'])
          for t in metrics if 'stopped_drivers' in metrics[t]]
    moving_spd = sum(a for a, _ in mv) / sum(n for _, n in mv) if mv and sum(n for _, n in mv) else np.nan

    # ── walkers: share of active walker-steps stopped, completed trips per
    # walker spawned and replans per walker (runs that record them) ─────────
    active_w = sum(metrics[t].get('active_walkers', 0) for t in metrics)
    w_stopped = (sum(metrics[t]['stopped_walkers'] for t in metrics) / active_w
                 if active_w and all('stopped_walkers' in metrics[t] for t in metrics) else np.nan)
    walker_ids = {k for k, v in agents.items() if v.get('type') == 'AstarWalker'}
    w_done = (sum(1 for t in metrics for aid in metrics[t].get('arrivals', {}) if aid in walker_ids)
              / num_walkers if num_walkers else np.nan)
    w_trips = [agents[k]['trip'] for k in walker_ids if 'trip' in agents[k]]
    w_replans = np.mean([t['replans'] for t in w_trips]) if w_trips else np.nan

    # ── city layout (for spatial maps) and jaywalking map ────────────────
    city = None
    jw_map = None
    if os.path.isfile(pfile):
        try:
            raw = json.load(open(pfile))
            grid = np.array(raw['city_grid'])
            city = SimpleNamespace(city_grid=grid, shape=grid.shape,
                                   obstacles=[tuple(o) for o in raw.get('obstacles', [])],
                                   potholes=[tuple(p) for p in raw.get('potholes', [])])
            # Distinct walkers that used each road cell
            cells = defaultdict(set)
            for t in metrics:
                for aid, positions in metrics[t].get('jaywalking', {}).items():
                    for p in positions:
                        cells[tuple(p)].add(aid)
            jw_map = np.zeros(grid.shape)
            for p, ids in cells.items():
                jw_map[p] = len(ids)
        except Exception as e:
            print(f'  [WARN] Could not read layout of {job_dir}: {e}')

    return {
        'dir':            job_dir,
        'name':           os.path.basename(job_dir),
        'num_drivers':    num_drivers,
        'num_walkers':    num_walkers,
        'collision_rate': collision_rate,
        'runover_rate':   runover_rate,
        'jw_rate':        jw_rate,
        'prer':           prer,
        'avg_spd_driver': avg_spd,
        'stopped_share':  stopped_share,
        'trips_done':     trips_done,
        'ft_planned':     ft_plan,
        'lt_planned':     lt_plan,
        'lt_executed':    lt_exec,
        'ww_planned':     ww_plan,
        'ft_executed':    ft_exec,
        'ww_executed':    ww_exec,
        'replans':        replans,
        'moving_spd':     moving_spd,
        'w_stopped':      w_stopped,
        'w_done':         w_done,
        'w_replans':      w_replans,
        'city':           city,
        'jw_map':         jw_map,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Discover experiment groups
# ──────────────────────────────────────────────────────────────────────────────
def discover_experiments(base_dir):
    """
    Returns {exp_group_name: [job_data, ...]} sorted by job name.
    A job is any sub-directory containing metrics.txt.
    An exp-group is the immediate parent of job dirs.
    """
    groups = defaultdict(list)

    # Collect all metrics.txt files two levels deep (exp_*/job_*/metrics.txt)
    pattern2 = os.path.join(base_dir, '*', '*', 'metrics.txt')
    pattern1 = os.path.join(base_dir, '*', 'metrics.txt')

    found = glob.glob(pattern2) + glob.glob(pattern1)
    for mpath in found:
        job_dir   = os.path.dirname(mpath)
        group_dir = os.path.dirname(job_dir)
        # If job IS directly in base, group = base_dir
        if os.path.normpath(group_dir) == os.path.normpath(base_dir):
            group_name = '(root)'
        else:
            group_name = os.path.basename(group_dir)
        data = load_job(job_dir)
        if data:
            groups[group_name].append(data)

    # Sort jobs within each group
    for g in groups:
        groups[g].sort(key=lambda d: d['name'])
    return dict(groups)


def num(s):
    """Parse a number written in a directory name ('0p5' -> 0.5)."""
    v = float(s.replace('p', '.'))
    return int(v) if v.is_integer() else v


def groups_matching(groups, pattern):
    """[(match, jobs)] for the groups whose name matches pattern."""
    return [(m, jobs) for g, jobs in sorted(groups.items())
            for m in [re.fullmatch(pattern, g)] if m]

# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def out(filename):
    return os.path.join(OUT_DIR, filename)


def mean_band(ax, xs, runs, color, label, marker='o', ls='-'):
    """Mean over seeds as a line; seed min-max as a band when there are several."""
    runs = np.array(runs, dtype=float)
    mean = np.nanmean(runs, axis=0)
    ax.plot(xs, mean, marker=marker, ls=ls, color=color, label=label)
    if len(runs) > 1:
        ax.fill_between(xs, np.nanmin(runs, axis=0), np.nanmax(runs, axis=0),
                        color=color, alpha=.18, linewidth=0)


def annotated_grid(mat, betas, rhos, label, filename, fmt='{:.3f}', vmin=0, cmap=PlotStyle.CMAP_RATE):
    """Heatmap over the (beta, rho) grid with each value written in its cell;
    vmin=None scales the colors to the data range instead of starting at 0."""
    fig, ax = plt.subplots(figsize=(PlotStyle.COLUMN_W, PlotStyle.COLUMN_W * .8))
    vmax = np.nanmax(mat) if np.nanmax(mat) > 0 else 1
    if vmin is None:
        vmin = np.nanmin(mat)
    im = ax.imshow(mat, cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, label=label)
    cb.outline.set_visible(False)
    ax.set_xticks(range(len(betas)), [str(b) for b in betas])
    ax.set_yticks(range(len(rhos)), [str(r) for r in rhos])
    ax.set_xlabel(r'Heuristic weight $\beta$')
    ax.set_ylabel(r'Risk sensitivity $\rho$')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(rhos)):
        for j in range(len(betas)):
            if not np.isnan(mat[i, j]):
                r, g, b, _ = im.cmap(im.norm(mat[i, j]))     # dark cells get white text
                ax.text(j, i, fmt.format(mat[i, j]), ha='center', va='center', fontsize=7.5,
                        color='white' if .299 * r + .587 * g + .114 * b < .5 else 'black')
    PlotStyle.save(fig, out(filename))

# ──────────────────────────────────────────────────────────────────────────────
# Experiment-specific analysis functions
# ──────────────────────────────────────────────────────────────────────────────
def analyse_obstacle_sweeps(groups):
    """Pedestrian-only sweeps (exp_nodrivers_seed_S_beta_B): PRER vs sidewalk
    obstruction with one line per beta, and road-use maps at MAP_DENSITY."""
    by_beta = defaultdict(list)          # beta -> [{density: job}] (one per seed)
    for m, jobs in groups_matching(groups, r'exp_nodrivers_seed_(\d+)_beta_([\dp.]+)'):
        sweep = {int(mm.group(1)): j for j in jobs
                 for mm in [re.fullmatch(r'obstacles_(\d+)', j['name'])] if mm}
        if sweep:
            by_beta[num(m.group(2))].append(sweep)
    if not by_beta:
        return

    # Two panels on a shared x axis: time spent on the road (PRER) and share of
    # walkers that used the road at least once; beta from cautious to reckless
    betas = sorted(by_beta)
    colors = (PlotStyle.CAUTIOUS_TO_RECKLESS if len(betas) <= 3 else PlotStyle.CATEGORICAL)
    fig, axes = plt.subplots(1, 2, sharex=True, constrained_layout=True,
                             figsize=(PlotStyle.TEXT_W * .85, 2.5))
    fig.get_layout_engine().set(wspace=.15)
    panels = [('prer', 1, 'PRER', 'Fraction of time spent on road cells'),
              ('jw_rate', 100, 'Pedestrians (%)', 'Proportion of pedestrians jaywalking')]
    for ax, (key, scale, ylabel, title) in zip(axes, panels):
        # Line styles repeat the color order so the curves stay distinct in grayscale print
        for beta, c, ls in zip(betas, colors, ['-', '--', '-.', ':']):
            densities = sorted(set.intersection(*[set(s) for s in by_beta[beta]]))
            runs = [[scale * s[d][key] for d in densities] for s in by_beta[beta]]
            mean_band(ax, densities, runs, c, rf'$\beta = {beta}$', marker=None, ls=ls)
        ax.set_title(title, loc='left')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max(max(s) for b in betas for s in by_beta[b]))
        ax.set_ylim(bottom=0)
        ax.grid(axis='x', visible=False)
    axes[1].set_ylim(0, 100)
    fig.supxlabel('Sidewalk obstruction (% of sidewalk cells)', fontsize=9)
    axes[0].legend(loc='lower right', handlelength=1.5)
    PlotStyle.save(fig, out('prer_vs_obstacles.png'))

    # Road cells used by walkers, one panel per beta (first seed), same scale
    panels = [(b, by_beta[b][0].get(MAP_DENSITY)) for b in betas]
    panels = [(b, j) for b, j in panels if j and j['jw_map'] is not None]
    if panels:
        maps = [j['jw_map'] for _, j in panels]
        nz = np.concatenate([m[m > 0] for m in maps])
        heatmap_panels(panels[0][1]['city'], maps, [rf'$\beta$={b}' for b, _ in panels],
                       'Walkers using the road cell', out(f'jaywalking_map_o{MAP_DENSITY}.png'),
                       cities=[j['city'] for _, j in panels],
                       vmax=np.percentile(nz, 99) if nz.size else None)


def analyse_road_risk(base_dir=None, seeds=None):
    """Choice of the road risk of walkers (jobs R_<risk>): PRER against the road
    risk for beta = rho = 1, one line per sidewalk obstruction level; lines are
    means over seeds and bands span the seeds."""
    base_dir = base_dir or BASE_DIR
    runs = defaultdict(lambda: defaultdict(dict))     # level -> seed -> {risk: prer}
    for mfile in glob.glob(os.path.join(base_dir, 'exp_road_risk_seed_*', 'r_*', 'obstacles_*', 'metrics.txt')):
        job_dir = os.path.dirname(mfile)
        seed = int(re.search(r'exp_road_risk_seed_(\d+)', job_dir).group(1))
        if seeds is not None and seed not in seeds:
            continue
        risk = num(re.search(r'r_([\dp.]+)', job_dir).group(1))
        level = int(re.search(r'obstacles_(\d+)', job_dir).group(1))
        job = load_job(job_dir)
        if job:
            runs[level][seed][risk] = job['prer']
    if not runs:
        return
    fig, ax = plt.subplots(constrained_layout=True, figsize=(PlotStyle.COLUMN_W, 2.4))
    levels = sorted(runs)
    for level, c, ls in zip(levels, PlotStyle.ordinal_colors(len(levels), 'viridis', .15, .75), ['-', '--', '-.']):
        complete = [r for r in runs[level].values() if len(r) == max(len(v) for v in runs[level].values())]
        risks = sorted(complete[0])
        mean_band(ax, risks, [[r[k] for k in risks] for r in complete], c, f'{level}%', ls=ls)
    ax.axvline(5, color='gray', lw=.8, ls=':')
    ax.set_xlabel('Road risk of walkers')
    ax.set_ylabel('PRER')
    ax.set_ylim(bottom=0)
    ax.grid(axis='x', visible=False)
    ax.legend(title='Sidewalk obstruction', fontsize=7, title_fontsize=7)
    PlotStyle.save(fig, out('prer_road_risk.png'))
    for level in levels:
        complete = list(runs[level].values())
        risks = sorted(complete[0])
        print(f'  road risk, obstruction {level}%: n={len(complete)} ' +
              ' '.join(f'r{k}={np.mean([r[k] for r in complete if k in r]):.3f}' for k in risks))


def analyse_interaction(groups):
    """Interaction sweeps (exp_interaction_seed_S, jobs obstacles_O_nw_W_nd_D):
    driver speed, collisions per driver and runovers per walker vs number of
    drivers; one panel per obstruction level, one line per walker population."""
    runs = defaultdict(list)             # (o, nw, nd) -> [job] over seeds
    for _, jobs in groups_matching(groups, r'exp_interaction_seed_(\d+)'):
        for j in jobs:
            mm = re.fullmatch(r'obstacles_(\d+)_nw_(\d+)_nd_(\d+)', j['name'])
            if mm:
                runs[tuple(int(x) for x in mm.groups())].append(j)
    if not runs:
        return

    obst = sorted({k[0] for k in runs})
    nws  = sorted({k[1] for k in runs})
    colors = dict(zip(nws, PlotStyle.ordinal_colors(len(nws))))

    for key, ylabel, fname, skip_nw0 in [
            ('avg_spd_driver', 'Mean driver speed (km/h)', 'interaction_speed.png', False),
            ('collision_rate', 'Collisions per driver',           'interaction_collisions.png', False),
            ('runover_rate',   'Runovers per walker',             'interaction_runovers.png', True)]:
        fig, axes = plt.subplots(1, len(obst), sharey=True, squeeze=False,
                                 figsize=(min(PlotStyle.TEXT_W, 3.2 * len(obst)), 2.6),
                                 constrained_layout=True)
        for ax, o in zip(axes[0], obst):
            for nw in nws:
                if skip_nw0 and nw == 0:
                    continue
                nds = sorted(k[2] for k in runs if k[0] == o and k[1] == nw)
                if not nds:
                    continue
                n_seeds = min(len(runs[(o, nw, d)]) for d in nds)
                vals = [[runs[(o, nw, d)][s][key] for d in nds] for s in range(n_seeds)]
                mean_band(ax, nds, vals, colors[nw], str(nw))
            ax.set_title(f'{o}% sidewalk obstruction')
            ax.set_xlabel('Number of drivers')
            ax.set_ylim(bottom=0)
        axes[0][0].set_ylabel(ylabel)
        axes[0][-1].legend(title='Walkers', loc='best')
        PlotStyle.save(fig, out(fname))


def load_grid(groups, pattern):
    """{(beta, rho): [job]} for the groups matching pattern (jobs beta_B_rho_R)."""
    grid = {}
    for _, jobs in groups_matching(groups, pattern):
        for j in jobs:
            mm = re.fullmatch(r'(?:beta|w)_([\dp.]+)_rho_([\dp.]+)', j['name'])
            if mm:
                grid.setdefault((num(mm.group(1)), num(mm.group(2))), []).append(j)
    return grid


# Driver-side heatmaps over the (beta, rho) grid: file suffix, key, label,
# format, color-scale minimum (None: data range). Speed maps (keys containing
# 'spd') use the red-green speed colormap
DRIVER_GRID_MAPS = [
    ('driver_speed',  'avg_spd_driver', 'Mean driver speed (km/h)',   '{:.1f}', None),
    ('moving_speed',  'moving_spd',     'Mean speed when moving (km/h)', '{:.1f}', None),
    ('stopped_share', 'stopped_share',  'Stationary share of driver time',  '{:.2f}', None),
    ('replans',       'replans',        'Replans per driver',               '{:.1f}', 0),
    ('wrongway_plan', 'ww_planned',     'Wrong-way cells per 100 (planned)', '{:.2f}', 0),
    ('wrongway_exec', 'ww_executed',    'Wrong-way cells per 100 (driven)',  '{:.2f}', 0),
]
# At rho = 0 about half of all cells are driven the wrong way, which would
# compress the color scale of the other rows: these maps show rho > 0 only
POSITIVE_RHO_KEYS = {'ww_planned', 'ww_executed'}


def grid_cmap(key):
    return PlotStyle.CMAP_SPEED if 'spd' in key else PlotStyle.CMAP_RATE


def driver_grid_maps(matrix, betas, rhos, prefix=''):
    """Annotated heatmaps of DRIVER_GRID_MAPS; matrix(key, rhos) -> values."""
    for suffix, key, label, fmt, vmin in DRIVER_GRID_MAPS:
        rows = [r for r in rhos if r > 0] if key in POSITIVE_RHO_KEYS else rhos
        mat = matrix(key, rows)
        if not np.all(np.isnan(mat)):
            annotated_grid(mat, betas, rows, label, f'{prefix}{suffix}_heatmap.png',
                           fmt=fmt, vmin=vmin, cmap=grid_cmap(key))


def grid_matrix(grid, betas, rhos, key):
    mat = np.full((len(rhos), len(betas)), np.nan)
    for i, r in enumerate(rhos):
        for j, b in enumerate(betas):
            if (b, r) in grid:
                mat[i, j] = np.mean([job[key] for job in grid[(b, r)]])
    return mat


def analyse_lane_closures(groups):
    """Lane-closure grid (exp_lane_closures): driver heatmaps as for the
    (beta, rho) grid, and wrong-way driving vs rho with and without closures."""
    lc = load_grid(groups, r'exp_lane_closures.*')
    if not lc:
        return
    base = load_grid(groups, r'exp_beta_rho_grid.*')
    betas = sorted({k[0] for k in lc})
    rhos  = sorted({k[1] for k in lc})
    driver_grid_maps(lambda key, rows: grid_matrix(lc, betas, rows, key), betas, rhos, prefix='lc_')

    # Wrong-way cells per 100 vs rho > 0 (at rho = 0 about half of all cells
    # are wrong-way and would hide the rest): one color per beta, closures
    # solid and the same configuration without closures dashed; error bars
    # span the seeds. Thin lines and offsets keep the six series apart
    rhos_pos = [r for r in rhos if r > 0]
    fig, axes = plt.subplots(1, 2, constrained_layout=True,
                             figsize=(PlotStyle.TEXT_W * .85, 2.5))
    colors = PlotStyle.CAUTIOUS_TO_RECKLESS if len(betas) <= 3 else PlotStyle.CATEGORICAL
    xs = np.arange(len(rhos_pos))

    def series(grid, b, key):
        vals = [[j[key] for j in grid.get((b, r), [])] for r in rhos_pos]
        mean = np.array([np.mean(v) if v else np.nan for v in vals])
        lo = np.array([mean[i] - min(v) if v else np.nan for i, v in enumerate(vals)])
        hi = np.array([max(v) - mean[i] if v else np.nan for i, v in enumerate(vals)])
        return mean, [lo, hi]

    style = dict(lw=.9, ms=3, capsize=1.5, elinewidth=.6, capthick=.6)
    for ax, (key, title) in zip(axes, [('ww_planned', 'Planned routes'),
                                       ('ww_executed', 'Driven cells')]):
        for k, (b, c) in enumerate(zip(betas, colors)):
            off = (k - (len(betas) - 1) / 2) * .1
            m, err = series(lc, b, key)
            ax.errorbar(xs + off - .02, m, yerr=err, color=c, marker='o', **style,
                        label=rf'$\beta = {b}$, closures')
            m, err = series(base, b, key)
            ax.errorbar(xs + off + .02, m, yerr=err, color=c, marker='o', ls='--', mfc='white',
                        **style, label=rf'$\beta = {b}$, no closures')
        ax.set_xticks(list(xs), [str(r) for r in rhos_pos])
        ax.set_title(title, loc='left')
        ax.set_ylim(bottom=0)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Wrong-way cells per 100')
    fig.supxlabel(r'Risk sensitivity $\rho$', fontsize=9)
    axes[0].legend(fontsize=7, handlelength=2.2)
    PlotStyle.save(fig, out('lc_wrongway_vs_rho.png'))


def analyse_beta_rho_grid(groups):
    """(beta, rho) sweep (exp_beta_rho_grid, jobs beta_B_rho_R): annotated
    heatmaps of PRER, runovers, collisions and the driver-side indicators
    (speed, stationary share, illegal maneuvers), and PRER vs beta per rho."""
    grid = {}
    for _, jobs in groups_matching(groups, r'exp_beta_rho_grid.*'):
        for j in jobs:
            mm = re.fullmatch(r'(?:beta|w)_([\dp.]+)_rho_([\dp.]+)', j['name'])
            if mm:
                grid.setdefault((num(mm.group(1)), num(mm.group(2))), []).append(j)
    if not grid:
        return

    betas = sorted({k[0] for k in grid})
    rhos  = sorted({k[1] for k in grid})

    def matrix(key, rows=rhos):
        return grid_matrix(grid, betas, rows, key)

    prer = matrix('prer')
    annotated_grid(prer, betas, rhos, r'PRER', 'prer_heatmap.png', fmt='{:.2f}')
    annotated_grid(matrix('runover_rate'), betas, rhos, 'Runovers per walker', 'runovers_heatmap.png')
    coll = matrix('collision_rate')
    if np.nanmax(coll) > 0:
        annotated_grid(coll, betas, rhos, 'Collisions per driver', 'collisions_heatmap.png')
    driver_grid_maps(matrix, betas, rhos)

    fig, ax = plt.subplots(figsize=(PlotStyle.COLUMN_W, PlotStyle.COLUMN_W * .7))
    markers = ['o', 's', '^', 'D', 'v', 'P']
    for i, (r, c) in enumerate(zip(rhos, PlotStyle.ordinal_colors(len(rhos), 'viridis'))):
        ax.plot(betas, prer[i], marker=markers[i % len(markers)], color=c, label=rf'$\rho$={r}')
    ax.set_xticks(betas)
    ax.set_xlabel(r'Heuristic weight $\beta$')
    ax.set_ylabel(r'PRER')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')
    PlotStyle.save(fig, out('prer_lines.png'))


def analyse_usecase(groups):
    """Use-case scenario (exp_usecase_seed_S): spatial maps over the city of
    walker and driver counts, mean driver speed and jaywalking."""
    for m, jobs in groups_matching(groups, r'exp_usecase_seed_(\d+)'):
        suffix = '' if m.group(1) == '0' else f'_seed_{m.group(1)}'
        for j in jobs:
            city = j['city']
            if city is None:
                continue
            load = lambda n: np.load(os.path.join(j['dir'], n)) if os.path.isfile(os.path.join(j['dir'], n)) else None
            wc, dc, ds = load('walker_count_hm.npy'), load('driver_count_hm.npy'), load('driver_speed_hm.npy')
            size = (PlotStyle.COLUMN_W * 1.15, PlotStyle.COLUMN_W)
            maps = [('walker_count', wc, 'Walkers visiting the cell', PlotStyle.CMAP_COUNT),
                    ('driver_count', dc, 'Drivers visiting the cell', PlotStyle.CMAP_COUNT),
                    ('jaywalking', j['jw_map'], 'Walkers using the road cell', PlotStyle.CMAP_COUNT)]
            road = np.isin(np.array([[c[0] for c in row] for row in city.city_grid]),
                           list('rtlzp'))
            if dc is not None:
                dc = np.where(road, dc, 0)
            if ds is not None and dc is not None:
                # Speed only on road cells visited often enough to give a stable mean
                speed = np.where(dc >= MIN_SPEED_VISITS, ds, np.nan)
                maps.append(('speed', speed, 'Mean driver speed (km/h)', PlotStyle.CMAP_SPEED))
            for name, data, label, cmap in maps:
                if data is None or not np.nansum(data):
                    continue
                vmax = np.nanpercentile(data[data > 0], 99) if name != 'speed' else None
                fig, ax = plt.subplots(figsize=size)
                urban_heatmap(city, data, ax=ax, cmap=cmap, label=label, vmax=vmax)
                PlotStyle.save(fig, out(f'uc_heatmap_{name}{suffix}.png'))

# ──────────────────────────────────────────────────────────────────────────────
# Text report
# ──────────────────────────────────────────────────────────────────────────────
def print_summary_table(group_name, jobs):
    print(f'\n{"="*86}')
    print(f'  Experiment group: {group_name}  ({len(jobs)} jobs)')
    print(f'{"="*86}')
    print(f'  {"Job":<32} {"Walkers":>8} {"Drivers":>8} {"Coll/Drv":>9} {"Runo/Ped":>9} {"PRER":>8} {"JW frac":>8} {"Speed":>7} {"Stopped":>8} {"Done":>6} {"MovSpd":>7} {"WW pl":>6} {"WW dr":>6} {"FT pl":>6} {"FT dr":>6} {"Repl":>5} {"WStop":>6} {"WDone":>6} {"WRepl":>6}')
    print(f'  {"-"*32} {"-"*8} {"-"*8} {"-"*9} {"-"*9} {"-"*8} {"-"*8} {"-"*7} {"-"*8} {"-"*6} {"-"*7} {"-"*6} {"-"*6} {"-"*6} {"-"*6} {"-"*5} {"-"*6} {"-"*6} {"-"*6}')
    for j in jobs:
        print(f'  {j["name"]:<32} '
              f'{j["num_walkers"]:>8} '
              f'{j["num_drivers"]:>8} '
              f'{j["collision_rate"]:>9.4f} '
              f'{j["runover_rate"]:>9.4f} '
              f'{j["prer"]:>8.4f} '
              f'{j["jw_rate"]:>8.4f} '
              f'{j["avg_spd_driver"]:>7.1f} '
              f'{j["stopped_share"]:>8.3f} '
              f'{j["trips_done"]:>6.2f} '
              f'{j["moving_spd"]:>7.1f} '
              f'{j["ww_planned"]:>6.2f} '
              f'{j["ww_executed"]:>6.2f} '
              f'{j["ft_planned"]:>6.2f} '
              f'{j["ft_executed"]:>6.2f} '
              f'{j["replans"]:>5.1f} '
              f'{j["w_stopped"]:>6.3f} '
              f'{j["w_done"]:>6.2f} '
              f'{j["w_replans"]:>6.2f}')


def safe_corr(x, y):
    """Return Pearson r, or nan if either series is constant."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return np.corrcoef(x, y)[0, 1]


def print_key_findings(all_groups):
    """Print a concise summary of the most important results across all groups."""
    print(f'\n{"#"*70}')
    print('  KEY FINDINGS ACROSS ALL EXPERIMENTS')
    print(f'{"#"*70}')

    if not any(all_groups.values()):
        print('  (no data found)')
        return

    # Trend: PRER as obstacles increase
    for gname, jobs in sorted(all_groups.items()):
        pts = sorted([(int(m.group(1)), j) for j in jobs
                      for m in [re.fullmatch(r'obstacles_(\d+)', j['name'])] if m],
                     key=lambda x: x[0])
        if len(pts) >= 3:
            denses = [p[0] for p in pts]
            prer_v = [p[1]['prer'] for p in pts]
            r = safe_corr(denses, prer_v)
            print(f'\n  [{gname}] PRER vs obstacle density: Pearson r={r:+.3f}  '
                  f'(PRER {prer_v[0]:.3f} at {denses[0]}% -> {prer_v[-1]:.3f} at {denses[-1]}%)')

    # β–ρ grid: impact of behavioral weights
    for gname, jobs in sorted(all_groups.items()):
        pts = [(num(m.group(1)), num(m.group(2)), j)
               for j in jobs
               for m in [re.fullmatch(r'(?:beta|w)_([\dp.]+)_rho_([\dp.]+)', j['name'])] if m]
        if len(pts) >= 4:
            betas = sorted(set(p[0] for p in pts))
            rhos  = sorted(set(p[1] for p in pts))
            print(f'\n  [{gname}] β–ρ grid ({len(betas)} betas × {len(rhos)} rhos = {len(pts)} runs)')
            for rho in rhos:
                sub = sorted([(p[0], p[2]['prer']) for p in pts if p[1] == rho])
                x = [s[0] for s in sub]; y = [s[1] for s in sub]
                print(f'    ρ={rho}: PRER vs β  r={safe_corr(x, y):+.3f}  (range {min(y):.4f}–{max(y):.4f})')
            print()
            for beta in betas:
                sub = sorted([(p[1], p[2]['prer']) for p in pts if p[0] == beta])
                x = [s[0] for s in sub]; y = [s[1] for s in sub]
                print(f'    β={beta}: PRER vs ρ  r={safe_corr(x, y):+.3f}  (range {min(y):.4f}–{max(y):.4f})')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f'Scanning {BASE_DIR} for experiment results …')
    groups = discover_experiments(BASE_DIR)

    if not groups:
        print(f'No experiment results found under {BASE_DIR}.')
        print('Ensure experiments have been run with Experiments.py and that')
        print('metrics.txt and agents.txt exist in each job subdirectory.')
        return

    total_jobs = sum(len(v) for v in groups.values())
    print(f'Found {len(groups)} experiment group(s) with {total_jobs} total job(s).\n')

    for group_name, jobs in sorted(groups.items()):
        print_summary_table(group_name, jobs)

    analyse_obstacle_sweeps(groups)
    analyse_interaction(groups)
    analyse_beta_rho_grid(groups)
    analyse_lane_closures(groups)
    analyse_road_risk()
    analyse_usecase(groups)

    print_key_findings(groups)

    print(f'\nAll figures saved to: {OUT_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
