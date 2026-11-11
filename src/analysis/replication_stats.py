"""
replication_stats.py
Run-level statistics over seeds for the long-run experiments E3 (traffic load,
exp_interaction_seed_S) and E4 (street market, exp_usecase_seed_S).

For each indicator and time window (whole run, and its first and last 2000
steps, which show whether the indicators are stationary) it prints the mean over seeds, the standard deviation and a 95%
confidence interval (Student t). For E3, whose two configurations share every
seed (same obstacle layout), it also reports the within-seed difference
heavy - light traffic with its 95% interval and a Wilcoxon signed-rank test.
For E4 it adds the market-area indicators.

Usage (from src/analysis):  python replication_stats.py [experiments_dir]
"""
import os, re, sys, glob, json
import numpy as np
from collections import Counter, defaultdict
from scipy import stats

E = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Experiments')
WINDOW = 2000
# Street-market area of E4 (rows 14-46, columns 26-33): the stalls and the streets around them
in_market = lambda x, y: 14 <= x <= 46 and 26 <= y <= 33


def window(m, a, did, lo, hi):
    """Indicators of one run restricted to the steps [lo, hi)."""
    ts = [str(t) for t in range(lo, hi)]
    ad = sum(m[t]['active_drivers'] for t in ts); sd = sum(m[t]['stopped_drivers'] for t in ts)
    r = {'drivers spawned': sum(lo <= a[k]['spawn_time'] < hi for k in did)}
    r['mean driver speed'] = np.mean([m[t]['avg_speed_drivers'] for t in ts if m[t]['active_drivers']] or [np.nan])
    r['moving driver speed'] = (sum(m[t]['avg_speed_drivers'] * m[t]['active_drivers'] for t in ts) / (ad - sd)
                                if ad > sd else np.nan)
    r['driver stationary share'] = sd / ad if ad else np.nan
    n = (hi - lo) / 1000
    r['driver arrivals /1000 steps'] = sum(1 for t in ts for x in m[t]['arrivals'] if x in did) / n
    r['walker arrivals /1000 steps'] = sum(1 for t in ts for x in m[t]['arrivals'] if x not in did) / n
    r['aborted trips /1000 steps'] = sum(len(m[t].get('aborted', ())) for t in ts) / n
    trips = [a[k]['trip'] for k in did if lo <= a[k]['spawn_time'] < hi]
    for kind in ('planned', 'executed'):
        c = Counter()
        for t in trips:
            c.update(t[kind])
        tot = sum(c.values())
        r[f'wrong-way /100 cells, {kind}'] = 100 * c['bw'] / tot if tot else np.nan
    r['vehicle collisions /1000 steps'] = sum(len(m[t]['collisions']) for t in ts) / n
    r['runovers /1000 steps'] = sum(len(m[t]['runovers']) for t in ts) / n
    r['replans per driver, median'] = np.median([t['replans'] for t in trips]) if trips else np.nan
    aw = sum(m[t]['active_walkers'] for t in ts); sw = sum(m[t]['stopped_walkers'] for t in ts)
    r['PRER'] = sum(len(v) for t in ts for v in m[t]['jaywalking'].values()) / aw
    r['walker stationary share'] = sw / aw
    return r


def market(m, a, did, p, d):
    """E4 market-area indicators of one run."""
    g = np.array([[c[0] for c in row] for row in p['city_grid']])
    roadlike = np.isin(g, list('rtz'))
    nm = np.array([[in_market(i, j) for j in range(g.shape[1])] for i in range(g.shape[0])])
    u = {'market share of road cells': (roadlike & nm).sum() / roadlike.sum()}
    T = len(m)
    for lo, hi, tag in ((0, WINDOW, 'early'), (0, T, 'all')):
        jw = Counter(tuple(q) for t in range(lo, hi) for ps in m[str(t)]['jaywalking'].values() for q in ps)
        u[f'market share of road-steps, {tag}'] = sum(v for q, v in jw.items() if in_market(*q)) / max(1, sum(jw.values()))
    ev = [tuple(c) for t in m for c, k in m[t]['illegal'] if k == 'bw']
    u['market share of wrong-way moves'] = sum(in_market(*c) for c in ev) / len(ev) if ev else np.nan
    dc, ds = np.load(d + 'driver_count_hm.npy'), np.load(d + 'driver_speed_hm.npy')
    mask = (dc >= 3) & roadlike
    u['driver speed, market'] = np.nanmean(ds[mask & nm]); u['driver speed, elsewhere'] = np.nanmean(ds[mask & ~nm])
    steps, jws = Counter(), set()
    for t in m:
        for aid, ps in m[t]['jaywalking'].items():
            steps[aid] += len(ps); jws.add(aid)
    for b in (1, 2, 3):
        ids = [k for k, v in a.items() if v['type'] == 'AstarWalker' and v['beta'] == b]
        u[f'jaywalking share, beta {b}'] = np.mean([k in jws for k in ids])
        u[f'road-steps per walker, beta {b}'] = np.mean([steps[k] for k in ids])
    return u


def ci(v):
    """Mean, sd and 95% t interval of the finite values of v."""
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (v.mean() if len(v) else np.nan), np.nan, (np.nan, np.nan), len(v)
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.std(ddof=1), (v.mean() - h, v.mean() + h), len(v)


def show(title, runs):
    """runs: {seed: {indicator: value}}"""
    print(f'\n== {title}  ({len(runs)} seeds)')
    for k in next(iter(runs.values())):
        m, s, (lo, hi), n = ci([r[k] for r in runs.values()])
        print(f'  {k:38s} {m:9.3f}  sd {s:8.3f}  95% CI [{lo:9.3f}, {hi:9.3f}]  n={n}')


def show_diff(title, a, b):
    """Within-seed differences b - a over the seeds present in both."""
    seeds = sorted(set(a) & set(b))
    print(f'\n== {title}  (within-seed differences, {len(seeds)} seeds)')
    for k in a[seeds[0]]:
        d = np.array([b[s][k] - a[s][k] for s in seeds], float)
        d = d[np.isfinite(d)]
        m, s, (lo, hi), n = ci(d)
        p = stats.wilcoxon(d).pvalue if n >= 6 and np.any(d != 0) else np.nan
        print(f'  {k:38s} {m:9.3f}  95% CI [{lo:9.3f}, {hi:9.3f}]  Wilcoxon p={p:.3g}  n={n}')


def main():
    runs = defaultdict(dict)        # (experiment, job, window) -> {seed: indicators}
    for d in sorted(glob.glob(os.path.join(E, 'exp_interaction_seed_*/*/')) +
                    sorted(glob.glob(os.path.join(E, 'exp_usecase_seed_*/*/')))):
        if not os.path.isfile(d + 'agents.txt'):
            continue
        seed = int(re.search(r'seed_(\d+)', d).group(1))
        m = json.load(open(d + 'metrics.txt')); a = json.load(open(d + 'agents.txt'))
        did = {k for k, v in a.items() if v['type'] == 'AstarDriver'}
        exp = 'E4' if 'usecase' in d else 'E3'
        job = os.path.basename(d.rstrip('/'))
        T = len(m)
        for lo, hi, tag in ((0, T, 'whole run'), (0, WINDOW, 'first 2000 steps'), (T - WINDOW, T, 'last 2000 steps')):
            runs[(exp, job, tag)][seed] = window(m, a, did, lo, hi)
        if exp == 'E4':
            runs[(exp, job, 'market area')][seed] = market(m, a, did, json.load(open(d + 'parameters.txt')), d)
    for key in sorted(runs):
        show(' / '.join(key), runs[key])
    for tag in ('first 2000 steps', 'whole run', 'last 2000 steps'):
        light = runs.get(('E3', 'obstacles_10_nw_200_nd_020', tag)); heavy = runs.get(('E3', 'obstacles_10_nw_200_nd_100', tag))
        if light and heavy:
            show_diff(f'E3 / 100 - 20 drivers / {tag}', light, heavy)


if __name__ == '__main__':
    main()
