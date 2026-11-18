# Risk-aware urban agents

Multiagent simulation of pedestrians and drivers on a grid city. Every agent plans
its route with a risk-aware weighted A\*: the heuristic weight β sets how greedy the
search is, and the risk sensitivity ρ scales the risk of cells and maneuvers that is
added to the path cost.

## Layout

| Path | Contents |
|---|---|
| `src/model/` | Agents (`AgentBase.py`, `AgentImpl.py`) and the city model (`UrbanModelling.py`) |
| `src/utils/AgentUtils.py` | Weighted A\* (`weighted_astar`), city, obstacle and lane-closure generation |
| `src/run_experiment.py` | Runs one job of experiments E1–E6 for one seed |
| `src/analysis/Experiments.py` | Experiment definitions used by the runner |
| `src/analysis/analysis.py` | Figures and summary statistics of the paper |
| `src/analysis/replication_stats.py` | Means, standard deviations, 95% intervals and within-seed differences (E4, E5) |
| `src/analysis/design_figures.py` | Figures of the model design section: grid cell types and example routes for several (β, ρ) |
| `src/analysis/robustness.py` | Stationarity over time, market runover ratio, grid indicators and the E6 sensitivity figure |
| `src/tests/` | Tests of the path planner and of the driver rules |

## Setup

Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All commands below are run from `src/`.

## Tests

```bash
../.venv/bin/python -m unittest discover tests
```

`tests/test_astar.py` includes a worked example with its expected path and cost. It
also checks the heading-aware search against an exhaustive search, and checks that
β > 1 stays within the β bound.

## Running the experiments

```bash
../.venv/bin/python run_experiment.py JOB SEED [--out DIR]
```

| Job | Experiment |
|---|---|
| `E1_1`, `E1_3`, `E1_5` | Pedestrians only, sidewalk obstruction 0–25%, β = 1, 3, 5 |
| `E2_1`, `E2_2`, `E2_3`, `E2_5` | (β, ρ) grid, one β per job |
| `E3_1`, `E3_3`, `E3_5` | (β, ρ) grid with eight one-lane closures |
| `E4_20`, `E4_100` | Traffic load: 20 or 100 drivers, 10 000 steps |
| `E5` | Street market, 10 000 steps |
| `E6_base`, `E6_pert`, `E6_tau2`, `E6_tau6` | Sensitivity to the assumptions: E2 setting, β = 3, ρ = 0.25, 1, 5, with the reference risk values, with each nonzero risk value multiplied by a log-uniform factor in [1/2, 2] drawn from the seed, or with a perception–reaction time of 0.6 s or 1.8 s instead of 1.2 s |

Output goes to `../Experiments/exp_<name>_seed_<SEED>/<job>/`. A job that already
exists there is skipped. Jobs are independent, so they can run in parallel:

```bash
for s in $(seq 0 19); do for j in E4_20 E4_100 E5; do echo "$j $s"; done; done \
  | xargs -P 6 -L 1 ../.venv/bin/python run_experiment.py
```

**Seed protocol.** The seed initializes Python's and NumPy's global generators and
the model's own generators. These drive the spawn counts, the agent parameters, and
the placement of obstacles, market stalls and lane closures. The same job and seed
reproduce the same run exactly. The configurations compared within an experiment use
the same seeds (for example, light and heavy traffic in E4 share the obstacle layout
of each seed), so they can be compared seed by seed.

## Analysis

```bash
cd analysis
../../.venv/bin/python analysis.py ../../Experiments ../../Experiments/analysis
../../.venv/bin/python replication_stats.py ../../Experiments
../../.venv/bin/python robustness.py ../../Experiments ../../Experiments/analysis
```

The choice of the road risk of walkers (Fig. 6 of the paper) is produced by
`PYTHONPATH=. python -m analysis.road_risk_calibration run ../Experiments/exp_road_risk_detour 10`
followed by `... plot ../Experiments/exp_road_risk_detour ../Experiments/analysis`: walkers only,
between 30 start–goal pairs stratified by detour factor, 10 seeds.

The paper uses seeds 0–9 for E1, E2 and E3, 0–29 for E4 and E5, 0–39 for `E6_base`
and `E6_pert`, 0–19 for `E6_tau2` and `E6_tau6`.
