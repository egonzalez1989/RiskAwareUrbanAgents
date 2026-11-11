"""
Shared figure style for every plot of the project (city grids, routes,
spatial heatmaps and metric graphs), so all figures in the paper read as
one system.

  - Categorical colors follow the Okabe-Ito palette (colorblind safe).
  - Ordered variables (e.g. number of walkers) use shades of one sequential
    colormap instead of unrelated categorical colors.
  - Figure widths match the paper: one column or the full text width.
"""
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# Okabe-Ito categorical palette
CATEGORICAL = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
               '#CC79A7', '#56B4E9', '#F0E442', '#000000']

# Sequential maps: counts and rates use one hue ramp; speed goes from red (slow) to green (fast)
CMAP_COUNT = 'YlOrRd'
CMAP_RATE = 'YlOrRd'
CMAP_SPEED = 'RdYlGn'

# Figure widths in inches (single column / full text width)
COLUMN_W = 3.5
TEXT_W = 7.16

RC = {
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': .3,
    'grid.linewidth': .6,
    'lines.linewidth': 1.6,
    'lines.markersize': 4.5,
    'axes.prop_cycle': mpl.cycler(color=CATEGORICAL),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
}


def apply():
    """Apply the project style to all following matplotlib figures."""
    mpl.rcParams.update(RC)


def ordinal_colors(n, cmap='viridis', lo=.15, hi=.85):
    """n colors from a sequential colormap, for an ordered variable."""
    return [plt.get_cmap(cmap)(v) for v in np.linspace(lo, hi, max(n, 1))]


# Ordered behavioral levels from cautious to reckless (e.g. beta = 1, 3, 5):
# Okabe-Ito green, amber and vermillion, distinguishable with color blindness
CAUTIOUS_TO_RECKLESS = ['#009E73', '#E69F00', '#D55E00']


def masked_cmap(name, lo=0.):
    """Colormap whose masked/under values are transparent, so cells without
    data let the city layout show through. lo > 0 drops the palest part of
    the map so the lowest values stay visible over light cells."""
    cmap = plt.get_cmap(name)
    if lo > 0:
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            f'{name}_from_{lo}', cmap(np.linspace(lo, 1, 256)))
    cmap = cmap.copy()
    cmap.set_bad((0, 0, 0, 0))
    cmap.set_under((0, 0, 0, 0))
    return cmap


def save(fig, path, pdf=False):
    """Save a figure as PNG (300 dpi) and optionally as vector PDF."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    if pdf:
        fig.savefig(os.path.splitext(path)[0] + '.pdf')
    plt.close(fig)
    print(f'  -> saved {path}')
