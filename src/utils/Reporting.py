import json, os
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from utils.AnimationUtils import plot_city
from utils import PlotStyle

'''
Generates plots for resulting collisions, runovers, and jaywalking.
pamrams:
    - model: executed model
'''
def generate_heatmaps(city, metrics, outdir = None, maps = ['runover', 'collision', 'jaywalking'], figsize = None, vmin = None, vmax = None):
    # Heatmap for runovers
    if 'runover' in maps:
        #xcol, ycol = [], []
        hm = np.zeros(city.shape, dtype=int)
        d = metrics['runovers']
        for coll in d:
            hm[coll] += d[coll]
        outfile = os.path.join(outdir, 'runovers.png') if outdir else None
        #urban_heatmap(city, (xcol, ycol), outfile = outfile, figsize = figsize, vmin = vmin, vmax = vmax)
        urban_heatmap(city, hm, title='Runovers', outfile=outfile, figsize=figsize, vmin=vmin, vmax=vmax)

    # Heatmap for collisions
    if 'collision' in maps:
        #xcol, ycol = [], []
        hm = np.zeros(city.shape, dtype=int)
        d = metrics['collisions']
        for coll in d:
            hm[coll] += d[coll]
        outfile = os.path.join(outdir, 'collisions.png') if outdir else None
        #urban_heatmap(city, (xcol, ycol), outfile = outfile, figsize = figsize, vmin = vmin, vmax = max)
        urban_heatmap(city, hm, title='Collisions', outfile=outfile, figsize=figsize, vmin=vmin, vmax=vmax)

    # Heatmap for jaywalking
    if 'jaywalking' in maps:
        #xcol, ycol = [], []
        hm = np.zeros(city.shape, dtype=int)
        d = metrics['jaywalking']
        for id in d:
            points = set([tuple(pos) for pos in d[id]])
            for point in points:
                # xcol.append(point[1])
                # ycol.append(point[0])
                hm[point] += 1
        outfile = os.path.join(outdir, 'jaywalking.png') if outdir else None
        urban_heatmap(city, hm, title='Jaywalking', outfile = outfile, figsize = figsize, vmin = vmin, vmax = vmax)


def urban_heatmap(city, data_grid=None, points=None, title=None, outfile=None, figsize=None,
                  vmin=None, vmax=None, cmap=None, ax=None, label=None, colorbar=True):
    """Heatmap drawn over the city layout.

    The city is drawn faded and without arrows so it only gives context;
    cells with no data (zero or NaN) stay transparent and the heat is drawn
    at full opacity, so colors match the colorbar. With ax given, the map is
    drawn there (for multi-panel figures) and the image is returned."""
    s = city.shape
    own = ax is None
    if own:
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(111)
    # Speed maps use the full red-green range and keep cells at speed 0 (always stopped);
    # their blocks are gray so they are not read as fast (green) cells
    speed = cmap == PlotStyle.CMAP_SPEED
    plot_city(city, ax, alpha='66', arrows=False, block_color='#BBBBBB' if speed else None)
    cm = PlotStyle.masked_cmap(cmap or PlotStyle.CMAP_COUNT, lo=0 if speed else .2)
    if data_grid is not None:
        empty = ~np.isfinite(data_grid) if speed else ~np.isfinite(data_grid) | (np.asarray(data_grid) <= 0)
        data = np.ma.masked_where(empty, np.asarray(data_grid, dtype=float))
        im = ax.imshow(data, cmap=cm, vmin=vmin if vmin is not None else 0, vmax=vmax,
                       interpolation='nearest', zorder=2.5)
    else:
        data, _, _ = np.histogram2d(points[0], points[1], bins=[s[0], s[1]], range=([0, s[0]], [0, s[1]]))
        im = ax.imshow(np.ma.masked_less_equal(data.T, 0), cmap=cm, vmin=vmin, vmax=vmax, zorder=2.5)
    if title:
        ax.set_title(title)
    if colorbar:
        clipped = vmax is not None and np.nanmax(data) > vmax
        cb = plt.colorbar(im, ax=ax, fraction=.046, pad=.02, extend='max' if clipped else 'neither')
        cb.outline.set_visible(False)
        if label:
            cb.set_label(label)
    if own:
        if outfile:
            PlotStyle.save(f, outfile)
        else:
            plt.show()
            plt.close(f)
    return im


def heatmap_panels(city, grids, titles, label, outfile=None, cmap=None, ncols=None,
                   vmax=None, panel_w=2.4, cities=None):
    """Several urban heatmaps side by side with one shared color scale, so
    the panels can be compared directly. cities optionally gives one city
    per panel (e.g. different obstacle layouts)."""
    n = len(grids)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    h, w = city.shape
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(panel_w * ncols + .8, panel_w * nrows * h / w + .3),
                             constrained_layout=True)
    if vmax is None:
        vmax = max(float(np.nanmax(g)) for g in grids) or 1
    im = None
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.set_axis_off()
            continue
        im = urban_heatmap((cities or [city] * n)[k], grids[k], title=titles[k], ax=ax,
                           vmin=0, vmax=vmax, cmap=cmap, colorbar=False)
    clipped = max(float(np.nanmax(g)) for g in grids) > vmax
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.9, pad=.01,
                      extend='max' if clipped else 'neither')
    cb.outline.set_visible(False)
    cb.set_label(label)
    if outfile:
        PlotStyle.save(fig, outfile)
    return fig

'''
def generate_summary(mfiles, outdir = None, figsize = None):
    nc, nr, nj, dat, wat = [], [], [], [], []

    for f in mfiles:
        metrics = json.load(open(f, 'r'))
        nc.append(sum(len([metrics[t]['collisions'] for t in metrics.keys()])))
        nr.append(sum(len([metrics[t]['runovers'] for t in metrics.keys()])))
        nj.append(sum(len([metrics[t]['jaywalking'] for t in metrics.keys()])))
        dat.append(np.mean([metrics[t]['avg_speed_drivers'] for t in metrics.keys()]))
        wat.append(np.mean([metrics[t]['avg_speed_walkers'] for t in metrics.keys()]))

    #draw_plot(nc, 'Collisions', xlabel='collisions', ylabel='number of drivers', outfile=os.path.join(outdir, 'collisions.png'), figsize=figsize)
    #draw_plot(nr, 'Runovers', xlabel='runovers', ylabel='number of drivers', outfile=os.path.join(outdir, 'runovers.png'), figsize=figsize)
    #draw_plot(nj, 'Jaywalking', xlabel = '% obstacles', ylabel = 'number of agents', outfile = os.path.join(outdir, 'jaywalking.png'), figsize = figsize)
    np.save(os.path.join(outdir, 'collisions.npy'), nc)
    np.save(os.path.join(outdir, 'runovers.npy'), nr)
    np.save(os.path.join(outdir, 'jaywalking.npy'), nj)
    np.save(os.path.join(outdir, 'avg_driver_speed.npy'), dat)
    np.save(os.path.join(outdir, 'avg_walker_speed.npy'), wat)
'''

def draw_plot(data, title, xlabel, ylabel, color = 'k', outfile = None, figsize = None):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data, color=color)
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
    plt.close()

def load_heatmaps(metrics):
    if isinstance(metrics, str):
        f = open(metrics, 'r')
        metrics = json.load(f)

    coll, rover, jwalk = defaultdict(int), defaultdict(int), defaultdict(list)
    for t in metrics.keys():
        for coord in metrics[t]['collisions']:
            coll[tuple(coord)] += 1
        for coord in metrics[t]['runovers']:
            rover[tuple(coord)] += 1
        for key, val in metrics[t]['jaywalking'].items():
            jwalk[key].extend(val)

    map_dict = {'collisions': coll, 'runovers': rover, 'jaywalking': jwalk}
    return map_dict




#def generate_run_summary(results_dir):




if __name__ == '__main__':
    from analysis.Experiments import load_parameters
    from model.UrbanModelling import CityModel
    import numpy as np

    fsize = (8, 6)
    #base_dir = '../experiments/exp_nodrivers_seed_0_weight_{}/obstacles_05'
    #base_dir = '../experiments/exp_seed_0_wweight_1_dweight_1/obstacles_{}_nw_{}_nd_{}'
    #results_dir = base_dir.format(10, '200', '100')

    results_dir = '../experiments/exp_usecase_seed_0_wweight_1_3_dweight_1_5/usecase'
    params = load_parameters('{}/parameters.txt'.format(results_dir))
    params['city_grid'] = np.array(params['city_grid'])
    params['initial_driver_count'] = 0
    params['initial_walker_count'] = 0
    model = CityModel(params)
    model.setup()
    city = model.city#City(model, shape=params['city_grid'].shape)
    #for w in (1, 3, 5):
        #hmaps = load_heatmaps(base_dir.format(w) + '/metrics.txt')
        #jmap = generate_heatmaps(city, hmaps, maps=['jaywalking'], vmin=0, vmax=40)


    # hmap = np.load(results_dir + '/driver_speed_hm.npy')
    # hmap[city.city_grid == 's  '] = -1
    # hmap[city.city_grid == 'b  '] = -1
    # cmap = plt.get_cmap('RdYlGn')
    # cmap.set_under('w')
    # urban_heatmap(city, data_grid=hmap, figsize=fsize, cmap=cmap, vmin=0, vmax = 60)
    # hmap = np.load(results_dir + '/driver_count_hm.npy')
    # urban_heatmap(city, data_grid=hmap, figsize=fsize, vmin=0, vmax = 150)
    # hmap = np.load(results_dir + '/walker_count_hm.npy')
    # urban_heatmap(city, data_grid=hmap, figsize=fsize, vmin=0, vmax = 100)
    hmap = np.load(results_dir + '/walker_count_hm.npy')
    hmap[city.city_grid == 's  '] = -1
    for pos in city.zebras:
        hmap[tuple(pos)] = -1
    cmap = plt.get_cmap('Reds')
    cmap.set_under('w')
    urban_heatmap(city, data_grid=hmap, figsize=fsize, vmin=0, vmax = 10)


