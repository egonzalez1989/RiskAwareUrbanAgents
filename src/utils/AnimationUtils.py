"""
Improved visualization layer for the urban traffic agent-based model.

Design goals over the original version:
  - One central COLORS/SIZES palette instead of magic strings scattered
    through every function, so re-theming is a one-line change.
  - Robust alpha handling via matplotlib.colors.to_hex(..., keep_alpha=True)
    instead of blindly string-concatenating a 2-digit hex suffix onto a
    color (the original approach breaks if the base color isn't already a
    plain 6-digit hex, and silently produces wrong colors otherwise).
  - Higher contrast + a real legend, so a still frame is readable without
    the reader already knowing the color code.
  - Agents get a soft drop shadow + outline so they read clearly against
    both light (sidewalk) and dark (road) cells.
  - Predicted future path is drawn as a fading dotted line instead of
    disconnected dots.
  - Collision/runover markers fade out over their trailing window instead
    of all rendering at full opacity regardless of age.
  - Red is reserved for events (collisions, runovers, heat), so obstacles
    use a dark outline instead of a red one.
  - city_legend() names every ground type drawn by plot_city, and
    plot_route() uses neutral start/goal symbols and colorblind-safe routes.

Public function names/signatures are unchanged from the original module,
so this is a drop-in replacement: animation_plot, plot_city, plot_agents,
plot_collisions, draw_arrow, plot_stamp, plot_route, plot_next.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patheffects as pe
import agentpy as ap

from utils.AgentUtils import manhattan
from utils.PlotStyle import CATEGORICAL

# --------------------------------------------------------------------------
# Palette & sizing — tweak here, nowhere else.
# --------------------------------------------------------------------------
COLORS = {
    'sidewalk':    '#EDEAE3',
    'road':        '#3B4250',
    'parking':     '#4E7FAE',
    'goal':        '#3FA76B',
    'obstacle':    '#8E97A3',
    'obstacle_edge': '#20242C',
    'zebra':       '#FFFFFF',
    'grid':        '#00000022',
    'ink':         '#20242C',
    'pothole':     '#7A5230',
    'walker':      '#2FAE60',
    'car_active':  '#2E86DE',
    'car_parked':  '#AAB2BD',
    'shadow':      '#000000',
    'collision':   '#E63946',
    'runover':     '#B3122A',
    'arrow_fill':  '#FFFFFF',
    'arrow_edge':  '#FFFFFF',
    'text_light':  '#FFFFFF',
    'panel_bg':    '#20242Ccc',
}

SIZES = {
    'walker': 26,
    'car': 46,
}


def _hex_alpha(color, alpha_255_hex):
    """Combine a base color with a 2-digit hex alpha suffix (e.g. '80'),
    the way the original code intended, but done safely through
    matplotlib's color machinery instead of raw string concatenation."""
    alpha = int(alpha_255_hex, 16) / 255.0
    return to_hex(to_rgba(color, alpha=alpha), keep_alpha=True)


def animation_plot(model, ax, alpha='80'):
    ax.set_facecolor(COLORS['sidewalk'])
    plot_city(model.city, ax, alpha=alpha)
    plot_agents(model.city, ax)
    plot_collisions(model.metrics, ax)
    _add_legend(ax)


###################################
def plot_collisions(metrics, ax, trail=10):
    """Draw a fading trail of recent collisions/runovers so older events
    read as history rather than competing with the current frame."""
    T = max(metrics.keys())
    for t in range(max(0, T - trail), T):
        age = T - t
        fade = max(0.15, 1 - age / trail)
        for key, color in (('collisions', COLORS['collision']), ('runovers', COLORS['runover'])):
            if key in metrics[t]:
                for (row, col) in metrics[t][key]:
                    # soft glow behind the mark
                    ax.add_patch(Circle((col, row), radius=0.55, color=color,
                                         alpha=0.18 * fade, zorder=4, lw=0))
                    ax.text(col, row, "\u2739", fontsize=26, color=color,
                            alpha=fade, ha='center', va='center', zorder=5,
                            fontweight='bold')

    ncollisions = sum(len(metrics[t].get('collisions', [])) for t in range(0, T))
    nrunovers = sum(len(metrics[t].get('runovers', [])) for t in range(0, T))

    ax.text(
        0.01, 0.99,
        f"t = {T}\ncollisions: {ncollisions}   runovers: {nrunovers}",
        transform=ax.transAxes, ha='left', va='top', fontsize=10,
        color=COLORS['text_light'], zorder=10,
        bbox=dict(boxstyle='round,pad=0.4', fc=COLORS['panel_bg'], ec='none'),
    )


def plot_agents(city, ax, show_speed=False, show_id=False):
    agents = city.agents
    fig_w, fig_h = ax.figure.get_size_inches()
    h, w = city.city_grid.shape
    marker_dict = {(-1, 0): '^', (1, 0): 'v', (0, 1): '>', (0, -1): '<', (0, 0): 'o'}

    for agent in agents:
        pos, dir = agent.position, agent.direction
        is_walker = agent.agent_type == 'walker'
        if is_walker:
            color, size = COLORS['walker'], SIZES['walker']
        elif agent.active:
            color, size = COLORS['car_active'], SIZES['car']
        else:
            color, size = COLORS['car_parked'], SIZES['car']

        msize = 0.9 * size * fig_w / w
        marker = marker_dict[dir]
        x, y = pos[1] - .5, pos[0] - .5

        # Predicted future path: fading dotted line + shrinking dots
        n = len(agent.next_real)
        if n:
            xs = [x] + [agent.next_real[i][1] - .5 for i in range(n)]
            ys = [y] + [agent.next_real[i][0] - .5 for i in range(n)]
            for i in range(n):
                a = (n - i) / n
                ax.plot(xs[i:i + 2], ys[i:i + 2], '-', color=color,
                         linewidth=1.4, alpha=0.35 * a, zorder=2)
                ax.plot([xs[i + 1]], [ys[i + 1]], 'o', color=color,
                         markersize=msize / 6, alpha=0.55 * a, zorder=2)

        # Drop shadow for contrast against both light and dark cells
        ax.plot(x + 0.08, y + 0.08, marker, markersize=msize, color=COLORS['shadow'],
                 alpha=0.25, zorder=2.5)
        # Agent marker with a light outline
        ax.plot(x, y, marker, markersize=msize, color=color,
                 markeredgecolor='white', markeredgewidth=0.8, zorder=3)

        if show_speed:
            ax.text(pos[1] - 1, pos[0] - .25, str(int(agent.speed)),
                    color=COLORS['text_light'], fontsize=7, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', fc=COLORS['panel_bg'], ec='none'))
        if show_id:
            ax.text(pos[1] - 1, pos[0], str(agent.id),
                    color=COLORS['text_light'], fontsize=7, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', fc=COLORS['panel_bg'], ec='none'))


###################################
def plot_city(city, ax, alpha='ff', arrows=True, gridlines=False, block_color=None):
    """Draw the city layout. arrows=False hides turn arrows and parking
    labels (useful as a quiet background for heatmaps); gridlines=True draws
    faint cell borders; block_color overrides the color of the blocks."""
    city_grid, obstacles, potholes = city.city_grid, city.obstacles, city.potholes
    h, w = city_grid.shape
    # Marker sizes follow the cell size on screen, so small panels stay readable
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    cell = min(bbox.width / w, bbox.height / h) * 72

    # numeric codes for the grid -> color lookup
    b, p, r, s, z = -1, 9, 10, 1, 2
    values = dict(zip(['b', 'p', 'r', 't', 'l', 's', 'z'], [b, p, r, r, r, s, z]))
    grid = np.array([values[city_grid[i, j][0]] for i in range(h) for j in range(w)]).reshape((h, w))

    color_dict = {
        s: _hex_alpha(COLORS['sidewalk'], alpha),
        z: _hex_alpha(COLORS['zebra'], alpha),
        b: _hex_alpha(block_color or COLORS['goal'], alpha),
        r: _hex_alpha(COLORS['road'], alpha),
        p: _hex_alpha(COLORS['parking'], alpha),
    }
    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)

    for obst in obstacles:
        ax.plot([obst[1]], [obst[0]], 'H', markersize=min(7, 1.1 * cell),
                 markerfacecolor=COLORS['obstacle'], markeredgecolor=COLORS['obstacle_edge'],
                 markeredgewidth=min(1, .08 * cell), zorder=3)

    for pth in potholes:
        ax.plot([pth[1]], [pth[0]], 's', markersize=min(5, .55 * cell),
                 markerfacecolor=COLORS['pothole'], markeredgecolor=COLORS['pothole'],
                 markeredgewidth=0.5, alpha=int(alpha, 16) / 255.)

    if gridlines:
        ax.set_xticks(np.arange(-.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-.5, h, 1), minor=True)
        ax.grid(which='minor', color=COLORS['grid'], linewidth=.4)
    ax.grid(which='major', visible=False)
    _frame(ax, h, w)
    if not arrows:
        return

    parking_cells = [tuple(x) for x in np.argwhere(
        np.array([city_grid[i, j][0] == 'p' for i in range(h) for j in range(w)]).reshape((h, w)))]
    for pc in parking_cells:
        ax.add_patch(Circle((pc[1], pc[0]), radius=0.22, facecolor='white',
                             edgecolor=COLORS['parking'], linewidth=1, alpha=0.85, zorder=1.5))
        ax.text(pc[1], pc[0], 'P', color=COLORS['parking'], fontsize=8,
                fontweight='bold', ha='center', va='center', zorder=2)

    rt_cells = [tuple(x) for x in np.argwhere(
        np.array([city_grid[i, j][0] == 't' for i in range(h) for j in range(w)]).reshape((h, w)))]
    for cell in rt_cells:
        dirc = city_grid[cell][1:]
        if dirc == 'NE':
            draw_arrow(ax, dirc, (cell[0] + 2, cell[1]))
            if cell[0] + 2 < h and cell[1] > 0:
                pos = (cell[0] + 2, cell[1] - 1)
                draw_arrow(ax, city_grid[pos][1:], pos)
        if dirc == 'NW':
            draw_arrow(ax, dirc, (cell[0], cell[1] + 2))
            if cell[0] + 1 < h and cell[1] + 2 < w:
                pos = (cell[0] + 1, cell[1] + 2)
                draw_arrow(ax, city_grid[pos][1:], pos)
        if dirc == 'SE':
            draw_arrow(ax, dirc, (cell[0], cell[1] - 2))
            if cell[0] > 0 and cell[1] > 1:
                pos = (cell[0] - 1, cell[1] - 2)
                draw_arrow(ax, city_grid[pos][1:], pos)
        if dirc == 'SW':
            draw_arrow(ax, dirc, (cell[0] - 2, cell[1]))
            if cell[0] > 1 and cell[1] + 1 < w:
                pos = (cell[0] - 2, cell[1] + 1)
                draw_arrow(ax, city_grid[pos][1:], pos)


def _frame(ax, h, w):
    ax.set_xlim(-.5, w - .5)
    ax.set_ylim(h - .5, -.5)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_arrow(ax, dir, pos):
    if dir == 'NE':
        ax.arrow(pos[1], pos[0] + .35, 0, -.5, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
        ax.arrow(pos[1], pos[0] + .05, .25, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    if dir == 'NW':
        ax.arrow(pos[1] + .35, pos[0], -.5, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
        ax.arrow(pos[1] + .05, pos[0], 0, -.25, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    if dir == 'SE':
        ax.arrow(pos[1] - .35, pos[0], .5, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
        ax.arrow(pos[1] - .05, pos[0], 0, .25, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    if dir == 'SW':
        ax.arrow(pos[1], pos[0] - .35, 0, .5, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
        ax.arrow(pos[1], pos[0] - .05, -.25, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    elif dir == 'N ':
        ax.arrow(pos[1], pos[0] + .35, 0, -.5, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    elif dir == 'S ':
        ax.arrow(pos[1], pos[0] - .35, 0, .5, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    elif dir == ' E':
        ax.arrow(pos[1] - .35, pos[0], .5, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    elif dir == ' W':
        ax.arrow(pos[1] + .35, pos[0], -.5, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')


def _add_legend(ax):
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['walker'],
               markersize=8, label='Pedestrian'),
        Line2D([0], [0], marker='v', color='none', markerfacecolor=COLORS['car_active'],
               markersize=9, label='Moving car'),
        Line2D([0], [0], marker='v', color='none', markerfacecolor=COLORS['car_parked'],
               markersize=9, label='Parked car'),
        Line2D([0], [0], marker='H', color='none', markerfacecolor=COLORS['obstacle'],
               markeredgecolor=COLORS['obstacle_edge'], markersize=8, label='Obstacle'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor=COLORS['pothole'],
               markersize=7, label='Pothole'),
        Line2D([0], [0], marker='*', color='none', markerfacecolor=COLORS['collision'],
               markersize=11, label='Collision'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7.5, framealpha=0.9,
              facecolor='white', edgecolor='none', borderpad=0.6, labelspacing=0.5)


def plot_stamp(model, ax):
    plot_city(model.city, ax, alpha='80')


def city_legend(ax, loc='upper left', extras=('parking', 'obstacle', 'pothole', 'turn'), **kwargs):
    """Legend naming every ground type drawn by plot_city, plus the
    optional elements listed in extras."""
    def patch(color, label):
        return Patch(facecolor=color, edgecolor='#00000033', linewidth=.5, label=label)
    handles = [patch(COLORS['road'], 'Road'), patch(COLORS['sidewalk'], 'Sidewalk'),
               patch(COLORS['zebra'], 'Zebra crossing'), patch(COLORS['goal'], 'Building')]
    optional = {
        'parking': patch(COLORS['parking'], 'Parking'),
        'obstacle': Line2D([0], [0], marker='H', color='none', markerfacecolor=COLORS['obstacle'],
                           markeredgecolor=COLORS['obstacle_edge'], markersize=7, label='Obstacle'),
        'pothole': Line2D([0], [0], marker='s', color='none', markerfacecolor=COLORS['pothole'],
                          markersize=5, label='Pothole'),
        'turn': Line2D([0], [0], marker=r'$\rightarrow$', color='none', markerfacecolor=COLORS['ink'],
                       markeredgecolor=COLORS['ink'], markersize=9, label='Permitted turn'),
    }
    handles += [optional[k] for k in extras]
    style = dict(fontsize=7.5, framealpha=.92, facecolor='white', edgecolor='none',
                 borderpad=.6, labelspacing=.5, frameon=True)
    style.update(kwargs)
    return ax.legend(handles=handles, loc=loc, **style)


def _mark_endpoints(ax, start, goal):
    """Start and goal symbols that differ in shape and tone and stay neutral
    so they never compete with route colors: the start is an origin mark
    (white disc, dark ring, center dot) and the goal a dark square
    with a white flag. Returns legend handles."""
    dark, light = COLORS['ink'], 'white'
    flag = '$\u2691$'      # flag glyph, present in matplotlib's bundled DejaVu Sans
    # Start: origin mark
    ax.plot(start[1], start[0], 'o', ms=13, mfc=light, mec=dark, mew=2.4, zorder=5)
    ax.plot(start[1], start[0], 'o', ms=4.5, mfc=dark, mec=dark, zorder=5.1)
    # Goal: flag on a dark tile, white halo to lift it off the road
    ax.plot(goal[1], goal[0], marker='s', ls='none', ms=14, mfc=dark, mec=light, mew=1.5, zorder=5)
    ax.plot(goal[1] + .03, goal[0], marker=flag, ls='none', ms=9, mfc=light, mec=light, mew=0, zorder=5.1)

    start_h = (Line2D([], [], marker='o', ls='none', ms=12, mfc=light, mec=dark, mew=2.2),
               Line2D([], [], marker='o', ls='none', ms=4, mfc=dark, mec=dark))
    goal_h = (Line2D([], [], marker='s', ls='none', ms=13, mfc=dark, mec=dark),
              Line2D([], [], marker=flag, ls='none', ms=9, mfc=light, mec=light, mew=0))
    return [_Labeled(start_h, 'Start'), _Labeled(goal_h, 'Goal')]


class _Labeled(tuple):
    """Tuple of artists drawn as one legend entry, carrying its own label."""
    def __new__(cls, artists, label):
        obj = super().__new__(cls, artists)
        obj.label = label
        return obj

    def get_label(self):
        return self.label


def plot_route(city, ax, agents, cmap=[], crop=None):
    """Planned routes of several agents sharing start and goal.

    Routes are solid lines in colorblind-safe colors, offset from each other
    so overlapping segments stay visible, with an arrowhead every few cells
    showing travel direction. Start and goal are neutral symbols so they are
    not confused with any route color. The legend reports beta, rho and the
    ratio D of route length to shortest (Manhattan) length for each route,
    with the Start and Goal entries in a column to their right.
    crop=(r0, r1, c0, c1) limits the view to a window of the grid."""
    cmap = list(cmap) if cmap else CATEGORICAL[:len(agents)]

    plot_city(city, ax, alpha='aa')
    h, w = city.city_grid.shape
    dir_to_vector = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
    n = len(agents)
    handles = []

    for i, agent in enumerate(agents):
        # Perpendicular offset: routes fan out symmetrically around the cell center
        off = (i - (n - 1) / 2) * .18
        xs, ys, dr, dc = [], [], 0, 0
        current = agent.start
        while current != agent.goal:
            (row, col), d = current, agent.instructions[current]
            dr, dc = dir_to_vector[d]
            xs.append(col + off * (dr != 0)); ys.append(row + off * (dc != 0))
            current = (row + dr, col + dc)
        xs.append(agent.goal[1] + off * (dr != 0)); ys.append(agent.goal[0] + off * (dc != 0))
        line, = ax.plot(xs, ys, '-', color=cmap[i], linewidth=2.4, solid_capstyle='round', zorder=3,
                        path_effects=[pe.Stroke(linewidth=3.6, foreground='white', alpha=.8), pe.Normal()])
        for k in range(2, len(xs) - 1, 4):
            ax.annotate('', xy=(xs[k + 1], ys[k + 1]), xytext=(xs[k], ys[k]), zorder=3.5,
                        arrowprops=dict(arrowstyle='-|>,head_width=.35,head_length=.6',
                                        color=cmap[i], linewidth=0, shrinkA=0, shrinkB=0))
        ratio = len(agent.instructions) / (manhattan(agent.start, agent.goal) + 1)
        line.set_label(r'$\beta$={}, $\rho$={}, D={:.1f}'.format(agent.beta, agent.rho, ratio))
        handles.append(line)

    endpoints = _mark_endpoints(ax, agents[0].start, agents[0].goal)

    if crop:
        r0, r1, c0, c1 = crop
        ax.set_xlim(c0 - .5, c1 - .5)
        ax.set_ylim(r1 - .5, r0 - .5)
    # Legend below the map so it never hides a route: routes in the first
    # column, Start and Goal in the second (columns are filled top to bottom,
    # so both are padded with invisible entries to the same length)
    rows = max(len(handles), len(endpoints))
    blank = Line2D([], [], ls='none', label=' ')
    entries = handles + [blank] * (rows - len(handles)) + endpoints + [blank] * (rows - len(endpoints))
    ax.legend(handles=entries, labels=[e.get_label() for e in entries],
              loc='upper center', bbox_to_anchor=(.5, -.01), ncol=2, columnspacing=2.5,
              fontsize=12, handlelength=2.2, frameon=False,
              handler_map={tuple: HandlerTuple(ndivide=1)})
    ax.set_axis_off()


def plot_next(city_grid, ax, agents, cmap=[]):
    plot_city(city_grid, ax)
    ax.plot([11], [11], 'v', markersize=20, markerfacecolor='y',
             markeredgecolor='r', markeredgewidth=2)

    for i in range(len(agents)):
        agent = agents[i]
        cur_grid = agent.start
        cur_pos = agent.real_position
        m = 'v' if i == 0 else '>'
        ax.plot([cur_pos[1]], [cur_pos[0]], cmap[i] + m, markersize=10)
        for l in range(len(agent.next_positions)):
            dir = agent.ACTION_MAP[agent.instructions[cur_grid]]
            new_pos, new_grid = agent.intent_move(cur_pos, dir)
            ax.plot([new_pos[1]], [new_pos[0]], cmap[i] + m, markersize=5)
            cur_pos, cur_grid = new_pos, new_grid
        ax.plot([cur_pos[1]], [cur_pos[0]], cmap[i] + m, markersize=5)