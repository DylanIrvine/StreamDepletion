# figures

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.image as mimage
from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea
import numpy as np
from pathlib import Path
from matplotlib.patches import Polygon, Rectangle

# default colours. The app lets the user change them
DEFAULT_COLOURS = {
    'Stream and depletion': '#0089D0',
    'Pumping': '#F37021',
    'Bores and comparison': '#CC004C',
    'Extra 1': '#0DB14B',
    'Extra 2': '#6460AA',
    'Extra 3': '#FCB711',
}
BADGE_ICON = Path(__file__).resolve().parents[1] / 'logo' / 'HydroCAPTURE_badge.png'
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
SOIL = '#D7B984'
BED = '#7A5C3A'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 9,
    'font.weight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})


def new_fig(width_cm=16, height_cm=9):
    return plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))


BADGE_PT = 14       # height of the app icon in the 'Made using' badge, in points


def add_badge(fig, x, y, transform):
    # 'Made using HydroCAPTURE, Dylan Irvine' with the app icon, right-aligned at (x, y). Text only if the icon is missing
    try:
        img = mimage.imread(BADGE_ICON)
    except (FileNotFoundError, OSError):
        img = None
    parts = [TextArea('Made using HydroCAPTURE, Dylan Irvine', textprops=dict(fontsize=6, color='0.5'))]
    if img is not None:
        parts.append(OffsetImage(img, zoom=BADGE_PT / img.shape[0]))
    packed = HPacker(children=parts, align='center', pad=0, sep=3)
    fig.add_artist(AnnotationBbox(packed, (x, y), xycoords=transform, box_alignment=(1, 0.5),
                                  frameon=False, pad=0, annotation_clip=False))

def add_watermark(fig):
    # the 'Made using' badge bottom right, in a strip below the
    # plot so it can be cropped off
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    below = fig.transFigure + mtransforms.ScaledTranslation(0, -(BADGE_PT / 2 + 4) / 72, fig.dpi_scale_trans)
    add_badge(fig, box.x1 / width, box.y0 / height, below)


def legend_above(ax, ncols=3):
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncols=ncols, borderaxespad=0.2,
              handlelength=1.5, columnspacing=1.2)


def date_axis(ax, dates):
    # a few evenly spaced date ticks, trimmed to the data
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlim(dates.iloc[0], dates.iloc[-1])


def bore_colours(c, n):
    cycle = [c['Bores and comparison'], c['Extra 1'], c['Extra 2'], c['Extra 3']]
    return [cycle[i % len(cycle)] for i in range(n)]


# %% conceptual diagram

def _aerial(ax, bores, unit, factor, c, observations=()):
    xs = [b['x'] / factor for b in bores]
    ys = [b['y'] / factor for b in bores]
    oxs = [o['x'] / factor for o in observations]
    oys = [o['y'] / factor for o in observations]
    all_x, all_y = xs + oxs, ys + oys
    reach = max(max(abs(x) for x in all_x), 1e-9)
    half_y = max(max(abs(y) for y in all_y), 0.35 * reach)
    left = -1.25 * reach if min(all_x) < 0 else -0.25 * reach
    right = 1.25 * reach if max(all_x) > 0 else 0.25 * reach
    ax.axvspan(-0.03 * reach, 0.03 * reach, color=c['Stream and depletion'], alpha=0.6, lw=0)
    ax.text(0, 1.22 * half_y, 'Stream', ha='center', va='bottom', color=c['Stream and depletion'])
    for b, x, y, colour in zip(bores, xs, ys, bore_colours(c, len(bores))):
        ax.plot([0, x], [y, y], color='0.5', lw=0.8, ls='--')
        ax.plot(x, y, 'o', color=colour, ms=6, zorder=3)
        ax.annotate(b['name'], (x, y), textcoords='offset points', xytext=(0, 7), ha='center')
    for o, x, y in zip(observations, oxs, oys):
        ax.plot(x, y, '^', mfc='white', mec='0.2', mew=1.1, ms=6, zorder=3)
        ax.annotate(o['name'], (x, y), textcoords='offset points', xytext=(0, -12), ha='center',
                    color='0.25', fontsize=8)
    x0, y0 = xs[0], ys[0]
    ax.annotate('d', (x0 / 2, y0), textcoords='offset points', xytext=(0, -11), ha='center')
    if observations:
        ax.plot([], [], 'o', color=bore_colours(c, 1)[0], ms=5, label='Pumping bore')
        ax.plot([], [], '^', mfc='white', mec='0.2', ms=5, label='Observation bore')
        ax.legend(loc='upper right', fontsize=7, handletextpad=0.3, borderaxespad=0.2)
    ax.set_xlim(left, right)
    ax.set_ylim(-1.2 * half_y, 1.2 * half_y)
    ax.set_xlabel(f'x: distance from the stream ({unit})')
    ax.set_ylabel(f'y: distance along the stream ({unit})')
    ax.text(0.98, 0.02, 'Aerial view', transform=ax.transAxes, color='0.4', ha='right', va='bottom')


def _section(ax, key, c):
    # schematic only. The stream is on the left and one bore on the right
    blue = c['Stream and depletion']
    ground, base, level, bore_x = 5.0, 0.6, 4.55, 7.6
    two_layer = key == 'hunt03'
    aquitard_base = 2.9

    ax.add_patch(Rectangle((0, base), 10, ground - base, color=SOIL, lw=0))
    if two_layer:
        ax.add_patch(Rectangle((0, aquitard_base), 10, ground - aquitard_base, color=BED, alpha=0.3, lw=0))
        ax.plot([0, 10], [aquitard_base, aquitard_base], color=BED, lw=0.8)
    ax.add_patch(Rectangle((0, base - 0.35), 10, 0.35, facecolor='0.75', edgecolor='0.5', hatch='///', lw=0))
    ax.plot([0, 10], [ground, ground], color='0.3', lw=1)
    ax.plot([0, 10], [base, base], color='0.3', lw=1)

    # stream
    if key in ('glover', 'hantush'):
        ax.add_patch(Rectangle((1.0, base), 0.7, level - base, color=blue, alpha=0.75, lw=0))
        ax.add_patch(Rectangle((1.0, level), 0.7, ground - level, color='white', lw=0))
        if key == 'hantush':
            ax.add_patch(Rectangle((1.7, base), 0.14, level - base, color=BED, lw=0))
            ax.annotate("Streambed: K', b'", (1.84, 3.0), xytext=(2.4, 3.3),
                        arrowprops=dict(arrowstyle='-', color='0.3', lw=0.6))
        stream_edge = 1.7
    else:
        bottom = 4.0
        ax.add_patch(Polygon([(0.9, ground), (1.1, bottom), (1.9, bottom), (2.1, ground)], color='white', lw=0))
        ax.add_patch(Polygon([(0.98, level), (1.1, bottom), (1.9, bottom), (2.02, level)], color=blue, alpha=0.75, lw=0))
        ax.annotate('', (1.1, 4.25), (1.9, 4.25), arrowprops=dict(arrowstyle='<->', color='k', lw=0.6))
        ax.text(1.5, 4.29, 'w', ha='center', va='bottom', fontsize=8)
        if key == 'hunt99':
            ax.add_patch(Rectangle((1.1, bottom - 0.14), 0.8, 0.14, color=BED, lw=0))
            ax.annotate("Streambed: K', b'", (1.5, bottom - 0.14), xytext=(1.2, 2.9),
                        arrowprops=dict(arrowstyle='-', color='0.3', lw=0.6))
        else:
            ax.annotate('', (1.5, aquitard_base), (1.5, bottom), arrowprops=dict(arrowstyle='<->', color='k', lw=0.6))
            ax.text(1.58, (aquitard_base + bottom) / 2, "B''", va='center')
            ax.annotate('', (2.9, aquitard_base), (2.9, level - 0.05), arrowprops=dict(arrowstyle='<->', color='k', lw=0.6))
            ax.text(2.98, 3.6, "B'", va='center')
            ax.text(4.9, 3.5, "Aquitard: K', σ", ha='center', va='center')
        stream_edge = 2.0
    ax.text(1.4, ground + 0.12, 'Stream', ha='center', va='bottom', color=blue)

    # water table, and for the two-layer case the head in the pumped aquifer
    x = np.linspace(stream_edge, 10, 300)
    cone = np.exp(-np.abs(x - bore_x) / 0.9)
    table = level - (0.25 if two_layer else 1.3) * cone - 0.05 * (x - stream_edge) / 8
    ax.plot(x, table, color=blue, lw=1.2)
    ax.plot([0, 1.0 if key in ('glover', 'hantush') else 0.98], [level, level], color=blue, lw=1.2)
    ax.text(9.9, table[-1] + 0.1, 'Water table', ha='right', va='bottom', color=blue, fontsize=8)
    if two_layer:
        head = level - 0.15 - 1.9 * cone
        ax.plot(x, head, color=blue, lw=1, ls='--')
        ax.annotate('Head in the pumped aquifer', (6.6, np.interp(6.6, x, head)), xytext=(2.3, 2.35),
                    color=blue, fontsize=8, arrowprops=dict(arrowstyle='-', color=blue, lw=0.6))

    # bore
    screen_top = 2.2 if two_layer else 2.6
    ax.plot([bore_x, bore_x], [screen_top, ground + 0.45], color='k', lw=1.6)
    ax.plot([bore_x, bore_x], [base + 0.2, screen_top], color='k', lw=1.6, ls=(0, (1, 1)))
    ax.annotate('', (bore_x, ground + 1.0), (bore_x, ground + 0.45),
                arrowprops=dict(arrowstyle='->', color=c['Bores and comparison'], lw=1.4))
    ax.text(bore_x + 0.15, ground + 0.7, 'Q', va='center', color=c['Bores and comparison'])
    ax.text(bore_x + 0.15, ground + 0.12, 'Bore', va='bottom')
    ax.annotate('', (1.5, ground + 1.25), (bore_x, ground + 1.25), arrowprops=dict(arrowstyle='<->', color='k', lw=0.6))
    ax.text((1.5 + bore_x) / 2, ground + 1.3, 'd', ha='center', va='bottom')
    ax.text(4.9, 1.4, 'Aquifer: T, S', ha='center', va='center')
    ax.text(5.2, base - 0.18, 'Impermeable base', ha='center', va='center', fontsize=8)

    # bottom of the impermeable base sits on the axes edge, level with the x-axis of the aerial view
    ax.set_xlim(0, 10)
    ax.set_ylim(base - 0.35, 6.9)
    ax.axis('off')
    ax.text(0.02, -0.06, 'Cross section, not to scale', transform=ax.transAxes, color='0.4')


def diagram(key, bores, unit, factor, c, observations=()):
    fig, axes = plt.subplots(1, 2, figsize=(17 / 2.54, 8 / 2.54), gridspec_kw=dict(width_ratios=[1, 1.25]))
    _aerial(axes[0], bores, unit, factor, c, observations)
    _section(axes[1], key, c)
    fig.tight_layout()
    return fig


# %% results

def depletion(ts, bores, unit, factor, c, show_bores=False):
    fig, ax = new_fig()
    dates = ts['Date']
    ax.step(dates, ts['Pumping, total'] / factor, where='post', color=c['Pumping'], lw=1, label='Pumping')
    if 'Depletion, low' in ts:
        ax.fill_between(dates, ts['Depletion, low'] / factor, ts['Depletion, high'] / factor,
                        color=c['Stream and depletion'], alpha=0.25, lw=0, label='Uncertainty range')
    if show_bores and len(bores) > 1:
        for b, colour in zip(bores, bore_colours(c, len(bores))):
            ax.plot(dates, ts[f"Depletion, {b['name']}"] / factor, color=colour, lw=0.9, label=b['name'])
    ax.plot(dates, ts['Depletion, total'] / factor, color=c['Stream and depletion'], lw=1.6, label='Stream depletion')
    date_axis(ax, dates)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f'Rate ({unit})')
    legend_above(ax, ncols=4)
    return fig


def cumulative(ts, unit, factor, c):
    fig, ax = new_fig()
    dates = ts['Date']
    ax.plot(dates, ts['Cumulative volume pumped'] / factor, color=c['Pumping'], lw=1.4, label='Pumped')
    ax.plot(dates, ts['Cumulative volume depleted'] / factor, color=c['Stream and depletion'], lw=1.6,
            label='Taken from the stream')
    date_axis(ax, dates)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f'Cumulative volume ({unit})')
    legend_above(ax)
    return fig


def comparison(days, curves, c):
    # depletion under constant pumping for the chosen solution and the Glover and Balmer case
    fig, ax = new_fig()
    colours = [c['Stream and depletion'], c['Bores and comparison'], c['Extra 1'], c['Extra 2']]
    for (name, values), colour in zip(curves.items(), colours):
        ax.plot(days, 100 * values, color=colour, lw=1.6, label=name)
    ax.set_xlim(0, days[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel('Days of constant pumping')
    ax.set_ylabel('Depletion (% of pumping rate)')
    legend_above(ax)
    return fig


def distance(dist, values, bores, at_days, unit, factor, c):
    fig, ax = new_fig()
    ax.plot(dist / factor, 100 * values, color=c['Stream and depletion'], lw=1.6)
    for b, colour in zip(bores, bore_colours(c, len(bores))):
        d = abs(b['x'])
        ax.plot(d / factor, 100 * np.interp(d, dist, values), 'o', color=colour, ms=6, label=b['name'])
    ax.set_xlim(0, dist[-1] / factor)
    ax.set_ylim(0, 100)
    ax.set_xlabel(f'Distance from the stream ({unit})')
    ax.set_ylabel(f'Depletion after {at_days:g} days (% of pumping rate)')
    legend_above(ax, ncols=4)
    return fig


def monthly(table, unit, factor, c):
    fig, ax = new_fig()
    x = np.arange(len(table))
    ax.bar(x - 0.2, table['Volume pumped'] / factor, width=0.4, color=c['Pumping'], label='Pumped')
    ax.bar(x + 0.2, table['Volume depleted'] / factor, width=0.4, color=c['Stream and depletion'],
           label='Taken from the stream')
    step = max(1, len(table) // 12)
    ticks = x[::step]
    ax.set_xticks(ticks, [f"{MONTHS[m - 1]} {y}" for y, m in zip(table['Year'].iloc[::step], table['Month'].iloc[::step])],
                  rotation=45, ha='right')
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set_ylabel(f'Monthly volume ({unit})')
    legend_above(ax)
    return fig


def drawdown(dates, analysis, observations, unit, factor, c, limit_date=None):
    # drawdown at each observation bore, plotted downwards, with its trigger level
    fig, ax = new_fig(height_cm=10)
    colours = [c['Stream and depletion'], c['Bores and comparison'], c['Extra 1'], c['Extra 2'],
               c['Extra 3'], c['Pumping']]
    banded = analysis['n_cases'] > 1
    for i, ob in enumerate(observations):
        colour = colours[i % len(colours)]
        name = ob['name']
        if banded:
            ax.fill_between(dates, analysis['low'][name] / factor, analysis['high'][name] / factor,
                            color=colour, alpha=0.2, lw=0)
        ax.plot(dates, analysis['entered'][name] / factor, color=colour, lw=1.6, label=name)
        level = ob.get('trigger')
        if level is not None and np.isfinite(level):
            ax.axhline(level / factor, color=colour, lw=1, ls='--')
            ax.annotate(f'Trigger, {name}', (1.0, level / factor), xycoords=('axes fraction', 'data'),
                        xytext=(-2, 2), textcoords='offset points', ha='right', va='bottom',
                        color=colour, fontsize=7)
    if limit_date is not None:
        ax.axvline(limit_date, color='0.35', lw=0.9, ls=':')
        ax.annotate('Depletion reaches the limit', (limit_date, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(3, -3), textcoords='offset points', ha='left', va='top', color='0.35', fontsize=7)
    date_axis(ax, dates)
    ax.set_ylim(bottom=0)
    ax.invert_yaxis()
    ax.set_ylabel(f'Drawdown ({unit})')
    legend_above(ax, ncols=4)
    return fig
