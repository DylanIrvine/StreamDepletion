# pumping schedules, superposition and summary numbers. Metres, days, m3/d throughout

import itertools

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from . import solutions

# parameters that can be given an uncertainty, by solution
UNCERTAIN = {
    'Glover and Balmer (1954)': {'T': 'Transmissivity', 'S': 'Storage coefficient'},
    'Hantush (1965)': {'T': 'Transmissivity', 'S': 'Storage coefficient', 'L': 'Streambed leakance'},
    'Hunt (1999)': {'T': 'Transmissivity', 'S': 'Storage coefficient', 'lam': 'Streambed conductance'},
    'Hunt (2003)': {'T': 'Transmissivity', 'S': 'Storage coefficient',
                    'K_aq': 'Aquitard hydraulic conductivity', 'sigma': 'Aquitard specific yield'},
}


# %% schedules: a table of daily pumping rates, one column per bore

def constant_schedule(bores, start, n_days, pump_days):
    dates = pd.date_range(start, periods=n_days, freq='D')
    on = np.arange(n_days) < pump_days
    return pd.DataFrame({b['name']: np.where(on, b['rate'], 0.0) for b in bores}, index=dates)


def seasonal_schedule(bores, start, n_days, months, pump_years):
    # pump in the chosen calendar months, for the first pump_years years
    dates = pd.date_range(start, periods=n_days, freq='D')
    last = pd.Timestamp(start) + pd.DateOffset(years=int(pump_years))
    on = dates.month.isin(months) & (dates < last)
    return pd.DataFrame({b['name']: np.where(on, b['rate'], 0.0) for b in bores}, index=dates)


def series_schedule(bores, table, n_days, to_m3d=1.0):
    # uploaded rates: each row holds until the next date. Columns are matched by bore name
    table = table.copy()
    table.columns = [str(c).strip() for c in table.columns]
    date_col = table.columns[0]
    table[date_col] = pd.to_datetime(table[date_col], errors='coerce', dayfirst=True)
    table = table.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    missing = [b['name'] for b in bores if b['name'] not in table.columns]
    if missing:
        raise ValueError(f"The pumping file has no column for: {', '.join(missing)}")
    if table.empty:
        raise ValueError('The pumping file has no rows with a readable date.')
    dates = pd.date_range(table.index[0].normalize(), periods=n_days, freq='D')
    rates = table[[b['name'] for b in bores]].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    rates = rates[~rates.index.duplicated(keep='last')]
    return rates.reindex(dates, method='ffill').fillna(0.0) * to_m3d


def series_template(bores, unit):
    # one year of monthly rows as an example, ending with pumping switched off
    dates = pd.date_range(pd.Timestamp.today().normalize().replace(month=1, day=1), periods=13, freq='MS')
    out = pd.DataFrame({'Date': dates.strftime('%d/%m/%Y')})
    for b in bores:
        out[b['name']] = [b['rate_entered'] if 5 <= d.month <= 10 and i < 12 else 0
                          for i, d in enumerate(dates)]
    return out.to_csv(index=False).encode(), f'Rates are in {unit}. Each row holds until the next date.'


# %% superposition

def bore_depletion(rates, d, solution, p):
    # depletion at the end of each day, for daily pumping rates.
    # each day of pumping is a step on then a step off, summed over all days
    n = len(rates)
    f = solutions.response(solution, np.arange(n + 1, dtype=float), d, p)
    kernel = np.diff(f)
    return np.convolve(np.asarray(rates, dtype=float), kernel)[:n]


def total_depletion(schedule, bores, solution, p):
    per_bore = {b['name']: bore_depletion(schedule[b['name']].to_numpy(), abs(b['x']), solution, p)
                for b in bores}
    return pd.DataFrame(per_bore, index=schedule.index)


def parameter_cases(p, percent):
    # the entered values, then every combination of low and high values
    keys = [k for k, v in percent.items() if v > 0]
    cases = [dict(p)]
    for signs in itertools.product((-1, 1), repeat=len(keys)):
        q = dict(p)
        for key, sign in zip(keys, signs):
            q[key] = p[key] * (1 + sign * percent[key] / 100)
        cases.append(q)
    return cases


def uncertainty_band(schedule, bores, solution, p, percent):
    # lowest and highest total depletion over the entered values and every combination
    # of low and high values. The entered case is included because after pumping stops
    # the low and high curves can cross, leaving the entered case outside them
    if not any(v > 0 for v in percent.values()):
        return None
    totals = np.array([total_depletion(schedule, bores, solution, q).sum(axis=1).to_numpy()
                       for q in parameter_cases(p, percent)])
    return totals.min(axis=0), totals.max(axis=0)


# %% observation bores

def observation_drawdown(schedule, bores, observations, solution, p):
    # drawdown at the end of each day at each observation bore, summed over the pumping bores
    n = len(schedule)
    days = np.arange(n + 1, dtype=float)
    out = {}
    for ob in observations:
        total = np.zeros(n)
        for b in bores:
            kernel = np.diff(solutions.drawdown(solution, days, b, ob, p))
            total += fftconvolve(schedule[b['name']].to_numpy(dtype=float), kernel)[:n]
        out[ob['name']] = total
    return pd.DataFrame(out, index=schedule.index)


def first_reached(values, level):
    # index of the first day at or above the level, or None
    hit = np.nonzero(np.asarray(values) >= level)[0]
    return int(hit[0]) if len(hit) else None


def trigger_analysis(schedule, bores, observations, solution, p, percent, limit=None):
    # drawdown at each observation bore for every parameter case, its range, and the drawdown
    # on the first day that total stream depletion reaches the limit (m3/d), for each case
    cases = parameter_cases(p, percent)
    drawdowns, at_limit, limit_days = [], [], []
    for q in cases:
        dd = observation_drawdown(schedule, bores, observations, solution, q)
        drawdowns.append(dd)
        if limit is not None:
            dep = total_depletion(schedule, bores, solution, q).sum(axis=1).to_numpy()
            day = first_reached(dep, limit)
            limit_days.append(day)
            at_limit.append(None if day is None else dd.iloc[day])
    stack = np.array([dd.to_numpy() for dd in drawdowns])
    return dict(entered=drawdowns[0], low=pd.DataFrame(stack.min(axis=0), index=schedule.index,
                                                      columns=drawdowns[0].columns),
                high=pd.DataFrame(stack.max(axis=0), index=schedule.index, columns=drawdowns[0].columns),
                n_cases=len(cases), limit=limit, limit_days=limit_days, at_limit=at_limit)


def trigger_summary(analysis, observations, dates, banded, length=('m', 1.0), dd=('m', 1.0)):
    # one row per observation bore, in the chosen distance and drawdown units.
    # Observation x and y, trigger levels and drawdowns are held in metres
    lu, lf = length
    du, df = dd
    rows = []
    entered, high = analysis['entered'], analysis['high']
    for ob in observations:
        name = ob['name']
        row = {'Observation bore': name, f'x ({lu})': ob['x'] / lf, f'y ({lu})': ob['y'] / lf,
               f'Largest drawdown ({du})': entered[name].max() / df}
        if banded:
            row[f'Largest drawdown, highest case ({du})'] = high[name].max() / df
        level = ob.get('trigger')
        has_level = level is not None and np.isfinite(level)
        row[f'Trigger level ({du})'] = level / df if has_level else None
        if has_level:
            day = first_reached(entered[name], level)
            row['Trigger first reached'] = 'Not reached' if day is None else str(dates[day].date())
            if banded:
                day = first_reached(high[name], level)
                row['Trigger first reached, earliest case'] = 'Not reached' if day is None else str(dates[day].date())
        if analysis['limit'] is not None:
            first = analysis['at_limit'][0]
            row[f'Drawdown when depletion reaches the limit ({du})'] = None if first is None else first[name] / df
            if banded:
                values = [a[name] / df for a in analysis['at_limit'] if a is not None]
                row[f'Lowest case ({du})'] = min(values) if values else None
                row[f'Highest case ({du})'] = max(values) if values else None
        rows.append(row)
    return pd.DataFrame(rows)


# %% results tables

def time_series(schedule, depletion, band=None):
    out = pd.DataFrame(index=schedule.index)
    out.index.name = 'Date'
    out['Day'] = np.arange(1, len(out) + 1)
    for name in schedule.columns:
        out[f'Pumping, {name}'] = schedule[name]
    out['Pumping, total'] = schedule.sum(axis=1)
    for name in depletion.columns:
        out[f'Depletion, {name}'] = depletion[name]
    out['Depletion, total'] = depletion.sum(axis=1)
    if band is not None:
        out['Depletion, low'] = band[0]
        out['Depletion, high'] = band[1]
    # volumes: rates are per day, depletion is averaged over each day
    total = out['Depletion, total'].to_numpy()
    out['Cumulative volume pumped'] = out['Pumping, total'].cumsum()
    out['Cumulative volume depleted'] = np.cumsum(0.5 * (total + np.concatenate(([0.0], total[:-1]))))
    pumped = out['Cumulative volume pumped'].to_numpy()
    out['Depleted as % of pumped'] = np.where(pumped > 0, 100 * out['Cumulative volume depleted'] / np.maximum(pumped, 1e-30), np.nan)
    return out.reset_index()


def monthly(ts):
    d = ts.set_index('Date')
    total = d['Depletion, total'].to_numpy()
    d = d.assign(depleted=0.5 * (total + np.concatenate(([0.0], total[:-1]))))
    g = d.groupby([d.index.year, d.index.month])
    out = pd.DataFrame({'Volume pumped': g['Pumping, total'].sum(), 'Volume depleted': g['depleted'].sum()})
    out.index.names = ['Year', 'Month']
    return out.reset_index()


def crossing_time(solution, d, p, fraction):
    # days of constant pumping until depletion reaches the fraction. None if over about 270 years
    t = np.geomspace(0.01, 1.0e5, 300)
    f = solutions.response(solution, t, d, p)
    if f[-1] < fraction:
        return None
    if f[0] >= fraction:
        return float(t[0])
    return float(np.exp(np.interp(fraction, f, np.log(t))))


def bore_summary(bores, solution, p):
    rows = []
    for b in bores:
        d = abs(b['x'])
        at = solutions.response(solution, np.array([30.0, 90.0, 365.0, 3650.0]), d, p)
        rows.append({
            'Bore': b['name'], 'Distance to stream (m)': d,
            'Stream depletion factor, d²S/T (days)': d * d * p['S'] / p['T'],
            'Depletion after 30 days (% of pumping)': 100 * at[0],
            'Depletion after 90 days (% of pumping)': 100 * at[1],
            'Depletion after 1 year (% of pumping)': 100 * at[2],
            'Depletion after 10 years (% of pumping)': 100 * at[3],
            'Days to 50% depletion': crossing_time(solution, d, p, 0.5),
            'Days to 90% depletion': crossing_time(solution, d, p, 0.9),
        })
    return pd.DataFrame(rows)
