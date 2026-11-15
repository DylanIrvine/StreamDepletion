# tests (no network needed). Run with: pytest tests/

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from hydrocapture import engine, solutions as so, units, workbook

APP = str(Path(__file__).resolve().parents[1] / 'app.py')
BORES = [dict(name='Bore 1', x=500.0, y=0.0, rate=864.0, rate_entered=10.0),
         dict(name='Bore 2', x=-1200.0, y=300.0, rate=432.0, rate_entered=5.0)]
P99 = dict(T=500.0, S=0.1, lam=1.0)
P03 = dict(T=1000.0, S=0.001, K_aq=1.0, B_sat=20.0, B_below=15.0, sigma=0.1, width=5.0)


def test_glover_published_values():
    # Glover and Balmer (1954) table 1, wells 1 to 3 (SI seconds and metres)
    values = [so.glover(1.5777e8, d, 0.1, 0.2)[0] for d in (1000, 5000, 10000)]
    assert values == pytest.approx([0.9365, 0.6906, 0.4259], abs=5e-4)
    assert so.glover(0, 100, 10, 0.1)[0] == 0


def test_hunt99_limits():
    t = np.array([1.0, 10.0, 100.0, 1826.0])
    # very high conductance gives Glover, very low gives almost nothing
    assert so.hunt99(t, 1000, 8640, 0.2, 1e7) == pytest.approx(so.glover(t, 1000, 8640, 0.2), abs=1e-5)
    assert so.hunt99(t, 1000, 8640, 0.2, 1e-9).max() < 1e-6
    assert (so.hunt99(t, 1000, 8640, 0.2, 0.864) < so.glover(t, 1000, 8640, 0.2)).all()


def test_hantush_matches_hunt99():
    t = np.array([5.0, 50.0, 500.0])
    T, lam = 800.0, 2.0
    assert so.hantush(t, 300, T, 0.05, 2 * T / lam) == pytest.approx(so.hunt99(t, 300, T, 0.05, lam))


def test_hunt03_against_independent_references():
    # USGS STRMDEPL08 example 3 (as in the pycap-dss tests)
    T = 0.0115740740740741 * 86400
    K = 1.1574074074074073e-05 * 86400
    days = np.array([50.0, 100.0, 200.0, 300.0])
    got = so.hunt03(days, 500, T, 0.001, K, 20, 15, 0.1, 5)
    # Talbot inversion of the Laplace-domain form at 30 digits (mpmath)
    assert got == pytest.approx([0.0611102, 0.102585, 0.158928, 0.199134], abs=2e-6)
    # STRMDEPL08 and the series form share small numerical errors of up to about 0.001
    assert got == pytest.approx(np.array([0.0340, 0.0567, 0.0885, 0.1110]) / 0.557, abs=1e-3)
    assert got == pytest.approx(so._hunt03_closed_form(days, 500, T, 0.001, K, 20, 15, 0.1, 5), abs=1e-3)
    # 1D finite-volume model of the same system (grid refined to 1.25 m), converging upward
    fv = so.hunt03(np.array([96.6, 344.0]), 500, T, 0.001, K, 20, 15, 0.1, 5)
    assert fv == pytest.approx([0.10014, 0.21383], abs=1e-4)


def test_hunt03_shape_and_limits():
    t = np.arange(0, 4000.0)
    dep = so.hunt03(t, 500, **P03)
    assert (np.diff(dep) >= -1e-9).all() and dep[0] == 0 and dep[-1] < 1
    # slower than Hunt (1999) with the same streambed conductance
    lam = P03['K_aq'] * P03['width'] / P03['B_below']
    assert (dep[100:] <= so.hunt99(t[100:], 500, 1000.0, 0.001, lam) + 1e-9).all()
    # with no aquitard storage it is Hunt (1999)
    p = dict(P03, sigma=1e-12)
    assert so.hunt03(t[1:], 500, **p) == pytest.approx(so.hunt99(t[1:], 500, 1000.0, 0.001, lam), abs=1e-6)


# %% drawdown

def _theis(t, r2, T, S):
    from scipy import special
    return special.exp1(r2 * S / (4 * T * t)) / (4 * np.pi * T)


def test_drawdown_limits():
    t = np.array([1.0, 10.0, 100.0, 1000.0])
    d, x, y, T, S = 300.0, 150.0, 100.0, 500.0, 0.1
    r1, r2 = (x - d) ** 2 + y * y, (x + d) ** 2 + y * y
    glover = _theis(t, r1, T, S) - _theis(t, r2, T, S)
    assert so.drawdown_glover(t, d, x, y, T, S) == pytest.approx(glover)
    assert so.drawdown_glover(t, d, -x, y, T, S).max() == 0
    assert so.drawdown_hunt99(t, d, x, y, T, S, 1e7) == pytest.approx(glover, rel=1e-5)
    assert so.drawdown_hunt99(t, d, x, y, T, S, 1e-9) == pytest.approx(_theis(t, r1, T, S), rel=1e-5)
    assert so.drawdown_hantush(t, d, x, y, T, S, 1e-6) == pytest.approx(glover, rel=1e-5)
    assert so.drawdown_hantush(t, d, x, y, T, S, 1e9) == pytest.approx(
        _theis(t, r1, T, S) + _theis(t, r2, T, S), rel=1e-5)
    assert np.isnan(so.drawdown_hantush(t, d, -x, y, T, S, 10.0)).all()
    assert so.drawdown_glover(np.array([0.0]), d, x, y, T, S)[0] == 0


def test_drawdown_against_quadrature():
    # Hunt (1999) eq. 31 and the Hantush (1965) image integral, by adaptive quadrature
    from scipy import integrate, special
    T, S, d = 500.0, 0.1, 300.0
    for x, y, lam in [(150.0, 400.0, 0.01), (-500.0, 300.0, 1.0), (299.0, 0.0, 100.0)]:
        for t in (10.0, 1000.0):
            a = S / (4 * T * t)
            smear = integrate.quad(lambda th: np.exp(-th) * special.exp1(
                ((abs(x) + d + 2 * T * th / lam) ** 2 + y * y) * a), 0, np.inf, limit=300)[0]
            ref = (special.exp1(((x - d) ** 2 + y * y) * a) - smear) / (4 * np.pi * T)
            assert so.drawdown_hunt99(np.array([t]), d, x, y, T, S, lam)[0] == pytest.approx(ref, rel=5e-4)
    for x, y, L in [(150.0, 0.0, 1.0), (600.0, -200.0, 100.0)]:
        t = 100.0
        a = S / (4 * T * t)
        smear = integrate.quad(lambda xi: np.exp(-xi / L) / L * special.exp1(((x + d + xi) ** 2 + y * y) * a),
                               0, np.inf, limit=300)[0]
        ref = (special.exp1(((x - d) ** 2 + y * y) * a) + special.exp1(((x + d) ** 2 + y * y) * a)
               - 2 * smear) / (4 * np.pi * T)
        assert so.drawdown_hantush(np.array([t]), d, x, y, T, S, L)[0] == pytest.approx(ref, rel=1e-5)


def test_hunt03_drawdown():
    # fixed-Talbot inversion with complex Bessel functions and adaptive quadrature
    T, S, d = 500.0, 0.001, 300.0
    cases = [((150.0, 100.0, 0.5), 1.0, 7.022526e-05), ((150.0, 100.0, 0.5), 10.0, 2.232130e-04),
             ((-200.0, 0.0, 0.5), 10.0, 2.994378e-05), ((150.0, 100.0, 0.002), 1000.0, 9.347264e-04),
             ((400.0, -500.0, 0.05), 1000.0, 5.624793e-04)]
    for (x, y, K), t, ref in cases:
        p = dict(T=T, S=S, K_aq=K, B_sat=10.0, B_below=8.0, sigma=0.1, width=10.0)
        assert so.drawdown_hunt03(np.array([t]), d, x, y, **p)[0] == pytest.approx(ref, rel=1e-4)
    # no aquitard storage gives Hunt (1999); interpolation on long runs matches the direct values
    p = dict(T=T, S=S, K_aq=0.5, B_sat=10.0, B_below=8.0, sigma=1e-12, width=10.0)
    t = np.array([1.0, 30.0, 900.0])
    assert so.drawdown_hunt03(t, d, 150.0, 100.0, **p) == pytest.approx(
        so.drawdown_hunt99(t, d, 150.0, 100.0, T, S, 0.5 * 10.0 / 8.0), rel=1e-6)
    p['sigma'] = 0.1
    t = np.arange(0, 3000.0)
    fast = so.drawdown_hunt03(t, d, 150.0, 100.0, **p)
    check = np.array([1, 5, 50, 400, 2999])
    assert fast[check] == pytest.approx(so._drawdown_hunt03_exact(t[check], d, 150.0, 100.0, **p), rel=2e-5)


def test_observation_superposition_and_triggers():
    obs = [dict(name='Obs 1', x=250.0, y=100.0, trigger=0.3), dict(name='Obs 2', x=-150.0, y=0.0, trigger=None)]
    schedule = engine.constant_schedule(BORES, '2026-01-01', 800, 365)
    dd = engine.observation_drawdown(schedule, BORES, obs, 'Hunt (1999)', P99)
    t = np.arange(1, 801.0)
    direct = sum(b['rate'] * (so.drawdown('Hunt (1999)', t, b, obs[0], P99)
                              - so.drawdown('Hunt (1999)', np.maximum(t - 365, 0), b, obs[0], P99)) for b in BORES)
    assert dd['Obs 1'].to_numpy() == pytest.approx(direct, rel=1e-9, abs=1e-12)
    # the position along the stream now matters
    moved = [dict(BORES[0], y=2000.0), BORES[1]]
    assert engine.observation_drawdown(schedule, moved, obs, 'Hunt (1999)', P99)['Obs 1'].max() < dd['Obs 1'].max()
    limit = 0.2 * 1296.0
    a = engine.trigger_analysis(schedule, BORES, obs, 'Hunt (1999)', P99, dict(T=20, S=0, lam=0), limit)
    assert a['n_cases'] == 3 and (a['low'] <= a['entered'] + 1e-12).all().all()
    dep = engine.total_depletion(schedule, BORES, 'Hunt (1999)', P99).sum(axis=1).to_numpy()
    day = engine.first_reached(dep, limit)
    assert a['limit_days'][0] == day and dep[day] >= limit > dep[day - 1]
    assert a['at_limit'][0]['Obs 1'] == pytest.approx(dd['Obs 1'].iloc[day])
    ts = engine.time_series(schedule, engine.total_depletion(schedule, BORES, 'Hunt (1999)', P99))
    table = engine.trigger_summary(a, obs, ts['Date'], True)
    first = table.iloc[0]
    assert first['Lowest case (m)'] <= first['Drawdown when depletion reaches the limit (m)'] <= first['Highest case (m)']
    assert first['Trigger first reached, earliest case'] <= first['Trigger first reached']


def test_superposition_matches_direct_sum():
    schedule = engine.constant_schedule(BORES, '2026-01-01', 2000, 365)
    dep = engine.total_depletion(schedule, BORES, 'Hunt (1999)', P99)
    t = np.arange(1, 2001.0)
    direct = 864 * (so.hunt99(t, 500, **P99) - so.hunt99(np.maximum(t - 365, 0), 500, **P99))
    assert dep['Bore 1'].to_numpy() == pytest.approx(direct, abs=1e-8)
    # depletion keeps going after pumping stops, then falls
    assert dep['Bore 1'].iloc[400] > 0 and dep['Bore 1'].iloc[-1] < dep['Bore 1'].iloc[364]
    # the other bank counts by distance only
    mirrored = [dict(BORES[1], x=1200.0)]
    assert engine.total_depletion(schedule, mirrored, 'Hunt (1999)', P99)['Bore 2'].to_numpy() == pytest.approx(
        dep['Bore 2'].to_numpy())


def test_volumes_and_band():
    schedule = engine.seasonal_schedule(BORES, '2026-01-01', 365 * 30, [5, 6, 7, 8, 9, 10], 3)
    dep = engine.total_depletion(schedule, BORES, 'Glover and Balmer (1954)', dict(T=500.0, S=0.1))
    band = engine.uncertainty_band(schedule, BORES, 'Glover and Balmer (1954)', dict(T=500.0, S=0.1), dict(T=30, S=0))
    ts = engine.time_series(schedule, dep, band)
    assert schedule.index[150].month == 5 and schedule['Bore 1'].iloc[150] == 864 and schedule['Bore 1'].iloc[10] == 0
    assert schedule.iloc[365 * 4:].sum().sum() == 0
    # the stream cannot lose more than was pumped, and in time it loses nearly all of it
    assert ts['Cumulative volume depleted'].iloc[-1] <= ts['Cumulative volume pumped'].iloc[-1]
    assert ts['Depleted as % of pumped'].iloc[-1] > 80
    assert (ts['Depletion, low'] <= ts['Depletion, total'] + 1e-9).all()
    assert (ts['Depletion, high'] >= ts['Depletion, total'] - 1e-9).all()
    assert engine.uncertainty_band(schedule, BORES, 'Glover and Balmer (1954)', dict(T=500.0, S=0.1), dict(T=0, S=0)) is None
    m = engine.monthly(ts)
    assert m['Volume pumped'].sum() == pytest.approx(ts['Cumulative volume pumped'].iloc[-1])


def test_uploaded_series():
    csv, _ = engine.series_template(BORES, 'L/s')
    table = pd.read_csv(io.BytesIO(csv))
    schedule = engine.series_schedule(BORES, table, 500, units.FLOW['L/s'])
    assert list(schedule.columns) == ['Bore 1', 'Bore 2']
    assert schedule.index[0].month == 1 and schedule['Bore 1'].iloc[0] == 0
    assert schedule['Bore 1'].iloc[140] == pytest.approx(864) and schedule['Bore 2'].iloc[140] == pytest.approx(432)
    assert schedule.iloc[-1].sum() == 0
    with pytest.raises(ValueError):
        engine.series_schedule(BORES, table.drop(columns=['Bore 2']), 500)


def test_summary_and_workbook():
    import openpyxl
    summary = engine.bore_summary(BORES, 'Glover and Balmer (1954)', dict(T=500.0, S=0.1))
    row = summary.iloc[0]
    assert row['Stream depletion factor, d²S/T (days)'] == pytest.approx(50)
    # for Glover, 50% depletion comes at about 1.1 times the stream depletion factor
    assert row['Days to 50% depletion'] == pytest.approx(50 * 1.099, rel=0.01)
    schedule = engine.constant_schedule(BORES, '2026-01-01', 400, 100)
    ts = engine.time_series(schedule, engine.total_depletion(schedule, BORES, 'Hunt (1999)', P99))
    data = workbook.build(ts, engine.monthly(ts), summary, pd.DataFrame(BORES),
                          pd.DataFrame([('a', 'b')], columns=['Setting', 'Value']))
    assert openpyxl.load_workbook(io.BytesIO(data), read_only=True).sheetnames == [
        'Time series', 'Monthly', 'Summary', 'Bores', 'Settings']
    data = workbook.build(ts, engine.monthly(ts), summary, pd.DataFrame(BORES),
                          pd.DataFrame([('a', 'b')], columns=['Setting', 'Value']),
                          pd.DataFrame({'Date': ts['Date'], 'Drawdown': 0.0}), pd.DataFrame({'Obs': ['A']}))
    assert openpyxl.load_workbook(io.BytesIO(data), read_only=True).sheetnames == [
        'Time series', 'Monthly', 'Summary', 'Bores', 'Drawdown', 'Observation bores', 'Settings']


def test_units():
    assert units.FLOW['L/s'] == pytest.approx(86.4)
    assert units.FLOW['ML/d'] == 1000 and units.TRANSMISSIVITY['m²/s'] == 86400


@pytest.mark.parametrize('choice', list(so.SOLUTIONS))
def test_app_runs_each_solution(choice):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.selectbox[0].set_value(choice).run()
    assert not at.exception
    next(b for b in at.button if b.label == 'Run').click().run()
    assert not at.exception and not at.error
    assert any(h.value == '7. Results' for h in at.header)


def test_app_observation_bores_and_limit():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox[0].set_value('Hunt (1999)').run()
    limit = next(r for r in at.radio if r.label == 'Link triggers to a stream depletion limit')
    limit.set_value('% of the peak pumping rate').run()
    for n in at.number_input:
        if n.label.startswith('Transmissivity (± %)'):
            n.set_value(20.0)
    at.run()
    next(b for b in at.button if b.label == 'Run').click().run()
    assert not at.exception and not at.error
    assert any(s.value == 'Observation bores and trigger levels' for s in at.subheader)
