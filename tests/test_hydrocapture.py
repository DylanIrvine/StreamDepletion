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


def test_hunt03_against_strmdepl08():
    # USGS STRMDEPL08 example 3, as used in the pycap-dss tests
    T = 0.0115740740740741 * 86400
    K = 1.1574074074074073e-05 * 86400
    days = np.array([50.0, 100.0, 200.0, 300.0])
    got = so._hunt03_exact(days, 500, T, 0.001, K, 20, 15, 0.1, 5)
    assert got == pytest.approx(np.array([0.0340, 0.0567, 0.0885, 0.1110]) / 0.557, abs=5e-4)


def test_hunt03_interpolation_and_shape():
    t = np.arange(0, 4000.0)
    fast = so.hunt03(t, 500, **{k: v for k, v in P03.items()})
    check = np.array([1, 7, 90, 1500, 3999])
    exact = so._hunt03_exact(t[check], 500, **P03)
    assert fast[check] == pytest.approx(exact, abs=1e-4)
    assert (np.diff(fast) >= -1e-9).all() and fast[0] == 0 and fast[-1] < 1
    # slower than Hunt (1999) with the same streambed conductance
    lam = P03['K_aq'] * P03['width'] / P03['B_below']
    assert (fast[100:] <= so.hunt99(t[100:], 500, 1000.0, 0.001, lam) + 1e-9).all()


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
