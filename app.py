# HydroCAPTURE
# stream depletion from pumping bores, using analytical solutions.
# Run with: streamlit run app.py

import datetime as dt
import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from hydrocapture import VERSION as APP_VERSION
from hydrocapture import engine, plots, run_counter, solutions, units, workbook

# %% page setup

APP_NAME = 'HydroCAPTURE'
# app address, used in the citation
APP_URL = 'https://hydrocapture.streamlit.app/'
LOGO = 'logo/HydroCAPTURE_logo.png'
FAVICON = 'logo/HydroCAPTURE_icon.png'
MONTHS = plots.MONTHS

st.set_page_config(page_title=APP_NAME, page_icon=FAVICON)

# %% header

head_text, head_logo = st.columns([3, 1], vertical_alignment='center')
with head_text:
    st.title(APP_NAME)
    st.subheader('Stream depletion from pumping bores')
    st.write('Dylan Irvine, Charles Darwin University.')
    st.caption(f'Version {APP_VERSION}')
with head_logo:
    st.image(LOGO, width='stretch')

st.markdown(
    'A bore pumping near a stream takes part of its water from the stream, either by drawing water '
    'out of it or by intercepting groundwater that would have reached it. HydroCAPTURE estimates '
    'how much, and how quickly, using four analytical solutions. Place one or more bores, describe '
    'the aquifer and the stream, set a pumping schedule, and download the results as one Excel '
    'workbook with plots. Optional observation bores give the drawdown that monitoring would see, '
    'to help set trigger levels.\n\n'
    'These solutions suit screening and teaching. They assume a uniform aquifer and a straight '
    'stream, so they do not replace a site-specific model.')

if 'run_count' not in st.session_state:
    st.session_state['run_count'] = run_counter.get_runs()
if st.session_state['run_count'] is not None:
    st.caption(f"This tool has been run {st.session_state['run_count']:,} times.")

with st.expander('**Privacy and data handling**'):
    st.markdown(
        '**Your data**\n'
        '- Everything you enter, and any pumping file you upload, is held in memory on the server '
        'for your session and discarded when it ends. Nothing is saved to a database or to '
        'permanent storage.\n'
        '- The results exist only in the files you download.\n'
        '- Nothing you enter is sent to any other service.\n\n'
        '**You**\n'
        '- No login, and no record kept by this app of who you are or where you are. The app does '
        'not log IP addresses or locations.\n'
        '- The only thing recorded is a single running total: the number of completed runs. It '
        'has no link to any user, location or file.\n\n'
        '**What this does not cover**\n'
        '- Processing happens on the server that hosts the app, not on your computer. If you are '
        'using the public web version, your inputs travel to that host (which may be outside '
        'Australia). The hosting provider and the Streamlit framework may use their own cookies '
        'and keep their own server logs, under their own privacy policies.\n'
        '- If your data are confidential or subject to a data agreement, run the app on your own '
        'machine. The code is open source (MIT).')

with st.expander('**Background: the solutions and their equations**'):
    st.markdown(
        'Each solution gives the stream depletion rate $Q_s$ as a fraction of the pumping rate '
        '$Q$, for a bore at distance $d$ from the stream that has pumped at a constant rate for '
        'time $t$. $T$ is transmissivity and $S$ is the storage coefficient (specific yield for an '
        'unconfined aquifer).')
    st.markdown('**Glover and Balmer (1954).** ' + solutions.SOLUTIONS['Glover and Balmer (1954)']['summary'])
    st.latex(r'\frac{Q_s}{Q} = \mathrm{erfc}\left(\sqrt{\frac{S d^2}{4 T t}}\right)')
    st.markdown('**Hunt (1999).** ' + solutions.SOLUTIONS['Hunt (1999)']['summary'] +
                r" The streambed conductance is $\lambda = K' w / b'$, for a streambed of hydraulic "
                r"conductivity $K'$, thickness $b'$ and width $w$.")
    st.latex(r'\frac{Q_s}{Q} = \mathrm{erfc}\left(\sqrt{\frac{S d^2}{4 T t}}\right) - '
             r'\exp\left(\frac{\lambda^2 t}{4 S T} + \frac{\lambda d}{2 T}\right)'
             r'\mathrm{erfc}\left(\sqrt{\frac{\lambda^2 t}{4 S T}} + \sqrt{\frac{S d^2}{4 T t}}\right)')
    st.markdown('**Hantush (1965).** ' + solutions.SOLUTIONS['Hantush (1965)']['summary'] +
                r" The streambed is described by its leakance $L = K b' / K'$, where $K$ is the "
                r'hydraulic conductivity of the aquifer. The equation has the same form as Hunt '
                r'(1999), with $\lambda$ replaced by $2T / L$.')
    st.markdown('**Hunt (2003).** ' + solutions.SOLUTIONS['Hunt (2003)']['summary'] +
                r" The aquitard has hydraulic conductivity $K'$, saturated thickness $B'$, thickness "
                r"$B''$ beneath the stream and specific yield $\sigma$. The stream has width $w$ and "
                r"$\lambda = K' w / B''$. The result is the Hunt (1999) expression less a correction "
                r'for water released from the aquitard:')
    st.latex(r"\frac{Q_s}{Q} = \left[\frac{Q_s}{Q}\right]_{1999} - \lambda' \int_0^1 "
             r"F(\alpha, t')\, G(\alpha, t')\, d\alpha, \quad t' = \frac{T t}{S d^2}, \quad "
             r"\lambda' = \frac{\lambda d}{T}")
    st.markdown(
        r"$F$ and $G$ are given in Hunt (2003), equations 46 to 52. They depend on $S/\sigma$ and "
        r"on $K' d^2 / (B' T)$. HydroCAPTURE evaluates the same solution in its Laplace-domain form, "
        r"which is Hunt (1999) with the storage term $Sp$ replaced by "
        r"$Sp + \varepsilon\sigma p/(\sigma p + \varepsilon)$, $\varepsilon = K'/B'$, and inverts it "
        r"numerically (Stehfest 1970). This agrees with an independent finite-volume model to better "
        r"than 0.0001 and avoids small errors (up to about 0.001) in the series form." '\n\n'
        '**Drawdown at observation bores.** For a bore at $(d, 0)$ and an observation bore at '
        '$(x, y)$, with $W$ the Theis well function and $u(r) = r^2 S / (4 T t)$:')
    st.latex(r'\text{Glover and Balmer: } s = \frac{Q}{4\pi T}\left[W(u(r_1)) - W(u(r_2))\right]')
    st.latex(r'\text{Hantush: } s = \frac{Q}{4\pi T}\left[W(u(r_1)) + W(u(r_2)) - 2\int_0^\infty '
             r'e^{-\theta} W\!\left(u\left(\sqrt{(x + d + L\theta)^2 + y^2}\right)\right) d\theta\right]')
    st.latex(r'\text{Hunt (1999): } s = \frac{Q}{4\pi T}\left[W(u(r_1)) - \int_0^\infty e^{-\theta} '
             r'W\!\left(u\left(\sqrt{(|x| + d + 2T\theta/\lambda)^2 + y^2}\right)\right) d\theta\right]')
    st.markdown(
        r'where $r_1 = \sqrt{(x - d)^2 + y^2}$ and $r_2 = \sqrt{(x + d)^2 + y^2}$. For Hunt (2003) '
        r'the Hunt (1999) drawdown is used in Laplace space with the storage term above, giving the '
        r'drawdown in the pumped aquifer. Glover and Balmer gives zero drawdown across the stream, and '
        r'Hantush has no aquifer there, so observation bores across the stream need Hunt (1999) or '
        r'Hunt (2003).' '\n\n'
        '**Changing rates and several bores.** The solutions are linear, so each change in pumping '
        'rate is treated as a new bore switched on at that time, and the effects are added. The '
        'same applies to several bores: the total depletion is the sum for each bore.\n\n'
        '**Assumptions common to all four**\n'
        '- The aquifer is uniform, the same in all directions, and extends a long way.\n'
        '- Flow is horizontal, and drawdown is small compared with the saturated thickness.\n'
        '- The stream is straight, very long, stays connected to the aquifer, and its level does '
        'not change.\n'
        '- Pumping does not change recharge or evapotranspiration.\n'
        '- Only the distance of a bore from the stream matters for depletion. Its position along '
        'the stream does not change the total, but it does change drawdown at an observation bore.')

with st.expander('**How to cite and references**'):
    st.write(
        'This app:\n\n'
        f'Irvine, D.J. ({dt.date.today().year}). {APP_NAME} (version {APP_VERSION}) '
        f'[Web application]. Charles Darwin University. {APP_URL}\n\n'
        'Please also cite the solution you use:\n\n'
        'Glover, R.E., and Balmer, G.G. (1954). River depletion resulting from pumping a well near '
        'a river. Eos, Transactions American Geophysical Union 35(3), 468-470.\n\n'
        'Hantush, M.S. (1965). Wells near streams with semipervious beds. Journal of Geophysical '
        'Research 70(12), 2829-2838.\n\n'
        'Hunt, B. (1999). Unsteady stream depletion from ground water pumping. Ground Water 37(1), '
        '98-102.\n\n'
        'Hunt, B. (2003). Unsteady stream depletion when pumping from semiconfined aquifer. Journal '
        'of Hydrologic Engineering 8(1), 12-19.\n\n'
        'Other references:\n\n'
        'Jenkins, C.T. (1968). Techniques for computing rate and volume of stream depletion by '
        'wells. Ground Water 6(2), 37-46.\n\n'
        'Stehfest, H. (1970). Algorithm 368: Numerical inversion of Laplace transforms. '
        'Communications of the ACM 13(1), 47-49.\n\n'
        'Theis, C.V. (1935). The relation between the lowering of the piezometric surface and the '
        'rate and duration of discharge of a well using ground-water storage. Eos, Transactions '
        'American Geophysical Union 16(2), 519-524.\n\n'
        'Theis, C.V. (1941). The effect of a well on the flow of a nearby stream. Eos, Transactions '
        'American Geophysical Union 22(3), 734-738.\n\n'
        'Hunt (2003) is checked against the series form in pycap-dss, U.S. Geological Survey '
        '(public domain). https://github.com/DOI-USGS/pycap-dss')

# %% 1. solution

st.header('1. Solution')
solution = st.selectbox('Analytical solution', list(solutions.SOLUTIONS), index=2)
info = solutions.SOLUTIONS[solution]
key = info['key']
st.markdown(f"{info['summary']}\n\n{info['when']}")

with st.expander('Plot colours'):
    st.caption('Change these to match your organisation. They apply to every plot.')
    colours = {}
    picker_columns = st.columns(3)
    for i, (role, default) in enumerate(plots.DEFAULT_COLOURS.items()):
        colours[role] = picker_columns[i % 3].color_picker(role, default)

# %% 2. bores

st.header('2. Bores')
st.caption('The stream runs along the line x = 0. Give each bore an x value (distance from the '
           'stream, negative for the other bank), a y value (position along the stream) and a '
           'pumping rate. Add a row for each bore.')
col_u1, col_u2 = st.columns(2)
length_unit = col_u1.selectbox('Distance unit', list(units.LENGTH))
flow_unit = col_u2.selectbox('Pumping rate unit', list(units.FLOW))

default_bores = pd.DataFrame({'Name': ['Bore 1'], 'x': [500.0], 'y': [0.0], 'Pumping rate': [10.0]})
bore_table = st.data_editor(
    default_bores, num_rows='dynamic', width='stretch', hide_index=True,
    column_config={
        'Name': st.column_config.TextColumn('Name', required=True),
        'x': st.column_config.NumberColumn(f'x ({length_unit})', required=True),
        'y': st.column_config.NumberColumn(f'y ({length_unit})', required=True, default=0.0),
        'Pumping rate': st.column_config.NumberColumn(f'Pumping rate ({flow_unit})', required=True, min_value=0.0),
    })

problems = []
bores = []
for _, row in bore_table.dropna(subset=['Name', 'x', 'Pumping rate']).iterrows():
    bores.append(dict(name=str(row['Name']).strip(),
                      x=float(row['x']) * units.LENGTH[length_unit],
                      y=float(0.0 if pd.isna(row['y']) else row['y']) * units.LENGTH[length_unit],
                      rate=float(row['Pumping rate']) * units.FLOW[flow_unit],
                      rate_entered=float(row['Pumping rate'])))
if not bores:
    problems.append('Add at least one bore.')
if len({b['name'] for b in bores}) < len(bores):
    problems.append('Give each bore a different name.')
if any(b['x'] == 0 for b in bores):
    problems.append('A bore cannot sit on the stream (x = 0).')

st.subheader('Observation bores (optional)')
st.caption('Monitoring bores where drawdown is wanted, for example to set trigger levels. Use the '
           'same x and y as the pumping bores. A trigger level is optional. Delete the row if none '
           'are needed.')
dd_unit = st.selectbox('Drawdown unit', list(units.DRAWDOWN))
default_obs = pd.DataFrame({'Name': ['Obs 1'], 'x': [250.0], 'y': [100.0], 'Trigger level': [None]},
                           ).astype({'Trigger level': float})
obs_table = st.data_editor(
    default_obs, num_rows='dynamic', width='stretch', hide_index=True, key='obs_table',
    column_config={
        'Name': st.column_config.TextColumn('Name', required=True),
        'x': st.column_config.NumberColumn(f'x ({length_unit})', required=True),
        'y': st.column_config.NumberColumn(f'y ({length_unit})', required=True, default=0.0),
        'Trigger level': st.column_config.NumberColumn(f'Trigger level, drawdown ({dd_unit})', min_value=0.0),
    })
observations = []
for _, row in obs_table.dropna(subset=['Name', 'x']).iterrows():
    level = row['Trigger level']
    observations.append(dict(name=str(row['Name']).strip(),
                             x=float(row['x']) * units.LENGTH[length_unit],
                             y=float(0.0 if pd.isna(row['y']) else row['y']) * units.LENGTH[length_unit],
                             trigger=None if pd.isna(level) else float(level) * units.DRAWDOWN[dd_unit]))
if len({o['name'] for o in observations}) < len(observations):
    problems.append('Give each observation bore a different name.')
for o in observations:
    for b in bores:
        if np.hypot(o['x'] - b['x'], o['y'] - b['y']) < 0.5:
            problems.append(f"Observation bore {o['name']} sits on pumping bore {b['name']}. Drawdown "
                            'in a pumping bore needs its radius and losses, which these solutions do not include.')
if observations and key == 'hantush' and any(o['x'] * b['x'] < 0 for o in observations for b in bores):
    problems.append('Hantush (1965) has no aquifer across the stream. Put observation bores on the same '
                    'side as the pumping bores, or use Hunt (1999) or Hunt (2003).')
if observations and key == 'glover' and any(o['x'] * b['x'] <= 0 for o in observations for b in bores):
    st.caption('With Glover and Balmer the stream fully penetrates the aquifer, so a bore on the other '
               'side of the stream causes no drawdown at an observation bore.')

limit_mode = st.radio('Link triggers to a stream depletion limit', ['No limit', 'Depletion rate',
                      '% of the peak pumping rate'], horizontal=True,
                      help='Reports the drawdown at each observation bore on the day total stream '
                           'depletion first reaches this limit.')
limit_value = None
if limit_mode == 'Depletion rate':
    limit_value = st.number_input(f'Depletion limit ({flow_unit})', value=5.0, min_value=0.0, format='%g')
elif limit_mode == '% of the peak pumping rate':
    limit_value = st.number_input('Depletion limit (% of the peak total pumping rate)', value=25.0,
                                  min_value=0.0, max_value=100.0, format='%g')
if limit_mode != 'No limit' and not observations:
    problems.append('Add an observation bore to link triggers to a depletion limit.')

if bores and not any(b['x'] == 0 for b in bores):
    figure = plots.diagram(key, bores, length_unit, units.LENGTH[length_unit], colours, observations)
    st.pyplot(figure)
    plt.close(figure)
    st.caption('Conceptual set-up for the chosen solution. The symbols match the inputs below.')

# %% 3. aquifer and stream


def value_with_unit(label, default, unit_options, key, disabled=False, min_value=0.0, help=None):
    # a number box with a unit drop-down beside it. Returns the value in metres and days
    col_value, col_unit = st.columns([3, 1])
    value = col_value.number_input(label, value=default, min_value=min_value, format='%g',
                                   key=f'{key}_value', disabled=disabled, help=help)
    unit = col_unit.selectbox('Unit', list(unit_options), key=f'{key}_unit', disabled=disabled)
    return value * unit_options[unit], f'{value:g} {unit}'


st.header('3. Aquifer and stream')
entered = {}

t_mode = st.radio('Transmissivity', ['Enter transmissivity', 'Enter hydraulic conductivity and saturated thickness'],
                  horizontal=True)
direct_t = t_mode == 'Enter transmissivity'
T_direct, entered['Transmissivity T'] = value_with_unit('Transmissivity, T', 500.0, units.TRANSMISSIVITY, 'T',
                                                        disabled=not direct_t)
K_aquifer, entered['Aquifer hydraulic conductivity K'] = value_with_unit(
    'Aquifer hydraulic conductivity, K', 10.0, units.CONDUCTIVITY, 'K', disabled=direct_t)
b_aquifer, entered['Aquifer saturated thickness b'] = value_with_unit(
    'Aquifer saturated thickness, b', 50.0, units.LENGTH, 'b', disabled=direct_t)
if direct_t:
    T = T_direct
    del entered['Aquifer hydraulic conductivity K'], entered['Aquifer saturated thickness b']
else:
    T = K_aquifer * b_aquifer
    del entered['Transmissivity T']
    st.caption(f'T = K x b = {T:g} m²/d')

S = st.number_input('Storage coefficient, S (specific yield if unconfined)', value=0.1,
                    min_value=0.0, max_value=1.0, format='%g')
entered['Storage coefficient S'] = f'{S:g}'
p = dict(T=T, S=S)

if key == 'hantush':
    mode = st.radio('Streambed', ["Enter streambed leakance", "Enter streambed K' and thickness"], horizontal=True)
    direct = mode == 'Enter streambed leakance'
    L_direct, text_L = value_with_unit('Streambed leakance, L', 100.0, units.LENGTH, 'L', disabled=not direct)
    K_bed, text_K = value_with_unit("Streambed hydraulic conductivity, K'", 0.1, units.CONDUCTIVITY, 'Kbed', disabled=direct)
    b_bed, text_b = value_with_unit("Streambed thickness, b'", 1.0, units.LENGTH, 'bbed', disabled=direct)
    K_for_L, text_KL = value_with_unit('Aquifer hydraulic conductivity, K (for the leakance)', 10.0,
                                       units.CONDUCTIVITY, 'KL', disabled=direct or not direct_t)
    if direct:
        p['L'] = L_direct
        entered['Streambed leakance L'] = text_L
    else:
        K_used = K_for_L if direct_t else K_aquifer
        p['L'] = K_used * b_bed / K_bed if K_bed > 0 else np.nan
        entered["Streambed hydraulic conductivity K'"] = text_K
        entered["Streambed thickness b'"] = text_b
        entered['Aquifer K used for the leakance'] = f'{K_used:g} m/d'
        st.caption(f"L = K x b' / K' = {p['L']:g} m")

elif key == 'hunt99':
    mode = st.radio('Streambed', ['Enter streambed conductance', "Enter streambed K', thickness and stream width"],
                    horizontal=True)
    direct = mode == 'Enter streambed conductance'
    lam_direct, text_lam = value_with_unit('Streambed conductance, λ', 1.0, units.CONDUCTIVITY, 'lam', disabled=not direct)
    K_bed, text_K = value_with_unit("Streambed hydraulic conductivity, K'", 0.1, units.CONDUCTIVITY, 'Kbed', disabled=direct)
    b_bed, text_b = value_with_unit("Streambed thickness, b'", 1.0, units.LENGTH, 'bbed', disabled=direct)
    width, text_w = value_with_unit('Stream width, w', 10.0, units.LENGTH, 'w', disabled=direct)
    if direct:
        p['lam'] = lam_direct
        entered['Streambed conductance λ'] = text_lam
    else:
        p['lam'] = K_bed * width / b_bed if b_bed > 0 else np.nan
        entered["Streambed hydraulic conductivity K'"] = text_K
        entered["Streambed thickness b'"] = text_b
        entered['Stream width w'] = text_w
        st.caption(f"λ = K' x w / b' = {p['lam']:g} m/d")

elif key == 'hunt03':
    st.caption('T and S above describe the pumped aquifer. The values below describe the aquitard '
               'that holds the stream.')
    p['K_aq'], entered["Aquitard hydraulic conductivity K'"] = value_with_unit(
        "Aquitard hydraulic conductivity, K'", 0.1, units.CONDUCTIVITY, 'Kaq')
    p['B_sat'], entered["Aquitard saturated thickness B'"] = value_with_unit(
        "Aquitard saturated thickness, B'", 10.0, units.LENGTH, 'Bsat')
    p['B_below'], entered["Aquitard thickness beneath the stream B''"] = value_with_unit(
        "Aquitard thickness beneath the stream, B''", 8.0, units.LENGTH, 'Bbelow')
    p['sigma'] = st.number_input('Aquitard specific yield, σ', value=0.1, min_value=0.0, max_value=1.0, format='%g')
    entered['Aquitard specific yield σ'] = f"{p['sigma']:g}"
    p['width'], entered['Stream width w'] = value_with_unit('Stream width, w', 10.0, units.LENGTH, 'w')

if any((not np.isfinite(v)) or v <= 0 for v in p.values()):
    problems.append('Every aquifer and stream value must be greater than zero.')

# %% 4. pumping schedule

st.header('4. Pumping schedule')
col_s1, col_s2 = st.columns(2)
start_date = col_s1.date_input('Start date', value=dt.date(dt.date.today().year, 1, 1))
sim_years = col_s2.number_input('Length of the calculation (years)', value=5.0, min_value=0.1,
                                max_value=100.0, step=1.0)
n_days = int(round(sim_years * 365.25))

kind = st.radio('Pumping', ['Constant rate', 'Repeating season', 'Uploaded time series'], horizontal=True)
schedule_text = kind
pump_days = months = pump_years = series_file = None
if kind == 'Constant rate':
    pump_days = st.number_input('Days of pumping before the bores are switched off', value=365,
                                min_value=1, step=30)
    schedule_text = f'Constant rate for {pump_days} days'
elif kind == 'Repeating season':
    chosen = st.multiselect('Months with pumping', MONTHS, default=MONTHS[4:10])
    months = [MONTHS.index(m) + 1 for m in chosen]
    pump_years = st.number_input('Number of years with pumping', value=int(np.ceil(sim_years)),
                                 min_value=1, step=1)
    schedule_text = f"Pumping in {', '.join(chosen)} for {pump_years} years"
    if not months:
        problems.append('Choose at least one month with pumping.')
else:
    st.caption(f'Upload a CSV with a Date column, then one column per bore holding its pumping rate '
               f'in {flow_unit}. Each row holds until the next date, so end with a row of zeros if '
               'pumping stops. The calculation starts on the first date in the file, and the rates '
               'in the bore table are not used.')
    if bores:
        template, _ = engine.series_template(bores, flow_unit)
        st.download_button('Download a template (.csv)', template, file_name='HydroCAPTURE_pumping_template.csv',
                           mime='text/csv', on_click='ignore')
    series_file = st.file_uploader('Pumping time series (CSV)', type=['csv'])
    if series_file is None:
        problems.append('Upload a pumping time series, or choose another pumping option.')
    else:
        schedule_text = f'Uploaded time series ({series_file.name})'

# %% 5. uncertainty

st.header('5. Uncertainty')
st.caption('Optional. Give a plus or minus percentage for any value. The shaded ranges on the '
           'depletion and drawdown plots are the lowest and highest results over the entered values '
           'and every combination of low and high values.')
percent = {}
uncertain_columns = st.columns(len(engine.UNCERTAIN[solution]))
for column, (name, label) in zip(uncertain_columns, engine.UNCERTAIN[solution].items()):
    percent[name] = column.number_input(f'{label} (± %)', value=0.0, min_value=0.0, max_value=95.0, step=5.0)
if key in ('hantush', 'hunt99') and 'L' not in p and 'lam' not in p:
    percent = {k: v for k, v in percent.items() if k in p}

# %% 6. run

st.header('6. Run')
col_o1, col_o2, col_o3 = st.columns(3)
out_flow_unit = col_o1.selectbox('Rate unit for the results', list(units.FLOW))
out_volume_unit = col_o2.selectbox('Volume unit for the results', list(units.VOLUME))
FIGURE_DPI = col_o3.radio('Figure download resolution (dpi)', [300, 600], horizontal=True)

for problem in problems:
    st.info(problem)

if st.button('Run', type='primary', disabled=bool(problems)):
    try:
        with st.spinner('Calculating...'):
            if kind == 'Constant rate':
                schedule = engine.constant_schedule(bores, start_date, n_days, pump_days)
            elif kind == 'Repeating season':
                schedule = engine.seasonal_schedule(bores, start_date, n_days, months, pump_years)
            else:
                table = pd.read_csv(io.BytesIO(series_file.getvalue()))
                schedule = engine.series_schedule(bores, table, n_days, units.FLOW[flow_unit])
            depletion = engine.total_depletion(schedule, bores, solution, p)
            band = engine.uncertainty_band(schedule, bores, solution, p, percent)
            ts = engine.time_series(schedule, depletion, band)
            result = dict(ts=ts, monthly=engine.monthly(ts), summary=engine.bore_summary(bores, solution, p),
                          bores=bores, solution=solution, key=key, p=dict(p), n_days=n_days,
                          length_unit=length_unit, flow_unit=out_flow_unit, volume_unit=out_volume_unit)

            # everything written out is in the result units
            f, v = units.FLOW[out_flow_unit], units.VOLUME[out_volume_unit]
            ts_out = ts.copy()
            for column in ts_out.columns:
                if column.startswith(('Pumping', 'Depletion')):
                    ts_out[column] = ts_out[column] / f
                    ts_out = ts_out.rename(columns={column: f'{column} ({out_flow_unit})'})
                elif column.startswith('Cumulative'):
                    ts_out[column] = ts_out[column] / v
                    ts_out = ts_out.rename(columns={column: f'{column} ({out_volume_unit})'})
            monthly_out = result['monthly'].copy()
            for column in ('Volume pumped', 'Volume depleted'):
                monthly_out[column] = monthly_out[column] / v
                monthly_out = monthly_out.rename(columns={column: f'{column} ({out_volume_unit})'})
            bores_out = pd.DataFrame([{
                'Bore': b['name'], f'x ({length_unit})': b['x'] / units.LENGTH[length_unit],
                f'y ({length_unit})': b['y'] / units.LENGTH[length_unit],
                f'Pumping rate ({flow_unit})': b['rate_entered']} for b in bores])
            settings = [('App', f'{APP_NAME} version {APP_VERSION}'),
                        ('Run date', dt.date.today().isoformat()),
                        ('Solution', solution),
                        ('Start date', str(ts['Date'].iloc[0].date())),
                        ('Length of the calculation (days)', n_days),
                        ('Pumping schedule', schedule_text),
                        ('Time step', 'Daily. Depletion is reported at the end of each day')]
            settings += [(name, value) for name, value in entered.items()]
            settings += [(f'Value used: {name} (metres and days)', f'{value:g}') for name, value in p.items()]
            settings += [(f'Uncertainty: {engine.UNCERTAIN[solution][name]} (± %)', value)
                         for name, value in percent.items()]
            settings += [('Rate unit for the results', out_flow_unit),
                         ('Volume unit for the results', out_volume_unit)]

            drawdown_out = obs_out = None
            if observations:
                limit = None
                if limit_mode == 'Depletion rate':
                    limit = limit_value * units.FLOW[flow_unit]
                    settings.append(('Stream depletion limit', f'{limit_value:g} {flow_unit}'))
                elif limit_mode == '% of the peak pumping rate':
                    limit = limit_value / 100 * schedule.sum(axis=1).max()
                    settings.append(('Stream depletion limit', f'{limit_value:g}% of the peak total '
                                     f'pumping rate ({limit:g} m³/d)'))
                analysis = engine.trigger_analysis(schedule, bores, observations, solution, p, percent, limit)
                banded = analysis['n_cases'] > 1
                dates = ts['Date']
                ddf = units.DRAWDOWN[dd_unit]
                obs_summary = engine.trigger_summary(analysis, observations, dates, banded,
                                                     (length_unit, units.LENGTH[length_unit]), (dd_unit, ddf))
                drawdown_out = pd.DataFrame({'Date': dates, 'Day': ts['Day']})
                for o in observations:
                    drawdown_out[f"Drawdown, {o['name']} ({dd_unit})"] = analysis['entered'][o['name']].to_numpy() / ddf
                    if banded:
                        drawdown_out[f"Drawdown, {o['name']}, low ({dd_unit})"] = analysis['low'][o['name']].to_numpy() / ddf
                        drawdown_out[f"Drawdown, {o['name']}, high ({dd_unit})"] = analysis['high'][o['name']].to_numpy() / ddf
                obs_out = obs_summary
                limit_day = analysis['limit_days'][0] if analysis['limit_days'] else None
                result.update(analysis=analysis, observations=observations, obs_summary=obs_summary,
                              dd_unit=dd_unit, limit=limit,
                              limit_date=None if limit_day is None else dates.iloc[limit_day])
                settings += [(f"Observation bore {o['name']}",
                              f"x = {o['x'] / units.LENGTH[length_unit]:g} {length_unit}, "
                              f"y = {o['y'] / units.LENGTH[length_unit]:g} {length_unit}"
                              + ('' if o['trigger'] is None else f", trigger {o['trigger'] / ddf:g} {dd_unit}"))
                             for o in observations]
            settings_table = pd.DataFrame(settings, columns=['Setting', 'Value']).astype({'Value': str})
            result['xlsx'] = workbook.build(ts_out, monthly_out, result['summary'], bores_out, settings_table,
                                            drawdown_out, obs_out)
            st.session_state['result'] = result

        new_total = run_counter.increment_runs()
        if new_total is not None:
            st.session_state['run_count'] = new_total
    except Exception as error:
        st.error(f'Run failed: {error}')

# %% 7. results

FIGURES = {}


def show(fig, name):
    # render once, show it, keep the png for the zip
    plots.add_watermark(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    FIGURES[name] = buffer.getvalue()
    st.image(FIGURES[name], width='stretch')
    st.download_button(f'Download PNG ({FIGURE_DPI} dpi)', FIGURES[name], file_name=f'{name}.png',
                       mime='image/png', key=f'download_{name}', on_click='ignore')


if 'result' in st.session_state:
    r = st.session_state['result']
    ts, rb = r['ts'], r['bores']
    f, v = units.FLOW[r['flow_unit']], units.VOLUME[r['volume_unit']]
    prefix = 'HydroCAPTURE'

    st.header('7. Results')
    st.caption(f"Solution: {r['solution']}. Change any input and press Run again to update.")
    peak = ts['Depletion, total'].max()
    col1, col2, col3 = st.columns(3)
    col1.metric('Peak stream depletion', f"{peak / f:,.3g} {r['flow_unit']}")
    col2.metric('Volume pumped', f"{ts['Cumulative volume pumped'].iloc[-1] / v:,.4g} {r['volume_unit']}")
    col3.metric('Taken from the stream', f"{ts['Depleted as % of pumped'].iloc[-1]:.0f}% of pumped")

    st.download_button('Download the workbook (.xlsx)', r['xlsx'], file_name=f'{prefix}.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                       type='primary', on_click='ignore')

    st.subheader('Pumping and stream depletion')
    show_bores = st.checkbox('Show each bore', value=False) if len(rb) > 1 else False
    show(plots.depletion(ts, rb, r['flow_unit'], f, colours, show_bores), f'{prefix}_depletion')

    st.subheader('Cumulative volumes')
    show(plots.cumulative(ts, r['volume_unit'], v, colours), f'{prefix}_cumulative')

    st.subheader('Monthly volumes')
    st.caption('Depletion lags pumping, so the stream keeps losing water after the bores are switched off.')
    show(plots.monthly(r['monthly'], r['volume_unit'], v, colours), f'{prefix}_monthly')

    st.subheader('Response to constant pumping')
    names = [b['name'] for b in rb]
    chosen_bore = rb[names.index(st.selectbox('Bore', names))] if len(rb) > 1 else rb[0]
    days = np.arange(0, r['n_days'] + 1, dtype=float)
    d = abs(chosen_bore['x'])
    curves = {r['solution']: solutions.response(r['solution'], days, d, r['p'])}
    if r['key'] != 'glover':
        curves['Glover and Balmer (1954)'] = solutions.glover(days, d, r['p']['T'], r['p']['S'])
        st.caption('Glover and Balmer (1954) is the case with no streambed or aquitard, shown for comparison.')
    show(plots.comparison(days, curves, colours), f'{prefix}_constant_pumping')

    st.subheader('Depletion against distance')
    at_days = st.number_input('Days of constant pumping', value=365, min_value=1, step=30)
    far = 2 * max(abs(b['x']) for b in rb)
    dist = np.linspace(far / 200, far, 80)
    values = np.array([solutions.response(r['solution'], np.array([float(at_days)]), x, r['p'])[0] for x in dist])
    show(plots.distance(dist, values, rb, at_days, r['length_unit'], units.LENGTH[r['length_unit']], colours),
         f'{prefix}_distance')

    st.subheader('Summary by bore')
    st.dataframe(r['summary'], width='stretch', hide_index=True)

    if r.get('observations'):
        st.subheader('Observation bores and trigger levels')
        a = r['analysis']
        show(plots.drawdown(ts['Date'], a, r['observations'], r['dd_unit'], units.DRAWDOWN[r['dd_unit']],
                            colours, r.get('limit_date')), f'{prefix}_drawdown')
        st.dataframe(r['obs_summary'], width='stretch', hide_index=True)
        notes = ['Drawdown is the change caused by the pumping bores, from the start of the '
                 'calculation. Add it to the expected seasonal low to compare with a water level.']
        if a['n_cases'] > 1:
            notes.append('The earliest case is the first day the highest drawdown over the uncertainty '
                         'range reaches the trigger.')
        if r.get('limit') is not None:
            if r.get('limit_date') is None:
                notes.append('Stream depletion does not reach the limit in this calculation with the '
                             'entered values.')
            else:
                notes.append(f"With the entered values, stream depletion first reaches the limit on "
                             f"{r['limit_date'].date()}. A trigger set at the drawdown on that day is "
                             'reached at the same time. If the uncertainty range is used, the lowest '
                             'case is the trigger that is reached no later than the depletion limit in '
                             'every case calculated.')
            notes.append('Depletion keeps rising after pumping stops while drawdown recovers, so a '
                         'drawdown trigger does not guard against depletion that arrives after pumping '
                         'ends. Check the depletion plot for that lag.')
        st.caption(' '.join(notes))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f'{prefix}.xlsx', r['xlsx'])
        for name, data in FIGURES.items():
            archive.writestr(f'{name}.png', data)
    st.download_button('Download everything (.zip)', zip_buffer.getvalue(),
                       file_name=f'{prefix}_outputs.zip', mime='application/zip', on_click='ignore')
