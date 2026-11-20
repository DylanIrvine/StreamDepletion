# HydroCAPTURE

Stream depletion from pumping bores, using analytical solutions.

Current version: 0.2.0

A Streamlit app that estimates how much water one or more pumping bores take
from a nearby stream, and how quickly, and the drawdown this causes at
observation bores (for example, to help set trigger levels). It is part of the same suite as HydroSILO,
HydroPOINT, HydroSTITCH and the Hydrochem Workbench.

## Using it

1. Choose a solution. The page says what each one assumes and when to use it.
2. Add the bores: a name, x (distance from the stream), y (position along the
   stream) and a pumping rate. The stream runs along x = 0. A diagram shows the
   bores from above and the set-up in cross section.
   Optionally add observation bores (name, x, y and a trigger level), and a
   stream depletion limit to link the triggers to.
3. Enter the aquifer and stream values. Where there is a choice, such as
   transmissivity or hydraulic conductivity and thickness, pick one and the
   matching boxes become active. Each value has its own unit drop-down.
4. Set the pumping schedule: a constant rate that stops after a set time, a
   season that repeats each year, or an uploaded time series (a template can be
   downloaded).
5. Optionally give a plus or minus percentage for each value.
6. Run, then download the workbook, single figures, or everything as a zip.

## Solutions

| Solution | Stream | Streambed | Extra inputs |
|---|---|---|---|
| Glover and Balmer (1954) | fully penetrating | none | none |
| Hantush (1965) | fully penetrating | lower-permeability bed | leakance L, or K', b' and aquifer K |
| Hunt (1999) | partially penetrating | streambed conductance | conductance, or K', b' and width w |
| Hunt (2003) | in an aquitard above the pumped aquifer | aquitard | K', B', B'', specific yield, width w |

Hantush (1965) and Hunt (1999) give the same family of curves. They differ in
how the streambed is described.

## Method notes

- **Units.** Calculations are in metres and days. Inputs and results can use
  other units.
- **Time step.** Daily. Depletion is reported at the end of each day.
- **Changing rates.** Each day of pumping is a step on and a step off, and the
  effects are added (superposition). This is exact for daily rates.
- **Several bores.** The total is the sum for each bore. Only the distance from
  the stream matters, so y changes the diagram but not the result, and a bore on
  the other bank counts by its distance.
- **Hunt (2003).** Evaluated in its Laplace-domain form, which is Hunt (1999)
  with the storage term Sp replaced by Sp + εσp/(σp + ε), ε = K'/B', and
  inverted numerically (Stehfest 1970). This matches an independent 1D
  finite-volume model to better than 0.0001 and a 30-digit Talbot inversion to
  2e-6. The series form used in STRMDEPL08 and pycap-dss differs from it by up
  to about 0.001 (kept in the code as `_hunt03_closed_form` for comparison).
- **Observation bores.** Drawdown uses each solution's own drawdown: Theis with
  an image bore (Glover and Balmer), the semipervious-boundary image integral
  (Hantush 1965), Hunt (1999) equations 29 to 31, and for Hunt (2003) the Hunt
  (1999) drawdown in Laplace space with the storage term above. Drawdown is
  summed over pumping bores and over the daily schedule, so the position along
  the stream (y) matters here. Glover and Balmer gives zero drawdown across the
  stream and Hantush has no aquifer there.
- **Triggers.** For each observation bore the app reports the first date the
  trigger level is reached, and the drawdown on the day total stream depletion
  first reaches the chosen limit. With an uncertainty range it also reports the
  earliest date over all cases, and the lowest and highest drawdown at the
  limit. The lowest is the trigger that is reached no later than the limit in
  every case calculated. Depletion keeps rising after pumping stops while
  drawdown recovers, so a drawdown trigger does not cover that lag.
- **Uncertainty.** The range is the lowest and highest total over the entered
  values and every combination of low and high values. It is a range of cases,
  not a statistical confidence interval.
- **Summary.** The stream depletion factor is d²S/T (Jenkins 1968). Times to 50%
  and 90% depletion are for constant pumping.

## Checks

The test suite compares the code with published values: Glover and Balmer (1954)
table 1, and the U.S. Geological Survey STRMDEPL08 example for Hunt (2003), plus
a 30-digit Talbot inversion and a finite-volume model for Hunt (2003). Drawdown
is checked against adaptive quadrature of Hunt (1999) and Hantush (1965), and
against an independent Talbot inversion for Hunt (2003). It also checks the
limits (very high or very low streambed conductance or leakance, and Hunt 2003
without aquitard storage), and that superposition matches a direct sum.

```bash
pip install pytest openpyxl
pytest tests/
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository layout

```
HydroCAPTURE/
  app.py                    Streamlit front end
  hydrocapture/
    solutions.py            the four solutions
    engine.py               schedules, superposition, uncertainty, summary tables
    units.py                unit conversions
    plots.py                diagram and result figures
    workbook.py             xlsx writer
    run_counter.py          shared run total (Upstash Redis over REST)
  tests/                    pytest suite (no network needed)
  logo/                     header logo, browser-tab icon and svg source
  requirements.txt
  .streamlit/config.toml    dark theme
  LICENSE
```

## Privacy and the run counter

- Inputs and any uploaded pumping file are held in memory for the session.
  Nothing is written to a database or permanent storage, and nothing is sent to
  another service.
- The only thing recorded is a running total of completed runs, kept in an
  Upstash Redis database under the key `hydrocapture_run_count`.

The counter is optional. To turn it on, add this to the app's Secrets on
Streamlit Cloud, or to `.streamlit/secrets.toml` locally:

```
[upstash]
url = "https://<name>.upstash.io"
token = "<REST token>"
```

`.streamlit/secrets.toml` is in `.gitignore`.

## How to cite

Irvine, D.J. (2026). HydroCAPTURE (version 0.2.0) [Web application]. Charles
Darwin University. https://hydrocapture.streamlit.app/

Please also cite the solution used. The references are listed in the app.

## Licence

Copyright (c) 2026 Charles Darwin University. All rights reserved. See `LICENSE`. The series form of Hunt (2003), kept for comparison, is
adapted from pycap-dss (U.S. Geological Survey, public domain).

## Credits

Dylan Irvine, Charles Darwin University.
