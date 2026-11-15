# analytical stream depletion solutions.
# each returns depletion as a fraction of the pumping rate, for a bore pumping at a
# constant rate from time zero. Units: metres and days

import numpy as np
from scipy import special as sps


def _times(t):
    return np.atleast_1d(np.asarray(t, dtype=float))


def glover(t, d, T, S):
    # Glover and Balmer (1954): fully penetrating stream, no streambed resistance
    t = _times(t)
    out = np.zeros_like(t)
    on = t > 0
    out[on] = sps.erfc(np.sqrt(S * d * d / (4 * T * t[on])))
    return out


def hunt99(t, d, T, S, lam):
    # Hunt (1999): partially penetrating stream with streambed conductance lam [m/d]
    t = _times(t)
    out = np.zeros_like(t)
    on = t > 0
    a = np.sqrt(S * d * d / (4 * T * t[on]))
    b = lam * lam * t[on] / (4 * S * T) + lam * d / (2 * T)
    c = np.sqrt(lam * lam * t[on] / (4 * S * T)) + a
    # exp(b) * erfc(c) written with erfcx so large arguments do not overflow
    out[on] = sps.erfc(a) - np.exp(b - c * c) * sps.erfcx(c)
    return np.clip(out, 0, 1)


def hantush(t, d, T, S, L):
    # Hantush (1965): fully penetrating stream behind a semipervious streambed.
    # L is the streambed leakance [m] = aquifer K x streambed thickness / streambed K.
    # same curve as Hunt (1999) with lam = 2T / L
    return hunt99(t, d, T, S, 2 * T / L)


# %% Hunt (2003)
# the correction integral is adapted from pycap-dss (U.S. Geological Survey, public
# domain), which follows Hunt (2003) equations 46 to 52

_N = np.arange(60, dtype=float)
_LOG_BINOM = sps.gammaln(2 * _N + 1) - 2 * sps.gammaln(_N + 1)
_BINOM = sps.binom(2 * _N, _N)


def _G(alpha, epsilon, dK, dtime):
    a = epsilon * dK * dtime * (1 - alpha ** 2)
    b = dK * dtime * alpha ** 2
    ab = a + b
    root = np.sqrt(np.maximum(a * b, 1e-300))
    term1 = np.where(ab < 80, np.exp(-np.minimum(ab, 80)) * sps.i0(2 * root), 0.0)
    ratio = np.where(ab > 1e-300, root / np.maximum(ab, 1e-300), 0.0)[:, None]
    gamma = sps.gammainc(2 * _N[None, :] + 1, ab[:, None])
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        direct = _BINOM * gamma * ratio ** (2 * _N[None, :])
        logs = (_LOG_BINOM + np.where(gamma > 1e-300, np.log(gamma), -700)
                + 2 * _N[None, :] * np.where(ratio > 1e-300, np.log(ratio), -700))
        terms = np.where(_N[None, :] <= 8, direct, np.exp(logs))
    total = np.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=1)
    slope = np.where(ab > 1e-300, (b - a) / np.maximum(ab, 1e-300), 0.0)
    return np.clip(0.5 * (1 - term1 + slope * total), 0, 1)


def _F(alpha, dlam, dtime):
    root = np.sqrt(dtime)
    z = alpha * dlam * root / 2 + 1 / (2 * alpha * root)
    a = dlam / 2 + dtime * alpha ** 2 * dlam ** 2 / 4
    lead = np.exp(-1 / (4 * dtime * alpha ** 2))
    near = lead * np.sqrt(dtime / np.pi) - alpha * dtime * dlam / 2 * sps.erfcx(z) * np.exp(a - z * z)
    # asymptotic form for large z
    far = lead / (2 * alpha * z * np.sqrt(np.pi)) * (
        1 + 2 / (dlam * (1 + (1 / (dlam * dtime * alpha ** 2)) ** 2))
        * (1 - 3 / (2 * z ** 2) + 15 / (4 * z ** 4) - 105 / (8 * z ** 6)))
    return np.where(np.abs(z) < 3, near, far)


def _hunt03_exact(t, d, T, S, K_aq, B_sat, B_below, sigma, width, n_quad=80):
    t = _times(t)
    dtime = T * t / (S * d * d)
    dlam = (K_aq * width / B_below) * d / T
    epsilon = S / sigma
    dK = (K_aq / B_sat) * d * d / T
    x, w = np.polynomial.legendre.leggauss(n_quad)
    alpha, weights = 0.5 * (x + 1), 0.5 * w
    out = np.zeros_like(t)
    for i, dt in enumerate(dtime):
        if dt <= 0:
            continue
        a = 1 / (2 * np.sqrt(dt))
        b = dlam / 2 + dt * dlam ** 2 / 4
        c = a + dlam * np.sqrt(dt) / 2
        base = sps.erfc(a) - sps.erfcx(c) * np.exp(b - c * c)
        correction = dlam * np.dot(weights, _F(alpha, dlam, dt) * _G(alpha, epsilon, dK, dt))
        out[i] = base - correction
    return np.clip(out, 0, 1)


def hunt03(t, d, T, S, K_aq, B_sat, B_below, sigma, width, n_grid=160):
    # Hunt (2003): bore in a semi-confined aquifer, stream in the overlying aquitard.
    # K_aq aquitard K [m/d], B_sat saturated aquitard thickness [m], B_below aquitard
    # thickness beneath the stream [m], sigma aquitard specific yield, width stream width [m].
    # the integral is slow, so it is solved on a log-spaced set of times and interpolated
    t = _times(t)
    out = np.zeros_like(t)
    on = t > 0
    if not on.any():
        return out
    if on.sum() <= n_grid:
        out[on] = _hunt03_exact(t[on], d, T, S, K_aq, B_sat, B_below, sigma, width)
        return out
    grid = np.geomspace(t[on].min(), t[on].max(), n_grid)
    values = _hunt03_exact(grid, d, T, S, K_aq, B_sat, B_below, sigma, width)
    out[on] = np.interp(np.log(t[on]), np.log(grid), values)
    return out


# %% registry used by the app

SOLUTIONS = {
    'Glover and Balmer (1954)': dict(
        key='glover', function=glover, parameters=('T', 'S'),
        summary='Stream cuts through the full aquifer thickness, with no streambed resistance.',
        when='The simplest and most conservative case: it gives the fastest and largest depletion. '
             'Use it for screening, or where the stream is large and well connected to the aquifer.'),
    'Hantush (1965)': dict(
        key='hantush', function=hantush, parameters=('T', 'S', 'L'),
        summary='Stream cuts through the full aquifer thickness, behind a streambed of lower permeability.',
        when='Use where a silted or clogged bed and banks slow the exchange, and the streambed is '
             'described by its thickness and hydraulic conductivity.'),
    'Hunt (1999)': dict(
        key='hunt99', function=hunt99, parameters=('T', 'S', 'lam'),
        summary='Shallow stream that only partly penetrates the aquifer, with a streambed conductance.',
        when='The usual choice for small to medium streams in an unconfined or confined aquifer. '
             'With a very high streambed conductance it returns the Glover and Balmer result.'),
    'Hunt (2003)': dict(
        key='hunt03', function=hunt03, parameters=('T', 'S', 'K_aq', 'B_sat', 'B_below', 'sigma', 'width'),
        summary='Bore pumps a semi-confined aquifer. The stream sits in an overlying aquitard that '
                'holds the water table.',
        when='Use where the stream is separated from the pumped aquifer by a lower-permeability '
             'layer. Depletion is slower, because drainage of the aquitard supplies part of the pumping.'),
}


def response(solution, t, d, p):
    # depletion fraction for the named solution. p is a dict of parameters
    s = SOLUTIONS[solution]
    return s['function'](t, d, *[p[name] for name in s['parameters']])
