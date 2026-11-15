# analytical stream depletion solutions.
# each returns depletion as a fraction of the pumping rate, for a bore pumping at a
# constant rate from time zero. Units: metres and days

from math import factorial

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


def _hunt03_closed_form(t, d, T, S, K_aq, B_sat, B_below, sigma, width, n_quad=80):
    # Hunt (2003) closed form as coded in pycap-dss and STRMDEPL08. Kept for comparison only:
    # it differs from the exact Laplace-domain result below by up to about 1e-3
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


# %% Laplace-domain solutions, inverted with Stehfest (1970)

def _stehfest_weights(N=14):
    V = np.zeros(N)
    for i in range(1, N + 1):
        total = 0.0
        for k in range((i + 1) // 2, min(i, N // 2) + 1):
            total += (k ** (N // 2) * factorial(2 * k)
                      / (factorial(N // 2 - k) * factorial(k) * factorial(k - 1)
                         * factorial(i - k) * factorial(2 * k - i)))
        V[i - 1] = (-1) ** (i + N // 2) * total
    return V


_STEHFEST = _stehfest_weights()


def _invert(fbar, t):
    # f(t) from its Laplace transform. fbar takes an array of p
    t = _times(t)
    out = np.zeros_like(t)
    on = t > 0
    ln2 = np.log(2.0)
    for i, v in enumerate(_STEHFEST, start=1):
        out[on] += v * fbar(i * ln2 / t[on])
    out[on] *= ln2 / t[on]
    return out


def _hunt03_storage(p, S, K_aq, B_sat, sigma):
    # Hunt (2003) aquifer storage term in Laplace space: aquifer storage plus delayed drainage
    # of the aquitard water table through the aquitard. With sigma = 0 it is S p (Hunt 1999)
    leak = K_aq / B_sat
    return S * p + leak * sigma * p / (sigma * p + leak)


def hunt03(t, d, T, S, K_aq, B_sat, B_below, sigma, width):
    # Hunt (2003): bore in a semi-confined aquifer, stream in the overlying aquitard.
    # K_aq aquitard K [m/d], B_sat saturated aquitard thickness [m], B_below aquitard
    # thickness beneath the stream [m], sigma aquitard specific yield, width stream width [m].
    # Hunt (1999) in Laplace space with the storage term above, inverted numerically
    lam = K_aq * width / B_below

    def fbar(p):
        mu = np.sqrt(_hunt03_storage(p, S, K_aq, B_sat, sigma) / T)
        return np.exp(-mu * d) * lam / (2 * T * mu + lam) / p

    return np.clip(_invert(fbar, t), 0, 1)


# %% drawdown at an observation bore, per unit pumping rate [m per m3/d]
# the pumping bore is at (d, 0) with d > 0, the observation bore at (x, y). Stream along x = 0

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def _smeared_theis(a, D, y, c):
    # I = integral over theta from 0 to infinity of exp(-theta) W(((D + c theta)^2 + y^2) a),
    # one value per entry of a = S / (4 T t). Written in terms of the distance xi = D + c theta
    # and integrated by Gauss-Legendre in ln(xi), out to where W is negligible (W(40) ~ 1e-19)
    a = np.atleast_1d(a)
    reach = np.sqrt(np.maximum(40.0 / a - y * y, 0.0))
    top = np.minimum(D + 40.0 * c, np.maximum(reach, D))
    lo, hi = np.log(D), np.log(top)
    z = 0.5 * (hi - lo)[:, None] * (_GL_X + 1) + lo
    w = 0.5 * (hi - lo)[:, None] * _GL_W
    xi = np.exp(z)
    f = np.exp(-(xi - D) / c) * sps.exp1((xi * xi + y * y) * a[:, None]) * xi / c
    return (w * f).sum(axis=1)


def _smeared_k0(kap, D, y, c):
    # Laplace-space counterpart of _smeared_theis: integral of exp(-theta) K0(kap r(theta)),
    # with r = sqrt((D + c theta)^2 + y^2), out to where K0 is negligible (K0(40) ~ 1e-18)
    kap = np.atleast_1d(kap)
    reach = np.sqrt(np.maximum((40.0 / kap) ** 2 - y * y, 0.0))
    top = np.minimum(D + 40.0 * c, np.maximum(reach, D))
    lo, hi = np.log(D), np.log(top)
    z = 0.5 * (hi - lo)[:, None] * (_GL_X + 1) + lo
    w = 0.5 * (hi - lo)[:, None] * _GL_W
    xi = np.exp(z)
    f = np.exp(-(xi - D) / c) * sps.k0(kap[:, None] * np.sqrt(xi * xi + y * y)) * xi / c
    return (w * f).sum(axis=1)


def drawdown_glover(t, d, x, y, T, S):
    # Theis with an image recharge bore (Glover and Balmer 1954). Zero across the stream
    t = _times(t)
    out = np.zeros_like(t)
    on = t > 0
    if x <= 0 or not on.any():
        return out
    a = S / (4 * T * t[on])
    r1, r2 = (x - d) ** 2 + y * y, (x + d) ** 2 + y * y
    out[on] = (sps.exp1(r1 * a) - sps.exp1(r2 * a)) / (4 * np.pi * T)
    return out


def drawdown_hantush(t, d, x, y, T, S, L):
    # Hantush (1965): semipervious boundary at x = 0, aquifer on the bore side only.
    # Theis + image + image smeared beyond the boundary with weight exp(-xi / L)
    t = _times(t)
    if x < 0:
        return np.full_like(t, np.nan)
    out = np.zeros_like(t)
    on = t > 0
    if not on.any():
        return out
    a = (S / (4 * T * t[on]))[:, None]
    r1, r2 = (x - d) ** 2 + y * y, (x + d) ** 2 + y * y
    integral = _smeared_theis(a[:, 0], x + d, y, L)
    out[on] = (sps.exp1(r1 * a[:, 0]) + sps.exp1(r2 * a[:, 0]) - 2 * integral) / (4 * np.pi * T)
    return out


def drawdown_hunt99(t, d, x, y, T, S, lam):
    # Hunt (1999) equations 29 to 31. Valid on both sides of the stream
    t = _times(t)
    out = np.zeros_like(t)
    on = t > 0
    if not on.any():
        return out
    a = (S / (4 * T * t[on]))[:, None]
    r1 = (x - d) ** 2 + y * y
    integral = _smeared_theis(a[:, 0], abs(x) + d, y, 2 * T / lam)
    out[on] = (sps.exp1(r1 * a[:, 0]) - integral) / (4 * np.pi * T)
    return out


def _drawdown_hunt03_exact(t, d, x, y, T, S, K_aq, B_sat, B_below, sigma, width):
    lam = K_aq * width / B_below
    r1 = np.sqrt((x - d) ** 2 + y * y)

    def fbar(p):
        kap = np.sqrt(_hunt03_storage(p, S, K_aq, B_sat, sigma) / T)
        smeared = _smeared_k0(kap, abs(x) + d, y, 2 * T / lam)
        return (sps.k0(kap * r1) - smeared) / (2 * np.pi * T * p)

    return _invert(fbar, t)


def drawdown_hunt03(t, d, x, y, T, S, K_aq, B_sat, B_below, sigma, width, n_grid=200):
    # drawdown in the pumped aquifer for Hunt (2003): the Hunt (1999) drawdown in Laplace
    # space with the delayed-drainage storage term. Solved on log-spaced times and interpolated
    t = _times(t)
    out = np.zeros_like(t)
    on = t > 0
    if not on.any():
        return out
    args = (d, x, y, T, S, K_aq, B_sat, B_below, sigma, width)
    if on.sum() <= n_grid:
        out[on] = _drawdown_hunt03_exact(t[on], *args)
        return out
    grid = np.geomspace(t[on].min(), t[on].max(), n_grid)
    out[on] = np.interp(np.log(t[on]), np.log(grid), _drawdown_hunt03_exact(grid, *args))
    return out


# %% registry used by the app

SOLUTIONS = {
    'Glover and Balmer (1954)': dict(
        key='glover', function=glover, drawdown=drawdown_glover, parameters=('T', 'S'),
        summary='Stream cuts through the full aquifer thickness, with no streambed resistance.',
        when='The simplest and most conservative case: it gives the fastest and largest depletion. '
             'Use it for screening, or where the stream is large and well connected to the aquifer.'),
    'Hantush (1965)': dict(
        key='hantush', function=hantush, drawdown=drawdown_hantush, parameters=('T', 'S', 'L'),
        summary='Stream cuts through the full aquifer thickness, behind a streambed of lower permeability.',
        when='Use where a silted or clogged bed and banks slow the exchange, and the streambed is '
             'described by its thickness and hydraulic conductivity.'),
    'Hunt (1999)': dict(
        key='hunt99', function=hunt99, drawdown=drawdown_hunt99, parameters=('T', 'S', 'lam'),
        summary='Shallow stream that only partly penetrates the aquifer, with a streambed conductance.',
        when='The usual choice for small to medium streams in an unconfined or confined aquifer. '
             'With a very high streambed conductance it returns the Glover and Balmer result.'),
    'Hunt (2003)': dict(
        key='hunt03', function=hunt03, drawdown=drawdown_hunt03,
        parameters=('T', 'S', 'K_aq', 'B_sat', 'B_below', 'sigma', 'width'),
        summary='Bore pumps a semi-confined aquifer. The stream sits in an overlying aquitard that '
                'holds the water table.',
        when='Use where the stream is separated from the pumped aquifer by a lower-permeability '
             'layer. Depletion is slower, because drainage of the aquitard supplies part of the pumping.'),
}


def response(solution, t, d, p):
    # depletion fraction for the named solution. p is a dict of parameters
    s = SOLUTIONS[solution]
    return s['function'](t, d, *[p[name] for name in s['parameters']])


def geometry(bore, observation):
    # distances for the drawdown functions: bore moved to the positive side if needed
    xb, xo = bore['x'], observation['x']
    if xb < 0:
        xb, xo = -xb, -xo
    return xb, xo, observation['y'] - bore['y']


def drawdown(solution, t, bore, observation, p, exact_until=2000.0, n_grid=600):
    # drawdown at the observation bore per unit pumping rate at the bore [m per m3/d].
    # Long runs: exact up to exact_until days, then log-spaced times and interpolation
    s = SOLUTIONS[solution]
    d, x, y = geometry(bore, observation)
    args = [p[name] for name in s['parameters']]
    t = _times(t)
    late = t > exact_until
    if late.sum() <= n_grid:
        return s['drawdown'](t, d, x, y, *args)
    out = np.zeros_like(t)
    out[~late] = s['drawdown'](t[~late], d, x, y, *args)
    grid = np.geomspace(exact_until, t[late].max(), n_grid)
    out[late] = np.interp(np.log(t[late]), np.log(grid), s['drawdown'](grid, d, x, y, *args))
    return out
