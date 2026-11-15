# reference implementation: depletion of a circular water body with a leaky bed by a
# nearby pumping bore. Metres and days throughout. See the derivation document.

import numpy as np
from math import factorial
from scipy import special as sps


# %% Stehfest (1970) numerical Laplace inversion

def _stehfest_weights(N=14):
    V = np.zeros(N)
    for i in range(1, N + 1):
        s = 0.0
        for k in range((i + 1) // 2, min(i, N // 2) + 1):
            s += (k ** (N // 2) * factorial(2 * k)
                  / (factorial(N // 2 - k) * factorial(k) * factorial(k - 1)
                     * factorial(i - k) * factorial(2 * k - i)))
        V[i - 1] = (-1) ** (i + N // 2) * s
    return V


_V = _stehfest_weights()


def invert(Fbar, t):
    # f(t) from its Laplace transform Fbar(p), for an array of t > 0
    t = np.atleast_1d(np.asarray(t, dtype=float))
    ln2 = np.log(2.0)
    out = np.zeros_like(t)
    for i, v in enumerate(_V, start=1):
        out += v * Fbar(i * ln2 / t)
    return out * ln2 / t


# %% Laplace-domain kernels (exponentially scaled Bessel functions avoid overflow)

def _parts(p, T, S, d, a, c, eps):
    k = np.sqrt((S * p + eps) / T)          # outside the water body
    kap = np.sqrt((S * p + c) / T)          # beneath the water body
    i0, i1 = sps.i0e(kap * a), sps.i1e(kap * a)
    k0a, k1a = sps.k0e(k * a), sps.k1e(k * a)
    delta = k * k1a * i0 + kap * i1 * k0a   # Delta / (exp(kap a) exp(-k a))
    return k, kap, i0, i1, k0a, k1a, delta


def a1(p, T, S, d, a, c, eps=0.0):
    # pumping kernel: qbar = (Q/p) a1(p) at fixed stage
    k, kap, i0, i1, k0a, k1a, delta = _parts(p, T, S, d, a, c, eps)
    k0d = sps.k0e(k * d) * np.exp(-k * (d - a))
    return c * k0d * i1 / (T * kap * delta)


def beta(p, T, S, d, a, c, eps=0.0):
    # stage kernel [m2/d]: change in leakage per unit pool drawdown (Laplace domain)
    k, kap, i0, i1, k0a, k1a, delta = _parts(p, T, S, d, a, c, eps)
    return (2 * np.pi * a * c * c * k * k1a * i1 / ((S * p + c) * kap * delta)
            + np.pi * a * a * c * S * p / (S * p + c))


# %% time-domain responses

def F(t, T, S, d, a, c, eps=0.0):
    # depletion fraction for constant pumping from t = 0, fixed pool stage
    return invert(lambda p: a1(p, T, S, d, a, c, eps) / p, t)


def G(t, T, S, d, a, c, eps=0.0):
    # leakage reduction [m3/d] for a unit (1 m) step drop in pool stage, no pumping
    return invert(lambda p: beta(p, T, S, d, a, c, eps) / p, t)


def closed_pool(t, T, S, d, a, c, Ap, eps=0.0):
    # constant-area closed pool, unit pumping: (depletion fraction, pool drawdown per m3/d)
    s = invert(lambda p: a1(p, T, S, d, a, c, eps) / p / (Ap * p + beta(p, T, S, d, a, c, eps)), t)
    q = invert(lambda p: Ap * a1(p, T, S, d, a, c, eps) / (Ap * p + beta(p, T, S, d, a, c, eps)), t)
    return q, s


def steady_fraction(T, d, a, c, eps):
    # long-time depletion fraction with groundwater ET (eps > 0), fixed stage
    k, kap = np.sqrt(eps / T), np.sqrt(c / T)
    delta = k * sps.k1(k * a) * sps.i0(kap * a) + kap * sps.i1(kap * a) * sps.k0(k * a)
    return c * sps.k0(k * d) * sps.i1(kap * a) / (T * kap * delta)


# %% pool water balance, paraboloid basin, daily (or finer) steps

def pool_balance(Q, P, E, T, S, d, R, D, y0, c, eps=0.0, inflow=0.0, q0=0.0, dt=1.0,
                 shape='paraboloid', area=None):
    # Q, P, E: arrays per step (m3/d, m/d, m/d). Returns depth y at the end of each step
    # and the pumping-induced change in leakage q (m3/d, step average).
    # The leaking footprint is fixed at the starting wetted radius.
    n = len(Q)
    if shape == 'paraboloid':
        a = R * np.sqrt(y0 / D)
        kv = np.pi * R * R / (2 * D)        # V = kv y^2
        A_of = lambda y: np.pi * R * R * y / D
    else:                                   # constant area: V = area * y
        a = np.sqrt(area / np.pi)
        A_of = lambda y: area
    tt = dt * np.arange(1, n + 1)
    # cumulative kernels at step ends, then step averages
    Fc = np.concatenate(([0.0], invert(lambda p: a1(p, T, S, d, a, c, eps) / p ** 2, tt)))
    Gc = np.concatenate(([0.0], invert(lambda p: beta(p, T, S, d, a, c, eps) / p ** 2, tt)))
    Fh, Gh = np.diff(Fc) / dt, np.diff(Gc) / dt
    Q = np.asarray(Q, float)
    dQ = np.diff(np.concatenate(([0.0], Q)))
    y = np.zeros(n)
    q = np.zeros(n)
    ds = np.zeros(n)
    prev = y0
    for i in range(n):
        pump = np.dot(dQ[:i + 1], Fh[i::-1])
        hist = np.dot(ds[:i], Gh[i:0:-1]) if i > 0 else 0.0
        known = pump - hist
        W = (P[i] - E[i]) * A_of(prev) + inflow - q0
        b = dt * Gh[0]
        if shape == 'paraboloid':
            rhs = kv * prev ** 2 + dt * (W - known + prev * Gh[0])
            yn = (-b + np.sqrt(b * b + 4 * kv * rhs)) / (2 * kv) if rhs > 0 else 0.0
        else:
            yn = (area * prev + dt * (W - known + prev * Gh[0])) / (area + b)
            yn = max(yn, 0.0)
        ds[i] = prev - yn
        q[i] = known - ds[i] * Gh[0]
        y[i] = yn
        prev = yn
        if yn <= 0:
            y[i:] = 0.0
            break
    return y, q
