"""
Assignment 3 - FinTech 545 (Quantitative Risk Management)
Jason Huang

Covers: Problem 1 (implied vol from a dirty option chain), Problem 2 (American options with discrete
dividends on a binomial tree), Problem 3 (VaR of an option book), Problem 4 (max Sharpe portfolio with a
put writing strategy), Problem 5 (alpha vs factor exposure with CAPM / FF3 / Carhart).

Run with:   python Assignment3.py
All the CSV files from the class repo must sit in the same folder as this script.
Every number in the written answers (Assignment3_Answers.pdf) is printed by this script, in order.
Each part is tagged like "--- 1(a) ..." so it's easy to find in outputs.txt.
Plots are saved as PNG files in the same folder (nothing pops up, so the script runs start to finish).

All option pricing (GBSM, Greeks, implied vol, binomial trees) is written by hand below, no pricing library.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # read the CSVs from the script's own folder
np.set_printoptions(precision=4, suppress=True, linewidth=120)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


# =====================================================================================
# Helper functions
# =====================================================================================

def header(title):
    """Print a banner so the output file is easy to scan."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def part(tag, title=""):
    """Print a small tag for each lettered part, e.g. --- 1(a) put call parity ---"""
    print(f"\n--- {tag} {title} ---")


def first_four_moments(x):
    """Mean, variance, skewness, excess kurtosis using the biased (divide by n) moments, same as Week 1."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.sum() / n
    d = x - mu
    cm2 = d @ d / n
    skew = np.sum(d ** 3) / n / cm2 ** 1.5
    kurt = np.sum(d ** 4) / n / cm2 ** 2
    return mu, cm2, skew, kurt - 3


def print_moments(label, x):
    mu, var, sk, ek = first_four_moments(x)
    print(f"{label:>8}: mean={mu: .6f}  std={np.sqrt(var):.6f}  skew={sk: .4f}  excess kurt={ek: .4f}")
    return mu, var, sk, ek


def aicc(ll, k, n):
    return -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_normal(x):
    """Normal MLE: sample mean and the n-1 standard deviation (same convention as the course code)."""
    x = np.asarray(x, float)
    return stats.norm(x.mean(), x.std(ddof=1)), 2


def fit_t(x):
    """Generalized (location-scale) Student t by maximum likelihood. Returns (frozen dist, number of params)."""
    x = np.asarray(x, float)
    k = np.mean((x - x.mean()) ** 4) / np.mean((x - x.mean()) ** 2) ** 2 - 3
    nu0 = min(max(6.0 / k + 4 if k > 0.05 else 50.0, 2.5), 100.0)   # t excess kurtosis = 6/(nu-4), solve for nu
    s0 = np.sqrt(x.var(ddof=1) * (nu0 - 2) / nu0)

    def nll(p):
        mu, ls, lnu = p                                              # search on logs so scale>0 and nu>2
        return -np.sum(stats.t.logpdf(x, df=2 + np.exp(lnu), loc=mu, scale=np.exp(ls)))

    res = optimize.minimize(nll, [x.mean(), np.log(s0), np.log(nu0 - 2)], method="Nelder-Mead",
                            options=dict(xatol=1e-10, fatol=1e-12, maxiter=5000))
    mu, s, nu = res.x[0], np.exp(res.x[1]), 2 + np.exp(res.x[2])
    return stats.t(df=nu, loc=mu, scale=s), 3


def _cutoff(x, alpha):
    """alpha quantile of a sample: average the two order statistics around position n*alpha (course convention)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return x, 0.5 * (x[int(np.ceil(n * alpha)) - 1] + x[int(np.floor(n * alpha)) - 1])


def var_sample(pnl, alpha):
    """VaR as a positive loss from a sample of P&L (or returns)."""
    return -_cutoff(pnl, alpha)[1]


def es_sample(pnl, alpha):
    """ES = average loss at or beyond the VaR cutoff, as a positive number."""
    x, c = _cutoff(pnl, alpha)
    return -x[x <= c].mean()


def var_es_normal(mu, s, alpha):
    """VaR and ES of a normal with mean mu and std s, both as positive losses."""
    z = stats.norm.ppf(alpha)
    return -(mu + s * z), -mu + s * stats.norm.pdf(z) / alpha


# ---------- Generalized Black Scholes Merton (Week 6) ----------
# b is the cost of carry: b = r for a non dividend stock, b = r - q with a continuous dividend yield q.

def _d1d2(S, K, T, r, b, sig):
    d1 = (np.log(S / K) + (b + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
    return d1, d1 - sig * np.sqrt(T)


def gbsm(call, S, K, T, r, b, sig):
    d1, d2 = _d1d2(S, K, T, r, b, sig)
    if call:
        return S * np.exp((b - r) * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp((b - r) * T) * stats.norm.cdf(-d1)


def gbsm_greeks(call, S, K, T, r, b, sig):
    """Closed form delta, gamma, vega (per 1.00 of vol), theta (per year, the change as time passes)."""
    d1, d2 = _d1d2(S, K, T, r, b, sig)
    carry = np.exp((b - r) * T)
    pdf = stats.norm.pdf(d1)
    gamma = carry * pdf / (S * sig * np.sqrt(T))
    vega = S * carry * pdf * np.sqrt(T)
    if call:
        delta = carry * stats.norm.cdf(d1)
        theta = (-S * carry * pdf * sig / (2 * np.sqrt(T)) - (b - r) * S * carry * stats.norm.cdf(d1)
                 - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
    else:
        delta = carry * (stats.norm.cdf(d1) - 1)
        theta = (-S * carry * pdf * sig / (2 * np.sqrt(T)) + (b - r) * S * carry * stats.norm.cdf(-d1)
                 + r * K * np.exp(-r * T) * stats.norm.cdf(-d2))
    return delta, gamma, vega, theta


def implied_vol(call, price, S, K, T, r, b, lo=1e-4, hi=5.0):
    """Invert GBSM with a bracketing root finder. Returns (vol, reason). vol is NaN when no vol works.
    A European price has to sit between its no arbitrage bounds:
      call: max(S e^{(b-r)T} - K e^{-rT}, 0) < C < S e^{(b-r)T}
      put : max(K e^{-rT} - S e^{(b-r)T}, 0) < P < K e^{-rT}
    Below the lower bound even a vol of zero is too expensive, so there's no answer."""
    fwd_s, pv_k = S * np.exp((b - r) * T), K * np.exp(-r * T)
    lower = max(fwd_s - pv_k, 0) if call else max(pv_k - fwd_s, 0)
    if price <= lower + 1e-12:
        return np.nan, f"price {price:.3f} <= lower bound {lower:.3f}"
    f = lambda s: gbsm(call, S, K, T, r, b, s) - price
    if f(lo) > 0:
        return np.nan, f"price below the vol={lo} value"
    if f(hi) < 0:
        return np.nan, "price above the vol=500% value"
    return optimize.brentq(f, lo, hi, xtol=1e-12), ""


# ---------- Binomial trees (Week 7) ----------
# CRR tree: u = e^{sig sqrt(dt)}, d = 1/u, pu = (e^{b dt} - d)/(u - d), discount e^{-r dt}.
# Written to work on a whole vector of starting prices at once, so the non recombining dividend tree
# (which starts a new tree at every node on a dividend date) stays fast.

def _tree_core(call, S0, K, r, b, sig, dt, n_steps, divs, american):
    """S0 = 1-D array of starting stock prices. divs = list of (step, amount), steps counted from this start.
    Returns an array of option values, one per starting price.
    Same logic as the course bt_american with discrete dividends:
      - roll back a normal tree until the first dividend step
      - at each node on the dividend step, the 'hold' value is a brand new tree started at (price - dividend)
        with the remaining time and remaining dividends; the 'exercise' value uses the cum dividend price
        (exercising just before the stock goes ex)."""
    S0 = np.atleast_1d(np.asarray(S0, float))
    u = np.exp(sig * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd_ = 1 - pu
    df = np.exp(-r * dt)
    z = 1.0 if call else -1.0

    def prices(j):
        i = np.arange(j + 1)                       # number of up moves at step j
        return S0[:, None] * u ** i * d ** (j - i)

    if not divs or divs[0][0] > n_steps:
        j_end = n_steps
        v = np.maximum(z * (prices(j_end) - K), 0)
    else:
        j_end, amt = divs[0]
        p = prices(j_end)
        rest = [(s - j_end, a) for s, a in divs[1:]]
        hold = _tree_core(call, (p - amt).ravel(), K, r, b, sig, dt, n_steps - j_end, rest, american).reshape(p.shape)
        v = np.maximum(hold, np.maximum(z * (p - K), 0)) if american else hold

    for j in range(j_end - 1, -1, -1):             # backward induction
        v = df * (pu * v[:, 1:] + pd_ * v[:, :-1])
        if american:
            v = np.maximum(v, z * (prices(j) - K))
    return v[:, 0]


def tree_price(call, S, K, T, r, b, sig, N, american=True, div_amts=(), div_times=()):
    """Price one option on an N step tree. Dividend times (years) must fall on a step: step = t / dt."""
    dt = T / N
    steps = [int(round(t / dt)) for t in div_times]
    for t, s in zip(div_times, steps):
        assert abs(s * dt - t) < 1e-9, f"dividend at {t} is not on a tree step for N={N}"
    divs = list(zip(steps, div_amts))
    return float(_tree_core(call, S, K, r, b, sig, dt, N, divs, american)[0])


# =====================================================================================
# PROBLEM 1 - Implied volatility from a dirty chain
# =====================================================================================
header("PROBLEM 1 - Implied volatility from a dirty chain")

S1, r1, q1 = 182.40, 0.0425, 0.006
b1 = r1 - q1                                           # cost of carry with a continuous dividend yield
days1 = (pd.Timestamp("2026-12-18") - pd.Timestamp("2026-11-13")).days
T1 = days1 / 365                                       # calendar days / 365, as the problem says
print(f"S = {S1}, r = {r1}, q = {q1}, days = {days1}, T = {T1:.6f}")

p1 = pd.read_csv(HERE / "problem1.csv")
p1["Mid"] = (p1.Bid + p1.Ask) / 2
p1["Spread"] = p1.Ask - p1.Bid
calls = p1[p1.Type == "Call"].set_index("Strike")
puts = p1[p1.Type == "Put"].set_index("Strike")
K1 = calls.index.to_numpy()

part("1(a)", "put call parity at every strike, using mids")
# Parity with a dividend yield: C - P = S e^{-qT} - K e^{-rT}. Gap = what the quotes say minus what parity says.
fwd_term = S1 * np.exp(-q1 * T1) - K1 * np.exp(-r1 * T1)
gap = calls.Mid.to_numpy() - puts.Mid.to_numpy() - fwd_term
band = calls.Spread.to_numpy() + puts.Spread.to_numpy()   # can't trade a gap smaller than both spreads
parity = pd.DataFrame({"C mid": calls.Mid.values, "P mid": puts.Mid.values, "S e^-qT - K e^-rT": fwd_term,
                       "gap": gap, "band": band, "flag": np.where(np.abs(gap) > band, "<-- FLAG", "")}, index=K1)
parity.index.name = "Strike"
print(parity.to_string())
flagged = K1[np.abs(gap) > band]
print(f"Flagged strikes: {[float(k) for k in flagged]}")

part("1(c)", "where the wings are (look before predicting)")
print("Cheapest quotes (mid <= 0.20), these are the far wings:")
print(p1[p1.Mid <= 0.20][["Type", "Strike", "Bid", "Ask", "Mid", "Spread"]].to_string(index=False))

part("1(d)", "implied vol from mid, bid, and ask")
rows = []
for typ, tab in (("Call", calls), ("Put", puts)):
    for K in K1:
        rec = {"Type": typ, "Strike": K}
        for col in ("Bid", "Mid", "Ask"):
            iv, why = implied_vol(typ == "Call", tab.loc[K, col], S1, K, T1, r1, b1)
            rec["IV " + col] = iv
            if why:
                rec.setdefault("note", []).append(f"{col}: {why}")
        rec["note"] = "; ".join(rec.get("note", []))
        rows.append(rec)
ivs = pd.DataFrame(rows)
print(ivs.to_string(index=False))
iv_c = ivs[ivs.Type == "Call"].set_index("Strike")
iv_p = ivs[ivs.Type == "Put"].set_index("Strike")

plt.figure(figsize=(8, 5))
plt.plot(K1, iv_c["IV Mid"], "o-", label="Call IV (mid)")
plt.plot(K1, iv_p["IV Mid"], "s-", label="Put IV (mid)")
plt.fill_between(K1, iv_c["IV Bid"], iv_c["IV Ask"], alpha=0.15, label="Call bid-ask IV")
plt.fill_between(K1, iv_p["IV Bid"], iv_p["IV Ask"], alpha=0.15, label="Put bid-ask IV")
plt.axvline(S1, color="gray", ls=":", label=f"Spot {S1}")
plt.xlabel("Strike")
plt.ylabel("Implied volatility")
plt.title("XYZ implied volatility, 12/18/2026 expiry")
plt.legend()
plt.tight_layout()
plt.savefig(HERE / "problem1_smile.png", dpi=130)
plt.close()
print("Saved problem1_smile.png")

part("1(e)", "vega and the vol change from a 1 cent price change")
# Vega for a call and a put at the same strike and vol is identical, so I use each quote's own mid IV.
vega_tab = pd.DataFrame(index=K1)
vega_tab.index.name = "Strike"
for typ, ivt in (("Call", iv_c), ("Put", iv_p)):
    v = np.array([gbsm_greeks(typ == "Call", S1, K, T1, r1, b1, s)[2] if np.isfinite(s) else np.nan
                  for K, s in zip(K1, ivt["IV Mid"])])
    vega_tab[f"{typ} vega"] = v                       # dollars per 1.00 (100 vol points)
    vega_tab[f"{typ} dIV per 1c (vol pts)"] = 0.01 / v * 100
print(vega_tab.to_string())

part("1(f)", "flagged strike: parity gap in vol points vs the call-put IV gap")
for K in flagged:
    i = list(K1).index(K)
    v = vega_tab.loc[K, "Put vega"]
    print(f"K={K}: parity gap {gap[i]:+.4f}, band {band[i]:.2f}. gap / vega = {gap[i] / v * 100:+.2f} vol points")
    print(f"        call IV {iv_c.loc[K, 'IV Mid']:.4f}, put IV {iv_p.loc[K, 'IV Mid']:.4f}, "
          f"call - put = {(iv_c.loc[K, 'IV Mid'] - iv_p.loc[K, 'IV Mid']) * 100:+.2f} vol points")
    # what the put should be worth if it sat on the smile (average of its neighbours' put IVs)
    nb = np.nanmean([iv_p.loc[K - 5, "IV Mid"], iv_p.loc[K + 5, "IV Mid"]])
    print(f"        neighbour average put IV {nb:.4f} -> put worth {gbsm(False, S1, K, T1, r1, b1, nb):.3f} vs quoted mid {puts.loc[K, 'Mid']:.3f}")

part("1(g)", "the call wing above 210")
wing = pd.DataFrame({"C bid": calls.Bid, "C ask": calls.Ask, "IV bid": iv_c["IV Bid"], "IV mid": iv_c["IV Mid"],
                     "IV ask": iv_c["IV Ask"], "put IV mid": iv_p["IV Mid"],
                     "dIV/1c": vega_tab["Call dIV per 1c (vol pts)"]}).loc[195:]
print(wing.to_string())
for K in (215.0, 220.0, 225.0):
    fair = gbsm(True, S1, K, T1, r1, b1, iv_p.loc[K, "IV Mid"])
    print(f"Call {K}: value at the put's IV = {fair:.4f}  (quoted 0.05 / 0.10)")

part("1(h)", "the smile without the two problems")
clean = pd.DataFrame({"Strike": K1, "moneyness K/S": K1 / S1, "put IV": iv_p["IV Mid"].values,
                      "call IV": iv_c["IV Mid"].values})
clean = clean[(clean.Strike != flagged[0]) if len(flagged) else True]
# use OTM options (puts below spot, calls above) and drop the stale call wing above 210
clean["OTM IV"] = np.where(clean.Strike < S1, clean["put IV"], clean["call IV"])
clean = clean[clean.Strike <= 210]
print(clean.to_string(index=False))
atm = clean.iloc[(clean.Strike - S1).abs().argsort()[:1]]
print(f"ATM-ish IV (K={atm.Strike.values[0]}) = {atm['OTM IV'].values[0]:.4f}; "
      f"140 put IV = {iv_p.loc[140.0, 'IV Mid']:.4f}; 210 call IV = {iv_c.loc[210.0, 'IV Mid']:.4f}")


# =====================================================================================
# PROBLEM 2 - American options and discrete dividends
# =====================================================================================
header("PROBLEM 2 - American options and discrete dividends")

S2, K2, T2, r2, sig2 = 48.50, 50.0, 0.5, 0.045, 0.30
D_amt = [0.25, 1.50]
D_t = [0.10, 0.40]
NBIG = 500                                             # main tree size (multiple of 5 so dividends land on steps)

part("2(a)", "early exercise condition D > X (1 - e^{-r(T - t)})")
for D, t in zip(D_amt, D_t):
    thresh = K2 * (1 - np.exp(-r2 * (T2 - t)))
    print(f"Dividend {D:.2f} at t={t}: X(1 - e^(-r(T-t))) = {thresh:.4f} -> "
          f"{'early exercise CAN be optimal just before it' if D > thresh else 'never optimal to exercise before it'}")

part("2(c)", f"prices with and without dividends (trees use N={NBIG})")
pv_div = sum(D * np.exp(-r2 * t) for D, t in zip(D_amt, D_t))
res2 = {}
res2["Euro call"] = (gbsm(True, S2, K2, T2, r2, r2, sig2),
                     tree_price(True, S2, K2, T2, r2, r2, sig2, NBIG, False, D_amt, D_t),
                     gbsm(True, S2 - pv_div, K2, T2, r2, r2, sig2))
res2["Amer call"] = (tree_price(True, S2, K2, T2, r2, r2, sig2, NBIG, True),
                     tree_price(True, S2, K2, T2, r2, r2, sig2, NBIG, True, D_amt, D_t), np.nan)
res2["Euro put"] = (gbsm(False, S2, K2, T2, r2, r2, sig2),
                    tree_price(False, S2, K2, T2, r2, r2, sig2, NBIG, False, D_amt, D_t),
                    gbsm(False, S2 - pv_div, K2, T2, r2, r2, sig2))
res2["Amer put"] = (tree_price(False, S2, K2, T2, r2, r2, sig2, NBIG, True),
                    tree_price(False, S2, K2, T2, r2, r2, sig2, NBIG, True, D_amt, D_t), np.nan)
tab2 = pd.DataFrame(res2, index=["No dividends", "With dividends (tree)", "GBSM, S - PV(div)"]).T
tab2["Change from dividends"] = tab2.iloc[:, 1] - tab2.iloc[:, 0]
print(f"PV of dividends = {pv_div:.4f}, so GBSM uses S = {S2 - pv_div:.4f}")
print("(No dividend column: GBSM for the Europeans, tree for the Americans)")
print(tab2.to_string())
eu_c_tree_nodiv = tree_price(True, S2, K2, T2, r2, r2, sig2, NBIG, False)
eu_p_tree_nodiv = tree_price(False, S2, K2, T2, r2, r2, sig2, NBIG, False)
print(f"Check: European tree with no dividends (N={NBIG}) call {eu_c_tree_nodiv:.4f}, put {eu_p_tree_nodiv:.4f} vs GBSM above")

part("2(d)", "convergence of the American values as N grows")
conv = []
for N in (20, 25, 50, 55, 100, 105, 200, 205, 300, 305, 500, 505):
    conv.append((N, tree_price(True, S2, K2, T2, r2, r2, sig2, N, True, D_amt, D_t),
                 tree_price(False, S2, K2, T2, r2, r2, sig2, N, True, D_amt, D_t)))
conv = pd.DataFrame(conv, columns=["N", "Amer call (div)", "Amer put (div)"]).set_index("N")
print(conv.to_string())
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(conv.index, conv["Amer call (div)"], "o-")
ax[0].set_title("American call with dividends")
ax[1].plot(conv.index, conv["Amer put (div)"], "o-", color="C1")
ax[1].set_title("American put with dividends")
for a in ax:
    a.set_xlabel("N (tree steps)")
plt.tight_layout()
plt.savefig(HERE / "problem2_convergence.png", dpi=130)
plt.close()
print("Saved problem2_convergence.png")

part("2(e)", "delta and gamma by central differences on the tree (N=200)")
NG = 200


def amer(call, S, divs=D_amt, N=NG):
    return tree_price(call, S, K2, T2, r2, r2, sig2, N, True, divs, D_t)


g_rows = []
for call in (True, False):
    v0 = amer(call, S2)
    for h in (1, 0.1, 0.01, 0.001):
        up, dn = amer(call, S2 + h), amer(call, S2 - h)
        g_rows.append(("Call" if call else "Put", h, (up - dn) / (2 * h), (up - 2 * v0 + dn) / h ** 2))
greek2 = pd.DataFrame(g_rows, columns=["Option", "bump h", "delta", "gamma"])
print(greek2.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
# For reference: the European GBSM gamma at S - PV(div), what a smooth model would say
print(f"Reference: European call GBSM gamma at S - PV(div): {gbsm_greeks(True, S2 - pv_div, K2, T2, r2, r2, sig2)[1]:.6f}")
# Look at the tree value on a fine grid of S to see the shape (piecewise linear between node crossings)
grid = np.linspace(S2 - 0.05, S2 + 0.05, 11)
vals = [amer(True, s) for s in grid]
print("American call value on a fine grid of S (step 0.01), and the slope between points:")
print("  S     :", " ".join(f"{s:8.2f}" for s in grid))
print("  value :", " ".join(f"{v:8.5f}" for v in vals))
print("  slope :", "         " + " ".join(f"{d:8.5f}" for d in np.diff(vals) / np.diff(grid)))

part("2(f)", "sensitivity to each dividend amount (central difference, bump 0.01)")
s_rows = []
for call in (True, False):
    for k in range(2):
        hi = list(D_amt); lo = list(D_amt)
        hi[k] += 0.01; lo[k] -= 0.01
        eu_hi = tree_price(call, S2, K2, T2, r2, r2, sig2, NG, False, hi, D_t)
        eu_lo = tree_price(call, S2, K2, T2, r2, r2, sig2, NG, False, lo, D_t)
        s_rows.append(("Call" if call else "Put", f"D{k + 1} ({D_amt[k]} at t={D_t[k]})",
                       (eu_hi - eu_lo) / 0.02, (amer(call, S2, hi) - amer(call, S2, lo)) / 0.02))
sens = pd.DataFrame(s_rows, columns=["Option", "Dividend", "European dV/dD", "American dV/dD"])
print(sens.to_string(index=False))
for k in range(2):
    print(f"For reference e^(-r t) for D{k + 1} = {np.exp(-r2 * D_t[k]):.4f}")

part("2(g)", "early exercise premium and the critical stock price before the 2nd dividend")
eep = pd.DataFrame({
    "No dividends": [res2["Amer call"][0] - res2["Euro call"][0], res2["Amer put"][0] - res2["Euro put"][0]],
    "With dividends": [res2["Amer call"][1] - res2["Euro call"][1], res2["Amer put"][1] - res2["Euro put"][1]]},
    index=["Call", "Put"])
print("Early exercise premium = American - European (GBSM without dividends, tree with them)")
print(eep.to_string())
# Just before the 2nd dividend (t = 0.4, 0.1 years left, no dividends after) the call is worth
# max(S - K, European call on S - 1.50 with 0.1 years left). With no more dividends the American call
# after the dividend is just a European call, so GBSM is exact for the hold value. Solve S - K = hold.
f_crit = lambda s: (s - K2) - gbsm(True, s - D_amt[1], K2, T2 - D_t[1], r2, r2, sig2)
s_star = optimize.brentq(f_crit, K2 + 1e-6, 200)
print(f"Critical cum dividend stock price at t=0.4: S* = {s_star:.4f}  "
      f"(exercise value {s_star - K2:.4f} = hold value {gbsm(True, s_star - D_amt[1], K2, T2 - D_t[1], r2, r2, sig2):.4f})")
# chance the stock gets above S* by t=0.4 under the tree's risk neutral lognormal (after the first dividend)
s_fwd = (S2 - D_amt[0] * np.exp(-r2 * D_t[0])) * np.exp(r2 * D_t[1])
prob = 1 - stats.norm.cdf((np.log(s_star / (S2 - D_amt[0] * np.exp(-r2 * D_t[0]))) - (r2 - sig2 ** 2 / 2) * D_t[1]) / (sig2 * np.sqrt(D_t[1])))
print(f"Rough risk neutral chance S(0.4) > S*: {prob:.3f}")

part("2(i)", "per dollar effect of each dividend on the call (from 2f)")
c_sens = sens[sens.Option == "Call"].set_index("Dividend")
print(c_sens.to_string())
r_ = c_sens["American dV/dD"].values
print(f"Ratio D1 effect / D2 effect, American call: {r_[0] / r_[1]:.2f}")


# =====================================================================================
# PROBLEM 3 - VaR of an option book
# =====================================================================================
header("PROBLEM 3 - VaR of an option book")

px3 = pd.read_csv(HERE / "problem3.csv")
books = pd.read_csv(HERE / "problem3_portfolios.csv")
S3 = float(px3.ABC.iloc[-1])                           # value everything at the last price
r3, DAYS, H = 0.0425, 255, 10                          # 255 trading days a year, 10 day horizon
rets3 = px3.ABC.pct_change().dropna().to_numpy()       # arithmetic returns
print(f"Last price S = {S3}, {len(rets3)} daily returns, r = {r3}, horizon = {H} trading days")


def book_value(book, S, days_passed=0):
    """Full revaluation of a book at stock price S (can be an array) after days_passed trading days.
    Implied vol is held constant over the horizon, as the problem says."""
    S = np.asarray(S, float)
    tot = np.zeros_like(S)
    for _, p in book.iterrows():
        if p.Type == "Stock":
            tot = tot + p.Holding * S
        else:
            T = (p.DaysToExpiry - days_passed) / DAYS
            tot = tot + p.Holding * gbsm(p.Type == "Call", S, p.Strike, T, r3, r3, p.ImpliedVol)
    return tot


def book_greeks(book, S):
    """Dollar Greeks of a book: delta (shares), gamma (per $1 move), theta (per year)."""
    dl = gm = th = 0.0
    for _, p in book.iterrows():
        if p.Type == "Stock":
            dl += p.Holding
        else:
            T = p.DaysToExpiry / DAYS
            d, g, _, t = gbsm_greeks(p.Type == "Call", S, p.Strike, T, r3, r3, p.ImpliedVol)
            dl += p.Holding * d; gm += p.Holding * g; th += p.Holding * t
    return dl, gm, th


names = list(dict.fromkeys(books.Portfolio))           # keep file order
part("3(a)", "value and Greeks of each book")
g3 = {}
for nm in names:
    bk = books[books.Portfolio == nm]
    dl, gm, th = book_greeks(bk, S3)
    g3[nm] = dict(value=book_value(bk, S3), delta=dl, gamma=gm, theta_yr=th, theta_10d=th * H / DAYS,
                  dollar_delta=dl * S3)
g3 = pd.DataFrame(g3).T
print(g3.to_string())
sig_d = rets3.std(ddof=1)
dS1 = S3 * sig_d * np.sqrt(H)                           # one standard deviation 10 day move in dollars
print(f"Daily return std {sig_d:.6f}, 10 day 1-sd move in ABC = ${dS1:.3f}")
print("Size of the gamma term 0.5*Gamma*(1sd move)^2 vs the delta term |delta|*(1sd move):")
for nm in names:
    print(f"  {nm:>16}: delta term {abs(g3.loc[nm, 'delta']) * dS1:9.2f}   gamma term {0.5 * g3.loc[nm, 'gamma'] * dS1 ** 2:+9.2f}"
          f"   theta 10d {g3.loc[nm, 'theta_10d']:+8.2f}")

part("3(c)", "moments of ABC returns, normal vs t by AICc")
print_moments("ABC", rets3)
fn, kn = fit_normal(rets3)
ft, kt = fit_t(rets3)
n3 = len(rets3)
ll_n, ll_t = fn.logpdf(rets3).sum(), ft.logpdf(rets3).sum()
print(f"Normal: mu={fn.mean():.6f} sigma={fn.std():.6f}  LL={ll_n:.3f}  AICc={aicc(ll_n, kn, n3):.3f}")
mu_t, sc_t = ft.kwds["loc"], ft.kwds["scale"]
nu_t = ft.kwds["df"]
print(f"t     : mu={mu_t:.6f} scale={sc_t:.6f} nu={nu_t:.4f}  LL={ll_t:.3f}  AICc={aicc(ll_t, kt, n3):.3f}")
print(f"AICc picks: {'t' if aicc(ll_t, kt, n3) < aicc(ll_n, kn, n3) else 'normal'}")

part("3(d)", "10 day delta normal VaR and ES (zero mean)")
# Delta normal: dP = delta * S * r_10, r_10 ~ N(0, sigma^2 * 10). Using the sample daily std.
dn = {}
for nm in names:
    s_pnl = abs(g3.loc[nm, "dollar_delta"]) * sig_d * np.sqrt(H)
    v5, e5 = var_es_normal(0, s_pnl, 0.05)
    v1, e1 = var_es_normal(0, s_pnl, 0.01)
    dn[nm] = {"VaR 5%": v5, "ES 5%": e5, "VaR 1%": v1, "ES 1%": e1}
dn = pd.DataFrame(dn).T
print(dn.to_string())

part("3(e)", "full revaluation Monte Carlo, 10 days, 100,000 paths per distribution")
NSIM = 100_000
rng = np.random.default_rng(545)
# zero mean: normal uses N(0, sigma); t uses the fitted scale and nu with location set to 0
R_norm = rng.normal(0, fn.std(), size=(NSIM, H))
R_t = stats.t.rvs(nu_t, loc=0, scale=sc_t, size=(NSIM, H), random_state=rng)
S10 = {"Normal": S3 * np.prod(1 + R_norm, axis=1), "t": S3 * np.prod(1 + R_t, axis=1)}   # compound the 10 days
sim_rows = []
pnl_sims = {}
for dist, s10 in S10.items():
    for nm in names:
        bk = books[books.Portfolio == nm]
        pnl = book_value(bk, s10, days_passed=H) - book_value(bk, S3)
        pnl_sims[(dist, nm)] = pnl
        v5, e5, v1, e1 = var_sample(pnl, .05), es_sample(pnl, .05), var_sample(pnl, .01), es_sample(pnl, .01)
        sim_rows.append((nm, dist, pnl.mean(), v5, e5, v1, e1, e5 / v5))
sim = pd.DataFrame(sim_rows, columns=["Book", "Dist", "Mean P&L", "VaR 5%", "ES 5%", "VaR 1%", "ES 1%", "ES/VaR 5%"])
sim = sim.sort_values(["Book", "Dist"], key=lambda c: c.map({n: i for i, n in enumerate(names)}) if c.name == "Book" else c)
print(sim.to_string(index=False))
print(f"Zero move P&L over 10 days (S unchanged, 10 days of decay):")
for nm in names:
    bk = books[books.Portfolio == nm]
    print(f"  {nm:>16}: {float(book_value(bk, S3, H) - book_value(bk, S3)):+.2f}")

part("3(f)", "plot of 10 day P&L vs price, with delta only line")
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
grid3 = np.linspace(S3 * 0.75, S3 * 1.25, 300)
for ax, nm in zip(axs.ravel(), names):
    bk = books[books.Portfolio == nm]
    full = book_value(bk, grid3, H) - book_value(bk, S3)
    ax.plot(grid3, full, label="Full revaluation (10 days later)")
    ax.plot(grid3, g3.loc[nm, "delta"] * (grid3 - S3), "--", label="Delta only")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(S3, color="gray", lw=0.5, ls=":")
    ax.set_title(nm)
    ax.set_xlabel("ABC price in 10 days")
    ax.set_ylabel("P&L ($)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(HERE / "problem3_pnl.png", dpi=130)
plt.close()
print("Saved problem3_pnl.png")

part("3(g)", "does the 2nd order (delta-gamma-theta) expansion reproduce full revaluation?")
dg_rows = []
for nm in names:
    dS = S10["Normal"] - S3
    approx = g3.loc[nm, "delta"] * dS + 0.5 * g3.loc[nm, "gamma"] * dS ** 2 + g3.loc[nm, "theta_10d"]
    full = pnl_sims[("Normal", nm)]
    dg_rows.append((nm, var_sample(approx, .05), var_sample(full, .05), es_sample(approx, .01), es_sample(full, .01),
                    approx.mean(), full.mean()))
dg = pd.DataFrame(dg_rows, columns=["Book", "VaR5 d-g-t", "VaR5 full", "ES1 d-g-t", "ES1 full", "mean d-g-t", "mean full"])
print("Same normal paths as 3(e):")
print(dg.to_string(index=False))

part("3(i)", "normal to t: % change in each number")
piv = sim.set_index(["Book", "Dist"])[["VaR 5%", "ES 5%", "VaR 1%", "ES 1%"]]
chg = (piv.xs("t", level="Dist") / piv.xs("Normal", level="Dist") - 1) * 100
print("Percent change going from normal to t:")
print(chg.loc[names].to_string(float_format=lambda v: f"{v:+.1f}%"))
for q in (0.05, 0.01, 0.001):
    print(f"10 day return quantile at {q}: normal {np.quantile(S10['Normal'] / S3 - 1, q):+.4f}, t {np.quantile(S10['t'] / S3 - 1, q):+.4f}")


# =====================================================================================
# PROBLEM 4 - A Sharpe ratio that sells insurance
# =====================================================================================
header("PROBLEM 4 - A Sharpe ratio that sells insurance")

p4 = pd.read_csv(HERE / "problem4.csv", parse_dates=["Month"]).set_index("Month")
assets = list(p4.columns)
RF_M = 0.03 / 12                                       # 3% a year, simple monthly (convention choice)
X4 = p4.to_numpy()
ex4 = X4 - RF_M
n4 = len(p4)

part("4(a)", "moments and annualized Sharpe ratios")
for a in assets:
    print_moments(a, p4[a])
sharpe = ex4.mean(0) / ex4.std(0, ddof=1) * np.sqrt(12)
print("Annualized Sharpe = mean monthly excess / monthly std * sqrt(12):")
for a, s in zip(assets, sharpe):
    print(f"  {a:>5}: {s:.4f}")
print("Effect of the single worst month on each asset:")
for j, a in enumerate(assets):
    worst = int(np.argmin(X4[:, j]))
    keep = np.delete(ex4[:, j], worst)
    print(f"  {a:>5}: worst month {p4.index[worst].date()} return {X4[worst, j]:+.4f}; "
          f"Sharpe without it {keep.mean() / keep.std(ddof=1) * np.sqrt(12):.4f}; "
          f"skew without it {first_four_moments(keep)[2]:+.3f}, excess kurt without it {first_four_moments(keep)[3]:.3f}")
putw_neg = (p4.PUTW < 0).sum()
print(f"PUTW: {putw_neg} losing months of {n4}, most common return {p4.PUTW.round(6).mode()[0]:.6f} "
      f"({(p4.PUTW.round(6) == p4.PUTW.round(6).mode()[0]).sum()} months)")
print(f"Correlation matrix:\n{p4.corr().to_string()}")

mu4 = X4.mean(0)
cov4 = np.cov(X4, rowvar=False)


def port_stats(w):
    m = w @ mu4
    s = np.sqrt(w @ cov4 @ w)
    return m, s, (m - RF_M) / s * np.sqrt(12)


def max_sharpe(bounds):
    """Long only (or capped) max Sharpe with weights summing to 1, by SLSQP. Several starts to be safe."""
    best = None
    for w0 in [np.full(4, .25), np.eye(4)[0] * .7 + .075, np.eye(4)[3] * .7 + .075]:
        w0 = np.clip(w0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = optimize.minimize(lambda w: -port_stats(w)[2], w0, method="SLSQP", bounds=bounds,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                                options=dict(ftol=1e-14, maxiter=1000))
        if best is None or res.fun < best.fun:
            best = res
    return best.x


part("4(c)", "max Sharpe portfolios: unconstrained, long only, long only with PUTW <= 20%")
w_unc = np.linalg.solve(cov4, mu4 - RF_M)
w_unc = w_unc / w_unc.sum()                             # tangency: w proportional to inverse(Sigma) (mu - rf), sum to 1
w_lo = max_sharpe([(0, 1)] * 4)
w_cap = max_sharpe([(0, 1)] * 3 + [(0, 0.2)])
W4 = {"Unconstrained": w_unc, "Long only": w_lo, "Long only, PUTW<=20%": w_cap}
wt = pd.DataFrame(W4, index=assets).T
print(wt.to_string(float_format=lambda v: f"{v:.4f}"))

# long only efficient frontier: min variance for each target mean.
# Start each solve from several points (equal weights and each single asset) and keep the best,
# because one start sometimes stalls on a bad corner. Only the part above the min variance point is "efficient".
w_mv = optimize.minimize(lambda w: w @ cov4 @ w, np.full(4, .25), method="SLSQP", bounds=[(0, 1)] * 4,
                         constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options=dict(ftol=1e-15)).x
targets = np.linspace(w_mv @ mu4, mu4.max(), 60)
front = []
for tm in targets:
    best = None
    for w0 in [np.full(4, .25)] + list(np.eye(4)):
        res = optimize.minimize(lambda w: w @ cov4 @ w * 1e4, w0, method="SLSQP", bounds=[(0, 1)] * 4,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                             {"type": "eq", "fun": lambda w, tm=tm: (w @ mu4 - tm) * 100}],
                                options=dict(ftol=1e-15, maxiter=1000))
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is not None:
        front.append((np.sqrt(best.x @ cov4 @ best.x) * np.sqrt(12), tm * 12))
front = np.array(front)
m_lo, s_lo, sr_lo = port_stats(w_lo)
plt.figure(figsize=(8, 5.5))
plt.plot(front[:, 0], front[:, 1], label="Long only efficient frontier")
xs = np.linspace(0, 0.12, 50)   # draw the CML over the low vol region where the tangency sits
plt.plot(xs, RF_M * 12 + sr_lo * xs, "--", label=f"Capital market line (Sharpe {sr_lo:.2f})")
for j, a in enumerate(assets):
    plt.scatter(np.sqrt(cov4[j, j] * 12), mu4[j] * 12, zorder=3)
    plt.annotate(a, (np.sqrt(cov4[j, j] * 12), mu4[j] * 12), textcoords="offset points", xytext=(5, 5))
plt.scatter(s_lo * np.sqrt(12), m_lo * 12, marker="*", s=200, color="k", label="Long only tangency", zorder=4)
plt.xlabel("Annualized volatility")
plt.ylabel("Annualized mean return")
plt.title("Long only frontier and CML")
plt.legend()
plt.tight_layout()
plt.savefig(HERE / "problem4_frontier.png", dpi=130)
plt.close()
print("Saved problem4_frontier.png")

part("4(d)", "Sharpe, historical and normal VaR / ES of each portfolio (monthly, % of value)")
rows4 = []
for nm, w in W4.items():
    rp = X4 @ w
    m, s, sr = port_stats(w)
    rec = {"Portfolio": nm, "Sharpe": sr}
    for a in (.05, .01):
        nv, ne = var_es_normal(rp.mean(), rp.std(ddof=1), a)
        rec[f"Hist VaR {a:.0%}"] = var_sample(rp, a)
        rec[f"Norm VaR {a:.0%}"] = nv
        rec[f"Hist ES {a:.0%}"] = es_sample(rp, a)
        rec[f"Norm ES {a:.0%}"] = ne
    rec["skew"], rec["xs kurt"] = first_four_moments(rp)[2:]
    rows4.append(rec)
risk4 = pd.DataFrame(rows4).set_index("Portfolio")
print(risk4.T.to_string())

part("4(e)", "CAPM regression of PUTW excess on EQ excess")


def ols(y, X, names_):
    """OLS with an intercept already in X. Returns a dict with coefficients, standard errors, t stats,
    R^2, residuals, Gaussian log likelihood (MLE sigma) and AICc (k = coefficients + 1 for sigma)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    s2 = e @ e / (n - p)                                # unbiased residual variance for standard errors
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    sig_mle = np.sqrt(e @ e / n)
    ll = np.sum(stats.norm.logpdf(e, 0, sig_mle))
    return dict(coef=pd.Series(beta, names_), se=pd.Series(se, names_), t=pd.Series(beta / se, names_),
                r2=r2, resid=e, ll=ll, aicc=aicc(ll, p + 1, n), n=n)


eq_ex = ex4[:, 0]
pw_ex = ex4[:, 3]
capm4 = ols(pw_ex, np.column_stack([np.ones(n4), eq_ex]), ["alpha", "beta"])
print(f"All {n4} months: alpha {capm4['coef']['alpha']:.6f}/month ({capm4['coef']['alpha'] * 12:.4%}/yr), "
      f"t = {capm4['t']['alpha']:.3f}; beta {capm4['coef']['beta']:.4f} (se {capm4['se']['beta']:.4f}); R^2 {capm4['r2']:.4f}")
down = eq_ex < -0.05
capm_dn = ols(pw_ex[down], np.column_stack([np.ones(down.sum()), eq_ex[down]]), ["alpha", "beta"])
up = ~down
capm_up = ols(pw_ex[up], np.column_stack([np.ones(up.sum()), eq_ex[up]]), ["alpha", "beta"])
print(f"EQ excess < -5% ({down.sum()} months): beta {capm_dn['coef']['beta']:.4f} (se {capm_dn['se']['beta']:.4f}), "
      f"alpha {capm_dn['coef']['alpha']:.5f}, R^2 {capm_dn['r2']:.4f}")
print(f"All other months ({up.sum()}): beta {capm_up['coef']['beta']:.4f} (se {capm_up['se']['beta']:.4f}), R^2 {capm_up['r2']:.4f}")
print(f"PUTW mean excess in the {down.sum()} down months: {pw_ex[down].mean():+.4f}; in the rest: {pw_ex[up].mean():+.4f}")

part("4(h)", "standard error of each mean, and bootstrapped long only tangency weights")
se_mu = X4.std(0, ddof=1) / np.sqrt(n4)
print(pd.DataFrame({"mean/month": mu4, "SE of mean": se_mu, "t = mean/SE": mu4 / se_mu,
                    "95% CI low (ann)": (mu4 - 1.96 * se_mu) * 12, "95% CI high (ann)": (mu4 + 1.96 * se_mu) * 12},
                   index=assets).to_string())
NB = 1000
rng4 = np.random.default_rng(4545)
boot = np.zeros((NB, 4))
for i in range(NB):
    idx = rng4.integers(0, n4, n4)                       # resample months with replacement
    Xb = X4[idx]
    mb, cb = Xb.mean(0), np.cov(Xb, rowvar=False)

    def neg_sr(w):
        return -((w @ mb - RF_M) / np.sqrt(w @ cb @ w))
    res = optimize.minimize(neg_sr, w_lo, method="SLSQP", bounds=[(0, 1)] * 4,
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    boot[i] = res.x
bt = pd.DataFrame({"point est": w_lo, "boot mean": boot.mean(0), "boot std": boot.std(0),
                   "5th pct": np.percentile(boot, 5, 0), "95th pct": np.percentile(boot, 95, 0),
                   "% of draws = 0": (boot < 1e-4).mean(0) * 100}, index=assets)
print(f"{NB} bootstrap resamples of the 240 months, long only max Sharpe each time:")
print(bt.to_string())


# =====================================================================================
# PROBLEM 5 - Alpha or factor exposure?
# =====================================================================================
header("PROBLEM 5 - Alpha or factor exposure?")

DAYS5 = 255                                            # annualize daily numbers with 255 days, same as Problem 3
funds = pd.read_csv(HERE / "problem5.csv", parse_dates=["Date"]).set_index("Date")
ff = pd.read_csv(HERE / "F-F_Research_Data_Factors_daily.CSV")
mom = pd.read_csv(HERE / "F-F_Momentum_Factor_daily.CSV")
mom.columns = [c.strip() for c in mom.columns]          # the file has "Mom   " with trailing spaces
ff["Date"] = pd.to_datetime(ff.Date.astype(str), format="%Y%m%d")
mom["Date"] = pd.to_datetime(mom.Date.astype(str), format="%Y%m%d")
fac = ff.merge(mom, on="Date").set_index("Date")
fac = fac / 100                                        # Fama French store percentages
df5 = funds.join(fac, how="inner")
print(f"Funds: {funds.index.min().date()} to {funds.index.max().date()}, {len(funds)} days; matched to factors: {len(df5)} days")
FN = ["FUND1", "FUND2"]
for f in FN:
    df5[f + "_ex"] = df5[f] - df5.RF

part("5(a)", "mean excess return, vol, Sharpe, CAPM")
for f in FN:
    x = df5[f + "_ex"]
    print(f"{f}: mean excess {x.mean() * DAYS5:.4%}/yr, vol {x.std(ddof=1) * np.sqrt(DAYS5):.4%}/yr, "
          f"Sharpe {x.mean() / x.std(ddof=1) * np.sqrt(DAYS5):.4f}")
specs = {"CAPM": ["Mkt-RF"], "FF3": ["Mkt-RF", "SMB", "HML"], "Carhart": ["Mkt-RF", "SMB", "HML", "Mom"]}
fits = {}
for f in FN:
    for m, cols in specs.items():
        X = np.column_stack([np.ones(len(df5))] + [df5[c] for c in cols])
        fits[(f, m)] = ols(df5[f + "_ex"], X, ["alpha"] + cols)
for f in FN:
    c = fits[(f, "CAPM")]
    print(f"{f} CAPM: alpha {c['coef']['alpha'] * DAYS5:.4%}/yr (t {c['t']['alpha']:.3f}), beta {c['coef']['Mkt-RF']:.4f}, R^2 {c['r2']:.4f}")

part("5(b)", "correlation of each CAPM residual with SMB, HML, Mom")
for f in FN:
    e = fits[(f, "CAPM")]["resid"]
    print(f"{f}: " + "  ".join(f"{c} {np.corrcoef(e, df5[c])[0, 1]:+.4f}" for c in ("SMB", "HML", "Mom")))

part("5(c)", "CAPM, FF3, Carhart for each fund")
for f in FN:
    for m in specs:
        o = fits[(f, m)]
        tab = pd.DataFrame({"coef": o["coef"], "std err": o["se"], "t": o["t"]})
        print(f"\n{f} - {m}:  R^2 {o['r2']:.4f}   AICc {o['aicc']:.2f}   alpha {o['coef']['alpha'] * DAYS5:.4%}/yr  (alpha t {o['t']['alpha']:.3f})")
        print(tab.to_string(float_format=lambda v: f"{v:.6f}"))

part("5(d)", "expected annual return from the 4 factor betas, last 10 years of factor data")
last = fac.index.max()
recent = fac[fac.index > last - pd.DateOffset(years=10)]
print(f"Factor window: {recent.index.min().date()} to {last.date()} ({len(recent)} days)")
fmean = recent.mean() * DAYS5
print("Average annual factor returns (x255): " + ", ".join(f"{k} {v:.4%}" for k, v in fmean.items()))
cc = specs["Carhart"]
for f in FN:
    o = fits[(f, "Carhart")]
    prem = sum(o["coef"][c] * fmean[c] for c in cc)
    a = o["coef"]["alpha"] * DAYS5
    print(f"{f}: RF {fmean['RF']:.4%} + factor premia {prem:.4%} = {fmean['RF'] + prem:.4%} without alpha; "
          f"{fmean['RF'] + prem + a:.4%} with alpha ({a:+.4%})")

part("5(e)", "worst daily momentum return repeated on $10M")
worst_day = fac.Mom.idxmin()
wm = fac.Mom.min()
print(f"Worst Mom day in the file: {worst_day.date()}, {wm:.4%}")
for f in FN:
    b = fits[(f, "Carhart")]["coef"]["Mom"]
    loss = b * wm * 10_000_000
    sd = df5[f].std(ddof=1)
    print(f"{f}: Mom beta {b:+.4f} -> P&L {loss:+,.0f} on $10M = {abs(b * wm) / sd:.2f} daily std devs "
          f"(daily std {sd:.4%}, so 1 sd = ${sd * 1e7:,.0f})")

part("5(f)", "factor premia during the fund sample and how they explain FUND1's alpha change")
smean = df5[["Mkt-RF", "SMB", "HML", "Mom", "RF"]].mean() * DAYS5
print("Average annual factor returns during the fund sample: " + ", ".join(f"{k} {v:.4%}" for k, v in smean.items()))
for f in FN:
    a_c = fits[(f, "CAPM")]["coef"]["alpha"] * DAYS5
    a_4 = fits[(f, "Carhart")]["coef"]["alpha"] * DAYS5
    co = fits[(f, "Carhart")]["coef"]
    contrib = {c: co[c] * smean[c] for c in ("SMB", "HML", "Mom")}
    mkt_shift = (co["Mkt-RF"] - fits[(f, "CAPM")]["coef"]["Mkt-RF"]) * smean["Mkt-RF"]
    print(f"{f}: CAPM alpha {a_c:+.4%} -> Carhart alpha {a_4:+.4%} (change {a_4 - a_c:+.4%})")
    print("   piece of return from each added factor (beta x sample premium): "
          + ", ".join(f"{k} {v:+.4%}" for k, v in contrib.items())
          + f", market beta change {mkt_shift:+.4%}; total {sum(contrib.values()) + mkt_shift:+.4%}")

part("5(h)", "expected return from 5(d) vs realized in the sample")
for f in FN:
    o = fits[(f, "Carhart")]
    prem = sum(o["coef"][c] * fmean[c] for c in cc)
    real = df5[f].mean() * DAYS5
    print(f"{f}: realized mean return in sample {real:.4%}/yr; factor model, recent 10 yrs, no alpha {fmean['RF'] + prem:.4%}; "
          f"same betas with the sample's own factor means {smean['RF'] + sum(o['coef'][c] * smean[c] for c in cc):.4%}")
print("SE of a factor's annual mean over 10 years (std*sqrt(255)/sqrt(10)): "
      + ", ".join(f"{c} {recent[c].std() * np.sqrt(DAYS5) / np.sqrt(10):.2%}" for c in ("Mkt-RF", "SMB", "HML", "Mom")))

print("\nDone.")
