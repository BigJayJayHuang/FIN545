"""
Assignment 2 - FinTech 545 (Quantitative Risk Management)
Jason Huang

Covers: Problem 1 (correlations with missing data), Problem 2 (volatility after a regime change),
Problem 3 (diversification and VaR), Problem 4 (Gaussian vs t copula), Problem 5 (model based
simulation and residual correlation).

Run with:   python Assignment2.py
The five CSV files (problem1.csv ... problem5.csv) must sit in the same folder as this script.
Every number in the written answers (Assignment2_Answers.pdf) is printed by this script, in order.
Plots are saved as PNG files in the same folder (nothing pops up, so the script runs start to finish).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, optimize
from scipy.linalg import solve_triangular
from scipy.special import gammaln
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # read the CSVs from the script's own folder
np.set_printoptions(precision=4, suppress=True, linewidth=120)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

ALPHA = 0.05  # default tail probability for VaR / ES (the assignment says 5% unless told otherwise)


# =====================================================================================
# Helper functions (these follow the Week 1-5 course code, written in Python)
# =====================================================================================

def header(title):
    """Print a banner so the output file is easy to scan."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def first_four_moments(x):
    """Mean, variance, skewness, excess kurtosis using the biased (divide by n) moments, same as Week 1."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.sum() / n
    d = x - mu
    cm2 = d @ d / n                               # second central moment = (biased) variance
    skew = np.sum(d ** 3) / n / cm2 ** 1.5
    kurt = np.sum(d ** 4) / n / cm2 ** 2          # raw kurtosis (normal = 3)
    return mu, cm2, skew, kurt - 3


def print_moments(label, x):
    mu, var, sk, ek = first_four_moments(x)
    print(f"{label:>8}: mean={mu: .6f}  var={var:.8f}  std={np.sqrt(var):.6f}  skew={sk: .4f}  excess kurt={ek: .4f}")
    return mu, var, sk, ek


def ew_weights(n_rows, lam):
    """Exponential weights, oldest row first, rescaled to sum to 1 (newest day has the biggest weight)."""
    w = (1 - lam) * lam ** np.arange(n_rows - 1, -1, -1)
    return w / w.sum()


def cov_to_corr(cov):
    inv_sd = 1 / np.sqrt(np.diag(cov))
    return cov * np.outer(inv_sd, inv_sd)


# ---------- non-PSD repairs (Week 3) ----------

def _to_corr(a):
    """Return (correlation matrix, std devs or None). Lets the repairs accept covariance or correlation."""
    a = np.array(a, dtype=float)
    if np.allclose(np.diag(a), 1.0):
        return a, None
    sd = np.sqrt(np.diag(a))
    return a / np.outer(sd, sd), sd


def rebonato_jackel(a, epsilon=0.0):
    """Rebonato and Jackel: floor the eigenvalues at epsilon, then rescale rows so the diagonal is 1 again."""
    out, sd = _to_corr(a)
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)                    # lift negative eigenvalues
    t = 1.0 / ((vecs * vecs) @ vals)                    # scaling that restores the 1s on the diagonal
    B = np.diag(np.sqrt(t)) @ vecs @ np.diag(np.sqrt(vals))
    out = B @ B.T                                       # B B' is always PSD
    return out if sd is None else out * np.outer(sd, sd)


def _clip_neg_eig(a):
    a = (a + a.T) / 2
    vals, vecs = np.linalg.eigh(a)
    return (vecs * np.maximum(vals, 0)) @ vecs.T


def higham(a, max_iter=200, tol=1e-12):
    """Higham (2002) nearest correlation matrix: alternate projections with Dykstra's correction.
    Projection 1 = nearest PSD matrix (clip eigenvalues). Projection 2 = set the diagonal to 1.
    The result is the PSD, unit-diagonal matrix closest to the input in Frobenius norm."""
    y, sd = _to_corr(a)
    y0 = y.copy()
    ds = np.zeros_like(y)
    last = np.inf
    for _ in range(max_iter):
        r = y - ds                       # remove the previous Dykstra correction
        x = _clip_neg_eig(r)             # project onto PSD
        ds = x - r                       # update correction
        y = x.copy()
        np.fill_diagonal(y, 1.0)         # project onto unit diagonal
        dist = np.sum((y - y0) ** 2)
        if abs(dist - last) < tol and np.linalg.eigvalsh(y).min() > -1e-10:
            break
        last = dist
    return y if sd is None else y * np.outer(sd, sd)


def chol_psd(a, tol=-1e-8):
    """Cholesky root that also works for PSD matrices that are singular (zero pivots allowed).
    np.linalg.cholesky refuses a matrix whose smallest eigenvalue is 0 (or -1e-12 from rounding),
    but a repaired matrix is exactly that, so we use the course version: a pivot between tol and 0 is treated as 0."""
    a = np.asarray(a, float)
    n = len(a)
    root = np.zeros((n, n))
    for j in range(n):
        piv = a[j, j] - root[j, :j] @ root[j, :j]
        if tol <= piv <= 0:
            piv = 0.0                                   # tiny negatives are rounding error
        if piv < 0:
            raise ValueError("matrix is not positive semi-definite")
        root[j, j] = np.sqrt(piv)
        if root[j, j] != 0.0:
            for i in range(j + 1, n):
                root[i, j] = (a[i, j] - root[i, :j] @ root[j, :j]) / root[j, j]
    return root


# ---------- distribution fitting (Week 2 / 4) ----------

def aicc(ll, k, n):
    return -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_normal(x):
    """Normal MLE: sample mean and the n-1 standard deviation (same convention as the course code)."""
    x = np.asarray(x, float)
    mu, s = x.mean(), x.std(ddof=1)
    return stats.norm(mu, s), 2


def fit_t(x):
    """Generalized (location-scale) Student t by maximum likelihood. Returns (frozen dist, number of params)."""
    x = np.asarray(x, float)
    k = np.mean((x - x.mean()) ** 4) / np.mean((x - x.mean()) ** 2) ** 2 - 3
    nu0 = min(max(6.0 / k + 4 if k > 0.05 else 50.0, 2.5), 100.0)   # t has excess kurtosis 6/(nu-4), solve for nu
    s0 = np.sqrt(x.var(ddof=1) * (nu0 - 2) / nu0)                    # scale that matches the sample variance

    def nll(p):
        mu, ls, lnu = p                                              # search on logs so scale>0 and nu>2
        return -np.sum(stats.t.logpdf(x, df=2 + np.exp(lnu), loc=mu, scale=np.exp(ls)))

    res = optimize.minimize(nll, [x.mean(), np.log(s0), np.log(nu0 - 2)], method="Nelder-Mead",
                            options=dict(xatol=1e-10, fatol=1e-12, maxiter=5000))
    mu, s, nu = res.x[0], np.exp(res.x[1]), 2 + np.exp(res.x[2])
    return stats.t(df=nu, loc=mu, scale=s), 3


# ---------- VaR / ES from a sample of P&L (Week 4/5 convention) ----------

def _cutoff(x, alpha):
    """5% quantile of a sample: average the two order statistics around position n*alpha."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return x, 0.5 * (x[int(np.ceil(n * alpha)) - 1] + x[int(np.floor(n * alpha)) - 1])


def var_sample(pnl, alpha=ALPHA):
    """VaR reported as a positive loss, from a vector of P&L."""
    return -_cutoff(pnl, alpha)[1]


def es_sample(pnl, alpha=ALPHA):
    """ES = average loss on the days at or beyond the VaR cutoff, as a positive number."""
    x, c = _cutoff(pnl, alpha)
    return -x[x <= c].mean()


def var_es_normal(pnl, alpha=ALPHA):
    """Normal VaR and ES from the sample mean and standard deviation of the P&L."""
    mu, s = np.mean(pnl), np.std(pnl, ddof=1)
    z = stats.norm.ppf(alpha)
    return -(mu + s * z), -mu + s * stats.norm.pdf(z) / alpha


# ---------- copula pieces (Week 5) ----------

def kendall_corr(X):
    """Correlation matrix from Kendall's tau: R_ij = sin(pi * tau_ij / 2). Repaired with Higham if not PSD."""
    n = X.shape[1]
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            tau = stats.kendalltau(X[:, i], X[:, j])[0]
            R[i, j] = R[j, i] = np.sin(np.pi * tau / 2)
    if np.linalg.eigvalsh(R).min() < 1e-8:
        R = higham(R)
    return R


def mvn_logpdf(Z, R):
    L = np.linalg.cholesky(R)
    q = np.sum(solve_triangular(L, Z.T, lower=True) ** 2, axis=0)   # squared Mahalanobis distance
    return -0.5 * (Z.shape[1] * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + q)


def mvt_logpdf(X, S, nu):
    d = X.shape[1]
    L = np.linalg.cholesky(S)
    q = np.sum(solve_triangular(L, X.T, lower=True) ** 2, axis=0)
    return (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)
            - np.sum(np.log(np.diag(L))) - 0.5 * (nu + d) * np.log1p(q / nu))


def gauss_copula_ll_t(U, R):
    """Per-day Gaussian copula log likelihood = joint normal density / product of marginal normal densities."""
    z = stats.norm.ppf(U)
    return mvn_logpdf(z, R) - stats.norm.logpdf(z).sum(axis=1)


def t_copula_ll_t(U, R, nu):
    """Per-day t copula log likelihood = joint t density / product of marginal t densities."""
    t = stats.t.ppf(U, nu)
    return mvt_logpdf(t, R, nu) - stats.t.logpdf(t, nu).sum(axis=1)


def profile_nu(ll_fn, lo=0.01, hi=0.49, n=200):
    """Search theta = 1/nu on a grid, then zoom in. Gives the best nu and its log likelihood."""
    th = np.linspace(lo, hi, n)
    lls = np.array([ll_fn(1 / t) for t in th])
    i = int(np.argmax(lls))
    fine = np.linspace(th[max(i - 1, 0)], th[min(i + 1, n - 1)], n)
    fl = np.array([ll_fn(1 / t) for t in fine])
    j = int(np.argmax(fl))
    return 1 / fine[j], fl[j]


def tail_dep_t(rho, nu):
    """Lower (= upper) tail dependence coefficient of the bivariate t copula."""
    return 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)


# =====================================================================================
# PROBLEM 1 - Correlations from mismatched histories
# =====================================================================================
header("PROBLEM 1 - Correlations from mismatched histories")

p1 = pd.read_csv(HERE / "problem1.csv")
cols = ["A", "B", "C", "D", "IDX"]
X1 = p1[cols]

# ---- Predict: how many days is each pair jointly observed? ----
obs = X1.notna().astype(int)
pair_counts = obs.T @ obs                     # entry (i,j) = number of days both i and j are observed
print("Days each pair is jointly observed (diagonal = days each series trades):")
print(pair_counts.to_string())
complete_mask = X1.notna().all(axis=1)
print(f"\nDays all five series trade (complete rows): {complete_mask.sum()} of {len(X1)}")

# ---- Fit (d): complete case and pairwise correlation matrices ----
R_cc = np.corrcoef(X1[complete_mask].to_numpy(), rowvar=False)   # complete case = drop any row with a blank

# Pairwise: for every pair use every day on which both are observed (each pair uses different days)
R_pw = np.eye(5)
for i in range(5):
    for j in range(i + 1, 5):
        both = X1[cols[i]].notna() & X1[cols[j]].notna()
        R_pw[i, j] = R_pw[j, i] = np.corrcoef(X1.loc[both, cols[i]], X1.loc[both, cols[j]])[0, 1]

# rough standard error of a correlation: (1 - rho^2) / sqrt(n - 1)
print("\nApprox. standard error of each pairwise correlation, (1 - rho^2)/sqrt(n - 1):")
se_pw = pd.DataFrame(np.nan, cols, cols)
for i in range(5):
    for j in range(5):
        if i != j:
            se_pw.iloc[i, j] = (1 - R_pw[i, j] ** 2) / np.sqrt(pair_counts.iloc[i, j] - 1)
print(se_pw.to_string())
print(f"Complete case uses only {complete_mask.sum()} rows, so its standard errors are about (1 - rho^2)/sqrt({complete_mask.sum() - 1}) = e.g. {(1 - 0.4 ** 2) / np.sqrt(complete_mask.sum() - 1):.3f} at rho=0.4")

print("\nComplete case correlation matrix:")
print(pd.DataFrame(R_cc, cols, cols).to_string())
print("\nPairwise correlation matrix:")
print(pd.DataFrame(R_pw, cols, cols).to_string())

ev_cc = np.linalg.eigvalsh(R_cc)
ev_pw = np.linalg.eigvalsh(R_pw)
print("\nEigenvalues, complete case:", ev_cc)
print("Eigenvalues, pairwise     :", ev_pw)


def try_chol(name, M):
    """Attempt a Cholesky factorization and report whether it worked.
    First the strict numpy version (needs strictly positive definite), then the PSD-tolerant course version."""
    try:
        np.linalg.cholesky(M)
        print(f"Cholesky of {name}: numpy SUCCEEDED")
    except np.linalg.LinAlgError as e:
        print(f"Cholesky of {name}: numpy FAILED ({e})")
    try:
        L = chol_psd(M)
        print(f"{'':>12}course chol_psd SUCCEEDED, max |L L' - M| = {np.abs(L @ L.T - M).max():.2e}")
    except ValueError as e:
        print(f"{'':>12}course chol_psd FAILED ({e})")


try_chol("complete case matrix", R_cc)
try_chol("pairwise matrix", R_pw)

# ---- Fit (e): covariance from pairwise correlations and full-history standard deviations ----
sd_full = X1.std(ddof=1).to_numpy()            # each series' own full history (blanks skipped)
print("\nFull history standard deviations:", {c: round(float(v), 6) for c, v in zip(cols, sd_full)})
# Portfolio: long 1 unit IDX, short the weighted basket 0.4A + 0.3B + 0.2C + 0.1D
w_te = np.array([-0.4, -0.3, -0.2, -0.1, 1.0])


def cov_from_corr(R):
    return R * np.outer(sd_full, sd_full)


def te_var(R):
    """Variance of the tracking portfolio w'Sigma w for a correlation matrix R (std devs held fixed)."""
    return w_te @ cov_from_corr(R) @ w_te


print(f"\nTracking portfolio variance, pairwise matrix      : {te_var(R_pw): .4e}  (std = {np.sign(te_var(R_pw)) * np.sqrt(abs(te_var(R_pw))):.5f})")
print(f"Tracking portfolio variance, complete case matrix : {te_var(R_cc): .4e}  (std = {np.sqrt(te_var(R_cc)):.5f})")

# ---- Fit (f): repair the pairwise matrix ----
R_rj = rebonato_jackel(R_pw)
R_hi = higham(R_pw)
fro = lambda M: np.linalg.norm(M - R_pw, "fro")
print("\nRepairs of the pairwise matrix:")
for name, M in [("Rebonato-Jackel", R_rj), ("Higham", R_hi)]:
    print(f"  {name:>16}: min eigenvalue = {np.linalg.eigvalsh(M).min(): .3e}, Frobenius distance from pairwise = {fro(M):.5f}, "
          f"tracking variance = {te_var(M): .4e}")
    try_chol(name + " matrix", M)  # a repaired matrix has smallest eigenvalue ~0 (PSD but singular), so strict Cholesky can fail by rounding

# ---- Reconcile (h): which entries did Higham move most? ----
diff_h = pd.DataFrame(R_hi - R_pw, cols, cols)
print("\nHigham change (repaired - pairwise):")
print(diff_h.to_string())
iu = np.triu_indices(5, 1)
order = np.argsort(-np.abs((R_hi - R_pw)[iu]))
print("Entries ranked by size of Higham change:")
for k in order[:5]:
    i, j = iu[0][k], iu[1][k]
    print(f"  {cols[i]}-{cols[j]}: pairwise={R_pw[i, j]: .4f}  Higham={R_hi[i, j]: .4f}  change={R_hi[i, j] - R_pw[i, j]: .4f}  (days observed together: {pair_counts.iloc[i, j]})")
diff_r = pd.DataFrame(R_rj - R_pw, cols, cols)
print("\nRebonato-Jackel change (repaired - pairwise):")
print(diff_r.to_string())

# eigenvector that carries the negative eigenvalue: which assets does the violation live in?
vals, vecs = np.linalg.eigh(R_pw)
print("\nEigenvector for the smallest eigenvalue of the pairwise matrix (A,B,C,D,IDX):", vecs[:, 0])

# ---- Reconcile (i): size of repair vs. size of estimator gap ----
print(f"\nFrobenius gap between complete case and pairwise : {np.linalg.norm(R_cc - R_pw, 'fro'):.5f}")
print(f"Frobenius size of Higham repair                  : {fro(R_hi):.5f}")
print(f"Frobenius size of Rebonato-Jackel repair         : {fro(R_rj):.5f}")
print(f"Tracking variance: complete={te_var(R_cc):.4e}, pairwise={te_var(R_pw):.4e}, Higham={te_var(R_hi):.4e}, R-J={te_var(R_rj):.4e}")

# sanity check on the premise of part (b): how close is IDX to the weighted basket?
rows = complete_mask
basket = (X1.loc[rows, ["A", "B", "C", "D"]] * np.array([0.4, 0.3, 0.2, 0.1])).sum(axis=1)
te_series = X1.loc[rows, "IDX"] - basket
true_te_var = (X1.loc[rows, cols].to_numpy() @ w_te).var(ddof=1)
print(f"\nActual variance of the tracking portfolio's return on the {rows.sum()} complete days (direct, no correlation matrix): {true_te_var:.4e}")
print(f"\nTracking error check on the {rows.sum()} complete days: std of (IDX - basket) = {te_series.std(ddof=1):.6f}, std of IDX = {X1.loc[rows, 'IDX'].std(ddof=1):.6f}")
print(f"  -> fraction of IDX variance not explained by the basket (on complete days): {te_series.var(ddof=1) / X1.loc[rows, 'IDX'].var(ddof=1):.5f}")


# =====================================================================================
# PROBLEM 2 - A volatility estimate after the regime changed
# =====================================================================================
header("PROBLEM 2 - A volatility estimate after the regime changed")

P = pd.read_csv(HERE / "problem2.csv")
price = P["Price"].to_numpy()
r_raw = price[1:] / price[:-1] - 1                 # arithmetic returns
r2 = r_raw - r_raw.mean()                          # remove the sample mean as the problem says
n2 = len(r2)
POS = 1_000_000                                    # $1,000,000 position
print(f"{n2} daily returns, sample mean removed (it was {r_raw.mean():.6f})")

# ---- Predict: plot the returns (and a rolling std to see the regime) ----
roll = pd.Series(r2).rolling(20).std()
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(r2, lw=0.8)
ax[0].set_title("Problem 2: daily returns (mean removed)")
ax[0].axvline(n2 - 30, color="red", ls="--", lw=1, label="last 30 days")
ax[0].legend()
ax[1].plot(roll, color="darkorange")
ax[1].set_title("20-day rolling standard deviation")
ax[1].set_xlabel("day")
plt.tight_layout()
plt.savefig(HERE / "problem2_returns.png", dpi=130)
plt.close()

# Regime length chosen by eye from the plot: the last 30 returns look different from the rest.
K_RECENT = 30
print(f"\nRegime split used (chosen by eye from the plot): last {K_RECENT} days vs. the first {n2 - K_RECENT}")

# ---- (b) effective sample size, half life, weight on recent days ----
print("\nlambda | n_eff (n=500 weights) | n_eff closed form (1+l)/(1-l) | half life (days) | weight on last "
      f"{K_RECENT} days | std error of vol (1/sqrt(2 n_eff))")
neff = {}
for lam in (0.94, 0.97):
    w = ew_weights(n2, lam)
    neff_fin = 1 / np.sum(w ** 2)                      # Week 3: 1 / sum of squared normalized weights
    neff_cf = (1 + lam) / (1 - lam)
    half = np.log(0.5) / np.log(lam)
    wrecent = w[-K_RECENT:].sum()
    neff[lam] = neff_fin
    print(f"{lam:6.2f} | {neff_fin:21.2f} | {neff_cf:29.2f} | {half:16.2f} | {wrecent:21.4f} | {1 / np.sqrt(2 * neff_fin):.4f}")
print(f"Equal weight: the last {K_RECENT} days carry {K_RECENT / n2:.4f} of the weight.")

# ---- (c) moments of the full sample ----
print("\nFirst four moments of the full sample:")
mu2, var2, sk2, ek2 = print_moments("returns", r2)
print(f"Under a normal, the std error of sample skewness is about sqrt(6/n) = {np.sqrt(6 / n2):.3f} and of excess kurtosis sqrt(24/n) = {np.sqrt(24 / n2):.3f}")
print(f"  -> skewness is {sk2 / np.sqrt(6 / n2):.1f} standard errors from 0, excess kurtosis is {ek2 / np.sqrt(24 / n2):.1f} standard errors from 0")

# ---- Fit (d): the five one day VaR estimates in dollars ----
sd_eq = r2.std(ddof=1)
# EW std: returns already have mean removed, so the EW second moment around zero is the variance
sd_ew = {lam: np.sqrt(np.sum(ew_weights(n2, lam) * r2 ** 2)) for lam in (0.97, 0.94)}
z05 = stats.norm.ppf(ALPHA)
t_dist, _ = fit_t(r2)
t_mu, t_s, t_nu = t_dist.kwds["loc"], t_dist.kwds["scale"], t_dist.kwds["df"]

var_est = {
    "1. Normal, equally weighted std": -z05 * sd_eq * POS,
    "2. Normal, EW std lambda=0.97":   -z05 * sd_ew[0.97] * POS,
    "3. Normal, EW std lambda=0.94":   -z05 * sd_ew[0.94] * POS,
    "4. Student t (MLE)":              -t_dist.ppf(ALPHA) * POS,
    "5. Historical simulation":        var_sample(r2) * POS,
}
print(f"\nDaily std: equal weight={sd_eq:.5f}, EW 0.97={sd_ew[0.97]:.5f}, EW 0.94={sd_ew[0.94]:.5f}")
print(f"Fitted t: location={t_mu:.6f}, scale={t_s:.6f}, nu={t_nu:.4f}   (t std = scale*sqrt(nu/(nu-2)) = {t_s * np.sqrt(t_nu / (t_nu - 2)):.5f})")
print("\n5% one day VaR on $1,000,000:")
for k, v in var_est.items():
    print(f"  {k:<34s} ${v:>12,.0f}")
print("Ranking, smallest to largest:", " < ".join(k.split('.')[0] for k, _ in sorted(var_est.items(), key=lambda kv: kv[1])))

# ---- Reconcile (f): std inside each regime, and the mixture explanation of kurtosis ----
old, new = r2[:-K_RECENT], r2[-K_RECENT:]
s_old, s_new = old.std(ddof=1), new.std(ddof=1)
print(f"\nStd in old regime (first {len(old)} days): {s_old:.5f}   Std in recent regime (last {K_RECENT} days): {s_new:.5f}   ratio = {s_new / s_old:.2f}")
print_moments("old", old)
print_moments("recent", new)
p_new = K_RECENT / n2
mix_var = (1 - p_new) * s_old ** 2 + p_new * s_new ** 2
mix_kurt = 3 * ((1 - p_new) * s_old ** 4 + p_new * s_new ** 4) / mix_var ** 2
print(f"Kurtosis of a 2-regime normal mixture with these stds and weights: {mix_kurt:.3f} (excess {mix_kurt - 3:.3f}); sample excess kurtosis = {ek2:.3f}")
print(f"t with nu={t_nu:.2f} implies excess kurtosis 6/(nu-4) = {6 / (t_nu - 4) if t_nu > 4 else float('inf'):.3f}")
print(f"VaR if we used each regime's own normal: old=${-z05 * s_old * POS:,.0f}, recent=${-z05 * s_new * POS:,.0f}")

# ---- Reconcile (g): is the EW gap bigger than noise? ----
gap = (sd_ew[0.94] - sd_ew[0.97]) / sd_ew[0.97]
print(f"\nRelative gap between the two EW vols (0.94 vs 0.97): {gap:.4f}")
for lam in (0.94, 0.97):
    print(f"  noise (std error / vol) at lambda={lam}: {1 / np.sqrt(2 * neff[lam]):.4f}")
print(f"  EW 0.94 vol vs equal weight vol, relative gap: {(sd_ew[0.94] - sd_eq) / sd_eq:.4f}")
print(f"  Dollar VaR gap 0.94 vs 0.97: ${var_est['3. Normal, EW std lambda=0.94'] - var_est['2. Normal, EW std lambda=0.97']:,.0f}")


# =====================================================================================
# PROBLEM 3 - When diversification raises VaR
# =====================================================================================
header("PROBLEM 3 - When diversification raises VaR")

p3 = pd.read_csv(HERE / "problem3.csv")
A3, B3 = p3["A"].to_numpy(), p3["B"].to_numpy()
n3 = len(A3)
print(f"{n3} scenarios; each bond bought at 90, pays 100 if no default -> no-default return = {100 / 90 - 1:.4f}")

# ---- (a) moments ----
print("\nFirst four moments of each bond's return:")
print_moments("A", A3)
print_moments("B", B3)

# ---- (b) counts of big losses ----
lossA, lossB = A3 < -0.20, B3 < -0.20
print(f"\nScenarios where A loses more than 20%: {lossA.sum()} ({lossA.mean():.2%})")
print(f"Scenarios where B loses more than 20%: {lossB.sum()} ({lossB.mean():.2%})")
print(f"Scenarios where BOTH lose more than 20%: {(lossA & lossB).sum()}")
print(f"Scenarios where AT LEAST ONE loses more than 20%: {(lossA | lossB).sum()} ({(lossA | lossB).mean():.2%})")
print(f"Worst single return: A={A3.min():.4f}, B={B3.min():.4f};  scenarios worse than -50%: A={np.sum(A3 < -0.5)}, B={np.sum(B3 < -0.5)}")
print(f"Share of A scenarios with a gain (return > 0): {np.mean(A3 > 0):.4f}")
print(f"Correlation of the two bonds' returns: {np.corrcoef(A3, B3)[0, 1]:.4f}")

# P&L for each choice (dollars)
pnl = {
    "$1M in A":       1e6 * A3,
    "$1M in B":       1e6 * B3,
    "$2M in A":       2e6 * A3,
    "$1M A + $1M B":  1e6 * A3 + 1e6 * B3,
}

# plot the P&L of the two choices
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(pnl["$2M in A"], bins=120, color="tab:blue", alpha=0.8)
ax[0].set_title("Concentrated: $2M in A")
ax[1].hist(pnl["$1M A + $1M B"], bins=120, color="tab:green", alpha=0.8)
ax[1].set_title("Diversified: $1M in each")
for a in ax:
    a.set_xlabel("P&L ($)")
    a.set_yscale("log")                    # log counts so the rare default scenarios are visible
plt.tight_layout()
plt.savefig(HERE / "problem3_pnl.png", dpi=130)
plt.close()

# ---- (c) historical VaR / ES at 5%, (d) normal VaR, (e) 1% ----
for alpha in (0.05, 0.01):
    print(f"\n--- alpha = {alpha:.0%} ---")
    print(f"{'Position':<16}{'Hist VaR':>14}{'Hist ES':>14}{'Normal VaR':>14}{'Normal ES':>14}")
    for name, p in pnl.items():
        nv, ne = var_es_normal(p, alpha)
        print(f"{name:<16}{var_sample(p, alpha):>14,.0f}{es_sample(p, alpha):>14,.0f}{nv:>14,.0f}{ne:>14,.0f}")

# ---- (f) subadditivity at 5% ----
print("\nSubadditivity check at 5% (historical):")
for label, f in [("VaR", var_sample), ("ES", es_sample)]:
    parts = f(pnl["$1M in A"]) + f(pnl["$1M in B"])
    comb = f(pnl["$1M A + $1M B"])
    print(f"  {label}: A + B separately = {parts:,.0f};  combined = {comb:,.0f};  combined - sum = {comb - parts:,.0f}  "
          f"-> {'SUBADDITIVE' if comb <= parts else 'VIOLATED'}")
print("Subadditivity check at 1% (historical):")
for label, f in [("VaR", lambda x: var_sample(x, 0.01)), ("ES", lambda x: es_sample(x, 0.01))]:
    parts = f(pnl["$1M in A"]) + f(pnl["$1M in B"])
    comb = f(pnl["$1M A + $1M B"])
    print(f"  {label}: A + B separately = {parts:,.0f};  combined = {comb:,.0f};  combined - sum = {comb - parts:,.0f}  "
          f"-> {'SUBADDITIVE' if comb <= parts else 'VIOLATED'}")

# mechanism: where does the 5% cutoff land?
default_prob_A = np.mean(A3 < -0.2)
print(f"\nProbability A defaults (loss > 20%) = {default_prob_A:.4f}; P(at least one of two defaults) = {(lossA | lossB).mean():.4f}")
print(f"5th percentile of A return = {np.percentile(A3, 5):.4f}; 5th percentile of the average of A and B = {np.percentile((A3 + B3) / 2, 5):.4f}")


# =====================================================================================
# PROBLEM 4 - Gaussian or t copula
# =====================================================================================
header("PROBLEM 4 - Gaussian or t copula")

p4 = pd.read_csv(HERE / "problem4.csv")
names4 = ["X1", "X2", "X3"]
X4 = p4[names4].to_numpy()
n4, d4 = X4.shape
VAL = 1_000_000                                    # $1,000,000 in each asset

# ---- (a) moments, and sensitivity to a single observation ----
print("First four moments:")
for j, nm in enumerate(names4):
    print_moments(nm, X4[:, j])
print("\nSensitivity: recompute after dropping the single most extreme observation of each series")
for j, nm in enumerate(names4):
    k = np.argmax(np.abs(X4[:, j] - X4[:, j].mean()))
    kept = np.delete(X4[:, j], k)
    print(f"  {nm}: dropped row {k} (value {X4[k, j]: .4f}, {abs(X4[k, j] - X4[:, j].mean()) / X4[:, j].std(ddof=1):.1f} std devs out)")
    print_moments(nm + " w/o", kept)

# ---- (b) pseudo observations (ranks scaled into (0,1)) and plots ----
Upseudo = (stats.rankdata(X4, axis=0)) / (n4 + 1)  # rank / (n+1) keeps everything strictly inside (0,1)
pairs = [(0, 1), (0, 2), (1, 2)]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, (i, j) in zip(ax, pairs):
    a.scatter(Upseudo[:, i], Upseudo[:, j], s=5, alpha=0.5)
    a.set_title(f"{names4[i]} vs {names4[j]} (ranks)")
    a.set_xlabel(names4[i])
    a.set_ylabel(names4[j])
plt.tight_layout()
plt.savefig(HERE / "problem4_ranks.png", dpi=130)
plt.close()

TAIL = 0.025
print(f"\nJoint tail days: both series in their own worst {TAIL:.1%} / best {TAIL:.1%}")
print(f"Expected under independence over {n4} days: {n4 * TAIL * TAIL:.3f}")
data_joint = {}
for (i, j) in pairs:
    lo = int(np.sum((Upseudo[:, i] <= TAIL) & (Upseudo[:, j] <= TAIL)))
    hi = int(np.sum((Upseudo[:, i] >= 1 - TAIL) & (Upseudo[:, j] >= 1 - TAIL)))
    data_joint[(i, j)] = (lo, hi)
    print(f"  {names4[i]}-{names4[j]}: both worst = {lo:2d}, both best = {hi:2d}   (Kendall tau = {stats.kendalltau(X4[:, i], X4[:, j])[0]:.4f})")

# ---- (c) fit normal and t to each margin, choose with AICc ----
print("\nMargin fits (AICc, lower is better):")
margins, U4 = [], np.zeros_like(X4)
for j, nm in enumerate(names4):
    x = X4[:, j]
    dn, kn = fit_normal(x)
    dt, kt = fit_t(x)
    a_n = aicc(np.sum(dn.logpdf(x)), kn, n4)
    a_t = aicc(np.sum(dt.logpdf(x)), kt, n4)
    best = dn if a_n < a_t else dt
    margins.append(best)
    U4[:, j] = best.cdf(x)                          # (d) transform each series to uniforms through the chosen margin
    print(f"  {nm}: Normal AICc={a_n:10.2f}   t AICc={a_t:10.2f}   -> {'Normal' if a_n < a_t else 't'}"
          f"   (t: nu={dt.kwds['df']:.2f}, scale={dt.kwds['scale']:.5f})")

# ---- (d) fit the copulas: same R from Kendall's tau for both, margins frozen ----
R4 = kendall_corr(U4)
print("\nR from Kendall's tau:")
print(pd.DataFrame(R4, names4, names4).to_string())
print("Eigenvalues of R:", np.linalg.eigvalsh(R4))

ll_g_t = gauss_copula_ll_t(U4, R4)          # per-day log likelihood, Gaussian
ll_g = ll_g_t.sum()
nu_hat, ll_t = profile_nu(lambda nu: t_copula_ll_t(U4, R4, nu).sum())
ll_tc_t = t_copula_ll_t(U4, R4, nu_hat)     # per-day log likelihood, t
k_g, k_t = 0, 1                              # margins and R are frozen; t copula has one free parameter (nu)
aicc_g, aicc_t = -2 * ll_g + 2 * k_g + (2 * k_g ** 2 + 2 * k_g) / (n4 - k_g - 1), -2 * ll_t + 2 * k_t + (2 * k_t ** 2 + 2 * k_t) / (n4 - k_t - 1)
bic_g, bic_t = k_g * np.log(n4) - 2 * ll_g, k_t * np.log(n4) - 2 * ll_t
print(f"\n{'':>10}{'logL':>12}{'nu':>10}{'AICc':>12}{'BIC':>12}")
print(f"{'Gaussian':>10}{ll_g:>12.3f}{'inf':>10}{aicc_g:>12.3f}{bic_g:>12.3f}")
print(f"{'t':>10}{ll_t:>12.3f}{nu_hat:>10.3f}{aicc_t:>12.3f}{bic_t:>12.3f}")
dBIC = 2 * (ll_t - ll_g) - np.log(n4)
print(f"logL difference (t - Gauss) = {ll_t - ll_g:.3f};  delta BIC = 2*dlogL - ln(m) = {dBIC:.3f};  delta AICc (Gauss - t) = {aicc_g - aicc_t:.3f}")

# profile of nu to see how flat it is
print("\nProfile log likelihood of the t copula at selected nu:")
for nu in (3, 4, 5, 6, 8, 10, 15, 20, 30, 50):
    print(f"  nu={nu:3d}: logL = {t_copula_ll_t(U4, R4, nu).sum():.3f}")

# ---- (e) simulate 100,000 days from each copula through the fitted margins ----
NSIM = 100_000
rng = np.random.default_rng(545)
L4 = np.linalg.cholesky(R4)
Z = rng.standard_normal((NSIM, d4)) @ L4.T           # correlated standard normals
Ug_sim = stats.norm.cdf(Z)                           # Gaussian copula uniforms
W = nu_hat / rng.chisquare(nu_hat, NSIM)             # shared mixing variable -> multivariate t
Ut_sim = stats.t.cdf(np.sqrt(W)[:, None] * Z, nu_hat)  # t copula uniforms (same Z so the comparison is clean)


def to_returns(Usim):
    """Push copula uniforms through each fitted margin's inverse CDF to get simulated returns."""
    return np.column_stack([margins[j].ppf(Usim[:, j]) for j in range(d4)])


pnl_hist = VAL * X4.sum(axis=1)                                   # historical P&L of the equal dollar portfolio
pnl_g = VAL * to_returns(Ug_sim).sum(axis=1)
pnl_t = VAL * to_returns(Ut_sim).sum(axis=1)
risk4 = {}
print(f"\nPortfolio ($1M in each of X1-X3) VaR / ES, {NSIM:,} simulated days:")
print(f"{'':>12}{'VaR 5%':>12}{'ES 5%':>12}{'VaR 1%':>12}{'ES 1%':>12}")
for nm, p in [("Gaussian", pnl_g), ("t copula", pnl_t), ("Historical", pnl_hist)]:
    risk4[nm] = [var_sample(p, 0.05), es_sample(p, 0.05), var_sample(p, 0.01), es_sample(p, 0.01)]
    print(f"{nm:>12}" + "".join(f"{v:>12,.0f}" for v in risk4[nm]))
chg = (np.array(risk4["t copula"]) / np.array(risk4["Gaussian"]) - 1) * 100
print(f"{'t vs Gauss %':>12}" + "".join(f"{v:>11.1f}%" for v in chg))

# ---- (f) implied joint tail days over 1,000 days from each simulation ----
print(f"\nImplied joint tail days per {n4} days (simulated copula uniforms, tail = {TAIL:.1%}):")
print(f"{'pair':>8}{'data lo':>9}{'data hi':>9}{'Gauss lo':>10}{'Gauss hi':>10}{'t lo':>8}{'t hi':>8}")
for (i, j) in pairs:
    row = []
    for Us in (Ug_sim, Ut_sim):
        lo = np.mean((Us[:, i] <= TAIL) & (Us[:, j] <= TAIL)) * n4
        hi = np.mean((Us[:, i] >= 1 - TAIL) & (Us[:, j] >= 1 - TAIL)) * n4
        row += [lo, hi]
    dl, dh = data_joint[(i, j)]
    print(f"{names4[i] + '-' + names4[j]:>8}{dl:>9d}{dh:>9d}{row[0]:>10.2f}{row[1]:>10.2f}{row[2]:>8.2f}{row[3]:>8.2f}")

# ---- (i) day by day contribution to ell_t - ell_Gauss ----
contrib = ll_tc_t - ll_g_t
outer = ((U4 < 0.05) | (U4 > 0.95)).any(axis=1)      # at least one series in its outer 5% (either side)
inner = ~outer                                       # no series in its outer 5% on either side
total = contrib.sum()
print(f"\nTotal ell_t - ell_Gauss = {total:.3f}")
print(f"Days with no series in its outer 5% on either side: {inner.sum()} of {n4} ({inner.mean():.1%})")
print(f"  their contribution: {contrib[inner].sum():.3f}  ({contrib[inner].sum() / total:.1%} of the total)")
print(f"Days with at least one series in its outer 5%: {outer.sum()}; contribution {contrib[outer].sum():.3f} ({contrib[outer].sum() / total:.1%})")
# split the outer days into 'joint outer' (2+ series outer on the same side) vs. single
n_out = ((U4 < 0.05) | (U4 > 0.95)).sum(axis=1)
print(f"Days with 2+ series in an outer 5%: {(n_out >= 2).sum()}; contribution {contrib[n_out >= 2].sum():.3f} ({contrib[n_out >= 2].sum() / total:.1%})")
print(f"Largest single day contribution: {contrib.max():.3f}; smallest: {contrib.min():.3f}; days where t beats Gauss: {(contrib > 0).sum()}")

# ---- (j) tail dependence for the most correlated pair ----
off = [(R4[i, j], i, j) for (i, j) in pairs]
rho_m, im, jm = max(off)
print(f"\nMost correlated pair: {names4[im]}-{names4[jm]} with rho = {rho_m:.4f}")
print(f"  t copula (nu={nu_hat:.2f}) lower tail dependence = {tail_dep_t(rho_m, nu_hat):.4f}")
print(f"  Gaussian copula tail dependence                  = 0 (limit as u->0 is 0 for any rho<1)")
for nu in (4, 6, 10, 20, 50):
    print(f"  reference: nu={nu:3d}, rho={rho_m:.3f} -> lambda = {tail_dep_t(rho_m, nu):.4f}")


# =====================================================================================
# PROBLEM 5 - Model based simulation and residual correlation
# =====================================================================================
header("PROBLEM 5 - Model based simulation and residual correlation")

p5 = pd.read_csv(HERE / "problem5.csv")
pr = p5[["MKT", "A", "B"]].to_numpy()
ret5 = pr[1:] / pr[:-1] - 1                        # arithmetic returns
rM, rA, rB = ret5[:, 0], ret5[:, 1], ret5[:, 2]
n5 = len(rM)
print(f"{n5} daily returns. Means: MKT={rM.mean():.6f}, A={rA.mean():.6f}, B={rB.mean():.6f} (we assume zero expected returns in the VaR)")

# ---- (b) OLS regressions r_i = alpha + beta * r_MKT + eps ----
Xd = np.column_stack([np.ones(n5), rM])


def ols(y):
    """OLS by least squares; returns alpha, beta, residuals and the residual std with n-2 degrees of freedom."""
    b = np.linalg.lstsq(Xd, y, rcond=None)[0]
    res = y - Xd @ b
    return b[0], b[1], res, np.sqrt(res @ res / (len(y) - 2))


aA, bA, eA, sA = ols(rA)
aB, bB, eB, sB = ols(rB)
rho_e = np.corrcoef(eA, eB)[0, 1]
print(f"\nStock A: alpha={aA: .6f}  beta={bA:.4f}  residual std={sA:.6f}  R^2={1 - eA.var() / rA.var():.4f}")
print(f"Stock B: alpha={aB: .6f}  beta={bB:.4f}  residual std={sB:.6f}  R^2={1 - eB.var() / rB.var():.4f}")
print(f"Correlation between the residuals: {rho_e:.4f}")
print(f"Market return std: {rM.std(ddof=1):.6f}")

# ---- (a) analytic look at the portfolio variance formula for the shortcut ----
cov_eps = np.cov(np.vstack([eA, eB]))                  # full residual covariance (2x2)
cov_eps_diag = np.diag(np.diag(cov_eps))               # shortcut: off diagonal set to zero
beta = np.array([bA, bB])
s2M = rM.var(ddof=1)

# ---- (c) simulation: market and errors normal, zero expected return ----
NS = 100_000
rng5 = np.random.default_rng(777)
zM = rng5.standard_normal(NS)
zE = rng5.standard_normal((NS, 2))
mkt_sim = np.sqrt(s2M) * zM                            # market ~ N(0, var of market returns), mean 0
eps_full = zE @ np.linalg.cholesky(cov_eps).T          # errors with the full residual covariance
eps_diag = zE @ np.linalg.cholesky(cov_eps_diag).T     # errors with off diagonal zeroed (same random numbers)
# alpha is set to zero because the problem tells us to assume zero expected returns
stk_full = mkt_sim[:, None] * beta[None, :] + eps_full
stk_diag = mkt_sim[:, None] * beta[None, :] + eps_diag
hold = {"P1 (long A, long B)": np.array([1e6, 1e6]), "P2 (long A, short B)": np.array([1e6, -1e6])}
print(f"\nSimulated 5% VaR ({NS:,} days), zero expected returns:")
print(f"{'':>24}{'full residual cov':>20}{'diag residual cov':>20}{'diag / full - 1':>18}")
sim_var = {}
for nm, w in hold.items():
    vf = var_sample(stk_full @ w)
    vd = var_sample(stk_diag @ w)
    sim_var[nm] = (vf, vd)
    print(f"{nm:>24}{vf:>20,.0f}{vd:>20,.0f}{(vd / vf - 1) * 100:>17.1f}%")

# Analytic variance (Week 2 formula): w' Sigma w with Sigma = s2M * beta beta' + Sigma_eps
print("\nPortfolio standard deviations from the Week 2 formula w' Sigma w  (Sigma = s2M*beta*beta' + Sigma_eps):")
for nm, w in hold.items():
    sf = np.sqrt(w @ (s2M * np.outer(beta, beta) + cov_eps) @ w)
    sd_ = np.sqrt(w @ (s2M * np.outer(beta, beta) + cov_eps_diag) @ w)
    print(f"  {nm}: full={sf:,.0f}  diag={sd_:,.0f}  -> normal VaR full={-z05 * sf:,.0f}, diag={-z05 * sd_:,.0f}")
    # split of the variance into market part and residual part
    mkt_part = s2M * (w @ beta) ** 2
    eps_part = w @ cov_eps @ w
    print(f"     variance split: market part={mkt_part:,.0f}, residual part (full)={eps_part:,.0f}, "
          f"residual part (diag)={w @ cov_eps_diag @ w:,.0f}; cross term lost by the shortcut = {w @ (cov_eps - cov_eps_diag) @ w:,.0f}")

# ---- (d) delta normal VaR from the sample covariance of the two stock returns ----
S = np.cov(np.vstack([rA, rB]))
print("\nSample covariance of (r_A, r_B):")
print(S)
print("Model implied covariance s2M*beta*beta' + Sigma_eps:")
print(s2M * np.outer(beta, beta) + cov_eps)
print("\nDelta normal VaR from the sample covariance (zero mean):")
dn_var = {}
for nm, w in hold.items():
    dn_var[nm] = -z05 * np.sqrt(w @ S @ w)
    print(f"  {nm}: ${dn_var[nm]:,.0f}   | simulated full residual cov VaR: ${sim_var[nm][0]:,.0f}   | difference: {(sim_var[nm][0] / dn_var[nm] - 1) * 100:.2f}%")
print(f"\nCorrelation between the two stock returns: {np.corrcoef(rA, rB)[0, 1]:.4f}")

print("\nDONE")
