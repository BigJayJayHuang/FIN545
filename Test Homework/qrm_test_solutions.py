import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats, integrate, optimize
from scipy.linalg import solve_triangular
from scipy.special import gammaln


#Missing Data Covariance
def missing_cov(x, skip_missing=True, corr=False):
    x = np.asarray(x, dtype=float)
    n_cols = x.shape[1]
    stat = np.corrcoef if corr else np.cov #pick correlation or covariance

    if skip_missing:
        complete_rows = ~np.isnan(x).any(axis=1) #rows with no NaN in any column
        return stat(x[complete_rows], rowvar=False)

    out = np.empty((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(i + 1):
            both_present = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j]) #rows where both columns have data
            pair = x[both_present][:, [i, j]]
            out[i, j] = out[j, i] = stat(pair, rowvar=False)[0, 1] #fill both sides of the symmetric matrix
    return out


#Exponential Weights
def ew_weights(n_rows, lam):
    w = (1 - lam) * lam ** np.arange(n_rows - 1, -1, -1) #weight = (1-lam) * lam^age, newest row counts most
    return w / w.sum() #rescale so weights sum to 1


#Exponentially Weighted Covariance
def ew_covar(x, lam):
    x = np.asarray(x, dtype=float)
    w = ew_weights(len(x), lam)
    centered = x - w @ x #subtract the weighted mean
    scaled = np.sqrt(w)[:, None] * centered #times sqrt(weight), so scaled' @ scaled is the weighted covariance
    return scaled.T @ scaled


#Covariance To Correlation
def cov_to_corr(cov):
    inv_sd = 1 / np.sqrt(np.diag(cov)) #1 / standard deviation
    return cov * np.outer(inv_sd, inv_sd) #cov_ij / (sd_i * sd_j)


#Mixed Lambda Covariance
def ew_cov_mixed(x, lam_var, lam_corr):
    sd = np.sqrt(np.diag(ew_covar(x, lam_var))) #standard deviations from the lam_var covariance
    corr = cov_to_corr(ew_covar(x, lam_corr)) #correlation from the lam_corr covariance
    return corr * np.outer(sd, sd) #rebuild covariance = corr * sd_i * sd_j


#Correlation Conversion Helper
def _to_corr_if_needed(a):
    a = np.array(a, dtype=float)
    if np.allclose(np.diag(a), 1.0, rtol=1.5e-8, atol=0): #already a correlation matrix (1s on the diagonal)
        return a, None
    sd = np.sqrt(np.diag(a))
    return a / np.outer(sd, sd), sd #covariance to correlation, keep sd to convert back later


#Near PSD
def near_psd(a, epsilon=0.0):
    out, sd = _to_corr_if_needed(a) #work on the correlation scale

    vals, vecs = np.linalg.eigh(out) #eigenvalues and eigenvectors
    vals = np.maximum(vals, epsilon) #lift negative eigenvalues up to epsilon
    t = 1.0 / ((vecs * vecs) @ vals) #scaling that puts 1s back on the diagonal
    B = np.diag(np.sqrt(t)) @ vecs @ np.diag(np.sqrt(vals)) #the new root
    out = B @ B.T #B @ B.T is always PSD

    return out if sd is None else out * np.outer(sd, sd) #back to covariance if we started with one


#Clip Negative Eigenvalues
def _clip_negative_eigenvalues(a):
    a = (a + a.T) / 2 #force exact symmetry
    vals, vecs = np.linalg.eigh(a)
    return (vecs * np.maximum(vals, 0)) @ vecs.T #rebuild with negative eigenvalues set to 0


#Higham Nearest PSD
def higham_nearest_psd(a, epsilon=1e-9, max_iter=100, tol=1e-9):
    yk, sd = _to_corr_if_needed(a)
    y0 = yk.copy() #original matrix, to measure distance
    delta_s = np.zeros_like(yk) #Dykstra correction term
    last_norm = np.inf

    for _ in range(max_iter):
        rk = yk - delta_s #remove the previous correction
        xk = _clip_negative_eigenvalues(rk) #project onto PSD matrices
        delta_s = xk - rk #update the correction
        yk = xk.copy()
        np.fill_diagonal(yk, 1.0) #project onto unit diagonal

        norm = np.sum((yk - y0) ** 2) #distance from the original matrix
        min_eig = np.linalg.eigvalsh((yk + yk.T) / 2).min() #smallest eigenvalue
        if abs(norm - last_norm) < tol and min_eig > -epsilon: #stop when change is tiny and matrix is PSD
            break
        last_norm = norm

    return yk if sd is None else yk * np.outer(sd, sd)


#Cholesky Root For PSD Matrices
def chol_psd(a, epsilon=-1e-8):
    a = np.asarray(a, dtype=float)
    n = len(a)
    root = np.zeros((n, n)) #lower-triangular root, built column by column

    for j in range(n):
        s = root[j, :j] @ root[j, :j] #squares already used in row j
        pivot_sq = a[j, j] - s #what is left for the diagonal
        if epsilon <= pivot_sq <= 0: #tiny negatives are rounding error, treat as 0
            pivot_sq = 0.0
        if pivot_sq < 0: #a real negative means not PSD
            raise ValueError("matrix is not positive semi-definite")
        root[j, j] = np.sqrt(pivot_sq)

        if root[j, j] != 0.0: #zero pivot means the rest of the column is 0
            for i in range(j + 1, n):
                s = root[i, :j] @ root[j, :j]
                root[i, j] = (a[i, j] - s) / root[j, j] #standard Cholesky formula
    return root


#Root Selection
def _root_of(cov, fix_method):
    try:
        return np.linalg.cholesky(cov) #fastest, needs positive definite
    except np.linalg.LinAlgError:
        pass
    try:
        return chol_psd(cov) #also handles PSD with zero eigenvalues
    except ValueError:
        return chol_psd(fix_method(cov)) #last resort: repair the matrix first


#Monte Carlo Simulation
def simulate_normal(n_sim, cov, mean=None, seed=1234, fix_method=near_psd):
    cov = np.asarray(cov, dtype=float)
    n = len(cov)
    root = _root_of(cov, fix_method) #cov = root @ root.T

    rng = np.random.default_rng(seed) #seeded so results repeat
    z = rng.standard_normal((n, n_sim)) #independent standard normals
    draws = (root @ z).T #correlate them, one row per simulation
    if mean is not None:
        draws = draws + np.asarray(mean) #add the mean
    return draws


#PCA Simulation
def simulate_pca(cov, n_sim, pct_explained=1.0, mean=None, seed=1234):
    cov = np.asarray(cov, dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = vals[::-1], vecs[:, ::-1] #largest eigenvalue first

    total = vals.sum()
    keep = np.where(vals >= 1e-8)[0] #drop zero eigenvalues
    if pct_explained < 1:
        cum = np.cumsum(vals / total) #cumulative share of variance explained
        n_needed = np.searchsorted(cum, pct_explained) + 1 #fewest components that reach pct_explained
        keep = keep[:n_needed]
    vals, vecs = vals[keep], vecs[:, keep]

    B = vecs * np.sqrt(vals) #root built from the kept components
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((len(vals), n_sim)) #one normal per kept component
    draws = (B @ r).T
    if mean is not None:
        draws = draws + np.asarray(mean)
    return draws


#Return Calculation
def return_calculate(prices, method="DISCRETE", date_column="Date"):
    stocks = [c for c in prices.columns if c != date_column] #every column except the date
    p = prices[stocks].to_numpy(dtype=float)
    ratio = p[1:] / p[:-1] #today's price / yesterday's price

    if method.upper() == "DISCRETE":
        rets = ratio - 1.0 #arithmetic return
    elif method.upper() == "LOG":
        rets = np.log(ratio) #log return
    else:
        raise ValueError('method must be "DISCRETE" or "LOG"')

    out = pd.DataFrame(rets, columns=stocks)
    out.insert(0, date_column, prices[date_column].iloc[1:].to_numpy()) #returns start on the second date
    return out


#Fitted Model Container
@dataclass
class FittedModel:
    dist: object #scipy distribution of the errors
    errors: np.ndarray #data minus the fitted mean
    u: np.ndarray #each error as a percentile (0 to 1)
    eval: Callable #percentile to value (inverse CDF)
    n_params: int #number of fitted parameters, used in AICc
    params: Optional[dict] = None #fitted numbers by name
    beta: Optional[np.ndarray] = None #regression coefficients (regression fits only)


#AICc
def aicc(loglik, k, n):
    return -2 * loglik + 2 * k + 2 * k * (k + 1) / (n - k - 1) #lower is better, penalizes extra parameters


#AICc Of Fitted Model
def aicc_of_fit(model, data):
    ll = np.sum(model.dist.logpdf(data)) #total log-likelihood of the data
    return aicc(ll, model.n_params, len(data))


#Normal Fit
def fit_normal(x):
    x = np.asarray(x, dtype=float)
    mu, sigma = x.mean(), x.std(ddof=1) #Normal MLE is the sample mean and std (n-1)
    dist = stats.norm(mu, sigma)
    return FittedModel(dist, x - mu, dist.cdf(x), dist.ppf, n_params=2,
                       params=dict(mu=mu, sigma=sigma))


#Excess Kurtosis
def _excess_kurtosis(x):
    d = x - x.mean()
    return np.mean(d ** 4) / np.mean(d ** 2) ** 2 - 3 #fourth moment / variance squared, minus 3


#T Starting Values
def _t_start(x):
    k = _excess_kurtosis(x)
    nu = 6.0 / k + 4 if k > 0.05 else 50.0 #t has excess kurtosis 6/(nu-4), solved for nu
    nu = min(max(nu, 2.5), 100.0) #keep the guess in a sensible range
    s = np.sqrt(x.var(ddof=1) * (nu - 2) / nu) #scale that matches the sample variance
    return nu, s


#Generalized T Fit
def fit_general_t(x):
    x = np.asarray(x, dtype=float)
    nu0, s0 = _t_start(x)

    def neg_loglik(p):
        mu, log_s, log_nu2 = p #search on logs so s > 0 and nu > 2
        return -np.sum(stats.t.logpdf(x, df=2 + np.exp(log_nu2), loc=mu, scale=np.exp(log_s))) #negative log-likelihood, minimizing it is MLE

    res = optimize.minimize(neg_loglik, [x.mean(), np.log(s0), np.log(nu0 - 2)], #Nelder-Mead needs no derivatives
                            method="Nelder-Mead",
                            options=dict(xatol=1e-10, fatol=1e-12, maxiter=5000))
    mu, s, nu = res.x[0], np.exp(res.x[1]), 2 + np.exp(res.x[2]) #convert back from logs

    dist = stats.t(df=nu, loc=mu, scale=s)
    return FittedModel(dist, x - mu, dist.cdf(x), dist.ppf, n_params=3,
                       params=dict(mu=mu, sigma=s, nu=nu))


#Regression With T Errors
def fit_regression_t(y, x):
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)]) #add an intercept column

    b0 = np.linalg.lstsq(X, y, rcond=None)[0] #OLS as the starting point
    nu0, s0 = _t_start(y - X @ b0) #starting nu and scale from the OLS residuals
    k = X.shape[1] #number of coefficients

    def neg_loglik(p):
        beta, log_s, log_nu2 = p[:k], p[k], p[k + 1] #first k values are the coefficients
        return -np.sum(stats.t.logpdf(y - X @ beta, df=2 + np.exp(log_nu2), scale=np.exp(log_s)))

    start = np.concatenate([b0, [np.log(s0), np.log(nu0 - 2)]])
    res = optimize.minimize(neg_loglik, start, method="BFGS", #fast first pass
                            options=dict(gtol=1e-9, maxiter=2000))
    res = optimize.minimize(neg_loglik, res.x, method="Nelder-Mead", #polish, since BFGS can stop early
                            options=dict(xatol=1e-10, fatol=1e-13, maxiter=20000, maxfev=20000))
    beta, s, nu = res.x[:k], np.exp(res.x[k]), 2 + np.exp(res.x[k + 1])

    dist = stats.t(df=nu, scale=s) #error distribution, centered at 0
    errors = y - X @ beta

    def eval_model(x_new, u): #predicted y at new x for percentile u
        X_new = np.column_stack([np.ones(len(x_new)), np.asarray(x_new, dtype=float)])
        return X_new @ beta + dist.ppf(u)

    return FittedModel(dist, errors, dist.cdf(errors), eval_model, n_params=k + 2, #k coefficients + scale + nu
                       params=dict(mu=0.0, sigma=s, nu=nu), beta=beta)


#Sample Moments
def _moments(x):
    m = x.mean()
    d = x - m
    m2 = np.mean(d ** 2)
    return m, x.var(ddof=1), np.mean(d ** 3) / m2 ** 1.5, np.mean(d ** 4) / m2 ** 2 - 3 #mean, variance, skewness, excess kurtosis


#NIG Method Of Moments
def fit_nig_moments(x):
    x = np.asarray(x, dtype=float)
    m, v, skew, kurt = _moments(x)
    if kurt <= 0:
        raise ValueError("NIG needs positive excess kurtosis")
    t = skew ** 2 / kurt #ratio that pins down beta/alpha
    if t >= 3 / 5: #a NIG cannot reach beyond this
        raise ValueError("sample is outside the range a NIG can reach")

    rho2 = t / (3 - 4 * t) #rho = beta / alpha
    rho = np.sign(skew) * np.sqrt(rho2)
    dg = 3 * (1 + 4 * rho2) / kurt #delta * gamma
    alpha = np.sqrt(dg / (v * (1 - rho2) ** 2)) #from the variance formula
    beta = rho * alpha
    gamma = alpha * np.sqrt(1 - rho2) #gamma = sqrt(alpha^2 - beta^2)
    delta = dg / gamma
    mu = m - delta * beta / gamma #mu from the mean
    return mu, alpha, beta, delta


#NIG Maximum Likelihood
def fit_nig_mle(x):
    a, b, loc, scale = stats.norminvgauss.fit(np.asarray(x, dtype=float)) #scipy's a = alpha*delta, b = beta*delta
    return loc, a / scale, b / scale, scale #convert to mu, alpha, beta, delta


#VaR From Distribution
def var_from_dist(dist, alpha=0.05):
    return -dist.ppf(alpha) #loss at the 5% quantile, as a positive number


#ES From Distribution
def es_from_dist(dist, alpha=0.05):
    var = var_from_dist(dist, alpha)
    lower = dist.ppf(1e-12) #far-left start for the integral
    area, _ = integrate.quad(lambda z: z * dist.pdf(z), lower, -var) #integral of x * pdf from far left up to -VaR
    return -area / alpha #average loss beyond VaR


#Sample Cutoff
def _cutoff(x, alpha):
    x = np.sort(x) #worst outcomes first
    n = len(x)
    hi = int(np.ceil(n * alpha))
    lo = int(np.floor(n * alpha))
    return x, 0.5 * (x[hi - 1] + x[lo - 1]) #average the two values around position n * alpha


#VaR From Sample
def var_from_sample(returns, alpha=0.05):
    _, cutoff = _cutoff(np.asarray(returns, dtype=float), alpha)
    return -cutoff


#ES From Sample
def es_from_sample(returns, alpha=0.05):
    x, cutoff = _cutoff(np.asarray(returns, dtype=float), alpha)
    return -x[x <= cutoff].mean() #average of everything at or below the cutoff


#Portfolio Risk Table
def risk_table(pnl, current_values, names, alpha=0.05):
    rows = []
    for j, name in enumerate(list(names) + ["Total"]): #each stock, then the whole portfolio
        p = pnl.sum(axis=1) if name == "Total" else pnl[:, j] #Total = sum of stock P&L in each simulation
        value = np.sum(current_values) if name == "Total" else current_values[j]
        var, es = var_from_sample(p, alpha), es_from_sample(p, alpha)
        rows.append([name, var, es, var / value, es / value]) #percent columns = dollars / current value
    return pd.DataFrame(rows, columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])


#Gaussian Copula VaR And ES
def copula_var_es(returns, portfolio, n_sim=100_000, seed=9):
    stocks = list(portfolio["Stock"])
    fitters = {"Normal": fit_normal, "T": fit_general_t} #which distribution to fit for each stock
    models = [fitters[d](returns[s].to_numpy()) #fit each stock on its own
              for s, d in zip(stocks, portfolio["Distribution"])]

    U = np.column_stack([m.u for m in models]) #percentile of every observation
    spearman = pd.DataFrame(U).corr(method="spearman").to_numpy() #correlation of the percentiles

    z = simulate_pca(spearman, n_sim, seed=seed) #correlated normals
    u_sim = stats.norm.cdf(z) #normals to percentiles (this is the copula)
    sim_returns = np.column_stack([m.eval(u_sim[:, j]) for j, m in enumerate(models)]) #percentiles to returns through each stock's own distribution

    values = (portfolio["Holding"] * portfolio["Starting Price"]).to_numpy() #dollar value of each position
    return risk_table(sim_returns * values, values, stocks) #P&L = return * value


#Risk Parity
def risk_parity(cov, risk_budget=None):
    cov = np.asarray(cov, dtype=float)
    n = len(cov)
    b = np.ones(n) if risk_budget is None else np.asarray(risk_budget, dtype=float) #risk budgets, equal by default

    objective = lambda w: 0.5 * w @ cov @ w - b @ np.log(w) #its minimum makes each risk contribution proportional to its budget
    gradient = lambda w: cov @ w - b / w #derivative, speeds up the solver

    res = optimize.minimize(objective, np.full(n, 1.0 / n), jac=gradient,
                            method="L-BFGS-B", bounds=[(1e-12, None)] * n, #weights must stay positive because of the log
                            options=dict(ftol=1e-15, gtol=1e-12, maxiter=5000))
    return res.x / res.x.sum() #scale so weights sum to 1


#Maximum Sharpe Ratio
def max_sharpe(cov, mean, rf, bounds=None):
    cov, mean = np.asarray(cov, dtype=float), np.asarray(mean, dtype=float)
    n = len(cov)
    bnds = [(0.0, 1.0)] * n if bounds is None else [tuple(b) for b in bounds] #default: long-only, 0 to 100% per asset

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ cov @ w, 1e-16)) #floor avoids dividing by 0
        return -(w @ mean - rf) / vol #negative Sharpe, so minimizing it maximizes Sharpe

    res = optimize.minimize(neg_sharpe, np.full(n, 1.0 / n), method="SLSQP", bounds=bnds, #SLSQP handles bounds and constraints
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], #weights sum to 1
                            options=dict(ftol=1e-14, maxiter=1000))
    return res.x


#Ex Post Attribution
def expost_attribution(weights, stock_returns, factor_returns, betas):
    w = np.asarray(weights, dtype=float).copy() #starting weights
    S = stock_returns.to_numpy(dtype=float)
    F = factor_returns.to_numpy(dtype=float)
    betas = np.asarray(betas, dtype=float)
    T, n_factors = F.shape

    factor_w = np.empty((T, n_factors)) #factor weights for each day
    port_ret = np.empty(T)
    for t in range(T):
        factor_w[t] = betas.T @ w #stock weights to factor weights
        w = w * (1 + S[t]) #weights drift with returns
        gross = w.sum() #portfolio growth factor
        w = w / gross #renormalize to sum to 1
        port_ret[t] = gross - 1 #portfolio return for the day
    alpha = port_ret - np.sum(factor_w * F, axis=1) #return the factors do not explain

    total = np.exp(np.log1p(port_ret).sum()) - 1 #compounded total return
    k = np.log(1 + total) / total #Carino scaling constant
    carino = np.log1p(port_ret) / port_ret / k #daily weights so contributions add up to the total

    names = list(factor_returns.columns) + ["Alpha", "Portfolio"]
    series = np.column_stack([F, alpha, port_ret])
    total_return = np.exp(np.log1p(series).sum(axis=0)) - 1 #compounded return of each series

    contrib = np.column_stack([F * factor_w, alpha]) #daily contribution of each factor and alpha
    return_attr = np.append((contrib * carino[:, None]).sum(axis=0), total) #Carino-weighted contributions, last value is the total

    X = np.column_stack([np.ones(T), port_ret])
    slopes = np.linalg.lstsq(X, contrib, rcond=None)[0][1] #regress each contribution on portfolio return, slope = its share of risk
    vol_attr = np.append(slopes * port_ret.std(ddof=1), port_ret.std(ddof=1)) #slope * portfolio volatility

    out = pd.DataFrame([total_return, return_attr, vol_attr], columns=names)
    out.insert(0, "Value", ["TotalReturn", "Return Attribution", "Vol Attribution"])
    return out


#Generalized Black Scholes Merton
def gbsm(call, S, K, T, rf, b, vol):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (b + vol ** 2 / 2) * T) / (vol * sqrt_t) #b is cost of carry (risk-free rate minus dividend yield)
    d2 = d1 - vol * sqrt_t
    carry = np.exp((b - rf) * T) #e^((b - r) * T)
    disc = np.exp(-rf * T) #discount factor

    if call:
        value = S * carry * stats.norm.cdf(d1) - K * disc * stats.norm.cdf(d2) #call price
        delta = carry * stats.norm.cdf(d1)
        theta = (-S * carry * stats.norm.pdf(d1) * vol / (2 * sqrt_t)
                 - (b - rf) * S * carry * stats.norm.cdf(d1)
                 - rf * K * disc * stats.norm.cdf(d2))
        rho = T * K * disc * stats.norm.cdf(d2)
    else:
        value = K * disc * stats.norm.cdf(-d2) - S * carry * stats.norm.cdf(-d1) #put price
        delta = carry * (stats.norm.cdf(d1) - 1)
        theta = (-S * carry * stats.norm.pdf(d1) * vol / (2 * sqrt_t)
                 + (b - rf) * S * carry * stats.norm.cdf(-d1)
                 + rf * K * disc * stats.norm.cdf(-d2))
        rho = -T * K * disc * stats.norm.cdf(-d2)

    gamma = stats.norm.pdf(d1) * carry / (S * vol * sqrt_t) #same for calls and puts
    vega = S * carry * stats.norm.pdf(d1) * sqrt_t
    return dict(value=value, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


#Binomial Tree Constants
def _tree_constants(T, r, b, vol, N):
    dt = T / N #length of one step
    u = np.exp(vol * np.sqrt(dt)) #up factor
    d = 1 / u #down factor, so the tree recombines
    pu = (np.exp(b * dt) - d) / (u - d) #risk-neutral probability of going up
    return u, d, pu, 1 - pu, np.exp(-r * dt) #also 1 - pu and the one-step discount factor


#Binomial Tree Roll Back
def _roll_back(values, S0, call, K, u, d, pu, pd, df, from_step): #S0 is an array, so many starting prices run at once
    z = 1 if call else -1 #payoff sign, +1 call and -1 put
    for j in range(from_step - 1, -1, -1): #step backward through the tree
        i = np.arange(j + 1) #number of up moves at each node
        stock = S0[:, None] * u ** i * d ** (j - i) #stock price at each node
        hold = df * (pu * values[:, 1:j + 2] + pd * values[:, :j + 1]) #value of waiting (discounted expectation)
        values = np.maximum(np.maximum(0.0, z * (stock - K)), hold) #American: larger of exercising now and waiting
    return values[:, 0] #value at the root


#Binomial Tree Without Dividends
def _plain_tree(S0, call, K, T, r, b, vol, N):
    u, d, pu, pd, df = _tree_constants(T, r, b, vol, N)
    z = 1 if call else -1
    i = np.arange(N + 1)
    final_stock = S0[:, None] * u ** i * d ** (N - i) #stock prices at expiry
    values = np.maximum(0.0, z * (final_stock - K)) #payoff at expiry
    return _roll_back(values, S0, call, K, u, d, pu, pd, df, N)


#American Option Price
def bt_american(call, S, K, T, r, b, vol, N):
    return _plain_tree(np.array([float(S)]), call, K, T, r, b, vol, N)[0]


#Binomial Tree With Discrete Dividends
def _dividend_tree(S0, call, K, T, r, amts, steps, vol, N):
    first = steps[0] if len(steps) else None #step of the next dividend
    if first is None or first > N: #no dividend left, plain tree (b = r)
        return _plain_tree(S0, call, K, T, r, r, vol, N)

    u, d, pu, pd, df = _tree_constants(T, r, r, vol, N)
    z = 1 if call else -1
    dt = T / N

    i = np.arange(first + 1)
    stock = S0[:, None] * u ** i * d ** (first - i) #stock prices just before the dividend
    n_starts = len(S0)
    after_drop = _dividend_tree((stock - amts[0]).ravel(), call, K, T - first * dt, r, #stock drops by the dividend, price the rest from there
                                amts[1:], [s - first for s in steps[1:]], vol, N - first)
    hold = after_drop.reshape(n_starts, first + 1) #one row per starting price
    values = np.maximum(np.maximum(0.0, z * (stock - K)), hold) #exercise before the dividend or hold through it

    return _roll_back(values, S0, call, K, u, d, pu, pd, df, first) #continue back to today


#American Option Price With Dividends
def bt_american_dividends(call, S, K, T, r, div_amts, div_steps, vol, N):
    return _dividend_tree(np.array([float(S)]), call, K, T, r,
                          list(div_amts), list(div_steps), vol, N)[0]


#Finite Difference Step
_STEP = np.cbrt(np.finfo(float).eps) #cube root of machine epsilon, standard finite-difference step


#Central Difference
def _central_diff(f, x):
    h = _STEP * max(abs(x), 1.0) #step scaled to the size of x
    return (f(x + h) - f(x - h)) / (2 * h) #(f(x+h) - f(x-h)) / 2h


#American Option Greeks
def american_greeks(call, S, K, T, r, b, vol, N=500):
    price = lambda S_=S, T_=T, r_=r, vol_=vol: bt_american(call, S_, K, T_, r_, b, vol_, N) #reprice, changing only the inputs we pass in

    value = price()
    delta = _central_diff(lambda s: price(S_=s), S) #bump the stock price
    theta = _central_diff(lambda t: price(T_=t), T) #bump time to maturity (dV/dT)
    rho = _central_diff(lambda r_: price(r_=r_), r) #bump the interest rate, b stays fixed
    vega = _central_diff(lambda v: price(vol_=v), vol) #bump volatility

    bump = 1.5 #bigger bump for gamma, tree prices are jagged
    gamma = (price(S_=S + bump) + price(S_=S - bump) - 2 * value) / bump ** 2 #second difference
    return dict(value=value, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


#Fix Correlation Matrix
def fix_correlation(R, tol=1e-8, ridge=1e-8):
    if np.linalg.eigvalsh((R + R.T) / 2).min() < -tol: #not PSD, repair with Higham
        R = higham_nearest_psd(R)
    R = (R + R.T) / 2
    if np.linalg.eigvalsh(R).min() < ridge: #too close to singular for Cholesky
        R = (R + ridge * np.eye(len(R))) / (1 + ridge) #add a small ridge, keeps 1s on the diagonal
    return R


#Kendall Correlation
def kendall_correlation(X):
    n = X.shape[1]
    tau = np.eye(n) #Kendall's tau matrix
    for i in range(n):
        for j in range(i + 1, n):
            tau[i, j] = tau[j, i] = stats.kendalltau(X[:, i], X[:, j])[0]
    R = np.sin(np.pi * tau / 2) #turns tau into a correlation for t and normal models
    np.fill_diagonal(R, 1.0)
    return fix_correlation(R)


#Multivariate T Log Density
def mvt_logpdf(X, mu, S, nu):
    d = X.shape[1] #number of dimensions
    L = np.linalg.cholesky(S) #S = L @ L.T
    z = solve_triangular(L, (X - mu).T, lower=True) #solves L z = (x - mu)
    q = np.sum(z ** 2, axis=0) #squared Mahalanobis distance
    log_det = 2 * np.sum(np.log(np.diag(L))) #log determinant of S
    return (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi) #multivariate t log-density
            - 0.5 * log_det - 0.5 * (nu + d) * np.log1p(q / nu))


#Multivariate Normal Log Density
def mvn_logpdf(X, R):
    d = X.shape[1]
    L = np.linalg.cholesky(R)
    z = solve_triangular(L, X.T, lower=True)
    q = np.sum(z ** 2, axis=0)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + q) #multivariate normal log-density


#Multivariate T Scale Matrix
def mvt_scale(sd, R, nu):
    d = sd * np.sqrt((nu - 2) / nu) #scale = variance * (nu-2)/nu, since variance = nu/(nu-2) * scale
    return np.outer(d, d) * R


#Profile Likelihood For Nu
def profile_nu(loglik, lo=0.01, hi=0.49, n=200):
    thetas = np.linspace(lo, hi, n) #search theta = 1/nu, keeps nu positive and finite
    lls = np.array([loglik(1 / th) for th in thetas]) #log-likelihood at each theta
    i = int(np.argmax(lls))

    fine = np.linspace(thetas[max(i - 1, 0)], thetas[min(i + 1, n - 1)], n) #zoom in around the best theta
    fine_lls = np.array([loglik(1 / th) for th in fine])
    j = int(np.argmax(fine_lls))
    return 1 / fine[j], fine_lls[j] #best nu and its log-likelihood


#Multivariate T Fit
def fit_multivariate_t(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    R = kendall_correlation(X) #correlation via Kendall's tau
    nu, ll = profile_nu(lambda nu_: np.sum(mvt_logpdf(X, mu, mvt_scale(sd, R, nu_), nu_))) #choose nu by maximum likelihood
    return mu, mvt_scale(sd, R, nu), nu, ll


#Gaussian Copula Log Likelihood
def gaussian_copula_ll(U, R):
    z = stats.norm.ppf(U) #percentiles to standard normals
    return np.sum(mvn_logpdf(z, R)) - np.sum(stats.norm.logpdf(z)) #joint normal density divided by the marginal densities


#T Copula Log Likelihood
def t_copula_ll(U, R, nu):
    t = stats.t.ppf(U, nu) #percentiles to t values
    return np.sum(mvt_logpdf(t, np.zeros(U.shape[1]), R, nu)) - np.sum(stats.t.logpdf(t, nu)) #joint t density divided by the marginal t densities


#Gaussian Copula Fit
def fit_gaussian_copula(U):
    R = kendall_correlation(U)
    return R, gaussian_copula_ll(U, R)


#T Copula Fit
def fit_t_copula(U):
    R = kendall_correlation(U)
    nu, ll = profile_nu(lambda nu_: t_copula_ll(U, R, nu_))
    return R, nu, ll


#Copula AICc
def copula_aicc(ll, k, m):
    return -2 * ll + 2 * k + (2 * k ** 2 + 2 * k) / (m - k - 1) #k = number of parameters, m = observations


#Copula BIC
def copula_bic(ll, k, m):
    return k * np.log(m) - 2 * ll


#Tail Dependence
def tail_dependence_t(rho, nu):
    return 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1) #chance both fall in the far left tail together


#T Copula Simulation
def simulate_t_copula(R, nu, n_sim, seed=1234):
    z = simulate_pca(R, n_sim, seed=seed) #correlated normals
    rng = np.random.default_rng(seed + 1) #different seed from z
    w = nu / rng.chisquare(nu, n_sim) #shared chi-square mixture turns normals into multivariate t
    return stats.t.cdf(np.sqrt(w)[:, None] * z, nu) #t values to percentiles


#Test Runner Settings
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FIN545_DATA", HERE / "data")) 
OUT = HERE / "my_output"

TOLERANCES = { #(relative, absolute) tolerance for each kind of test
    "exact": (1e-8, 1e-10),
    "optimizer": (1e-5, 1e-6),
    "sim": (0.10, 1e-6),
    "sim100k": (0.05, 1e-6),
}
SIM_COV_TOLERANCE = 0.03 #simulated covariances may differ by up to 3%


#Test Runner Helpers
def read(name):
    return pd.read_csv(DATA / name, encoding="utf-8-sig") #utf-8-sig drops a hidden BOM if present


def matrix(name):
    return read(name).to_numpy()


def frame(a):
    a = np.asarray(a)
    return pd.DataFrame(a, columns=[f"x{i + 1}" for i in range(a.shape[1])]) #name columns x1, x2, ... like the expected files


def one_col(name):
    return read(name).iloc[:, 0].to_numpy()


def sim_cov(draws):
    return frame(np.cov(draws, rowvar=False)) #covariance of the simulated draws


TESTS = [] #every registered test


def test(test_id, description, expected, tol="exact"): #decorator that registers a test with its expected file and tolerance
    def register(fn):
        TESTS.append((test_id, description, fn, expected, tol))
        return fn
    return register


#Test 1 Missing Data
@test("1.1", "Covariance, skip missing rows", "testout_1.1.csv")
def t1_1(): return frame(missing_cov(matrix("test1.csv"), skip_missing=True))

@test("1.2", "Correlation, skip missing rows", "testout_1.2.csv")
def t1_2(): return frame(missing_cov(matrix("test1.csv"), skip_missing=True, corr=True))

@test("1.3", "Covariance, pairwise", "testout_1.3.csv")
def t1_3(): return frame(missing_cov(matrix("test1.csv"), skip_missing=False))

@test("1.4", "Correlation, pairwise", "testout_1.4.csv")
def t1_4(): return frame(missing_cov(matrix("test1.csv"), skip_missing=False, corr=True))


#Test 2 Exponentially Weighted Covariance
@test("2.1", "EW covariance, lambda=0.97", "testout_2.1.csv")
def t2_1(): return frame(ew_covar(matrix("test2.csv"), 0.97))

@test("2.2", "EW correlation, lambda=0.94", "testout_2.2.csv")
def t2_2(): return frame(cov_to_corr(ew_covar(matrix("test2.csv"), 0.94)))

@test("2.3", "EW variance (0.97) with EW correlation (0.94)", "testout_2.3.csv")
def t2_3(): return frame(ew_cov_mixed(matrix("test2.csv"), 0.97, 0.94))


#Test 3 Nearest PSD
@test("3.1", "near_psd covariance", "testout_3.1.csv")
def t3_1(): return frame(near_psd(matrix("testout_1.3.csv"))) #input is the Test 1.3 output

@test("3.2", "near_psd correlation", "testout_3.2.csv")
def t3_2(): return frame(near_psd(matrix("testout_1.4.csv")))

@test("3.3", "Higham covariance", "testout_3.3.csv")
def t3_3(): return frame(higham_nearest_psd(matrix("testout_1.3.csv")))

@test("3.4", "Higham correlation", "testout_3.4.csv")
def t3_4(): return frame(higham_nearest_psd(matrix("testout_1.4.csv")))


#Test 4 Cholesky
@test("4.1", "chol_psd", "testout_4.1.csv")
def t4_1(): return frame(chol_psd(matrix("testout_3.1.csv"))) #input is the Test 3.1 output


#Test 5 Simulation
@test("5.1", "Normal simulation, PD input", "testout_5.1.csv", "sim_cov")
def t5_1(): return sim_cov(simulate_normal(100_000, matrix("test5_1.csv"))) #100,000 simulations, compare the covariance to the expected one

@test("5.2", "Normal simulation, PSD input", "testout_5.2.csv", "sim_cov")
def t5_2(): return sim_cov(simulate_normal(100_000, matrix("test5_2.csv")))

@test("5.3", "Normal simulation, non-PSD, near_psd fix", "testout_5.3.csv", "sim_cov")
def t5_3(): return sim_cov(simulate_normal(100_000, matrix("test5_3.csv"), fix_method=near_psd)) #test5_3.csv is the non-PSD matrix

@test("5.4", "Normal simulation, non-PSD, Higham fix", "testout_5.4.csv", "sim_cov")
def t5_4(): return sim_cov(simulate_normal(100_000, matrix("test5_3.csv"), fix_method=higham_nearest_psd))

@test("5.5", "PCA simulation, 99% explained", "testout_5.5.csv", "sim_cov")
def t5_5(): return sim_cov(simulate_pca(matrix("test5_2.csv"), 100_000, pct_explained=0.99)) #keep components explaining 99% of variance


#Test 6 Returns
@test("6.1", "Arithmetic returns", "testout6_1.csv")
def t6_1(): return return_calculate(read("test6.csv"), "DISCRETE", "Date")

@test("6.2", "Log returns", "testout6_2.csv")
def t6_2(): return return_calculate(read("test6.csv"), "LOG", "Date")


#Test 7 Distribution Fitting
@test("7.1", "Fit Normal", "testout7_1.csv")
def t7_1():
    p = fit_normal(one_col("test7_1.csv")).params
    return pd.DataFrame({"mu": [p["mu"]], "sigma": [p["sigma"]]})

@test("7.2", "Fit t (mu, sigma, nu)", "testout7_2.csv", "optimizer")
def t7_2():
    p = fit_general_t(one_col("test7_2.csv")).params
    return pd.DataFrame({"mu": [p["mu"]], "sigma": [p["sigma"]], "nu": [p["nu"]]})

@test("7.3", "Regression with t errors", "testout7_3.csv", "optimizer")
def t7_3():
    d = read("test7_3.csv")
    m = fit_regression_t(d["y"].to_numpy(), d.drop(columns="y").to_numpy()) #y against every other column
    p = m.params
    return pd.DataFrame({"mu": [p["mu"]], "sigma": [p["sigma"]], "nu": [p["nu"]],
                         "Alpha": [m.beta[0]], "B1": [m.beta[1]], "B2": [m.beta[2]], "B3": [m.beta[3]]})

@test("7.4", "AICc of the fitted t", "testout7_4.csv")
def t7_4():
    x = one_col("test7_2.csv")
    return pd.DataFrame({"AICC": [aicc_of_fit(fit_general_t(x), x)]})

def _nig_frame(params):
    return pd.DataFrame(dict(zip(["mu", "alpha", "beta", "delta"], [[v] for v in params]))) #one row: mu, alpha, beta, delta

@test("7.5", "NIG by method of moments", "testout7_5.csv")
def t7_5(): return _nig_frame(fit_nig_moments(one_col("test7_5.csv")))

@test("7.6", "NIG by maximum likelihood", "testout7_6.csv", "optimizer")
def t7_6(): return _nig_frame(fit_nig_mle(one_col("test7_5.csv")))


#Test 8 VaR And ES
def _t_parts():
    m = fit_general_t(one_col("test7_2.csv")) #the t fitted in Test 7.2
    return m, stats.t(m.params["nu"], scale=m.params["sigma"]) #second value is the same t centered at 0

@test("8.1", "VaR, Normal", "testout8_1.csv")
def t8_1():
    m = fit_normal(one_col("test7_1.csv"))
    centred = stats.norm(0, m.params["sigma"]) #centered at 0 gives VaR as a difference from the mean
    return pd.DataFrame({"VaR Absolute": [var_from_dist(m.dist)], "VaR Diff from Mean": [var_from_dist(centred)]})

@test("8.2", "VaR, t", "testout8_2.csv", "optimizer")
def t8_2():
    m, centred = _t_parts()
    return pd.DataFrame({"VaR Absolute": [var_from_dist(m.dist)], "VaR Diff from Mean": [var_from_dist(centred)]})

def _simulated_returns():
    m, _ = _t_parts()
    return m.eval(np.random.default_rng(0).random(10_000)) #10,000 random percentiles through the fitted t

@test("8.3", "VaR from simulation (compare to 8.2)", "testout8_3.csv", "sim")
def t8_3():
    s = _simulated_returns()
    return pd.DataFrame({"VaR Absolute": [var_from_sample(s)], "VaR Diff from Mean": [var_from_sample(s - s.mean())]}) #second value removes the sample mean

@test("8.4", "ES, Normal", "testout8_4.csv")
def t8_4():
    m = fit_normal(one_col("test7_1.csv"))
    centred = stats.norm(0, m.params["sigma"])
    return pd.DataFrame({"ES Absolute": [es_from_dist(m.dist)], "ES Diff from Mean": [es_from_dist(centred)]})

@test("8.5", "ES, t", "testout8_5.csv", "optimizer")
def t8_5():
    m, centred = _t_parts()
    return pd.DataFrame({"ES Absolute": [es_from_dist(m.dist)], "ES Diff from Mean": [es_from_dist(centred)]})

@test("8.6", "ES from simulation (compare to 8.5)", "testout8_6.csv", "sim")
def t8_6():
    s = _simulated_returns()
    return pd.DataFrame({"ES Absolute": [es_from_sample(s)], "ES Diff from Mean": [es_from_sample(s - s.mean())]})


#Test 9 Gaussian Copula
@test("9.1", "Portfolio VaR/ES, Gaussian copula", "testout9_1.csv", "sim100k")
def t9_1(): return copula_var_es(read("test9_1_returns.csv"), read("test9_1_portfolio.csv"))


#Test 10 Portfolio Optimization
@test("10.1", "Risk parity", "testout10_1.csv", "optimizer")
def t10_1(): return pd.DataFrame({"W": risk_parity(matrix("test5_2.csv"))})

@test("10.2", "Risk parity, half budget on X5", "testout10_2.csv", "optimizer")
def t10_2(): return pd.DataFrame({"W": risk_parity(matrix("test5_2.csv"), [1, 1, 1, 1, 0.5])})

@test("10.3", "Max Sharpe, w >= 0", "testout10_3.csv", "optimizer")
def t10_3():
    return pd.DataFrame({"W": max_sharpe(matrix("test5_3.csv"), read("test10_3_means.csv")["Mean"].to_numpy(), 0.04)}) #risk-free rate is 4%

@test("10.4", "Max Sharpe, 0.1 <= w <= 0.5", "testout10_4.csv", "optimizer")
def t10_4():
    bounds = np.column_stack([np.full(5, 0.1), np.full(5, 0.5)]) #each weight between 10% and 50%
    return pd.DataFrame({"W": max_sharpe(matrix("test5_3.csv"), read("test10_3_means.csv")["Mean"].to_numpy(), 0.04, bounds)})


#Test 11 Attribution
@test("11.1", "Ex-post attribution", "testout11_1.csv")
def t11_1():
    returns = read("test11_1_returns.csv")
    out = expost_attribution(read("test11_1_weights.csv")["W"].to_numpy(), returns, returns, np.eye(3)) #stocks are the factors, so betas are the identity
    return out.drop(columns="Alpha") #no alpha column in this test

@test("11.2", "Ex-post attribution to factors", "testout11_2.csv")
def t11_2():
    return expost_attribution(read("test11_2_weights.csv")["W"].to_numpy(),
                              read("test11_2_stock_returns.csv"), read("test11_2_factor_returns.csv"),
                              read("test11_2_beta.csv").iloc[:, 1:].to_numpy()) #first column of the beta file is the stock name


#Test 12 Options
def _options(file):
    o = read(file)
    return o[o["ID"].notna()] #drop blank rows

@test("12.1", "European options, GBSM + Greeks", "testout12_1.csv")
def t12_1():
    rows = []
    for _, o in _options("test12_1.csv").iterrows():
        g = gbsm(o["Option Type"] == "Call", o.Underlying, o.Strike, o.DaysToMaturity / o.DayPerYear, #b = risk-free rate minus dividend rate
                 o.RiskFreeRate, o.RiskFreeRate - o.DividendRate, o.ImpliedVol)
        rows.append([o.ID, g["value"], g["delta"], g["gamma"], g["vega"], g["rho"], g["theta"]])
    return pd.DataFrame(rows, columns=["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]).astype({"ID": int})

@test("12.2", "American options, continuous dividend + Greeks", "testout12_2.csv", "optimizer")
def t12_2():
    rows = []
    for _, o in _options("test12_1.csv").iterrows():
        g = american_greeks(o["Option Type"] == "Call", o.Underlying, o.Strike, o.DaysToMaturity / o.DayPerYear, #500-step tree by default
                            o.RiskFreeRate, o.RiskFreeRate - o.DividendRate, o.ImpliedVol)
        rows.append([o.ID, g["value"], g["delta"], g["gamma"], g["vega"], g["rho"], g["theta"]])
    return pd.DataFrame(rows, columns=["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]).astype({"ID": int})

@test("12.3", "American options, discrete dividends", "testout12_3.csv")
def t12_3():
    rows = []
    for _, o in _options("test12_3.csv").iterrows():
        amounts = [float(v) for v in o.DividendAmts.split(",")]
        steps = [2 * int(v) for v in o.DividendDates.split(",")] #tree has 2 steps per day, so dividend day * 2
        price = bt_american_dividends(o["Option Type"] == "Call", o.Underlying, o.Strike,
                                      o.DaysToMaturity / o.DayPerYear, o.RiskFreeRate,
                                      amounts, steps, o.ImpliedVol, int(2 * o.DaysToMaturity)) #N = 2 steps per day
        rows.append([o.ID, price])
    return pd.DataFrame(rows, columns=["ID", "Value"]).astype({"ID": int})


#Test 13 Multivariate T And T Copula
@lru_cache(maxsize=1) #fit once, reuse for every Test 13 check
def _t13():
    X = matrix("test13_returns.csv")
    models = [fit_general_t(X[:, j]) for j in range(X.shape[1])] #t fit for each column
    U = np.column_stack([m.u for m in models]) #percentiles
    _, ll_gauss = fit_gaussian_copula(U)
    R, nu, ll_t = fit_t_copula(U)
    return dict(X=X, models=models, U=U, ll_gauss=ll_gauss, R=R, nu=nu, ll_t=ll_t)

@test("13.1", "Correlation from Kendall's tau", "testout13_1.csv")
def t13_1(): return frame(kendall_correlation(matrix("test13_returns.csv")))

@test("13.2", "Multivariate t: mean", "testout13_2.csv")
def t13_2(): return pd.DataFrame({"mu": fit_multivariate_t(matrix("test13_returns.csv"))[0]})

@test("13.3", "Multivariate t: scale matrix", "testout13_3.csv")
def t13_3(): return frame(fit_multivariate_t(matrix("test13_returns.csv"))[1])

@test("13.4", "Multivariate t: nu and log-likelihood", "testout13_4.csv")
def t13_4():
    _, _, nu, ll = fit_multivariate_t(matrix("test13_returns.csv"))
    return pd.DataFrame({"nu": [nu], "ll": [ll]})

@test("13.5", "Gaussian copula log-likelihood", "testout13_5.csv", "optimizer")
def t13_5(): return pd.DataFrame({"ll": [_t13()["ll_gauss"]]})

@test("13.6", "t copula: nu and log-likelihood", "testout13_6.csv", "optimizer")
def t13_6(): return pd.DataFrame({"nu": [_t13()["nu"]], "ll": [_t13()["ll_t"]]})

@test("13.7", "Choose copula by AICc and BIC", "testout13_7.csv", "optimizer")
def t13_7():
    c = _t13()
    m = len(c["X"]) #number of observations
    rows = [["Gaussian", c["ll_gauss"], 0, copula_aicc(c["ll_gauss"], 0, m), copula_bic(c["ll_gauss"], 0, m)], #Gaussian copula has 0 extra parameters
            ["T", c["ll_t"], 1, copula_aicc(c["ll_t"], 1, m), copula_bic(c["ll_t"], 1, m)]] #t copula has 1 (nu)
    return pd.DataFrame(rows, columns=["Copula", "LL", "K", "AICC", "BIC"])

@test("13.8", "Lower tail dependence, t copula", "testout13_8.csv", "optimizer")
def t13_8():
    c = _t13()
    n = c["R"].shape[0]
    rows = [[i + 1, j + 1, c["R"][i, j], tail_dependence_t(c["R"][i, j], c["nu"])] #one row per pair of assets
            for i in range(n) for j in range(i + 1, n)]
    return pd.DataFrame(rows, columns=["I", "J", "Rho", "Lambda"])

@test("13.9", "Portfolio VaR/ES, t copula", "testout13_9.csv", "sim100k")
def t13_9():
    c = _t13()
    port = read("test13_portfolio.csv")
    u_sim = simulate_t_copula(c["R"], c["nu"], 100_000, seed=13) #simulated percentiles from the t copula
    sim_returns = np.column_stack([m.eval(u_sim[:, j]) for j, m in enumerate(c["models"])]) #percentiles to returns through each fitted t
    values = port["currentValue"].to_numpy() #current dollar value of each position
    return risk_table(sim_returns * values, values, list(port["Stock"]))


#Compare Results
def compare(mine, expected, tol):
    if list(mine.columns) != list(expected.columns) or mine.shape != expected.shape: #shape and column names must match
        return False, float("nan")
    text_cols = [c for c in expected.columns if not pd.api.types.is_numeric_dtype(expected[c])] #non-numeric columns must match exactly
    if text_cols and not all((mine[c].astype(str) == expected[c].astype(str)).all() for c in text_cols):
        return False, float("nan")
    a = mine.drop(columns=text_cols).to_numpy(dtype=float)
    b = expected.drop(columns=text_cols).to_numpy(dtype=float)

    if tol == "sim_cov":
        err = np.linalg.norm(a - b) / np.linalg.norm(b) #relative Frobenius norm of the difference
        return err < SIM_COV_TOLERANCE, err
    rtol, atol = TOLERANCES[tol]
    passed = np.allclose(a, b, rtol=rtol, atol=atol) #within relative + absolute tolerance
    return passed, np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12) if tol.startswith("sim") else np.abs(a - b)) #worst relative error for sims, worst absolute error otherwise


#Run All Tests
def main(filters):
    OUT.mkdir(exist_ok=True)
    selected = [t for t in TESTS if not filters or any(t[0] == f or t[0].startswith(f + ".") for f in filters)] #run only the tests named on the command line
    print(f"{'Test':<6}{'Description':<50}{'Result':<8}{'Worst error':>13}{'Seconds':>9}")
    print("-" * 86)
    failures = 0
    for test_id, description, fn, expected_file, tol in selected:
        start = time.time()
        try:
            mine = fn()
            mine.to_csv(OUT / expected_file, index=False) #save our answer
            passed, err = compare(mine, read(expected_file), tol)
        except Exception as e: #a crash counts as a failure
            passed, err = False, float("nan")
            print(f"   {test_id} raised {type(e).__name__}: {e}")
        failures += not passed
        label = {"sim_cov": "(rel. Frobenius)", "sim": "(relative)", "sim100k": "(relative)"}.get(tol, "")
        print(f"{test_id:<6}{description:<50}{'PASS' if passed else 'FAIL':<8}{err:>13.2e}{time.time() - start:>9.2f}  {label}")
    print("-" * 86)
    print(f"{len(selected) - failures} of {len(selected)} tests passed.   Our outputs are in {OUT.name}/")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0) #exit code 1 if any test failed
