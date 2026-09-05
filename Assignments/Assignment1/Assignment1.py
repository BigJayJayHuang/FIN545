import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt


def first_four_moments(sample):
    sample = np.asarray(sample)
    n = len(sample)

    mu_hat = np.sum(sample) / n #mean

    sim_corrected = sample - mu_hat #deviations from the mean
    cm2 = np.dot(sim_corrected, sim_corrected) / n #sum of squared deviations / n

    sigma2_hat = cm2 #variance(biased)
    skew_hat = np.sum(sim_corrected ** 3) / n / np.sqrt(cm2 ** 3)   #skewness
    kurt_hat = np.sum(sim_corrected ** 4) / n / cm2 ** 2            #raw kurtosis
    excess_kurt_hat = kurt_hat - 3 #excess kurtosis

    return mu_hat, sigma2_hat, skew_hat, excess_kurt_hat


#Quesitom 1. 
print("=" * 60)
print("QUESTION 1")
print("=" * 60)

df = pd.read_csv("problem1.csv").dropna()
x = df["x"].values

#Predict first four moments
mean_hat, var_hat, skew_hat, excess_kurt_hat = first_four_moments(x)

print(f"mean = {mean_hat:.6f}, variance = {var_hat:.6f}")
print(f"skewness = {skew_hat:.6f}, excess kurtosis = {excess_kurt_hat:.6f}")

mu_fit = mean_hat
sigma_fit = np.sqrt(var_hat) #Normal needs a standard deviation, not variance

#The 1% quantile is the value below which 1% of the distribution's
q01 = stats.norm.ppf(0.01, loc=mu_fit, scale=sigma_fit)
print(f"fitted Normal: mean = {mu_fit:.6f}, std = {sigma_fit:.6f}")
print(f"1% quantile = {q01:.6f}")

#Count how many actual observations fall below that fitted quantile
n_below = np.sum(x < q01)
print(f"observations below the 1% quantile: {n_below}")

#Question 2. 
print("=" * 60)
print("QUESTION 2")
print("=" * 60)

#observing the scatter plot
df2 = pd.read_csv("problem2.csv").dropna()
x2 = df2["x"].values
y2 = df2["y"].values
n2 = len(x2)

plt.figure(figsize=(7, 5))
plt.scatter(x2, y2, alpha=0.6, s=15)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Problem 2: y vs x")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show() 

#OLS
ols = sm.OLS(y2, sm.add_constant(x2)).fit()
alpha_ols, beta_ols = ols.params
se_beta_ols = ols.bse[1]
resid0 = y2 - (alpha_ols + beta_ols * x2)
 
#MLE, Normal error
def nll_normal(p, x, y):
    a, b, s = p
    return np.inf if s <= 0 else -np.sum(stats.norm.logpdf(y - (a + b*x), scale=s))
 
res_n = minimize(nll_normal, [alpha_ols, beta_ols, np.std(resid0)], args=(x2, y2), method="Nelder-Mead")
alpha_n, beta_n, sigma_n = res_n.x
ll_n = -res_n.fun
 
#MLE, t error
def nll_t(p, x, y):
    a, b, s, nu = p
    return np.inf if s <= 0 or nu <= 2 else -np.sum(stats.t.logpdf(y - (a + b*x), df=nu, scale=s))
 
res_t = minimize(nll_t, [alpha_ols, beta_ols, np.std(resid0), 5.0], args=(x2, y2), method="Nelder-Mead")
alpha_t, beta_t, scale_t, nu_t = res_t.x
ll_t = -res_t.fun
 
#AICc
def aicc(ll, k, n):
    return 2*k - 2*ll + (2*k**2 + 2*k) / (n - k - 1)
 
aicc_n, aicc_t = aicc(ll_n, 3, n2), aicc(ll_t, 4, n2)
 
print(f"OLS:    alpha={alpha_ols:.4f}, beta={beta_ols:.4f}, SE(beta)={se_beta_ols:.4f}")
print(f"Normal: alpha={alpha_n:.4f}, beta={beta_n:.4f}, sigma={sigma_n:.4f}, AICc={aicc_n:.3f}")
print(f"t:      alpha={alpha_t:.4f}, beta={beta_t:.4f}, scale={scale_t:.4f}, df={nu_t:.4f}, AICc={aicc_t:.3f}")
print(f"Preferred model: {'t' if aicc_t < aicc_n else 'Normal'}")
 
#95% / 99.5% quantiles of each fitted error
for lvl in [0.95, 0.995]:
    q_n = stats.norm.ppf(lvl, scale=sigma_n)
    q_t = stats.t.ppf(lvl, df=nu_t, scale=scale_t)
    print(f"{lvl:.1%}: Normal={q_n:.4f}, t={q_t:.4f}")

#Question 3.
print("=" * 60)
print("QUESTION 3")
print("=" * 60)

df3 = pd.read_csv("problem3.csv").dropna()
cols = ["x1", "x2", "x3", "x4"]

#Predict plot every pair
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ci in enumerate(cols):
    for j, cj in enumerate(cols):
        ax = axes[i, j]
        if i == j:
            ax.hist(df3[ci], bins=20, color="gray")
        else:
            ax.scatter(df3[cj], df3[ci], s=8, alpha=0.5)
        if i == 3:
            ax.set_xlabel(cj)
        if j == 0:
            ax.set_ylabel(ci)
plt.tight_layout()
plt.show()
 
#Fit Pearson vs Spearman
pearson = df3[cols].corr(method="pearson")
spearman = df3[cols].corr(method="spearman")
 
print("=== Pearson ===")
print(pearson.round(4))
print("\n=== Spearman ===")
print(spearman.round(4))
 
#find the pair with the largest gap between the two measures
gap = (pearson - spearman).abs()
gap_arr = gap.to_numpy().copy()
np.fill_diagonal(gap_arr, 0)
i, j = np.unravel_index(np.argmax(gap_arr), gap_arr.shape)
print(f"\nLargest gap: {cols[i]} vs {cols[j]}  "
      f"(Pearson={pearson.iloc[i,j]:.4f}, Spearman={spearman.iloc[i,j]:.4f}, "
      f"gap={gap.iloc[i,j]:.4f})")

#Question 4. 
print("=" * 60)
print("QUESTION 4")
print("=" * 60)

df4 = pd.read_csv("problem4.csv").dropna()
x1 = df4["x1"].values
x2 = df4["x2"].values
n4 = len(x1)
 
#Predict sample covariance matrix, blocks are Sigma11(s11)=Var(x1), Sigma22=Var(x2), Sigma12=Cov(x1,x2)
Sigma = np.cov(x1, x2, ddof=0)
s11, s22, s12 = Sigma[0, 0], Sigma[1, 1], Sigma[0, 1]
mu1, mu2 = np.mean(x1), np.mean(x2)
rho = s12 / np.sqrt(s11 * s22)
 
print("=== Covariance matrix ===")
print(Sigma)
print(f"mu1={mu1:.4f}, mu2={mu2:.4f}, rho={rho:.4f}")
 
#Conditional variance of x2 given x1: Sigma22 - Sigma12^2/Sigma11(constant, no x1 term)
cond_var = s22 - s12**2 / s11
reduction_factor = 1 - rho**2   #Var(x2|x1) / Var(x2) = 1 - rho^2
 
print(f"\nVar(x2) = {s22:.4f}")
print(f"Var(x2|x1) = {cond_var:.4f}")
print(f"reduction factor (1 - rho^2) = {reduction_factor:.4f}")
 
#Fit conditional mean of x2 given x1: mu2 + (Sigma12/Sigma11)*(x1 - mu1)
beta = s12 / s11   #this is exactly the OLS slope of regressing x2 on x1
alpha = mu2 - beta * mu1
 
xs = np.linspace(x1.min(), x1.max(), 200)
cond_mean_xs = alpha + beta * xs
band_half_width = stats.norm.ppf(0.975) * np.sqrt(cond_var)   #95% band, constant width
 
plt.figure(figsize=(7, 5))
plt.scatter(x1, x2, s=10, alpha=0.4, label="data")
plt.plot(xs, cond_mean_xs, color="black", label="conditional mean")
plt.fill_between(xs, cond_mean_xs - band_half_width, cond_mean_xs + band_half_width,
                  color="gray", alpha=0.3, label="95% band")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
plt.title("Problem 4: conditional mean of x2 given x1, with 95% band")
plt.tight_layout()
plt.savefig("problem4_band.png", dpi=150)
plt.show()
 
#fraction of points inside the band
pred = alpha + beta * x1
inside = np.abs(x2 - pred) <= band_half_width
print(f"\noverall coverage: {inside.mean():.4f}  ({inside.sum()} of {n4})")
 
#Split by distance of x1 from its mean, in units of std(x1)
std1 = np.std(x1, ddof=0)
dist = np.abs(x1 - mu1) / std1
buckets = {
    "within 1 sd": dist <= 1,
    "1 to 2 sd": (dist > 1) & (dist <= 2),
    "beyond 2 sd": dist > 2,
}
print("\ncoverage by bucket:")
for name, mask in buckets.items():
    cov = inside[mask].mean()
    print(f"  {name}: {cov:.4f}  (n={mask.sum()})")

#Question 5.
print("=" * 60)
print("QUESTION 5")
print("=" * 60)

df5 = pd.read_csv("problem5.csv").dropna()
x5 = df5["x"].values
n5 = len(x5)
 
#visuals series, ACF, PACF
fig, axes = plt.subplots(3, 1, figsize=(9, 9))
axes[0].plot(x5)
axes[0].set_title("Series")
plot_acf(x5, ax=axes[1], lags=20)
plot_pacf(x5, ax=axes[2], lags=20, method="ywm")
plt.tight_layout()
plt.show()

#numeric ACF/PACF values with the significance band, to back up the "cuts off vs decays" call
band = 1.96 / np.sqrt(n5)
acf_vals = acf(x5, nlags=10)
pacf_vals = pacf(x5, nlags=10, method="ywm")

print(f"\nsignificance band = +/-{band:.4f}")
print(f"{'lag':>4} {'ACF':>8} {'PACF':>8}")
for lag in range(1, 11):
    print(f"{lag:>4} {acf_vals[lag]:>8.4f} {pacf_vals[lag]:>8.4f}")

#Fit AR(1..3) and MA(1..3), compare with AICc
def aicc(aic, k, n):
    return aic + (2 * k**2 + 2 * k) / (n - k - 1)
 
print("=== AR / MA model comparison ===")
results = {}
for p in [1, 2, 3]:
    m = ARIMA(x5, order=(p, 0, 0)).fit()   #AR(p): order=(p,0,0)
    k = p + 2   #p AR coeffs + mean + sigma^2
    results[f"AR({p})"] = (m, aicc(m.aic, k, n5))
 
for q in [1, 2, 3]:
    m = ARIMA(x5, order=(0, 0, q)).fit()   #MA(q): order=(0,0,q)
    k = q + 2   #q MA coeffs + mean + sigma^2
    results[f"MA({q})"] = (m, aicc(m.aic, k, n5))
 
for name, (m, ac) in results.items():
    print(f"{name}: AICc = {ac:.3f}, params = {np.round(m.params, 4)}")
 
best = min(results, key=lambda k: results[k][1])
print(f"\nBest by AICc: {best}")
 
print("\n=== AR(2) vs AR(3) coefficients ===")
print("AR(2):", np.round(results["AR(2)"][0].params, 4))
print("AR(3):", np.round(results["AR(3)"][0].params, 4))
 