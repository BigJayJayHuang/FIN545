Assignment 3 - FinTech 545
Jason Huang

REQUIREMENTS
Python 3.9+ with these packages installed:
    numpy, pandas, scipy, matplotlib

Install with:
    pip install numpy pandas scipy matplotlib

HOW TO RUN
1. Keep Assignment3.py and the data files in the same folder:
    problem1.csv, problem3.csv, problem3_portfolios.csv, problem4.csv, problem5.csv,
    F-F_Research_Data_Factors_daily.CSV, F-F_Momentum_Factor_daily.CSV
2. Run:
    python Assignment3.py
   To also save the printed output to a text file (this is how outputs.txt was made):
    python Assignment3.py > outputs.txt
3. The script prints every number used in Assignment3_Answers.pdf, in order by problem (1 through 5).
   Each part is tagged like "--- 3(e) ..." so it's easy to match to the write up.
   All random draws use fixed seeds, so rerunning gives the same numbers. It takes about 15 seconds.
4. Plots are saved as PNG files in the same folder (no windows pop up):
    problem1_smile.png, problem2_convergence.png, problem3_pnl.png, problem4_frontier.png

FILES
    Assignment3.py           all code, with comments on the conventions used
    outputs.txt              the full printed output of the script, so the numbers can be read without running it
    Assignment3_Answers.pdf  written answers (Predict / Fit / Reconcile for each problem), with terminal output shown inline
    screenshots/             the terminal output images used in the PDF (same text as outputs.txt)
    *.csv / *.CSV            data (from the class repo)

CONVENTIONS (also commented in the code)
    - All option pricing is written by hand: generalized Black Scholes Merton (cost of carry b), its closed form
      Greeks, implied vol by a bracketing root finder (brentq), and CRR binomial trees.
    - VaR and ES are reported as positive losses. Sample VaR uses the course quantile (average of the two order
      statistics around n*alpha).
    - Problem 1: T = 35 calendar days / 365, b = r - q. A quote with a price below its no arbitrage lower bound
      gets no implied vol (reported as NaN with the reason).
    - Problem 2: the discrete dividend tree is the course's non recombining tree: at each node on a dividend step
      a new tree starts at (price - dividend). Exercise on a dividend step uses the cum dividend price. N is a
      multiple of 5 so both dividends land on a step. Main prices use N = 500; Greeks and dividend
      sensitivities use N = 200 with central differences.
    - Problem 3: 255 trading days a year, options valued with T = DaysToExpiry / 255, implied vol held fixed.
      Delta normal uses the sample daily std times sqrt(10). The simulation draws 10 daily returns with zero
      mean (normal: sigma from the fit; t: fitted scale and nu with location 0), compounds them, and fully
      revalues each book with 10 fewer days. 100,000 paths, seed 545.
    - Problem 4: monthly risk free rate = 3% / 12. Sharpe is annualized with sqrt(12). Unconstrained tangency
      is the closed form inverse(Sigma)(mu - rf) scaled to sum to 1; the constrained ones use SLSQP.
      Bootstrap: 1,000 resamples of the 240 months, seed 4545.
    - Problem 5: factor files are divided by 100. Daily numbers are annualized with 255 days (same as Problem 3).
      AICc counts the regression coefficients plus 1 for the residual variance. "Most recent 10 years" is
      2014-10-01 to 2024-09-30, the last 10 years in the factor files.
