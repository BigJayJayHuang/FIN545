Assignment 2 - FinTech 545
Jason Huang

REQUIREMENTS
Python 3.9+ with these packages installed:
    numpy, pandas, scipy, matplotlib

Install with:
    pip install numpy pandas scipy matplotlib

HOW TO RUN
1. Keep Assignment2.py and the five CSV files (problem1.csv through problem5.csv)
   in the same folder.
2. Run:
    python Assignment2.py
   To also save the printed output to a text file (this is how outputs.txt was made):
    python Assignment2.py > outputs.txt
3. The script prints every number used in Assignment2_Answers.pdf, in order by
   problem (1 through 5). All random draws use fixed seeds, so rerunning gives the
   same numbers. It takes a few seconds.
4. Plots are saved as PNG files in the same folder (no windows pop up):
    problem2_returns.png, problem3_pnl.png, problem4_ranks.png

FILES
    Assignment2.py           all code, with comments on the conventions used
    outputs.txt              the full printed output of the script, so the numbers can be read without running it
    Assignment2_Answers.pdf  written answers (Predict / Fit / Reconcile for each problem), with terminal output shown inline
    screenshots/             the terminal output images used in the PDF (same text as outputs.txt)
    problem1.csv - problem5.csv   data (from the class repo)

CONVENTIONS (also commented in the code)
    - VaR and ES are at 5% and reported as positive losses unless a part says otherwise.
    - Problem 2: the regime split is the last 30 days vs the first 470, chosen by eye from the plot.
    - Problem 4: the margins and R (from Kendall's tau) are frozen, so the Gaussian copula has 0
      free parameters and the t copula has 1 (nu), for AICc and BIC.
    - Problem 5: alpha is set to 0 in the simulation because the problem says to assume zero expected returns.
