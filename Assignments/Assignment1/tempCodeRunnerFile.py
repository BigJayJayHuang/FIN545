#Question 3.
df3 = pd.read_csv("problem3.csv").dropna()
cols = ["x1", "x2", "x3", "x4"]

# Predict: plot every pair
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