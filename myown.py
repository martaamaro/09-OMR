import pandas as pd
import matplotlib.pyplot as plt
import smogn

# === LOAD DATA ===
df = pd.read_csv(
    "https://raw.githubusercontent.com/nickkunz/smogn/master/data/housing.csv"
)

y_col = "SalePrice"

# Original distribution
y_original = df[y_col]

# === APPLY SMOGN ===
df_smogn = smogn.smoter(
    data=df,
    y=y_col
)

y_smogn = df_smogn[y_col]


# === PLOT ===
plt.figure(figsize=(6, 4))

# blue = original
plt.hist(y_original, bins=40, color="royalblue", alpha=0.6, label="empirical")

# red = oversampled (SMOGN result)
plt.hist(y_smogn, bins=40, histtype="step", color="red", linewidth=2, label="SMOGN")

plt.xlabel("Target variable")
plt.ylabel("Samples")
plt.title("SMOGN vs Original Distribution")
plt.legend()
plt.tight_layout()
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smogn

# load data
housing = pd.read_csv("data/housing.csv")
y_col = "SalePrice"
y_emp = housing[y_col]
# SMOGN with over- *and* under-sampling (this is the default behaviour)
housing_over_under = smogn.smoter(
    data=housing,
    y=y_col,
    under_samp=True          # hybrid: over + under
)
y_over_under = housing_over_under[y_col]

# SMOGN with *only* over-sampling
housing_over = smogn.smoter(
    data=housing,
    y=y_col,
    under_samp=False
)

y_over = housing_over[y_col]

bins = 40

fig, ax = plt.subplots(figsize=(6,4))

ax.hist(y_emp, bins=bins, alpha=0.6, label="empirical", color="C0")
ax.hist(y_over, bins=bins, histtype="step", linewidth=2, label="over-sampled", color="C1")
ax.hist(y_over_under, bins=bins, histtype="step", linewidth=2, label="over + under", color="C2")

ax.set_xlabel("Target variable domain")
ax.set_ylabel("Number of samples")
ax.set_title("SMOGN")
ax.legend()
plt.tight_layout()
plt.show()


#%%



# %%
