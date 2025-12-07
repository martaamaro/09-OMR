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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load data
housing = pd.read_csv("data/housing.csv")

# 2) Use only numeric features + target
numeric_cols = ["LotArea", "GrLivArea", "OverallQual", "YearBuilt", "SalePrice"]
data = housing[numeric_cols].dropna().copy()

target = "SalePrice"
y = data[target].to_numpy().astype(float)



#%%
# ---- relevance with distance kernel ----
from imbami import DensityDistanceRelevance, cSMOGN
y_scaled = (y - y.min()) / (y.max() - y.min())

dist_kernel = DensityDistanceRelevance()
dist_kernel.fit(
    y_scaled,
    emp_bandwidth_type="silverman",
    rel_data=None,
    rel_bandwidth_type="uniform",
    rel_bandwidth_factor=1.0,
)

rel_raw = dist_kernel.eval(y_scaled, centered=True)
rel_scaled = (rel_raw - rel_raw.min()) / (rel_raw.max() - rel_raw.min() + 1e-9)
relevance_values = pd.Series(rel_scaled, index=data.index, name="relevance")

#%%
# ---- run cSMOGN ----
sampler = cSMOGN(
    data=data,
    target_column=target,
    relevance_values=relevance_values,
)

new_data = sampler.run_sampling(
    oversample_rate=0.5,
    undersample_rate=0.5,
    knns=5,
    num_bins=10,
    allowed_bin_deviation=1,
    noise_factor=0.01,
    ignore_categorical_similarity=False,
    enable_undersampling=True,
)

print("Original shape:", data.shape)
print("After cSMOGN  :", new_data.shape)

# %%
# ---- visualize ----
plt.figure(figsize=(6,4))
plt.hist(data[target], bins=40, alpha=0.6, label="original")
plt.hist(new_data[target], bins=40, histtype="step", linewidth=2, label="cSMOGN")
plt.xlabel("SalePrice")
plt.ylabel("Number of samples")
plt.title("cSMOGN on housing.csv (numeric features only)")
plt.legend()
plt.tight_layout()
plt.show()
#%%
import pandas as pd

housing = pd.read_csv("data/housing.csv")

numeric_cols = ["LotArea", "GrLivArea", "OverallQual", "YearBuilt", "SalePrice"]
data = housing[numeric_cols].dropna().copy()

target = "SalePrice"

#%%
from imbami import DensityRatioRelevance   # or from imbami.density_ratio_relevance import ...

y = data[target].to_numpy().astype(float)
y_scaled = np.log10(y)   # prices → log-scale

ratio_kernel = DensityRatioRelevance()
ratio_kernel.fit(
    y_scaled,
    rel_data=None
)

relevance_values = pd.Series(
    ratio_kernel.eval(y_scaled),
    index=data.index,
)



# %%
from imbami import crbSMOGN

sampler = crbSMOGN(data=data, target_column=target, relevance_values=relevance_values)
new_data_crb = sampler.run_sampling(
    min_acceptable_relevance=0.8,
    max_acceptable_relevance=1.2,
    num_bins=10,
    allowed_bin_deviation=1,
    noise_factor=0.01,
    ignore_categorical_similarity=False,
    enable_undersampling=True,
)

# %%
import matplotlib.pyplot as plt

bins = 40
plt.figure(figsize=(6,4))

plt.hist(data[target], bins=bins, alpha=0.6, label="original", density=False)
plt.hist(new_data_crb[target], bins=bins,
         histtype="step", linewidth=2, label="crbSMOGN")

plt.xlabel("SalePrice")
plt.ylabel("Number of samples")
plt.title("crbSMOGN on housing.csv")
plt.legend()
plt.tight_layout()
plt.show()

# %%
