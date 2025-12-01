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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smogn

# load data
housing = pd.read_csv(
    r"C:\Users\marta\OneDrive - Universiteit Utrecht\Desktop\UU-MASTER\Year 1 - Semester 1\OMR\Project II\smogn\data\housing.csv"
)
y_col = "SalePrice"
housing_clean = housing.dropna()
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
    under_samp=False         # only oversample rare bins
)
y_over = housing_over[y_col]

bins = 30

fig, ax = plt.subplots(figsize=(4,4))

# empirical – blue filled
ax.hist(y_emp, bins=bins, alpha=0.8)

# oversampled only – red outline
ax.hist(y_over, bins=bins,
        histtype="step", linewidth=2)

# over + under – green outline
ax.hist(y_over_under, bins=bins,
        histtype="step", linewidth=2)

# colours similar to the paper
for p in ax.patches:            # fill of first hist
    p.set_facecolor("#1f77b4")  # blue-ish
    p.set_edgecolor("none")
for l in ax.lines[0:1]:
    l.set_color("red")
for l in ax.lines[1:2]:
    l.set_color("green")

# styling to look closer to the paper
ax.set_title("SMOGN", fontsize=16)
ax.set_xlabel("Target variable domain", fontsize=12)
ax.set_ylabel("Number of samples", fontsize=12)

# thicker axes
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# small legend below, like the caption
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

handles = [
    Patch(facecolor="#1f77b4", label="empirical data"),
    Line2D([], [], color="red",  lw=2, label="over-sampled data"),
    Line2D([], [], color="green", lw=2, label="over- and under-sampled data"),
]
ax.legend(handles=handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

plt.tight_layout()
plt.show()

##still work in progress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smogn
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------- LOAD DATA ----------
housing = pd.read_csv(
    r"C:\Users\marta\OneDrive - Universiteit Utrecht\Desktop\UU-MASTER\Year 1 - Semester 1\OMR\Project II\smogn\data\housing.csv"
)
y_col = "SalePrice"

data = housing.copy()

# ---------- IMPUTE MISSING VALUES (instead of dropna) ----------
# numeric columns → median
num_cols = data.select_dtypes(include=["int64", "float64"]).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# categorical columns → mode
cat_cols = data.select_dtypes(include=["object", "category"]).columns
for c in cat_cols:
    data[c] = data[c].fillna(data[c].mode().iloc[0])

# target from cleaned data
y_emp = data[y_col]

# You can tweak the quantiles / cut-points later; this is a reasonable start.
q10 = data[y_col].quantile(0.10)
q50 = data[y_col].quantile(0.50)
q90 = data[y_col].quantile(0.90)

rel_pts = [
    [q10,  1, 0],  # low rare region  (oversample)
    [q50,  0, 0],  # central "normal" region
    [q90,  0, 0],  # still normal
    [data[y_col].max(), 1, 0],  # high rare region (oversample)
]

# ---------- SMOGN CALLS ----------
# over + under sampling
data_over_under = smogn.smoter(
    data=data,
    y=y_col,
    k=5,
    under_samp=True,
    rel_method="manual",
    rel_ctrl_pts_rg=rel_pts,
)

y_over_under = data_over_under[y_col]

# over-sampling only
data_over = smogn.smoter(
    data=data,
    y=y_col,
    k=5,
    under_samp=False,
    rel_method="manual",
    rel_ctrl_pts_rg=rel_pts,
)

y_over = data_over[y_col]

# use common bin edges so shapes are comparable
bins = np.linspace(y_emp.min(), y_emp.max(), 30)

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(4, 4))

# empirical – blue filled
ax.hist(y_emp, bins=bins, alpha=0.8)

# oversampled only – red outline
ax.hist(y_over, bins=bins, histtype="step", linewidth=2)

# over + under – green outline
ax.hist(y_over_under, bins=bins, histtype="step", linewidth=2)

# color styling similar to the paper
for p in ax.patches:            # bars from first hist
    p.set_facecolor("#1f77b4")
    p.set_edgecolor("none")

ax.lines[0].set_color("red")    # over-sampled
ax.lines[1].set_color("green")  # over+under

# axes + labels
ax.set_title("SMOGN", fontsize=16)
ax.set_xlabel("Target variable domain", fontsize=12)
ax.set_ylabel("Number of samples", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# legend below, like in the paper
handles = [
    Patch(facecolor="#1f77b4", label="empirical data"),
    Line2D([], [], color="red",   lw=2, label="over-sampled data"),
    Line2D([], [], color="green", lw=2, label="over- and under-sampled data"),
]
ax.legend(handles=handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

plt.tight_layout()
plt.show()

