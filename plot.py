# plot_and_table_cml.py
# -----------------------------------------------------------
# Self-Contained Script to:
# 1) Compute the two OOS R² time‐series for the CML method:
#    - [t:end]-R²_t (cumulative from t to end)
#    - [1:t]-R²_t (cumulative from first OOS month to t)
# 2) Plot those two series in the “paper style”
# 3) Aggregate CML’s OOS R² into four forecast‐period averages
#    (e.g. 1964–1999, 2000–2007, 2008–2020, full sample)
#    and display them in a “table” format.
#
# Prerequisites:
#   * You have already run the replication code and produced
#     cml_outputs/oos_results.csv containing columns:
#       ['yyyymm', 'g_t', 'y_true', 'y_hat']
#   * Python packages: pandas, numpy, matplotlib, statsmodels, tqdm (optional)
#
# To run:
#   python plot_and_table_cml.py
#
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# ==================== CONFIGURATION ====================

OOS_RESULTS_CSV = "oos_results_short.csv"
SAVE_DIR        = Path("cml_outputs")
SAVE_DIR.mkdir(exist_ok=True)

# Define forecast‐period ranges (inclusive yyyymm):
# Adjust these to match your sample. Here we use the paper’s periods:
PERIODS = {
    "1964-1999": (196401, 199912),
    "2000-2007": (200001, 200712),
    "2008-2020": (200801, 202012),
    "full period": None  # meaning “use entire OOS span”
}

# ==================== STEP 1: LOAD OOS RESULTS ====================

print("\n1. Loading OOS results...")
res_df = pd.read_csv(SAVE_DIR / OOS_RESULTS_CSV, dtype={'yyyymm': int})
# Ensure sorted by yyyymm
res_df = res_df.sort_values("yyyymm").reset_index(drop=True)

# Extract arrays for convenience
months = res_df['yyyymm'].values       # e.g. [196412, 196501, ...]
y_true = res_df['y_true'].values       # actual return at next month
y_hat  = res_df['y_hat'].values        # forecasted return

# Keep a DataFrame padded for easy indexing
df_oos = res_df.set_index('yyyymm')

# ==================== STEP 2: COMPUTE [t:end]-R²_t SERIES ====================
print("2. Computing [t:end]-R²_t series...")

# Pre‐compute the in‐sample running mean of y_true up to each s
# We'll use this in denominator for both R² definitions
running_mean = {}
cum_sum = 0.0
for i, s in enumerate(months):
    cum_sum += y_true[i]
    running_mean[s] = cum_sum / (i + 1)

# Now compute [t:end]-R²_t for each t
r2_t_to_end = {}
for idx, t in enumerate(months):
    # For all s ≥ t, collect:
    mask = months >= t
    s_indices = np.where(mask)[0]
    if len(s_indices) == 0:
        r2_t_to_end[t] = np.nan
        continue

    # Numerator: sum[(y_true[s] - y_hat[s])² for s ≥ t]
    num = np.sum((y_true[s_indices] - y_hat[s_indices]) ** 2)

    # Denominator: sum[(y_true[s] - running_mean[s])² for s ≥ t]
    denom = 0.0
    for si in s_indices:
        s_month = months[si]
        ms = running_mean[s_month]
        denom += (y_true[si] - ms) ** 2

    # If denom is zero, set R2 = NaN to avoid division by zero
    if denom == 0.0:
        r2_t_to_end[t] = np.nan
    else:
        r2_t_to_end[t] = 1.0 - num / denom

# ==================== STEP 3: COMPUTE [1:t]-R²_t SERIES ====================
print("3. Computing [1:t]-R²_t series...")

r2_1_to_t = {}
first_month = months[0]  # earliest OOS month

# We also reuse running_mean stored above
for idx, t in enumerate(months):
    # Collect all s such that first_month ≤ s ≤ t
    mask = (months >= first_month) & (months <= t)
    s_indices = np.where(mask)[0]
    if len(s_indices) == 0:
        r2_1_to_t[t] = np.nan
        continue

    # Numerator: sum[(y_true[s] - y_hat[s])² for s in [first_month..t]]
    num = np.sum((y_true[s_indices] - y_hat[s_indices]) ** 2)

    # Denominator: sum[(y_true[s] - running_mean[s])² for s in [first_month..t]]
    denom = 0.0
    for si in s_indices:
        s_month = months[si]
        ms = running_mean[s_month]
        denom += (y_true[si] - ms) ** 2

    if denom == 0.0:
        r2_1_to_t[t] = np.nan
    else:
        r2_1_to_t[t] = 1.0 - num / denom

# Convert series to pandas Series for easy plotting
r2_to_end_series = pd.Series(r2_t_to_end)
r2_1_to_t_series = pd.Series(r2_1_to_t)

# ==================== STEP 4: PLOT R² SERIES ====================
print("4. Plotting R² time series...")

plt.figure(figsize=(10, 8))

# --- Upper subplot: [t:end]-R²_t ---
ax1 = plt.subplot(2, 1, 1)
ax1.plot(r2_to_end_series.index.values, r2_to_end_series.values, color='C0', label='CML')
ax1.axhline(0, color='grey', linestyle=':', linewidth=1)  # zero‐reference
ax1.set_title('[t : end] OOS R² (CML)')
ax1.set_ylabel('R² (%)')
ax1.set_xlim(r2_to_end_series.index.min(), r2_to_end_series.index.max())

# (Optional) Draw vertical dashed lines at period boundaries:
#   Suppose we mark 1980, 1990, 2000, 2010, 2020 as examples
vlines = [198001, 199001, 200001, 201001, 202001]
for vl in vlines:
    ax1.axvline(vl, color='r', linestyle='--', linewidth=0.8)

ax1.legend(loc='upper left')

# --- Lower subplot: [1:t]-R²_t ---
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(r2_1_to_t_series.index.values, r2_1_to_t_series.values, color='C0', label='CML')
ax2.axhline(0, color='grey', linestyle=':', linewidth=1)
ax2.set_title('[1 : t] OOS R² (CML)')
ax2.set_xlabel('yyyymm')
ax2.set_ylabel('R² (%)')
ax2.set_xlim(r2_1_to_t_series.index.min(), r2_1_to_t_series.index.max())

for vl in vlines:
    ax2.axvline(vl, color='r', linestyle='--', linewidth=0.8)

ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig(SAVE_DIR / "cml_oos_r2_series.png", dpi=300)
print("   → Saved plot to cml_oos_r2_series.png")

# ==================== STEP 5: AGGREGATE R² INTO TABLE ====================
print("5. Aggregating R² into table format (CML only)...")

# Prepare an output DataFrame with four rows: [t:end]-R2 and [1:t]-R2 for each forecast‐period
rows = []

for label, bounds in PERIODS.items():
    if bounds is None:
        # Full period: use all OOS months
        mask_to_end = (r2_to_end_series.index >= months.min())
        mask_1_to_t = (r2_1_to_t_series.index >= months.min())
    else:
        start, end = bounds
        # For [t:end]-R²: we only average R²_t for t in [start..end]
        mask_to_end = (r2_to_end_series.index >= start) & (r2_to_end_series.index <= end)
        # For [1:t]-R²: same indexing by t ∈ [start..end]
        mask_1_to_t = (r2_1_to_t_series.index >= start) & (r2_1_to_t_series.index <= end)

    # Compute average, drop NaNs
    avg_to_end = r2_to_end_series[mask_to_end].dropna().mean() * 100  # convert to percentage
    avg_1_to_t = r2_1_to_t_series[mask_1_to_t].dropna().mean() * 100

    rows.append({
        'Forecast period': label,
        '[t:end]-R² (CML)':   f"{avg_to_end:.3f}",
        '[1:t]-R² (CML)':     f"{avg_1_to_t:.3f}"
    })

table_df = pd.DataFrame(rows)

# Display table
print("\n=== CML OOS R² Averages by Forecast Period ===")
print(table_df.to_string(index=False))

# Save as CSV for posterity
table_df.to_csv(SAVE_DIR / "cml_oos_r2_table.csv", index=False)
print("   → Saved table to cml_oos_r2_table.csv")

# ==================== END ====================
