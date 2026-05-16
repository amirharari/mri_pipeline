"""
Quality Assurance 2: Cross-reference motion (scrub_percent) vs cleaning (confound_var_explained).

Tests whether high motion is associated with lower or higher variance explained by confounds.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TS_OUTPUT_DIR, QA_OUTPUT_DIR  # noqa: E402

TS_DIR  = TS_OUTPUT_DIR
OUT_DIR = QA_OUTPUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)


def load_merged_report():
    """Load all-runs report (_new) and fill confound_var_explained from the older report."""
    p_new = os.path.join(TS_DIR, "scrubbing_report_filtered_new.csv")
    p_old = os.path.join(TS_DIR, "scrubbing_report_filtered.csv")

    # Prefer the all-runs file for motion columns
    p_main = p_new if os.path.isfile(p_new) else p_old
    df = pd.read_csv(p_main)

    # Merge variance-explained from the other file if it has more data
    p_other = p_old if p_main == p_new else p_new
    if os.path.isfile(p_other):
        df_other = pd.read_csv(p_other)
        if "confound_var_explained" in df_other.columns:
            ve_other = df_other.dropna(subset=["confound_var_explained"])
            if len(ve_other) > df["confound_var_explained"].notna().sum() if "confound_var_explained" in df.columns else True:
                keys = ["subject", "session", "task", "acq", "run"]
                ve_map = ve_other.set_index(keys)["confound_var_explained"]
                df = df.set_index(keys)
                df["confound_var_explained"] = df["confound_var_explained"].combine_first(ve_map)
                df = df.reset_index()

    n_ve = df["confound_var_explained"].notna().sum() if "confound_var_explained" in df.columns else 0
    print(f"Loaded {len(df)} runs, variance-explained available for {n_ve}")
    return df


def main():
    df = load_merged_report()

    # Use scrub_percent and confound_var_explained
    scrub_col = "scrub_percent"
    ve_col = "confound_var_explained"

    if scrub_col not in df.columns:
        print(f"Column {scrub_col} not found. Available: {list(df.columns)}")
        return

    # confound_var_explained may be fraction (0-1) or percent
    if ve_col in df.columns:
        ve = df[ve_col].dropna()
        if (ve < 2).all() and (ve >= 0).all():
            df["var_exp_pct"] = df[ve_col] * 100
        else:
            df["var_exp_pct"] = df[ve_col]
    else:
        df["var_exp_pct"] = np.nan
        print("[WARNING] confound_var_explained not in report (skipped runs). Using scrub vs FD only.")

    # Drop rows with NaN for correlation
    valid = df.dropna(subset=[scrub_col, "var_exp_pct"])
    n_valid = len(valid)

    # Colors by subject
    subj_colors = {"sub-001": "#2196F3", "sub-002": "#F44336", "sub-003": "#4CAF50", "sub-004": "#FF9800"}
    colors = [subj_colors.get(s, "#9E9E9E") for s in df["subject"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter scrub vs var_explained
    ax = axes[0]
    for subj in df["subject"].unique():
        m = df["subject"] == subj
        ax.scatter(df.loc[m, scrub_col], df.loc[m, "var_exp_pct"],
                   c=subj_colors.get(subj, "#9E9E9E"), label=subj, alpha=0.8, s=60)

    if n_valid >= 3:
        # Regression line
        x = valid[scrub_col].values
        y = valid["var_exp_pct"].values
        r, p = stats.pearsonr(x, y)
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, np.poly1d(z)(x_line), "k--", lw=2, label=f"r={r:.3f}, p={p:.4f}")
        ax.set_title(f"Scrub % vs Variance Explained\nPearson r={r:.3f}, p={p:.4f} (n={n_valid})")
    else:
        ax.set_title("Scrub % vs Variance Explained\n(insufficient data for correlation)")

    ax.set_xlabel("Scrub percent (volumes censored)")
    ax.set_ylabel("Variance explained by confounds (%)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: scrub vs fd_mean_filtered (motion severity)
    ax = axes[1]
    fd_col = "fd_mean_filtered" if "fd_mean_filtered" in df.columns else "fd_mean_raw"
    if fd_col in df.columns:
        for subj in df["subject"].unique():
            m = df["subject"] == subj
            ax.scatter(df.loc[m, scrub_col], df.loc[m, fd_col],
                       c=subj_colors.get(subj, "#9E9E9E"), label=subj, alpha=0.8, s=60)
        fd_valid = df[[scrub_col, fd_col]].dropna()   # drop NaN before pearsonr
        r2, p2 = stats.pearsonr(fd_valid[scrub_col], fd_valid[fd_col])
        ax.set_title(f"Scrub % vs FD mean (filtered)\nr={r2:.3f}, p={p2:.4f} (n={len(fd_valid)})")
        ax.set_xlabel("Scrub percent")
        ax.set_ylabel(f"FD mean (mm) [{fd_col}]")
    else:
        ax.text(0.5, 0.5, "FD column not found", ha="center", va="center")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "motion_vs_cleaning.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Print correlation summary
    if n_valid >= 3:
        print(f"\nCorrelation: scrub_percent vs confound_var_explained")
        print(f"  Pearson r = {r:.4f}, p = {p:.4f}")
        if p < 0.05:
            print(f"  Significant: high motion {'associated with' if r > 0 else 'inversely related to'} variance explained")
        else:
            print(f"  Not significant at alpha=0.05")


if __name__ == "__main__":
    main()
