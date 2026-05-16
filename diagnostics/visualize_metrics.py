"""Visualize GCOR, scrubbing, and FD metrics from scrubbing_report.csv."""
import os
import pandas as pd
import matplotlib.pyplot as plt

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "timeseries_GSR_off", "scrubbing_report.csv")
OUTPUT_PATH = "gcor_scrubbing_dashboard.png"

def main():
    if not os.path.isfile(REPORT_PATH):
        print(f"Report not found: {REPORT_PATH}")
        return

    df = pd.read_csv(REPORT_PATH)
    ncols = 2
    nrows = 2
    if "gcor" in df.columns:
        nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))

    ax = axes[0, 0]
    if "gcor" in df.columns:
        ax.hist(df["gcor"].dropna(), bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(0.4, color="red", linestyle="--", label="GCOR=0.4 threshold")
        ax.set_xlabel("GCOR")
        ax.set_ylabel("Count")
        ax.set_title("GCOR distribution")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No GCOR column\n(Run extractor to populate)", ha="center", va="center")
        ax.set_title("GCOR distribution")

    ax = axes[0, 1]
    ax.scatter(df["fd_mean"], df["scrubbed_volumes"], alpha=0.7)
    ax.set_xlabel("FD mean (mm)")
    ax.set_ylabel("Scrubbed volumes")
    ax.set_title("FD mean vs scrubbed volumes per run")

    ax = axes[1, 0]
    run_labels = [f"{r.subject}\n{r.session}" for _, r in df.iterrows()]
    ax.bar(range(len(df)), df["scrubbed_volumes"], color="steelblue", alpha=0.8)
    ax.set_ylabel("Scrubbed volumes")
    ax.set_title("Scrubbed volumes per run")
    if len(df) <= 20:
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([])

    ax = axes[1, 1]
    ax.scatter(df["fd_mean"], df["fd_max"], alpha=0.7)
    ax.set_xlabel("FD mean (mm)")
    ax.set_ylabel("FD max (mm)")
    ax.set_title("FD mean vs FD max per run")

    if "gcor" in df.columns and nrows >= 3:
        ax = axes[2, 0]
        ax.scatter(df["fd_mean"], df["gcor"], c=(df["gcor"] > 0.4), cmap="RdYlGn_r", alpha=0.7)
        ax.axhline(0.4, color="red", linestyle="--")
        ax.set_xlabel("FD mean (mm)")
        ax.set_ylabel("GCOR")
        ax.set_title("FD mean vs GCOR (red=high GCOR)")

        ax = axes[2, 1]
        ax.scatter(df["dvars_mean"], df["gcor"], alpha=0.7)
        ax.set_xlabel("DVARS mean")
        ax.set_ylabel("GCOR")
        ax.set_title("DVARS vs GCOR")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
