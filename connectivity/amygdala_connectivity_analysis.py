import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import ttest_rel, ttest_ind
from nilearn import datasets, plotting, surface
from nilearn.plotting import plot_matrix
import plotly.graph_objects as go

from utils import (
    fetch_aal_spm12,
    fetch_fsaverage_surfaces,
    map_roi_to_indices,
)

# =============================================================================
# CONFIG
# =============================================================================
# It yields, yet remains whole Roots in the earth, spirit in the sky
scrubbed_volumes_threshold = 999  # Disabled - include all sessions even with high motion

PROJECT_ROOT = r"C:\aaf-files"  # Folder containing *_aal_ts.csv time series
OUTPUT_FOLDER = "t_test_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Show a matrix figure for each correlation matrix (off by default, as it can be many)
SHOW_CORRELATION_MATRICES = False

# Group table location (update if needed)
RANDOMIZATION_XLSX_PATH = "C:/Users/amirh/Downloads/RandomizationTable.xlsx"
report_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "timeseries_GSR_off", "scrubbing_report.csv")

# Which sessions to compare? (substring matching, case-insensitive)
# e.g., "MRI1" or "S1" for baseline; "MRI2" or "S2" for follow-up
BASELINE_SESSION_KEYWORDS = ("MRI1", "S1")
FOLLOWUP_SESSION_KEYWORDS = ("MRI2", "S2")

# Column names inside the randomization Excel
RANDOMIZATION_SUBJECT_COLUMN = "SubID"
RANDOMIZATION_GROUP_COLUMN = "Group_Simbol"
KETAMINE_GROUP_SYMBOLS = ("A",)   # which symbols are ketamine
CONTROL_GROUP_SYMBOLS = ("C",)    # which symbols are control

# Manual control for group analysis (auto-fallback if groups missing stays in code)
ANALYZE_BY_GROUP = False  # Disabled for single-subject 3-session analysis

# Group colors for A/B/C (UNKNOWN = gray)
GROUP_COLOR = {
    "A": "#d62728",   # red
    "B": "#ff7f0e",   # orange
    "C": "#1f77b4",   # blue
    "UNKNOWN": "#7f7f7f",
}

# =============================================================================
# Utilities
# =============================================================================
def filter_correlation_matrices_by_fd_motion_threshold(report_path, correlation_matrices):
    try:
        scrub_report = pd.read_csv(report_path)
    except FileNotFoundError:
        print(f"Warning: Scrubbing report not found at {report_path}, skipping motion filtering")
        return correlation_matrices
    except Exception as e:
        print(f"Warning: Error reading scrubbing report: {e}, skipping motion filtering")
        return correlation_matrices
    
    if "scrubbed_volumes" not in scrub_report.columns:
        print("Warning: 'scrubbed_volumes' column not found in report, skipping motion filtering")
        return correlation_matrices
    
    over_scrubbed_volumes = scrub_report[scrub_report["scrubbed_volumes"] > scrubbed_volumes_threshold]
    subject_and_session_to_delete = set(zip(over_scrubbed_volumes["subject"], over_scrubbed_volumes["session"]))

    def parse_subject_session(key):
        base = os.path.basename(key).replace("_aal_ts.csv", "")
        parts = base.split("_")
        return parts[0], parts[1]  # ('sub-010', 'ses-MRI1')

    deleted_str = "none" if not subject_and_session_to_delete else ", ".join(
        f"{sub}_{ses}" for sub, ses in sorted(subject_and_session_to_delete)
    )
    print(f"Deleted subjects and session: {deleted_str}")

    return {
        key: mat
        for key, mat in correlation_matrices.items()
        if (parse_subject_session(key) not in subject_and_session_to_delete)
    }


def load_group_table(
        xlsx_path: str,
        subject_col: str = "SubID",
        group_col: str = "Group_Simbol",
        ketamine_groups=("A", "B"),
        control_groups=("C",),
):
    try:
        data_frame = pd.read_excel(xlsx_path)
    except FileNotFoundError:
        print(f"Error: Group table file not found: {xlsx_path}")
        return {}, pd.DataFrame()
    except Exception as e:
        print(f"Error reading group table: {e}")
        return {}, pd.DataFrame()

    def normalize_subject_id(value: str) -> str:
        value = str(value).strip()
        if re.match(r"sub-\d+", value, flags=re.IGNORECASE):
            digits = re.findall(r"\d+", value)
            return f"sub-{int(digits[0]):03d}" if digits else value
        match = re.match(r".*?(\d+)", value)
        return f"sub-{int(match.group(1)):03d}" if match else value

    data_frame["_subject_norm"] = data_frame[subject_col].apply(normalize_subject_id)
    data_frame["_group_norm"] = data_frame[group_col].astype(str).str.strip().str.upper()

    subject_to_group = {}
    for _, row in data_frame.iterrows():
        group_symbol = row["_group_norm"]
        if group_symbol in ketamine_groups:
            subject_to_group[row["_subject_norm"]] = "ketamine"
        elif group_symbol in control_groups:
            subject_to_group[row["_subject_norm"]] = "control"
        else:
            subject_to_group[row["_subject_norm"]] = None

    return subject_to_group, data_frame


def subject_group_symbols(xlsx_path: str,
                          subject_col: str = "SubID",
                          group_col: str = "Group_Simbol") -> dict:
    """
    Returns dict: { 'sub-024': 'A', 'sub-031': 'C', ... } using the raw symbol from Excel.
    """
    try:
        df = pd.read_excel(xlsx_path)
    except FileNotFoundError:
        print(f"Error: Group symbols file not found: {xlsx_path}")
        return {}
    except Exception as e:
        print(f"Error reading group symbols: {e}")
        return {}

    def normalize_subject_id(value: str) -> str:
        value = str(value).strip()
        if re.match(r"sub-\d+", value, flags=re.IGNORECASE):
            digits = re.findall(r"\d+", value)
            return f"sub-{int(digits[0]):03d}" if digits else value
        match = re.match(r".*?(\d+)", value)
        return f"sub-{int(match.group(1)):03d}" if match else value

    df["_subject_norm"] = df[subject_col].apply(normalize_subject_id)
    df["_group_raw"]    = df[group_col].astype(str).str.strip().str.upper()
    return dict(zip(df["_subject_norm"], df["_group_raw"]))


def resolve_session_labels(
        amygdala_correlations: dict,
        baseline_keywords: tuple,
        followup_keywords: tuple,
) -> tuple:
    """
    Resolve which session labels in amygdala_correlations correspond to baseline and follow-up,
    using substring matching with the provided keyword tuples.

    Returns:
        (baseline_session_label, followup_session_label)
    """
    all_session_labels = set(session for (_, session) in amygdala_correlations.keys())

    def pick_label(keyword_tuple):
        keyword_tuple = tuple(k.lower() for k in keyword_tuple)
        for label in all_session_labels:
            lower_label = label.lower()
            if any(k in lower_label for k in keyword_tuple):
                return label
        return None

    baseline_label = pick_label(baseline_keywords)
    followup_label = pick_label(followup_keywords)

    if not baseline_label or not followup_label:
        raise ValueError(
            f"Could not resolve sessions. Found labels: {sorted(all_session_labels)}. "
            f"Baseline keywords: {baseline_keywords}, Follow-up keywords: {followup_keywords}"
        )
    return baseline_label, followup_label


def _order_sessions(all_session_labels):
    """
    Returns a sensible session order. Prefers MRI1<MRI2<MRI3<S1<S2<S3>, else falls back to lexicographic.
    """
    labels = list(all_session_labels)
    priority = {  # lower is earlier
        "mri1": 1, "mri2": 2, "mri3": 3,
        "s1": 1,   "s2": 2,   "s3": 3,
    }
    def key(lbl: str):
        l = lbl.lower().replace("ses-", "")
        return (priority.get(l, 99), l)
    return sorted(labels, key=key)

# =============================================================================
# Data I/O and correlation extraction
# =============================================================================
def compute_pearson_correlations(project_root: str) -> dict:
    """
    Create correlation matrices (Pandas DataFrame) from each *_aal_ts.csv found in project_root.

    Returns:
        dict[str, pd.DataFrame]: mapping from filename -> correlation matrix
    """
    correlation_matrices = {}

    for ts_file in os.listdir(project_root):
        if not ts_file.endswith(".csv"):
            continue
        file_path = os.path.join(project_root, ts_file)
        try:
            data_frame = pd.read_csv(file_path)
            if "Amygdala_L" in data_frame.columns and "Amygdala_R" in data_frame.columns:
                correlation_matrix = data_frame.corr()
                correlation_matrices[ts_file] = correlation_matrix
            else:
                print(f"Warning: {ts_file} missing amygdala columns (Amygdala_L / Amygdala_R)")
        except Exception as error:
            print(f"Error processing {ts_file}: {error}")

    if SHOW_CORRELATION_MATRICES:
        for filename, corr_mat in correlation_matrices.items():
            plot_matrix(corr_mat, vmax=0.8, vmin=-0.8, colorbar=True)
            plt.title(filename, fontsize=10)
            plt.tight_layout()
            plt.show()
            plt.close()

    return correlation_matrices


def extract_amygdala_correlations(correlation_matrices: dict) -> dict:
    amygdala_correlations = {}

    for file_name, corr_df in correlation_matrices.items():
        left_amygdala_corr = corr_df.loc["Amygdala_L", :].drop("Amygdala_L")
        right_amygdala_corr = corr_df.loc["Amygdala_R", :].drop("Amygdala_R")

        # Expect something like: sub-024_ses-MRI1_aal_ts.csv
        name_without_suffix = file_name.replace("_aal_ts.csv", "")
        name_parts = name_without_suffix.split("_")
        subject_id = name_parts[0]  # e.g., 'sub-024'
        session_label = name_parts[1]  # e.g., 'ses-MRI1'

        amygdala_correlations[(subject_id, session_label)] = {
            "Amygdala_L": left_amygdala_corr,
            "Amygdala_R": right_amygdala_corr,
        }

    return amygdala_correlations


# =============================================================================
# Build tidy DF for timelines and plot with group colors
# =============================================================================
def build_seed_target_long_df(
    amygdala_correlations: dict,
    seed: str,
    target: str,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns:
      ['subject','session','seed','target','pearson_r']
    One row per subject-session with the Pearson correlation taken from the matrix.
    """
    rows = []
    all_sessions = set(session for (_, session) in amygdala_correlations.keys())
    sessions = _order_sessions(all_sessions)

    for (sub, ses), seeds in amygdala_correlations.items():
        series = seeds.get(seed)
        if series is None:
            continue
        if target not in series.index:
            continue
        r = float(series[target])
        rows.append({
            "subject": sub,
            "session": ses,
            "seed": seed,
            "target": target,
            "pearson_r": r
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["session"] = pd.Categorical(df["session"], categories=sessions, ordered=True)
    df = df.sort_values(["subject","session"])
    return df


def plot_subject_timelines_by_group(df_long: pd.DataFrame,
                                    subj_to_group_symbol: dict,
                                    title: str,
                                    out_png: str) -> None:
    """
    Line chart: x=session, y=pearson_r; one line per subject, colored by group (A/B/C).
    Each line is annotated with the subject ID above its last data point.
    """
    if df_long is None or df_long.empty:
        print(f"No data to plot for {title}")
        return

    # attach group symbols
    df_long = df_long.copy()
    df_long["group"] = df_long["subject"].map(lambda s: subj_to_group_symbol.get(s, "UNKNOWN"))

    plt.figure(figsize=(12, 7))

    for (sub, gsym), sub_df in df_long.groupby(["subject","group"]):
        color = GROUP_COLOR.get(gsym, GROUP_COLOR["UNKNOWN"])
        x = sub_df["session"].astype(str)
        y = sub_df["pearson_r"].values

        # plot line
        plt.plot(x, y, marker="o", linewidth=2.0, alpha=0.9, color=color)

        # annotate subject ID above the last point
        if len(x) > 0:
            plt.text(x.iloc[-1], y[-1] + 0.02, sub,
                     fontsize=8, color=color, ha="center")

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.ylabel("Pearson r")
    plt.xlabel("Session")
    plt.title(title)

    # Group legend (only A/B/C, not subjects)
    present_groups = [g for g in sorted(df_long["group"].unique()) if g in GROUP_COLOR]
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=GROUP_COLOR[g], lw=3, label=f"Group {g}")
               for g in present_groups if g != "UNKNOWN"]
    if "UNKNOWN" in df_long["group"].unique():
        handles.append(Line2D([0],[0], color=GROUP_COLOR["UNKNOWN"], lw=3, label="Group ?"))
    if handles:
        plt.legend(handles=handles, title="Groups", loc="upper left", bbox_to_anchor=(1.02,1))

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Save tidy data (now includes 'group')
    csv_path = out_png.replace(".png", ".csv")
    df_long.to_csv(csv_path, index=False)
    print(f"Saved: {out_png}\nSaved data: {csv_path}")


# =============================================================================
# Within-subject change (baseline vs follow-up)
# =============================================================================
def analyze_session_changes(
        amygdala_correlations: dict,
        baseline_session_keywords: tuple,
        followup_session_keywords: tuple,
) -> pd.DataFrame:
    """
    For each (seed, region), compute paired t-test between follow-up and baseline across
    subjects that have both sessions.

    Returns:
        pd.DataFrame with columns:
        ['region','seed','mri1_mean','mri3_mean','mean_difference','t_statistic','p_value','n_subjects']
    """
    baseline_label, followup_label = resolve_session_labels(
        amygdala_correlations, baseline_session_keywords, followup_session_keywords
    )
    print(f"Comparing {baseline_label} (baseline) vs {followup_label} (follow-up)")

    session_to_subjects = {baseline_label: {}, followup_label: {}}
    for (subject_id, session_label), seed_series_dict in amygdala_correlations.items():
        if session_label in session_to_subjects:
            session_to_subjects[session_label][subject_id] = seed_series_dict

    common_subjects = set(session_to_subjects[baseline_label].keys()) & set(
        session_to_subjects[followup_label].keys()
    )
    print(f"Subjects with both sessions: {len(common_subjects)}")
    if len(common_subjects) < 1:
        print("Need at least 1 subject for comparison")
        return pd.DataFrame([])

    all_regions = set()
    for subject_id in common_subjects:
        all_regions.update(session_to_subjects[baseline_label][subject_id]["Amygdala_L"].index)
        all_regions.update(session_to_subjects[baseline_label][subject_id]["Amygdala_R"].index)

    print(f"Total unique regions found: {len(all_regions)}")
    print(f"Sample regions: {list(all_regions)[:5]}")

    results_rows = []
    for region_name in all_regions:
        for seed_name in ["Amygdala_L", "Amygdala_R"]:
            baseline_values = []
            followup_values = []

            for subject_id in common_subjects:
                baseline_series = session_to_subjects[baseline_label][subject_id][seed_name]
                followup_series = session_to_subjects[followup_label][subject_id][seed_name]
                if region_name in baseline_series and region_name in followup_series:
                    baseline_values.append(float(baseline_series[region_name]))
                    followup_values.append(float(followup_series[region_name]))

            if len(baseline_values) >= 1 and len(followup_values) >= 1:
                t_statistic, p_value = ttest_rel(baseline_values, followup_values)
                mean_difference = np.mean(followup_values) - np.mean(baseline_values)
                results_rows.append({
                    "region": region_name,
                    "seed": seed_name,
                    "mri1_mean": np.mean(baseline_values),
                    "mri3_mean": np.mean(followup_values),
                    "mean_difference": mean_difference,
                    "t_statistic": t_statistic,
                    "p_value": p_value,
                    "n_subjects": len(baseline_values),
                })

    results_data_frame = pd.DataFrame(results_rows)
    return results_data_frame


def plot_session_changes(results_data_frame: pd.DataFrame, output_folder: str) -> None:
    """
    Create bar plots for regions showing larger changes (or top-10 by |t| if none pass thresholds).
    """
    if results_data_frame is None or results_data_frame.empty:
        print("No results to plot")
        return

    significant_results = results_data_frame[
        (results_data_frame["p_value"] < 0.05) & (results_data_frame["t_statistic"].abs() > 2.0)
    ].copy()

    if significant_results.empty:
        print("No significant changes found (p < 0.05, |t| > 2.0), showing top 10 by |t|")
        significant_results = results_data_frame.reindex(
            results_data_frame["t_statistic"].abs().sort_values(ascending=False).index
        ).head(10).copy()

    for seed_name in ["Amygdala_L", "Amygdala_R"]:
        seed_results = significant_results[significant_results["seed"] == seed_name].copy()
        if seed_results.empty:
            print(f"No data for {seed_name}")
            continue

        seed_results["abs_t_stat"] = seed_results["t_statistic"].abs()
        seed_results = seed_results.sort_values("abs_t_stat", ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        colors = ["red" if val > 0 else "blue" for val in seed_results["t_statistic"]]
        ax1.barh(range(len(seed_results)), seed_results["t_statistic"], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(seed_results)))
        ax1.set_yticklabels(
            [r[:25] + "..." if len(r) > 25 else r for r in seed_results["region"]]
        )
        ax1.set_xlabel("T-Statistic")
        ax1.set_title(f"{seed_name} - T-Statistics (Follow-up vs Baseline)")
        ax1.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax1.grid(True, alpha=0.3)

        for i, (_, row) in enumerate(seed_results.iterrows()):
            p_text = f"p={row['p_value']:.3f}"
            ax1.text(row["t_statistic"], i, f" {p_text}", va="center", fontsize=8)

        ax2.barh(range(len(seed_results)), -np.log10(seed_results["p_value"]), color=colors, alpha=0.7)
        ax2.set_yticks(range(len(seed_results)))
        ax2.set_yticklabels(
            [r[:25] + "..." if len(r) > 25 else r for r in seed_results["region"]]
        )
        ax2.set_xlabel("-log10(p-value)")
        ax2.set_title(f"{seed_name} - Statistical Significance")
        ax2.axvline(x=-np.log10(0.05), color="red", linestyle="--", alpha=0.7, label="p=0.05")
        ax2.axvline(x=-np.log10(0.01), color="orange", linestyle="--", alpha=0.7, label="p=0.01")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figure_path = os.path.join(output_folder, f"{seed_name}_session_changes.png")
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        print(f"\n=== {seed_name} Session Changes ===")
        print("Top 10 regions by absolute t-statistic:")
        for _, row in seed_results.head(10).iterrows():
            direction_text = "increased" if row["mean_difference"] > 0 else "decreased"
            print(f"  {row['region']}: t={row['t_statistic']:.3f}, p={row['p_value']:.4f}, {direction_text}")

    print("\n=== Summary Statistics ===")
    for seed_name in ["Amygdala_L", "Amygdala_R"]:
        seed_rows = results_data_frame[results_data_frame["seed"] == seed_name]
        print(f"\n{seed_name}:")
        print(f"  Total regions tested: {len(seed_rows)}")
        print(f"  Significant changes (p < 0.05): {len(seed_rows[seed_rows['p_value'] < 0.05])}")
        print(f"  Large changes (|t| > 2.0): {len(seed_rows[seed_rows['t_statistic'].abs() > 2.0])}")
        print(f"  Mean t-statistic: {seed_rows['t_statistic'].mean():.3f}")
        print(f"  Mean p-value: {seed_rows['p_value'].mean():.3f}")

    results_csv = os.path.join(output_folder, "session_changes_results.csv")
    results_data_frame.to_csv(results_csv, index=False)
    print(f"\nSaved detailed results to: {results_csv}")


# =============================================================================
# Between-group deltas (follow-up - baseline) and volcano plot
# =============================================================================
def cohen_d_independent(group_x, group_y) -> float:
    """
    Cohen's d for independent samples (pooled SD).
    """
    array_x = np.asarray(group_x)
    array_y = np.asarray(group_y)
    n_x, n_y = len(array_x), len(array_y)
    if n_x < 2 or n_y < 2:
        return np.nan
    var_x, var_y = np.var(array_x, ddof=1), np.var(array_y, ddof=1)
    
    # Check for valid denominator before computing pooled SD
    if (n_x + n_y - 2) <= 0:
        return np.nan
    
    pooled_sd = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))
    
    if np.isnan(pooled_sd) or pooled_sd == 0:
        return np.nan
    return (np.mean(array_x) - np.mean(array_y)) / pooled_sd


def compute_subject_deltas(
        amygdala_correlations: dict,
        baseline_session_keywords: tuple,
        followup_session_keywords: tuple,
) -> dict:
    baseline_label, followup_label = resolve_session_labels(
        amygdala_correlations, baseline_session_keywords, followup_session_keywords
    )

    session_to_subjects = {baseline_label: {}, followup_label: {}}
    for (subject_id, session_label), seed_series_dict in amygdala_correlations.items():
        if session_label in session_to_subjects:
            session_to_subjects[session_label][subject_id] = seed_series_dict

    common_subjects = set(session_to_subjects[baseline_label].keys()) & set(
        session_to_subjects[followup_label].keys()
    )
    if not common_subjects:
        raise ValueError("No subjects with both baseline and follow-up sessions.")

    all_regions = set()
    for subject_id in common_subjects:
        all_regions.update(session_to_subjects[baseline_label][subject_id]["Amygdala_L"].index)

    deltas_by_seed_region = {}
    for seed_name in ["Amygdala_L", "Amygdala_R"]:
        for region_name in all_regions:
            key = (seed_name, region_name)
            deltas_by_seed_region[key] = {}
            for subject_id in common_subjects:
                baseline_series = session_to_subjects[baseline_label][subject_id][seed_name]
                followup_series = session_to_subjects[followup_label][subject_id][seed_name]
                if region_name in baseline_series and region_name in followup_series:
                    delta_value = float(followup_series[region_name]) - float(baseline_series[region_name])
                    deltas_by_seed_region[key][subject_id] = delta_value

    return deltas_by_seed_region


def between_group_tests(
        amygdala_correlations: dict,
        randomization_xlsx_path: str,
        output_folder: str,
        baseline_session_keywords: tuple,
        followup_session_keywords: tuple,
        subject_col: str = "SubID",
        group_col: str = "Group_Simbol",
) -> tuple:
    """
    Welch's t-test comparing (follow-up - baseline) between ketamine and control groups
    for each (seed, region). Also computes Cohen's d.

    Returns:
        (pd.DataFrame, bool): (results sorted by p-value, had_groups flag)
    """
    subject_to_group, _ = load_group_table(
        randomization_xlsx_path,
        subject_col=subject_col,
        group_col=group_col,
        ketamine_groups=KETAMINE_GROUP_SYMBOLS,
        control_groups=CONTROL_GROUP_SYMBOLS,
    )

    group_values = set(v for v in subject_to_group.values() if v is not None)
    has_ket = "ketamine" in group_values
    has_ctrl = "control" in group_values
    had_groups = has_ket and has_ctrl

    if not had_groups:
        print("No valid ketamine/control group symbols found in the randomization file.")
        return pd.DataFrame([]), False

    deltas_by_seed_region = compute_subject_deltas(
        amygdala_correlations, baseline_session_keywords, followup_session_keywords
    )

    rows = []
    for (seed_name, region_name), subject_to_delta in deltas_by_seed_region.items():
        ketamine_deltas = []
        control_deltas = []
        for subject_id, delta_value in subject_to_delta.items():
            group_label = subject_to_group.get(subject_id)
            if group_label == "ketamine":
                ketamine_deltas.append(delta_value)
            elif group_label == "control":
                control_deltas.append(delta_value)

        if len(ketamine_deltas) >= 2 and len(control_deltas) >= 2:
            t_statistic, p_value = ttest_ind(ketamine_deltas, control_deltas, equal_var=False)
            cohens_d = cohen_d_independent(ketamine_deltas, control_deltas)
            rows.append({
                "seed": seed_name,
                "region": region_name,
                "n_ketamine": len(ketamine_deltas),
                "n_control": len(control_deltas),
                "mean_delta_ketamine": np.mean(ketamine_deltas),
                "mean_delta_control": np.mean(control_deltas),
                "mean_diff_(ket-control)": np.mean(ketamine_deltas) - np.mean(control_deltas),
                "t_statistic": t_statistic,
                "p_value": p_value,
                "cohens_d": cohens_d,
            })

    results_data_frame = pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame([])
    if not results_data_frame.empty:
        out_path = os.path.join(output_folder, "between_group_followup_minus_baseline_amygdala.csv")
        results_data_frame.to_csv(out_path, index=False)
        print(f"Saved between-group results to: {out_path}")
    else:
        print("No regions had sufficient subjects in both groups for testing.")
    return results_data_frame, True


def volcano_plot(results_data_frame: pd.DataFrame, output_folder: str) -> None:
    """
    Simple volcano plot: x = mean difference (ketamine - control), y = -log10(p).
    One PNG per seed.
    """
    if results_data_frame is None or results_data_frame.empty:
        return

    for seed_name in results_data_frame["seed"].unique():
        per_seed = results_data_frame[results_data_frame["seed"] == seed_name].copy()
        if per_seed.empty:
            continue
        per_seed["neglog10p"] = -np.log10(per_seed["p_value"])
        plt.figure(figsize=(8, 6))
        plt.scatter(per_seed["mean_diff_(ket-control)"], per_seed["neglog10p"])
        plt.axhline(-np.log10(0.05), linestyle="--")
        plt.axvline(0.0, linestyle="--")
        plt.xlabel("Mean Δ difference (ketamine − control)")
        plt.ylabel("−log10 p")
        plt.title(f"{seed_name}: follow-up − baseline group difference")
        plt.tight_layout()
        fig_path = os.path.join(output_folder, f"volcano_{seed_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


# =============================================================================
# Visualization on atlas (optional best-effort label matching)
# =============================================================================
def create_brain_visualization(results_data_frame: pd.DataFrame, output_folder: str) -> None:
    """
    Attempt to visualize significant t-statistics per region on Harvard-Oxford atlas.
    Falls back to a bar plot if atlas mapping fails.
    """
    if results_data_frame is None or results_data_frame.empty:
        print("No results to visualize")
        return

    try:
        print("Loading Harvard-Oxford atlas...")
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_labels = atlas.labels  # list of strings

        for seed_name in ["Amygdala_L", "Amygdala_R"]:
            seed_rows = results_data_frame[results_data_frame["seed"] == seed_name].copy()
            if seed_rows.empty:
                print(f"No data for {seed_name}")
                continue

            filtered = seed_rows[
                (seed_rows["p_value"] < 0.05) & (seed_rows["t_statistic"].abs() > 2.0)
            ].copy()
            if filtered.empty:
                print(f"No significant changes for {seed_name}, showing top 10 by |t|")
                filtered = seed_rows.reindex(
                    seed_rows["t_statistic"].abs().sort_values(ascending=False).index
                ).head(10).copy()

            fig, ax = plt.subplots(figsize=(12, 8))
            filtered_sorted = filtered.sort_values("t_statistic", ascending=True)
            colors = ["red" if x > 0 else "blue" for x in filtered_sorted["t_statistic"]]
            ax.barh(range(len(filtered_sorted)), filtered_sorted["t_statistic"], color=colors, alpha=0.7)
            ax.set_yticks(range(len(filtered_sorted)))
            ax.set_yticklabels(
                [r[:30] + "..." if len(r) > 30 else r for r in filtered_sorted["region"]]
            )
            ax.set_xlabel("T-Statistic")
            ax.set_title(f"{seed_name} - Brain Regions with Connectivity Changes")
            ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path_bar = os.path.join(
                output_folder, f"{seed_name.replace(' ', '_')}_connectivity_changes.png"
            )
            plt.savefig(path_bar, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
            print(f"Created connectivity changes plot for {seed_name}")
    except Exception as error:
        print(f"Error creating brain visualization: {error}")
        print("This might be due to missing nilearn or atlas data")


# =============================================================================
# Convenience: plot between-group top regions (existing)
# =============================================================================
def plot_between_group_top_regions(between_group_results):
    for seed_val, group_df in between_group_results.groupby("seed"):
        top_10_regions = group_df.sort_values("p_value").head(10).copy()
        colors = ["red" if diff > 0 else "blue" for diff in top_10_regions["mean_diff_(ket-control)"]]

        plt.figure(figsize=(10, 6))
        plt.barh(top_10_regions["region"], top_10_regions["p_value"], color=colors)
        plt.xlabel("p-value")
        plt.ylabel("Region")
        plt.xlim(0, 0.2)
        plt.title(f"Top 10 Regions by Significance ({seed_val})")
        plt.gca().invert_yaxis()
        plt.grid(True, axis="x", alpha=0.3)

        for i, p in enumerate(top_10_regions["p_value"]):
            plt.text(-np.log10(p) + 0.05, i, f"p={p:.3e}", va="center", fontsize=8)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Ketamine Δ > Control Δ'),
            Patch(facecolor='blue', label='Ketamine Δ < Control Δ')
        ]
        plt.legend(handles=legend_elements, title="Group difference", loc="lower right")

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_FOLDER, f"top10_between_group_{seed_val}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        print(f"Saved plot for {seed_val} to: {output_path}")


# =============================================================================
# 3D Brain Visualization Functions
# (fetch_aal_spm12, fetch_fsaverage_surfaces, map_roi_to_indices imported from utils)
# =============================================================================

def sample_labels(atlas_img, white_mesh, pial_mesh):
    """Sample AAL labels onto surface"""
    try:
        lbl = surface.vol_to_surf(
            atlas_img, pial_mesh,
            inner_mesh=white_mesh, kind="line", n_samples=25, interpolation="nearest"
        )
    except TypeError:
        lbl = surface.vol_to_surf(atlas_img, pial_mesh)
    lbl = np.rint(np.asarray(lbl)).astype(np.int32)
    lbl[lbl < 0] = 0
    return lbl


def build_vertex_data(surf_mesh, labels_on_vertices, roi_index_to_val: dict, index_to_name: dict = None):
    """Build vertex data for surface rendering with hover labels"""
    coords, faces = surface.load_surf_mesh(surf_mesh)
    intens = np.zeros(coords.shape[0], dtype=float)
    names = np.empty(coords.shape[0], dtype=object)
    
    for idx_val, v in roi_index_to_val.items():
        mask = labels_on_vertices == idx_val
        if np.any(mask):
            intens[mask] = 0.0 if v is None or np.isnan(v) else float(v)
            if index_to_name is not None:
                names[mask] = index_to_name.get(idx_val, "")
            else:
                names[mask] = ""
    
    return coords, faces, intens, names


def compute_group_amygdala_correlation_changes_with_pvalues(
        amygdala_correlations: dict,
        group_symbol: str,
        baseline_keywords: tuple,
        followup_keywords: tuple,
        seed_name: str,
        subj_to_group_symbol: dict
) -> dict:
    """
    Compute mean correlation changes (ses2 - ses1) and p-values for a specific group and seed.
    
    Returns:
        dict mapping region_name -> {'mean_change': float, 'p_value': float}
    """
    baseline_label, followup_label = resolve_session_labels(
        amygdala_correlations, baseline_keywords, followup_keywords
    )
    
    # Get subjects in this group
    group_subjects = [sub for sub, grp in subj_to_group_symbol.items() if grp == group_symbol]
    
    if not group_subjects:
        print(f"Warning: No subjects found in group {group_symbol}")
        return {}
    
    # Get data for subjects in this group that have both sessions
    session_to_subjects = {baseline_label: {}, followup_label: {}}
    for (subject_id, session_label), seed_series_dict in amygdala_correlations.items():
        if subject_id in group_subjects and session_label in [baseline_label, followup_label]:
            if session_label not in session_to_subjects:
                session_to_subjects[session_label] = {}
            session_to_subjects[session_label][subject_id] = seed_series_dict
    
    common_subjects = set(session_to_subjects[baseline_label].keys()) & set(
        session_to_subjects[followup_label].keys()
    )
    
    if not common_subjects:
        print(f"Warning: No subjects in group {group_symbol} have both {baseline_label} and {followup_label}")
        return {}
    
    # Collect all regions
    all_regions = set()
    for subject_id in common_subjects:
        if seed_name in session_to_subjects[baseline_label][subject_id]:
            all_regions.update(session_to_subjects[baseline_label][subject_id][seed_name].index)
    
    # Compute changes and p-values for each region
    region_results = {}
    for region_name in all_regions:
        baseline_values = []
        followup_values = []
        
        for subject_id in common_subjects:
            if seed_name not in session_to_subjects[baseline_label][subject_id]:
                continue
            if seed_name not in session_to_subjects[followup_label][subject_id]:
                continue
            
            baseline_series = session_to_subjects[baseline_label][subject_id][seed_name]
            followup_series = session_to_subjects[followup_label][subject_id][seed_name]
            
            if region_name in baseline_series and region_name in followup_series:
                baseline_values.append(float(baseline_series[region_name]))
                followup_values.append(float(followup_series[region_name]))
        
        if len(baseline_values) >= 2 and len(followup_values) >= 2:
            # Paired t-test
            t_stat, p_value = ttest_rel(baseline_values, followup_values)
            mean_change = np.mean(followup_values) - np.mean(baseline_values)
            region_results[region_name] = {
                'mean_change': mean_change,
                'p_value': p_value
            }
    
    return region_results


def compute_between_group_pvalues(
        amygdala_correlations: dict,
        baseline_keywords: tuple,
        followup_keywords: tuple,
        seed_name: str,
        subj_to_group_symbol: dict
) -> dict:
    """
    Compute p-values for between-group differences (Group A - Group C) in correlation changes.
    
    Returns:
        dict mapping region_name -> {'p_value': float, 'mean_diff_a_minus_c': float}
    """
    baseline_label, followup_label = resolve_session_labels(
        amygdala_correlations, baseline_keywords, followup_keywords
    )
    
    # Get subjects in each group
    group_a_subjects = [sub for sub, grp in subj_to_group_symbol.items() if grp == "A"]
    group_c_subjects = [sub for sub, grp in subj_to_group_symbol.items() if grp == "C"]
    
    if not group_a_subjects or not group_c_subjects:
        print("Warning: Need both Group A and Group C subjects for between-group comparison")
        return {}
    
    # Compute deltas for each group
    def compute_deltas_for_group(group_subjects):
        session_to_subjects = {baseline_label: {}, followup_label: {}}
        for (subject_id, session_label), seed_series_dict in amygdala_correlations.items():
            if subject_id in group_subjects and session_label in [baseline_label, followup_label]:
                if session_label not in session_to_subjects:
                    session_to_subjects[session_label] = {}
                session_to_subjects[session_label][subject_id] = seed_series_dict
        
        common_subjects = set(session_to_subjects[baseline_label].keys()) & set(
            session_to_subjects[followup_label].keys()
        )
        
        deltas_by_region = {}
        for subject_id in common_subjects:
            if seed_name not in session_to_subjects[baseline_label][subject_id]:
                continue
            if seed_name not in session_to_subjects[followup_label][subject_id]:
                continue
            
            baseline_series = session_to_subjects[baseline_label][subject_id][seed_name]
            followup_series = session_to_subjects[followup_label][subject_id][seed_name]
            
            for region_name in baseline_series.index:
                if region_name in followup_series.index:
                    if region_name not in deltas_by_region:
                        deltas_by_region[region_name] = []
                    delta = float(followup_series[region_name]) - float(baseline_series[region_name])
                    deltas_by_region[region_name].append(delta)
        
        return deltas_by_region
    
    deltas_a = compute_deltas_for_group(group_a_subjects)
    deltas_c = compute_deltas_for_group(group_c_subjects)
    
    # Compute between-group p-values
    region_results = {}
    all_regions = set(deltas_a.keys()) & set(deltas_c.keys())
    
    for region_name in all_regions:
        if len(deltas_a[region_name]) >= 2 and len(deltas_c[region_name]) >= 2:
            # Independent samples t-test
            t_stat, p_value = ttest_ind(deltas_a[region_name], deltas_c[region_name], equal_var=False)
            mean_diff_a_minus_c = np.mean(deltas_a[region_name]) - np.mean(deltas_c[region_name])
            region_results[region_name] = {
                'p_value': p_value,
                'mean_diff_a_minus_c': mean_diff_a_minus_c
            }
    
    return region_results


def create_3d_brain_visualization_pvalues(
        region_results: dict,
        output_path: str,
        title: str,
        seed_name: str,
        value_key: str = 'p_value',
        value_label: str = 'P-Value'
):
    """
    Create 3D brain visualization showing p-values (or other values) for correlation changes.
    
    Parameters:
    -----------
    region_results : dict
        Mapping from region_name -> dict with value_key (e.g., 'p_value')
    output_path : str
        Path to save HTML file
    title : str
        Title for the visualization
    seed_name : str
        Seed name (e.g., 'Amygdala_L')
    value_key : str
        Key in region_results[region] to use for the value (default: 'p_value')
    value_label : str
        Label for the colorbar
    """
    print(f"\nCreating 3D visualization: {title}")
    
    # Load atlas and surfaces
    print("Loading atlas and brain surfaces...")
    _, atlas_img, atlas, name_to_index = fetch_aal_spm12()
    pial_L, pial_R, white_L, white_R = fetch_fsaverage_surfaces()
    
    # Sample labels onto surfaces
    lbl_L = sample_labels(atlas_img, white_L, pial_L)
    lbl_R = sample_labels(atlas_img, white_R, pial_R)
    
    # Map region results to atlas indices
    roi_to_val = {}
    for region_name, results_dict in region_results.items():
        if value_key in results_dict:
            val = results_dict[value_key]
            if not np.isnan(val):
                # For p-values, use -log10 for better visualization
                if value_key == 'p_value':
                    roi_to_val[region_name] = -np.log10(max(val, 1e-10))  # Avoid log(0)
                else:
                    roi_to_val[region_name] = val
    
    diff_map = map_roi_to_indices(roi_to_val, name_to_index)
    
    # Create reverse mapping: index -> name (for hover tooltips)
    index_to_name = {idx: name for name, idx in name_to_index.items()}
    
    # Build vertex data
    coords_L, faces_L, intens_L, names_L = build_vertex_data(pial_L, lbl_L, diff_map, index_to_name)
    coords_R, faces_R, intens_R, names_R = build_vertex_data(pial_R, lbl_R, diff_map, index_to_name)
    
    # Get color scale range
    all_intens = np.concatenate([intens_L[intens_L != 0], intens_R[intens_R != 0]])
    if len(all_intens) > 0:
        if value_key == 'p_value':
            vmax = np.percentile(np.abs(all_intens), 95)
            vmin = 0
        elif value_key == 'mean_change':
            vmax = np.percentile(np.abs(all_intens), 95)
            vmin = -vmax  # Symmetric for correlation changes
        else:
            vmax = np.percentile(np.abs(all_intens), 95)
            vmin = -vmax
    else:
        vmax = 3
        vmin = 0 if value_key == 'p_value' else -3
    
    print(f"Color scale range: [{vmin:.3f}, {vmax:.3f}]")
    
    # Create figure
    fig = go.Figure()
    
    # Configuration
    HEMISPHERE_SHIFT = 55
    BACKGROUND_COLOR = "black"
    TEXT_COLOR = "white"
    OPACITY = 1.0
    
    # Shift hemispheres
    coords_L_shift = coords_L.copy()
    coords_L_shift[:, 0] -= HEMISPHERE_SHIFT
    coords_R_shift = coords_R.copy()
    coords_R_shift[:, 0] += HEMISPHERE_SHIFT
    
    # Create customdata arrays for hover (region names and values)
    custom_L = np.stack([
        names_L.astype(str),
        intens_L
    ], axis=1)
    custom_R = np.stack([
        names_R.astype(str),
        intens_R
    ], axis=1)
    
    # Hover template showing region name and value
    if value_key == 'p_value':
        hover_tmpl = "<b>%{customdata[0]}</b><br>-log10(P-Value): %{customdata[1]:.3f}<extra></extra>"
        colorscale = 'Reds'  # Use reds for p-values (higher = more significant)
    elif value_key == 'mean_change':
        hover_tmpl = "<b>%{customdata[0]}</b><br>Correlation Change: %{customdata[1]:.3f}<extra></extra>"
        colorscale = 'RdBu_r'  # Diverging colormap for correlation changes
    else:
        # Use string formatting to include value_label, but keep %{customdata[...]} as Plotly template
        hover_tmpl = f"<b>%{{customdata[0]}}</b><br>{value_label}: %{{customdata[1]:.3f}}<extra></extra>"
        colorscale = 'RdBu_r'
    
    lighting = dict(ambient=0.35, diffuse=0.7, specular=0.6, roughness=0.4)
    
    # Left hemisphere
    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:, 0], y=coords_L_shift[:, 1], z=coords_L_shift[:, 2],
        i=faces_L[:, 0], j=faces_L[:, 1], k=faces_L[:, 2],
        intensity=intens_L, cmin=vmin, cmax=vmax, colorscale=colorscale,
        intensitymode='vertex',
        opacity=OPACITY, lighting=lighting,
        showscale=True,
        customdata=custom_L, hovertemplate=hover_tmpl,
        colorbar=dict(title=f"{value_label}<br>(if p-value: -log10)", len=0.6, y=0.5)
    ))
    
    # Right hemisphere
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:, 0], y=coords_R_shift[:, 1], z=coords_R_shift[:, 2],
        i=faces_R[:, 0], j=faces_R[:, 1], k=faces_R[:, 2],
        intensity=intens_R, cmin=vmin, cmax=vmax, colorscale=colorscale,
        intensitymode='vertex',
        opacity=OPACITY, lighting=lighting,
        showscale=False,
        customdata=custom_R, hovertemplate=hover_tmpl
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=24)),
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
    )
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="inline", full_html=True, auto_open=False)
    print(f"Saved 3D visualization to: {output_path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=== Amygdala Seed Connectivity Analysis ===")
    correlation_matrices = compute_pearson_correlations(PROJECT_ROOT)
    print(f"Found {len(correlation_matrices)} correlation matrices before scrubbing filter")
    filtered_correlation_matrices = filter_correlation_matrices_by_fd_motion_threshold(
        report_path, correlation_matrices
    )
    if not filtered_correlation_matrices:
        print("No correlation matrices after filtering. Check your data files and scrubbing thresholds.")
        raise SystemExit(1)

    # ✅ Use the FILTERED matrices going forward
    amygdala_correlations = extract_amygdala_correlations(filtered_correlation_matrices)

    # === NEW: Subject timelines colored by randomized group (A/B/C) ===
    subj_to_group_symbol = subject_group_symbols(
        RANDOMIZATION_XLSX_PATH,
        subject_col=RANDOMIZATION_SUBJECT_COLUMN,
        group_col=RANDOMIZATION_GROUP_COLUMN,
    )
    # === ALSO: medial orbitofrontal targets (AAL labels) ===
    pairs = [
        ("Amygdala_L", "Frontal_Med_Orb_L", "timeline_Amygdala_L_to_Frontal_Med_Orb_L_by_group.png"),
        ("Amygdala_R", "Frontal_Med_Orb_R", "timeline_Amygdala_R_to_Frontal_Med_Orb_R_by_group.png"),
    ]

    for seed_name, target_name, filename in pairs:
        df_tmp = build_seed_target_long_df(amygdala_correlations, seed=seed_name, target=target_name)
        plot_subject_timelines_by_group(
            df_tmp,
            subj_to_group_symbol=subj_to_group_symbol,
            title=f"{seed_name} <-> {target_name} connectivity per subject (MRI1->MRI2->MRI3)",
            out_png=os.path.join(OUTPUT_FOLDER, filename),
        )
    # Left <-> Left (Amygdala_L <-> Hippocampus_L)
    df_LL = build_seed_target_long_df(amygdala_correlations, seed="Amygdala_L", target="Hippocampus_L")
    plot_subject_timelines_by_group(
        df_LL,
        subj_to_group_symbol=subj_to_group_symbol,
        title="Amygdala_L <-> Hippocampus_L connectivity per subject (MRI1->MRI2->MRI3)",
        out_png=os.path.join(OUTPUT_FOLDER, "timeline_Amygdala_L_to_Hippocampus_L_by_group.png"),
    )

    # Right <-> Right (Amygdala_R <-> Hippocampus_R)
    df_RR = build_seed_target_long_df(amygdala_correlations, seed="Amygdala_R", target="Hippocampus_R")
    plot_subject_timelines_by_group(
        df_RR,
        subj_to_group_symbol=subj_to_group_symbol,
        title="Amygdala_R <-> Hippocampus_R connectivity per subject (MRI1->MRI2->MRI3)",
        out_png=os.path.join(OUTPUT_FOLDER, "timeline_Amygdala_R_to_Hippocampus_R_by_group.png"),
    )

    # === Skip group analysis for single subject ===
    print("\nSingle-subject analysis - finding top connectivity changes...")

    # === NEW: 3-Session Task-Specific Analysis (REST and GLASS) ===
    print("\n" + "="*80)
    print("3-SESSION TASK-SPECIFIC ANALYSIS - TOP 20 CONNECTIVITY CHANGES")
    print("="*80)
    
    # Analyze REST task across 3 sessions
    print("\n" + "-"*80)
    print("ANALYZING REST TASK: Session 1 -> Session 2 -> Session 3")
    print("-"*80)
    
    # Filter for only REST task
    rest_correlations = {k: v for k, v in amygdala_correlations.items() 
                        if 'rest' in str(k[1]).lower()}
    
    print(f"Found {len(rest_correlations)} REST sessions: {list(rest_correlations.keys())}")
    
    if len(rest_correlations) >= 2:
        # Compare ses-1 vs ses-2
        print("\nREST: Session 1 vs Session 2")
        rest_1v2 = analyze_session_changes(
            amygdala_correlations=rest_correlations,
            baseline_session_keywords=("ses-1",),
            followup_session_keywords=("ses-2",),
        )
        if not rest_1v2.empty:
            rest_1v2.to_csv(os.path.join(OUTPUT_FOLDER, "rest_ses1_vs_ses2_changes.csv"), index=False)
            print(f"Saved: rest_ses1_vs_ses2_changes.csv")
        
        # Compare ses-2 vs ses-3
        print("\nREST: Session 2 vs Session 3")
        rest_2v3 = analyze_session_changes(
            amygdala_correlations=rest_correlations,
            baseline_session_keywords=("ses-2",),
            followup_session_keywords=("ses-3",),
        )
        if not rest_2v3.empty:
            rest_2v3.to_csv(os.path.join(OUTPUT_FOLDER, "rest_ses2_vs_ses3_changes.csv"), index=False)
            print(f"Saved: rest_ses2_vs_ses3_changes.csv")
        
        # Compare ses-1 vs ses-3 (overall change)
        print("\nREST: Session 1 vs Session 3 (Overall)")
        rest_1v3 = analyze_session_changes(
            amygdala_correlations=rest_correlations,
            baseline_session_keywords=("ses-1",),
            followup_session_keywords=("ses-3",),
        )
        if not rest_1v3.empty:
            # Sort by absolute change and get top 20
            rest_1v3['abs_change'] = rest_1v3['mean_difference'].abs()
            rest_1v3_sorted = rest_1v3.sort_values('abs_change', ascending=False)
            rest_1v3_top20 = rest_1v3_sorted.head(20)
            
            rest_1v3_sorted.to_csv(os.path.join(OUTPUT_FOLDER, "rest_ses1_vs_ses3_changes_all.csv"), index=False)
            rest_1v3_top20.to_csv(os.path.join(OUTPUT_FOLDER, "rest_ses1_vs_ses3_changes_TOP20.csv"), index=False)
            print(f"\nSaved: rest_ses1_vs_ses3_changes_all.csv")
            print(f"Saved: rest_ses1_vs_ses3_changes_TOP20.csv")
            print(f"\nTOP 20 REST Changes (Session 1 vs Session 3):")
            for idx, row in rest_1v3_top20.iterrows():
                print(f"  {row['seed']} -> {row['region']}: {row['mean_difference']:+.3f} (p={row['p_value']:.3f})")
    
    # Analyze GLASS task across 3 sessions
    print("\n" + "-"*80)
    print("ANALYZING GLASS TASK: Session 1 -> Session 2 -> Session 3")
    print("-"*80)
    
    # Filter for only GLASS task
    glass_correlations = {k: v for k, v in amygdala_correlations.items() 
                         if 'glass' in str(k[1]).lower()}
    
    print(f"Found {len(glass_correlations)} GLASS sessions: {list(glass_correlations.keys())}")
    
    if len(glass_correlations) >= 2:
        # Compare ses-1 vs ses-2
        print("\nGLASS: Session 1 vs Session 2")
        glass_1v2 = analyze_session_changes(
            amygdala_correlations=glass_correlations,
            baseline_session_keywords=("ses-1",),
            followup_session_keywords=("ses-2",),
        )
        if not glass_1v2.empty:
            glass_1v2.to_csv(os.path.join(OUTPUT_FOLDER, "glass_ses1_vs_ses2_changes.csv"), index=False)
            print(f"Saved: glass_ses1_vs_ses2_changes.csv")
        
        # Compare ses-2 vs ses-3
        print("\nGLASS: Session 2 vs Session 3")
        glass_2v3 = analyze_session_changes(
            amygdala_correlations=glass_correlations,
            baseline_session_keywords=("ses-2",),
            followup_session_keywords=("ses-3",),
        )
        if not glass_2v3.empty:
            glass_2v3.to_csv(os.path.join(OUTPUT_FOLDER, "glass_ses2_vs_ses3_changes.csv"), index=False)
            print(f"Saved: glass_ses2_vs_ses3_changes.csv")
        
        # Compare ses-1 vs ses-3 (overall change)
        print("\nGLASS: Session 1 vs Session 3 (Overall)")
        glass_1v3 = analyze_session_changes(
            amygdala_correlations=glass_correlations,
            baseline_session_keywords=("ses-1",),
            followup_session_keywords=("ses-3",),
        )
        if not glass_1v3.empty:
            # Sort by absolute change and get top 20
            glass_1v3['abs_change'] = glass_1v3['mean_difference'].abs()
            glass_1v3_sorted = glass_1v3.sort_values('abs_change', ascending=False)
            glass_1v3_top20 = glass_1v3_sorted.head(20)
            
            glass_1v3_sorted.to_csv(os.path.join(OUTPUT_FOLDER, "glass_ses1_vs_ses3_changes_all.csv"), index=False)
            glass_1v3_top20.to_csv(os.path.join(OUTPUT_FOLDER, "glass_ses1_vs_ses3_changes_TOP20.csv"), index=False)
            print(f"\nSaved: glass_ses1_vs_ses3_changes_all.csv")
            print(f"Saved: glass_ses1_vs_ses3_changes_TOP20.csv")
            print(f"\nTOP 20 GLASS Changes (Session 1 vs Session 3):")
            for idx, row in glass_1v3_top20.iterrows():
                print(f"  {row['seed']} -> {row['region']}: {row['mean_difference']:+.3f} (p={row['p_value']:.3f})")
    
    # === 3D Brain Visualizations ===
    print("\n" + "="*80)
    print("CREATING 3D BRAIN VISUALIZATIONS")
    print("="*80)
    
    # Identify ses-1 and ses-2 labels dynamically
    all_sessions = set(session for (_, session) in amygdala_correlations.keys())
    ses1_label = None
    ses2_label = None
    for sess in all_sessions:
        sess_lower = sess.lower()
        if 'ses-1' in sess_lower or 'mri1' in sess_lower or 's1' in sess_lower:
            ses1_label = sess
        if 'ses-2' in sess_lower or 'mri2' in sess_lower or 's2' in sess_lower:
            ses2_label = sess
    
    if ses1_label and ses2_label:
        print(f"\nUsing sessions: Baseline={ses1_label}, Follow-up={ses2_label}")
        
        # Left Amygdala seed
        seed_name = "Amygdala_L"
        
        # 1. Group A: Mean correlation changes (ses2 - ses1) with hover labels
        print(f"\n1. Computing Group A mean correlation changes ({seed_name})...")
        group_a_results = compute_group_amygdala_correlation_changes_with_pvalues(
            amygdala_correlations, "A", ("ses-1", "MRI1", "S1"), ("ses-2", "MRI2", "S2"),
            seed_name, subj_to_group_symbol
        )
        if group_a_results:
            # Create visualization with mean changes (using 'mean_change' key)
            roi_to_mean_change = {roi: res['mean_change'] for roi, res in group_a_results.items()}
            create_3d_brain_visualization_pvalues(
                {roi: {'mean_change': val} for roi, val in roi_to_mean_change.items()},
                os.path.join(OUTPUT_FOLDER, "3d_group_A_mean_correlation_changes_LeftAmygdala.html"),
                f"Group A: Mean Left Amygdala Connectivity Changes (Change: {ses2_label} - {ses1_label})",
                seed_name,
                value_key='mean_change',
                value_label='Correlation Change'
            )
        
        # 2. Group C: Mean correlation changes (ses2 - ses1) with hover labels
        print(f"\n2. Computing Group C mean correlation changes ({seed_name})...")
        group_c_results = compute_group_amygdala_correlation_changes_with_pvalues(
            amygdala_correlations, "C", ("ses-1", "MRI1", "S1"), ("ses-2", "MRI2", "S2"),
            seed_name, subj_to_group_symbol
        )
        if group_c_results:
            # Create visualization with mean changes
            roi_to_mean_change = {roi: res['mean_change'] for roi, res in group_c_results.items()}
            create_3d_brain_visualization_pvalues(
                {roi: {'mean_change': val} for roi, val in roi_to_mean_change.items()},
                os.path.join(OUTPUT_FOLDER, "3d_group_C_mean_correlation_changes_LeftAmygdala.html"),
                f"Group C: Mean Left Amygdala Connectivity Changes (Change: {ses2_label} - {ses1_label})",
                seed_name,
                value_key='mean_change',
                value_label='Correlation Change'
            )
        
        # 3. Group A: P-values of correlation changes (ses2 - ses1)
        if group_a_results:
            print(f"\n3. Creating Group A p-value visualization ({seed_name})...")
            create_3d_brain_visualization_pvalues(
                group_a_results,
                os.path.join(OUTPUT_FOLDER, "3d_group_A_pvalues_LeftAmygdala.html"),
                f"Group A: P-Values of Left Amygdala Connectivity Changes ({ses2_label} - {ses1_label})",
                seed_name,
                value_key='p_value',
                value_label='P-Value'
            )
        
        # 4. Group C: P-values of correlation changes (ses2 - ses1)
        if group_c_results:
            print(f"\n4. Creating Group C p-value visualization ({seed_name})...")
            create_3d_brain_visualization_pvalues(
                group_c_results,
                os.path.join(OUTPUT_FOLDER, "3d_group_C_pvalues_LeftAmygdala.html"),
                f"Group C: P-Values of Left Amygdala Connectivity Changes ({ses2_label} - {ses1_label})",
                seed_name,
                value_key='p_value',
                value_label='P-Value'
            )
        
        # 5. Between-group: P-values of (Group A difference - Group C difference)
        print(f"\n5. Computing between-group p-values (A - C) ({seed_name})...")
        between_group_results = compute_between_group_pvalues(
            amygdala_correlations,
            ("ses-1", "MRI1", "S1"), ("ses-2", "MRI2", "S2"),
            seed_name, subj_to_group_symbol
        )
        if between_group_results:
            create_3d_brain_visualization_pvalues(
                between_group_results,
                os.path.join(OUTPUT_FOLDER, "3d_between_group_A_minus_C_pvalues_LeftAmygdala.html"),
                f"Between-Group: P-Values of (Group A - Group C) Connectivity Changes ({ses2_label} - {ses1_label})",
                seed_name,
                value_key='p_value',
                value_label='P-Value'
            )
    else:
        print("\nWarning: Could not identify ses-1 and ses-2 sessions for 3D visualization")
        print(f"Available sessions: {sorted(all_sessions)}")
    
    print("\n=== Analysis Complete ===")
