import pandas as pd
import numpy as np

scrub = pd.read_csv(r'outputs\timeseries_anatomical\scrubbing_report_filtered.csv')
qual  = pd.read_csv(r'outputs\quality_assurance\quality_by_session.csv')

print("=== OVERALL ===")
print("Total runs:", len(scrub))
print("High motion excluded:", int(scrub["high_motion_skip"].sum()))
print("Total volumes:", int(scrub["total_vols"].sum()))
scrub_total = int(scrub["scrubbed_volumes"].sum())
scrub_pct   = round(scrub_total / scrub["total_vols"].sum() * 100, 2)
print("Scrubbed volumes: %d (%.2f%% of all volumes)" % (scrub_total, scrub_pct))
print("Mean FD (notch-filtered): %.3f +/- %.3f mm" % (scrub["fd_mean_filtered"].mean(), scrub["fd_mean_filtered"].std()))
print("Mean confound var explained: %.1f%% +/- %.1f%%" % (scrub["confound_var_explained"].mean()*100, scrub["confound_var_explained"].std()*100))
print("Mean DVARS post-denoising: %.3f +/- %.3f" % (scrub["dvars_post"].mean(), scrub["dvars_post"].std()))
print("Mean GCOR (Schaefer-400): %.3f +/- %.3f" % (scrub["gcor_schaefer"].mean(), scrub["gcor_schaefer"].std()))

print("\n=== BY SESSION ===")
for ses in ["ses-1","ses-2","ses-3"]:
    s = scrub[scrub.session == ses]
    print("%s: n=%d  FD=%.3f+/-%.3f  scrub%%=%.1f  GCOR=%.3f  DVARS=%.3f" % (
        ses, len(s),
        s["fd_mean_filtered"].mean(), s["fd_mean_filtered"].std(),
        s["scrub_percent"].mean(),
        s["gcor_schaefer"].mean(),
        s["dvars_post"].mean()))

print("\n=== tSNR + QC-FC per subject/session ===")
print(qual[["subject","session","mean_tSNR","median_abs_qcfc","qcfc_dist_r"]].to_string(index=False))

print("\n=== OVERALL tSNR + QC-FC ===")
print("Mean tSNR: %.1f +/- %.1f" % (qual["mean_tSNR"].mean(), qual["mean_tSNR"].std()))
print("Median |QC-FC|: %.4f +/- %.4f" % (qual["median_abs_qcfc"].mean(), qual["median_abs_qcfc"].std()))
print("QC-FC dist-r: %.3f +/- %.3f (range %.3f to %.3f)" % (
    qual["qcfc_dist_r"].mean(), qual["qcfc_dist_r"].std(),
    qual["qcfc_dist_r"].min(), qual["qcfc_dist_r"].max()))
