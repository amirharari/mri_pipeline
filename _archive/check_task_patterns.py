import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

print("\n" + "="*80)
print("CHECKING IF SESSIONS ARE FROM THE SAME TASK")
print("="*80)

# Load data
ses1_glass = pd.read_csv("sub-001_ses-1_task-glass_aal_ts.csv")
ses2_glass = pd.read_csv("sub-001_ses-2_task-glass_aal_ts.csv")
ses3_glass = pd.read_csv("sub-001_ses-3_task-glass_aal_ts.csv")
ses3_music = pd.read_csv("sub-001_ses-3_task-music_run-2_aal_ts.csv")

print(f"\nSession 1 glass: {ses1_glass.shape}")
print(f"Session 2 glass: {ses2_glass.shape}")
print(f"Session 3 'glass': {ses3_glass.shape}")
print(f"Session 3 music run-2: {ses3_music.shape}")

# Check if ses3 glass IS actually music run-2
if ses3_glass.equals(ses3_music):
    print("\n⚠️  PROBLEM DETECTED!")
    print("Session 3 'glass' IS ACTUALLY music run-2!")
    print("You renamed music to glass, but they're DIFFERENT TASKS!")
else:
    print("\n✓ Session 3 glass is different from music run-2")

# Check Caudate_L first 10 values
print("\n" + "="*80)
print("COMPARING FIRST 10 TIMEPOINTS (Caudate_L):")
print("="*80)
print(f"\nSes3 'glass': {ses3_glass['Caudate_L'].head(10).values}")
print(f"Ses3 music-2: {ses3_music['Caudate_L'].head(10).values}")
print(f"Are they identical? {np.array_equal(ses3_glass['Caudate_L'].values, ses3_music['Caudate_L'].values)}")

# Calculate correlation between sessions
print("\n" + "="*80)
print("WITHIN-TASK CONSISTENCY CHECK")
print("="*80)

# For Caudate_L, calculate temporal correlation
# Take first 129 timepoints (common length)
min_len = min(len(ses1_glass), len(ses2_glass), len(ses3_glass))

corr_1v2, p_1v2 = pearsonr(
    ses1_glass['Caudate_L'].iloc[:min_len], 
    ses2_glass['Caudate_L'].iloc[:min_len]
)
corr_1v3, p_1v3 = pearsonr(
    ses1_glass['Caudate_L'].iloc[:min_len], 
    ses3_glass['Caudate_L'].iloc[:min_len]
)
corr_2v3, p_2v3 = pearsonr(
    ses2_glass['Caudate_L'].iloc[:min_len], 
    ses3_glass['Caudate_L'].iloc[:min_len]
)

print(f"\nTemporal correlation of Caudate_L BOLD patterns:")
print(f"  Session 1 vs 2: r = {corr_1v2:.3f} (p={p_1v2:.4f})")
print(f"  Session 1 vs 3: r = {corr_1v3:.3f} (p={p_1v3:.4f})")
print(f"  Session 2 vs 3: r = {corr_2v3:.3f} (p={p_2v3:.4f})")

print("\n💡 INTERPRETATION:")
if abs(corr_1v2) > 0.3 and abs(corr_1v3) < 0.1:
    print("⚠️  WARNING: Session 1 and 2 are correlated (same task)")
    print("⚠️  WARNING: Session 1 and 3 are NOT correlated (different tasks!)")
    print("\n🔴 PROBLEM: Session 3 'glass' appears to be from a DIFFERENT TASK!")
    print("   → Glass task should show similar patterns across sessions")
    print("   → Low correlation suggests session 3 is NOT actually glass task")
elif abs(corr_1v2) > 0.2 and abs(corr_1v3) > 0.2:
    print("✓ All sessions show similar patterns (same task)")
else:
    print("⚠️  Low correlations - but expected for different scan lengths/contexts")

print("\n" + "="*80 + "\n")
