import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "logs"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# collect CSV files
csv_files = sorted(glob.glob(os.path.join(LOG_DIR, "episode_*.csv")))
if len(csv_files) == 0:
    raise FileNotFoundError(f"No logs found in {LOG_DIR}. Run Session 1 first.")

# load per-episode dataframes
dfs = []
scores = []
lengths = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)
    if "reward" in df.columns:
        scores.append(df["reward"].sum())
    else:
        # try second last column if names differ
        scores.append(df.iloc[:, -2].sum())
    lengths.append(len(df))

scores = np.array(scores)
lengths = np.array(lengths)

# baseline statistics
mean_score = float(np.mean(scores))
max_score = float(np.max(scores))
std_score = float(np.std(scores, ddof=0))
median_score = float(np.median(scores))
p25 = float(np.percentile(scores, 25))
p75 = float(np.percentile(scores, 75))

print("=== Baseline statistics ===")
print(f"Episodes: {len(scores)}")
print(f"Mean score: {mean_score:.3f}")
print(f"Max score: {max_score:.3f}")
print(f"Std dev: {std_score:.3f}")
print(f"Median: {median_score:.3f}")
print(f"25th percentile: {p25:.3f}")
print(f"75th percentile: {p75:.3f}")

# save episode summary CSV
summary = pd.DataFrame({"episode": np.arange(len(scores)), "score": scores, "length": lengths})
summary.to_csv(os.path.join(OUT_DIR, "episode_summary.csv"), index=False)

# Plot: score distribution
plt.figure(figsize=(6,4))
plt.hist(scores, bins=min(20, max(5, len(scores)//1 or 1)))
plt.axvline(mean_score, linestyle='--', label=f"mean={mean_score:.2f}")
plt.axvline(max_score, linestyle=':', label=f"max={max_score:.2f}")
plt.title("Score distribution (random agent)")
plt.xlabel("Episode score")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"))
plt.close()

# Plot: rewards per episode
plt.figure(figsize=(8,4))
plt.plot(scores, marker='.', linewidth=0.8)
plt.title("Rewards per episode")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rewards_per_episode.png"))
plt.close()

# Determine which columns are the state features (defensive)
cols = dfs[0].columns.tolist()
exclude = set(["episode", "step", "reward", "done"])
state_cols = [c for c in cols if c not in exclude]

if len(state_cols) < 3:
    # fallback: try positional (assume last columns are reward/done, earlier are states)
    state_cols = [c for c in cols if c not in ("episode","step")]
    # remove reward/done if present
    state_cols = [c for c in state_cols if c.lower() not in ("reward","done")]
    # ensure at least 3 columns
    state_cols = state_cols[:3]

print("\nDetected state columns:", state_cols)

# Choose short vs long episodes by length percentiles
short_idx = np.where(lengths <= np.percentile(lengths, 25))[0]
long_idx = np.where(lengths >= np.percentile(lengths, 75))[0]
if len(short_idx) == 0:
    short_idx = np.array([int(np.argmin(lengths))])
if len(long_idx) == 0:
    long_idx = np.array([int(np.argmax(lengths))])

short_example = int(short_idx[0])
long_example = int(long_idx[-1])
print(f"Short example episode index: {short_example}  (length={lengths[short_example]})")
print(f"Long example episode index:  {long_example}  (length={lengths[long_example]})")

# Combined plot short vs long for top 3 state columns
top_cols = state_cols[:3]
plt.figure(figsize=(10, 6))
for i, c in enumerate(top_cols):
    plt.subplot(len(top_cols), 1, i+1)
    df_short = dfs[short_example]
    df_long = dfs[long_example]
    if c in df_short.columns:
        plt.plot(df_short["step"], df_short[c], label=f"short ep {short_example}", linewidth=0.9)
    if c in df_long.columns:
        plt.plot(df_long["step"], df_long[c], label=f"long ep {long_example}", linewidth=0.9)
    plt.ylabel(c)
    if i == 0:
        plt.title("State trajectories: short vs long episode")
    if i == len(top_cols)-1:
        plt.xlabel("step")
    plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "state_trajectories_short_vs_long.png"))
plt.close()

# Save individual trajectory images for the chosen episodes
def save_episode_plot(df, ep_idx, fname):
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(top_cols):
        plt.subplot(len(top_cols), 1, i+1)
        if c in df.columns:
            plt.plot(df["step"], df[c])
        plt.ylabel(c)
        if i == 0:
            plt.title(f"Episode {ep_idx} state trajectories")
    plt.xlabel("step")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()

save_episode_plot(dfs[short_example], short_example, f"state_trajectory_example_short_ep{short_example}.png")
save_episode_plot(dfs[long_example], long_example, f"state_trajectory_example_long_ep{long_example}.png")

print(f"\nSaved plots to {OUT_DIR}")

# Quick feature -> score correlation (per-episode averages)
feature_names = top_cols
feat_means = []
for df in dfs:
    # compute mean of each selected feature for that episode
    means = []
    for c in feature_names:
        if c in df.columns:
            means.append(df[c].dropna().astype(float).mean())
        else:
            means.append(np.nan)
    feat_means.append(means)
feat_means = np.array(feat_means, dtype=float)  

print("\nFeature means (per-episode) correlation with score:")
for i, c in enumerate(feature_names):
    col = feat_means[:, i]
    # remove NaNs
    valid = ~np.isnan(col)
    if valid.sum() > 1:
        corr = np.corrcoef(col[valid], scores[valid])[0,1]
    else:
        corr = np.nan
    print(f"  {c}: corr with score = {corr:.3f}")

print("\nEpisode summary saved to outputs/episode_summary.csv")
