"""Subject-level bootstrap CI for A0 vs A7 hypoglycemia event sensitivity (q=0.10 alarm, 30 min).

Reproduces the quantile-band event_sensitivity numbers in
results/tables/hypoglycemia_a0_vs_a7.csv (A0=0.5893, A7=0.6786) from the raw
per-subject test_diagnostics.npz, then does a subject-level (cluster)
bootstrap over the 12 test subjects to get 95% CIs for each arm and their
difference, since the point estimate rests on only 56 total events.
"""
import numpy as np

from twin.metrics.clinical import find_hypoglycemia_episodes, LOW_MAX

RNG_SEED = 42
N_BOOT = 5000
HORIZON_IDX = 0  # 30 min
Q_ALARM_IDX = 0  # q=0.10


def per_subject_counts(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    targets = d["targets"][:, HORIZON_IDX]
    q_pred = d["quantile_predictions"][:, Q_ALARM_IDX, HORIZON_IDX]
    subj = d["subject_index"]
    alarm_active = q_pred < LOW_MAX

    out = {}
    for s in np.unique(subj):
        mask = subj == s
        ref_s = targets[mask]
        alarm_s = alarm_active[mask]
        episodes = find_hypoglycemia_episodes(ref_s, threshold=LOW_MAX, min_duration_steps=3)
        detected = 0
        for start, end in episodes:
            lookback = max(0, start - 6)
            if np.any(alarm_s[lookback : start + 1]):
                detected += 1
        out[int(s)] = (len(episodes), detected)
    return out


def sensitivity(counts_by_subject, subject_sample):
    tot = sum(counts_by_subject[s][0] for s in subject_sample)
    det = sum(counts_by_subject[s][1] for s in subject_sample)
    return det / tot if tot > 0 else 0.0, tot, det


def bootstrap_ci(counts_by_subject, rng):
    subjects = np.array(sorted(counts_by_subject.keys()))
    n = len(subjects)
    sens_samples = []
    for _ in range(N_BOOT):
        sample = rng.choice(subjects, size=n, replace=True)
        s, _, _ = sensitivity(counts_by_subject, sample)
        sens_samples.append(s)
    sens_samples = np.array(sens_samples)
    return np.percentile(sens_samples, 2.5), np.percentile(sens_samples, 97.5), sens_samples


def main():
    a0 = per_subject_counts("artifacts/official/quantile-A0/official/test_diagnostics.npz")
    a7 = per_subject_counts("artifacts/official/quantile/official/test_diagnostics.npz")

    s0_point, tot0, det0 = sensitivity(a0, list(a0.keys()))
    s7_point, tot7, det7 = sensitivity(a7, list(a7.keys()))
    print(f"A0 point sensitivity: {det0}/{tot0} = {s0_point:.4f}")
    print(f"A7 point sensitivity: {det7}/{tot7} = {s7_point:.4f}")

    rng = np.random.default_rng(RNG_SEED)
    lo0, hi0, samp0 = bootstrap_ci(a0, rng)
    lo7, hi7, samp7 = bootstrap_ci(a7, rng)

    # paired difference: resample subject INDEX jointly (same subjects for both arms each draw)
    subjects = np.array(sorted(a0.keys()))
    n = len(subjects)
    diffs = []
    rng2 = np.random.default_rng(RNG_SEED + 1)
    for _ in range(N_BOOT):
        sample = rng2.choice(subjects, size=n, replace=True)
        s0, _, _ = sensitivity(a0, sample)
        s7, _, _ = sensitivity(a7, sample)
        diffs.append(s7 - s0)
    diffs = np.array(diffs)

    print(f"A0 95% CI: [{lo0:.4f}, {hi0:.4f}]")
    print(f"A7 95% CI: [{lo7:.4f}, {hi7:.4f}]")
    print(f"A7-A0 paired-bootstrap diff 95% CI: [{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}], mean diff={diffs.mean():.4f}")
    print(f"Fraction of bootstrap resamples with A7 >= A0: {(diffs >= 0).mean():.4f}")


if __name__ == "__main__":
    main()
