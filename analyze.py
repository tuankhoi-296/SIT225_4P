"""
SIT225 Assessment 4 (4P) - data processing & analysis pipeline
Nguyen Tuan Khoi Le (s226531772)

Reads the raw capture files in ./data/, cleans them, writes a processed CSV,
computes descriptive statistics, draws the required plots, and runs the
hypothesis tests from Assessment 3 (Part D.3):

  H1 (primary)     PIR OR ultrasonic fusion has a lower false-negative rate than
                   PIR alone, without a much higher false-positive rate on an
                   empty desk. Tested with McNemar's test, using the exact
                   binomial form when b+c < 25 -- stricter than the b+c < 10
                   Part D.3 set, so the exact test is reached in more cases
                   than the design required. Balanced accuracy and MCC are
                   reported as effect sizes.
  H2 (secondary)   Ultrasonic distance separates occupied from vacant desks.
                   Tested with a Mann-Whitney U test on 30 s window medians,
                   reported with the rank-biserial correlation.
  H3 (exploratory) Temperature/humidity change is used to label windows as
                   HVAC-active, so H1/H2 can be compared under each state.

Occupancy fusion rule (Part D.3):
    Presence(t) = PIR(t)  OR  ( |dist(t) - baseline| > epsilon )
where baseline and epsilon come from the 10-minute calibration window that is
excluded from the analysis, and the first 30 s window after every ground-truth
change is dropped to remove the button reaction time.

Run:
    python analyze.py
Outputs land in ./data/processed_all.csv, ./data/windows_all.csv and ./plots/.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no display needed, save to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
PLOT_DIR = BASE / "plots"

WINDOW_S = 30                              # analysis window length (Part D.3)
US_MIN_CM, US_MAX_CM = 2.0, 400.0         # physical range of the HC-SR04
EPS_SIGMA = 3.0                           # epsilon = 3 x calibration noise (cm)
EPS_MIN_CM = 6.0                          # floor so noise alone never "detects"


# --------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("raw_*.csv"))
    if not files:
        raise SystemExit("No raw_*.csv files found in ./data/. Run a capture "
                         "session (capture_ui.py) first.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        df["kind"] = "vacant" if "_vacant_" in f.name else "session"
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["host_time"] = pd.to_datetime(raw["host_time"])
    return raw


# --------------------------------------------------------------------------
# 2. VALIDATE
# --------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> dict:
    report = {}
    report["rows"] = len(df)
    report["sessions"] = df["session"].nunique()
    report["missing_per_col"] = df.isna().sum().to_dict()
    # A duplicate = same board sequence number twice inside one session.
    dup = df.dropna(subset=["seq"]).duplicated(subset=["session", "seq"]).sum()
    report["duplicate_samples"] = int(dup)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    return report


# --------------------------------------------------------------------------
# 3. CLEAN  +  4. DERIVE
# --------------------------------------------------------------------------

def clean_and_derive(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {}
    df = df.copy()

    # Duplicate samples: the board emits on its own crystal and the host reads
    # on the laptop clock, so roughly once every hundred ticks the host reads a
    # sample it has already recorded -- same sequence number, same sensor
    # fields, only host_time and age differ. That is one measurement, not two,
    # so the repeat is dropped before anything is computed from it.
    before = len(df)
    df = df.drop_duplicates(subset=["session", "seq"], keep="first")
    info["duplicates_removed"] = before - len(df)

    # Outliers: any distance outside the sensor's physical range is a sensor
    # error, not a real reading -> set to NaN (flagged, not silently kept).
    before = df["dist_cm"].notna().sum()
    bad = df["dist_cm"].notna() & (
        (df["dist_cm"] < US_MIN_CM) | (df["dist_cm"] > US_MAX_CM))
    df.loc[bad, "dist_cm"] = np.nan
    info["dist_outliers_removed"] = int(bad.sum())
    info["dist_valid"] = int(df["dist_cm"].notna().sum())
    info["dist_missing_pct"] = round(100 * df["dist_cm"].isna().mean(), 2)

    # Baseline + epsilon per session, from the calibration window only.
    # Normal session: phase == 'calib'. Vacant session: it has no calib phase,
    # so use its first 10 minutes (gt_age_s < 600) as the calibration proxy.
    # A 30 s guard band is dropped at BOTH ends of the calibration segment,
    # for the same reason transition windows are dropped from the hypothesis
    # tests. The operator is still at the desk when they press start and only
    # then walks out, and at the other end they are back in the chair a few
    # seconds before they press "I am back and seated". Both tails are the
    # seated distance, not the vacant baseline, and leaving either in inflates
    # the calibration standard deviation and with it epsilon.
    baselines, epsilons = {}, {}
    for sess, g in df.groupby("session"):
        if (g["phase"] == "calib").any():
            seg = g[g["phase"] == "calib"]
        else:
            seg = g[g["gt_age_s"] < 600]          # vacant session: first 10 min
        age = seg["gt_age_s"]
        calib = seg[(age >= WINDOW_S) & (age <= age.max() - WINDOW_S)]
        d = calib["dist_cm"].dropna()
        baselines[sess] = float(d.median())
        eps = max(EPS_MIN_CM, EPS_SIGMA * float(d.std()))
        epsilons[sess] = round(eps, 1)
    df["baseline_cm"] = df["session"].map(baselines)
    df["epsilon_cm"] = df["session"].map(epsilons)
    info["baselines"] = {k: round(v, 1) for k, v in baselines.items()}
    info["epsilons"] = epsilons

    # Median-filter the distance (Part D.3 uses median-filtered readings). A
    # 5-sample median removes isolated single-sample spikes (a passer-by or an
    # echo glitch) that would otherwise trip the deviation rule for a whole
    # window, while a real seated deviation lasts far longer and survives.
    df["dist_filt"] = df.groupby("session")["dist_cm"].transform(
        lambda s: s.rolling(5, center=True, min_periods=1).median())

    # Derived detection columns (the fusion rule).
    df["dev_cm"] = (df["dist_filt"] - df["baseline_cm"]).abs()
    df["us_present"] = (df["dev_cm"] > df["epsilon_cm"]).fillna(False)  # NaN->False
    df["pir_pred"] = df["pir_latch"] == 1                       # PIR-only method
    df["fusion_pred"] = df["pir_pred"] | df["us_present"]       # OR fusion

    # Time-of-day label from the session start hour (for the report).
    start_hour = df.groupby("session")["host_time"].transform("min").dt.hour
    df["time_of_day"] = pd.cut(start_hour, bins=[-1, 11, 16, 24],
                               labels=["morning", "afternoon", "evening"])

    # 30 s window index inside each phase; window 0 = transition window (dropped).
    df["window"] = (df["gt_age_s"] // WINDOW_S).astype(int)
    df["is_transition"] = df["window"] == 0
    return df, info


# --------------------------------------------------------------------------
# 5. WINDOW AGGREGATION (30 s)
# --------------------------------------------------------------------------

def make_windows(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[~df["is_transition"]].copy()
    grp = keep.groupby(["session", "kind", "phase", "window"], observed=True)
    win = grp.agg(
        occupied=("occupied", "max"),
        pir_win=("pir_pred", "any"),
        fusion_win=("fusion_pred", "any"),
        dist_median=("dist_cm", "median"),
        temp_mean=("temp_c", "mean"),
        hum_mean=("hum_pct", "mean"),
        hvac=("hvac", "max"),
        t_start=("host_time", "min"),
        n=("occupied", "size"),
    ).reset_index()
    win = win[win["n"] >= WINDOW_S // 2]          # need enough samples in a window
    win["occupied"] = win["occupied"].astype(bool)
    return win


# --------------------------------------------------------------------------
# 6-8. STATISTICS & HYPOTHESIS TESTS
# --------------------------------------------------------------------------

def confusion(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    p, n = tp + fn, tn + fp
    tpr = tp / p if p else float("nan")
    tnr = tn / n if n else float("nan")
    bal_acc = (tpr + tnr) / 2
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else float("nan")
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                fnr=(fn / p if p else float("nan")),
                fpr=(fp / n if n else float("nan")),
                bal_acc=bal_acc, mcc=mcc)


def mcnemar(pir_correct: np.ndarray, fusion_correct: np.ndarray) -> dict:
    # Discordant pairs between the two methods.
    b = int(np.sum(~pir_correct & fusion_correct))   # PIR wrong, fusion right
    c = int(np.sum(pir_correct & ~fusion_correct))   # PIR right, fusion wrong
    if b + c == 0:
        return dict(b=b, c=c, test="none", stat=float("nan"), p=1.0)
    # Part D.3 put the switch at b + c < 10, the minimum the continuity
    # correction needs. 25 is the stricter convention and is what runs here,
    # so the exact test is used in more cases than the design asked for.
    if b + c < 25:
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        return dict(b=b, c=c, test="exact", stat=min(b, c), p=p)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)            # continuity-corrected
    p = stats.chi2.sf(chi2, df=1)
    return dict(b=b, c=c, test="chi2_cc", stat=chi2, p=p)


def run_tests(win: pd.DataFrame) -> dict:
    out = {}

    # ---- H1: fusion vs PIR-only -----------------------------------------
    truth = win["occupied"].to_numpy()
    pir = win["pir_win"].to_numpy()
    fus = win["fusion_win"].to_numpy()
    out["pir"] = confusion(pir, truth)
    out["fusion"] = confusion(fus, truth)
    out["mcnemar"] = mcnemar(pir == truth, fus == truth)

    # Per-session false-negative / false-positive rates.
    per = {}
    for sess, g in win.groupby("session"):
        t = g["occupied"].to_numpy()
        per[sess] = dict(
            pir=confusion(g["pir_win"].to_numpy(), t),
            fusion=confusion(g["fusion_win"].to_numpy(), t),
            n_occ=int(t.sum()), n_vac=int((~t).sum()))
    out["per_session"] = per

    # ---- H2: distance occupied vs vacant --------------------------------
    occ = win.loc[win["occupied"], "dist_median"].dropna().to_numpy()
    vac = win.loc[~win["occupied"], "dist_median"].dropna().to_numpy()
    u, p = stats.mannwhitneyu(occ, vac, alternative="two-sided")
    rbc = 1 - 2 * u / (len(occ) * len(vac))           # rank-biserial correlation
    out["h2"] = dict(n_occ=len(occ), n_vac=len(vac),
                     occ_median=float(np.median(occ)),
                     vac_median=float(np.median(vac)),
                     U=float(u), p=float(p), rank_biserial=float(rbc))

    # ---- H3: split occupied windows by HVAC state -----------------------
    h3 = {}
    for state, label in [(1, "hvac_on"), (0, "hvac_off")]:
        sub = win[(win["occupied"]) & (win["hvac"] == state)]
        if len(sub):
            h3[label] = dict(
                n=len(sub),
                pir_fnr=confusion(sub["pir_win"].to_numpy(),
                                  sub["occupied"].to_numpy())["fnr"],
                fusion_fnr=confusion(sub["fusion_win"].to_numpy(),
                                     sub["occupied"].to_numpy())["fnr"],
                dist_median=float(sub["dist_median"].median()))
    out["h3"] = h3
    return out


def h3_temp_rate(win: pd.DataFrame) -> dict:
    """Can the DHT22 detect the HVAC state? Compare |dTemp/dt| on vs off.

    The rate is taken between consecutive 30 s window means, not between raw
    1 Hz samples. Each DHT22 read carries a few hundredths of a degree of its
    own noise and the output is quantised to 0.1 C, so the difference between
    two neighbouring raw samples is almost always either zero or one whole
    quantisation step -- that measures the sensor, not the room. Averaging the
    ~15 independent reads inside a window first cancels the read noise and
    leaves the drift the heater actually causes. Dividing by the real gap
    between window starts keeps phase boundaries honest.
    """
    d = win.dropna(subset=["temp_mean"]).sort_values(["session", "t_start"]).copy()
    gap_min = d.groupby("session")["t_start"].diff().dt.total_seconds() / 60.0
    d["dtemp"] = (d.groupby("session")["temp_mean"].diff().abs() / gap_min)
    on = d.loc[d["hvac"] == 1, "dtemp"].dropna()
    off = d.loc[d["hvac"] == 0, "dtemp"].dropna()
    u, p = stats.mannwhitneyu(on, off, alternative="greater")
    return dict(on_rate=float(on.mean()), off_rate=float(off.mean()),
                U=float(u), p=float(p))


# --------------------------------------------------------------------------
# 9. PLOTS
# --------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, win: pd.DataFrame):
    PLOT_DIR.mkdir(exist_ok=True)

    # Plot 1: time series (distance) for one normal session, shaded by occupancy.
    sess = df[df["kind"] == "session"]["session"].iloc[0]
    g = df[df["session"] == sess].copy()
    t0 = g["host_time"].min()
    mins = (g["host_time"] - t0).dt.total_seconds() / 60
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(mins, g["dist_cm"], color="#1b5fb4", lw=0.8, label="Distance (cm)")
    ax1.axhline(g["baseline_cm"].iloc[0], color="#94a3b8", ls="--", lw=1,
                label="Vacant baseline")
    ax1.set_xlabel("Minutes since session start")
    ax1.set_ylabel("Ultrasonic distance (cm)", color="#1b5fb4")
    # Shade occupied periods.
    occ = g["occupied"].to_numpy()
    for i in range(1, len(occ)):
        if occ[i] == 1:
            ax1.axvspan(mins.iloc[i - 1], mins.iloc[i], color="#1f8a4c", alpha=0.06)
    ax2 = ax1.twinx()
    ax2.plot(mins, g["pir_latch"], color="#b4501f", lw=0.7, alpha=0.7, label="PIR")
    ax2.set_ylabel("PIR (0/1)", color="#b4501f")
    ax2.set_ylim(-0.1, 1.3)
    ax1.set_title(f"Plot 1  Distance & PIR over time  (green = occupied)  [{sess}]")
    ax1.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot1_timeseries.png", dpi=130)
    plt.close(fig)

    # Plot 2: distribution of window-median distance, occupied vs vacant.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(win.loc[win["occupied"], "dist_median"].dropna(), bins=30,
            color="#1f8a4c", alpha=0.7, label="Occupied")
    ax.hist(win.loc[~win["occupied"], "dist_median"].dropna(), bins=30,
            color="#b4501f", alpha=0.7, label="Vacant")
    ax.set_xlabel("30 s window median distance (cm)")
    ax.set_ylabel("Number of windows")
    ax.set_title("Plot 2  Distance distribution: occupied vs vacant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot2_distribution.png", dpi=130)
    plt.close(fig)

    # Plot 3: box plot of distance by occupancy state (the H2 relationship).
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    data = [win.loc[~win["occupied"], "dist_median"].dropna(),
            win.loc[win["occupied"], "dist_median"].dropna()]
    ax.boxplot(data, tick_labels=["Vacant", "Occupied"], showfliers=True)
    ax.set_ylabel("30 s window median distance (cm)")
    ax.set_title("Plot 3  Ultrasonic distance by occupancy state")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot3_boxplot.png", dpi=130)
    plt.close(fig)

    # Plot 4 (optional, H3): temperature over time coloured by HVAC state.
    fig, ax = plt.subplots(figsize=(10, 4.0))
    mins = (g["host_time"] - t0).dt.total_seconds() / 60
    ax.plot(mins, g["temp_c"], color="#0d9488", lw=1.1, label="Temperature (C)")
    for i in range(1, len(g)):
        if g["hvac"].iloc[i] == 1:
            ax.axvspan(mins.iloc[i - 1], mins.iloc[i], color="#f59e0b", alpha=0.07)
    ax.set_xlabel("Minutes since session start")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Plot 4  Temperature vs time  (orange = heater on)  [H3]")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot4_hvac_temp.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def pct(x):
    return "nan" if x != x else f"{100 * x:.1f}%"


def main():
    raw = load_raw()
    vrep = validate(raw)
    df, cinfo = clean_and_derive(raw)

    # Export the processed per-second dataset and the 30 s windows.
    keep_cols = ["host_time", "session", "kind", "phase", "seq", "age",
                 "pir", "pir_latch", "dist_cm", "us_retry", "temp_c", "hum_pct",
                 "us_err", "dht_err", "occupied", "gt_age_s", "hvac",
                 "baseline_cm", "epsilon_cm", "dist_filt", "dev_cm", "us_present",
                 "pir_pred", "fusion_pred", "time_of_day",
                 "window", "is_transition"]
    processed = df[keep_cols]
    processed.to_csv(DATA_DIR / "processed_all.csv", index=False)

    win = make_windows(df)
    win.to_csv(DATA_DIR / "windows_all.csv", index=False)

    make_plots(df, win)
    res = run_tests(win)

    # ---- console report -------------------------------------------------
    line = "=" * 68
    print(line)
    print("SIT225 4P  -  data processing & analysis")
    print(line)
    print(f"Raw rows: {vrep['rows']:>6}   sessions: {vrep['sessions']}"
          f"   duplicate samples: {vrep['duplicate_samples']}")
    print("Missing values per column (raw):")
    for c, m in vrep["missing_per_col"].items():
        if m:
            print(f"    {c:<12} {m}")
    print(f"Duplicate samples removed: {cinfo['duplicates_removed']}"
          f"   rows kept for analysis: {vrep['rows'] - cinfo['duplicates_removed']}")
    print(f"Distance outliers removed: {cinfo['dist_outliers_removed']}"
          f"   distance missing: {cinfo['dist_missing_pct']}%")
    print(f"Per-session baseline / epsilon (cm): ")
    for s in cinfo["baselines"]:
        print(f"    {s}:  baseline={cinfo['baselines'][s]:>6}  "
              f"epsilon={cinfo['epsilons'][s]}")
    print(f"Analysis windows (transition windows dropped): {len(win)}")

    print("\n--- Descriptive statistics (processed) ---")
    desc = df[["dist_cm", "temp_c", "hum_pct"]].describe(
        percentiles=[.25, .5, .75, .95]).round(2)
    print(desc.to_string())
    corr = df[["dist_cm", "temp_c", "hum_pct", "occupied"]].corr().round(3)
    print("\nCorrelation matrix:")
    print(corr.to_string())

    print("\n--- H1: fusion vs PIR-only (pooled windows) ---")
    p_, f_ = res["pir"], res["fusion"]
    print(f"  PIR-only : FNR={pct(p_['fnr'])}  FPR={pct(p_['fpr'])}  "
          f"bal_acc={p_['bal_acc']:.3f}  MCC={p_['mcc']:.3f}")
    print(f"  Fusion   : FNR={pct(f_['fnr'])}  FPR={pct(f_['fpr'])}  "
          f"bal_acc={f_['bal_acc']:.3f}  MCC={f_['mcc']:.3f}")
    m = res["mcnemar"]
    print(f"  McNemar  : b(PIR wrong,fusion right)={m['b']}  "
          f"c(PIR right,fusion wrong)={m['c']}  test={m['test']}  "
          f"p={m['p']:.3g}")

    print("\n  Per session (FNR PIR -> fusion, FPR PIR -> fusion):")
    for s, v in res["per_session"].items():
        print(f"    {s}: occ_win={v['n_occ']:>2} vac_win={v['n_vac']:>2}  "
              f"FNR {pct(v['pir']['fnr'])}->{pct(v['fusion']['fnr'])}   "
              f"FPR {pct(v['pir']['fpr'])}->{pct(v['fusion']['fpr'])}")

    print("\n--- H2: distance occupied vs vacant ---")
    h2 = res["h2"]
    print(f"  occupied windows={h2['n_occ']} (median {h2['occ_median']:.1f} cm)  "
          f"vacant windows={h2['n_vac']} (median {h2['vac_median']:.1f} cm)")
    print(f"  Mann-Whitney U={h2['U']:.0f}  p={h2['p']:.3g}  "
          f"rank-biserial r={h2['rank_biserial']:.3f}")

    print("\n--- H3: by HVAC state (exploratory) ---")
    for label, v in res["h3"].items():
        print(f"  {label}: windows={v['n']}  PIR FNR={pct(v['pir_fnr'])}  "
              f"fusion FNR={pct(v['fusion_fnr'])}  dist median={v['dist_median']:.1f} cm")
    tr = h3_temp_rate(win)
    print(f"  Temp change rate |dT/dt|:  HVAC on={tr['on_rate']:.3f} C/min  "
          f"off={tr['off_rate']:.3f} C/min  (Mann-Whitney p={tr['p']:.3g})")

    print("\nWrote: data/processed_all.csv, data/windows_all.csv, plots/*.png")
    print(line)


if __name__ == "__main__":
    main()
