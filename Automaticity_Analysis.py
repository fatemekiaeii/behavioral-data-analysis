"""
Final automaticity analysis script:
Integrated ITS for H1-H3 + targeted early-window guidance tests for H4.

Key design decisions
--------------------
1. No forced common sample between eye-tracking and log data.
   - Attentional organization uses all available eye-tracking data.
   - Behavioral organization, motor efficiency, and task-execution efficiency use all available log data.

2. No manual participant removals.
   - Participants are only excluded from a specific model if required variables are missing.

3. H1-H3 are tested with integrated interrupted time-series mixed models.
   - H1: immediate level shifts after updates.
   - H2: post-update recovery slopes.
   - H3: pre-update automaticity amplification of immediate disruption.

4. H4 is tested with targeted early-window buffering models.
   - Guidance is theoretically expected to affect the first post-update exposure.
   - Therefore, H4 uses last N pre-update bins and first N post-update bins.
   - This avoids diluting guidance effects across later adaptation periods.

5. Direction alignment:
   Higher values always mean theoretically better:
   - attentional organization = - gaze transition entropy
   - behavioral organization = - behavioral transition entropy
   - motor efficiency = - NAUC
   - task-execution efficiency = 1 - (unique actions / total actions)

Output files
------------
eye_panel_final.csv
log_panel_final.csv
integrated_its_H1_H3_summary.csv
early_window_H4_summary.csv
final_hypothesis_summary_combined.csv
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

EYE_PATH  = Path("binned_entropy_pupil_60sec_dataset.csv")
LOG_PATH  = Path("combined_log_data.xlsx")
LOG_SHEET = "combined_data"

BIN_SECONDS        = 60
BIN_COL            = "bin60"
PAUSE_THRESHOLD_MS = 200

# H4 early-window size.
# Recommended: 1 or 2 for highly immediate guidance effects.
# Use 3 as a robustness check if needed.
EARLY_WINDOW_BINS = 2

REQUIRED_BLOCKS = {"baseline_task", "game1", "game2"}

RAW_GAZE_ENTROPY       = "entropy_transition"
RAW_BEHAVIOR_ENTROPY   = "entropy_behavioral"
RAW_MOTOR_INEFFICIENCY = "NAUC"

ATTENTIONAL_ORG = "attentional_organization"
BEHAVIORAL_ORG = "behavioral_organization"
MOTOR_EFFICIENCY = "motor_efficiency"
TASK_EXECUTION_EFFICIENCY = "task_execution_efficiency"

CONSTRUCTS = [
    ("a", ATTENTIONAL_ORG, "attentional organization", "eye"),
    ("b", BEHAVIORAL_ORG, "behavioral organization", "log"),
    ("c", MOTOR_EFFICIENCY, "motor efficiency", "log"),
    ("d", TASK_EXECUTION_EFFICIENCY, "task-execution efficiency", "log"),
]

GUIDANCE_MAP = {"NC": 0, "C": 1}

CONDITION_MAP = {
    "p01": "S1-NC", "p02": "S1-C",  "p03": "S2-NC", "p04": "S2-C",
    "p05": "S3-NC", "p06": "S3-C",  "p07": "S4-NC", "p08": "S4-C",
    "p09": "S5-NC", "p10": "S5-C",  "p11": "S6-NC", "p12": "S6-C",
    "p13": "S1-NC", "p14": "S1-C",  "p15": "S2-NC", "p16": "S2-C",
    "p17": "S3-NC", "p18": "S3-C",  "p19": "S3-C",  "p20": "S4-NC",
    "p21": "S4-C",  "p22": "S5-NC", "p23": "S5-C",  "p24": "S3-NC",
    "p25": "S6-NC", "p26": "S6-C",  "p27": "S1-NC", "p28": "S1-C",
    "p29": "S2-NC", "p30": "S2-C",  "p31": "S3-NC", "p32": "S3-C",
    "p33": "S4-NC", "p34": "S4-C",  "p35": "S5-NC", "p36": "S5-C",
    "p37": "S6-NC", "p38": "S6-C",  "p39": "S1-NC", "p40": "S1-C",
    "p41": "S2-NC", "p42": "S2-C",  "p43": "S3-NC", "p44": "S3-C",
    "p45": "S4-NC", "p46": "S4-C",  "p47": "S5-NC", "p48": "S5-C",
    "p49": "S6-NC", "p50": "S6-C",  "p51": "S1-NC", "p52": "S1-C",
}

RE_VARIANCE_TERMS = {
    "Group Var",
    "Group x global_time_c Cov",
    "global_time_c Var",
}

# =============================================================================
# BASIC HELPERS
# =============================================================================

def pid_clean(x) -> Optional[str]:
    n = pd.Series([str(x).lower().strip()]).str.extract(r"p?(\d+)")[0].iloc[0]
    return f"p{int(n):02d}" if pd.notna(n) else np.nan


def block_clean(x) -> str:
    x = str(x).lower().strip().replace("_", " ")
    mapping = {
        "baseline": "baseline_task",
        "baseline task": "baseline_task",
        "baseline_task": "baseline_task",
        "game1": "game1", "game 1": "game1", "g1": "game1",
        "game2": "game2", "game 2": "game2", "g2": "game2",
    }
    return mapping.get(x, x)


def find_and_rename_bin(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    for c in ["bin60", "bin30", "bin", "time_bin", "bin_id"]:
        if c in df.columns:
            return df.rename(columns={c: BIN_COL})
    raise ValueError(f"No bin column in {dataset_name}. Columns: {df.columns.tolist()}")


def add_condition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sequence_condition"] = df["participant_id"].map(CONDITION_MAP)
    df["condition_label"] = df["sequence_condition"].str.extract(r"-(NC|C)$")
    df["guidance"] = df["condition_label"].map(GUIDANCE_MAP)
    return df


def print_sample(df: pd.DataFrame, name: str):
    print(f"\n{name}")
    print("  Participants:", df["participant_id"].nunique())
    print("  Rows:        ", len(df))
    print("  Guidance counts by participant:")
    print(df.drop_duplicates("participant_id")["guidance"].value_counts(dropna=False).sort_index())


def parse_payload(x):
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x) if isinstance(x, str) else {}
    except Exception:
        return {}


def entropy(labels: Iterable) -> float:
    labels = pd.Series(labels).dropna().astype(str)
    labels = labels[labels != ""]
    if len(labels) <= 1:
        return 0.0
    p = labels.value_counts(normalize=True).to_numpy()
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(len(p))) if len(p) > 1 else 0.0


def transition_entropy(labels: Iterable) -> float:
    labels = [x for x in pd.Series(labels).dropna().astype(str) if x != ""]
    if len(labels) <= 1:
        return 0.0
    pairs = [f"{labels[i]}->{labels[i + 1]}" for i in range(len(labels) - 1)]
    return entropy(pairs)


def cell(x, y, w, h, grid: int = 10):
    try:
        x, y, w, h = map(float, [x, y, w, h])
        if w <= 0 or h <= 0:
            return np.nan
        gx = int(np.clip(np.floor(x / w * grid), 0, grid - 1))
        gy = int(np.clip(np.floor(y / h * grid), 0, grid - 1))
        return f"cell_{gx}_{gy}"
    except Exception:
        return np.nan


def action_label(row: pd.Series) -> str:
    tid = str(row.get("targetId", "") or "").strip()
    tcls = str(row.get("targetClass", "") or "").strip()
    ttag = str(row.get("targetTag", "") or "").strip()

    if tid:
        target = f"id:{tid}"
    elif tcls:
        target = f"class:{tcls}"
    elif ttag:
        target = f"tag:{ttag}"
    else:
        target = cell(row.get("x"), row.get("y"), row.get("viewportWidth"), row.get("viewportHeight"))

    return f"{row.get('type')}|{target}"


def task_execution_efficiency(labels: Iterable) -> float:
    """
    Higher = more focused/non-exploratory task execution.

    Operationalized as inverse action diversity:
        1 - (unique actions / total actions)

    Bins with <= 1 click return NaN.
    """
    labels = pd.Series(labels).dropna().astype(str)
    labels = labels[labels != ""]
    n = len(labels)
    if n <= 1:
        return np.nan
    return 1.0 - (labels.nunique() / n)

# =============================================================================
# NAUC
# =============================================================================

def segment_nauc(points: pd.DataFrame) -> Optional[Dict[str, float]]:
    pts = points[["x", "y"]].to_numpy(dtype=float)
    if len(pts) < 2:
        return None

    vec = pts[-1] - pts[0]
    ideal_dist = float(np.linalg.norm(vec))
    if ideal_dist <= 0:
        return None

    unit = vec / ideal_dist
    rel = pts - pts[0]
    proj = rel @ unit
    closest = np.outer(proj, unit) + pts[0]
    dev = np.linalg.norm(pts - closest, axis=1)

    order = np.argsort(proj)
    dp = np.abs(np.diff(proj[order]))
    dv = dev[order]
    auc = float(np.nansum((dv[:-1] + dv[1:]) / 2 * dp))
    return {"auc": auc, "ideal_dist": ideal_dist}


def compute_nauc(g: pd.DataFrame) -> float:
    g = g[g["type"].astype(str).isin(["mousemove", "click"])].copy()
    for c in ["x", "y", "ts"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    g = g.dropna(subset=["x", "y", "ts"]).sort_values("ts").reset_index(drop=True)
    if len(g) < 2:
        return np.nan

    endpoints = {0, len(g) - 1}
    endpoints |= set(g.index[g["type"].astype(str).eq("click")].tolist())

    dt_ms = g["ts"].diff().to_numpy()
    endpoints |= set(np.where(dt_ms >= PAUSE_THRESHOLD_MS)[0].tolist())

    endpoints = sorted(i for i in endpoints if 0 <= i < len(g))
    if len(endpoints) < 2:
        return np.nan

    total_auc = total_ideal = 0.0
    for a, b in zip(endpoints[:-1], endpoints[1:]):
        if b <= a:
            continue
        out = segment_nauc(g.iloc[a:b + 1])
        if out:
            total_auc += out["auc"]
            total_ideal += out["ideal_dist"]

    return (total_auc / total_ideal) if total_ideal > 0 else np.nan

# =============================================================================
# LOAD DATA: NO MANUAL REMOVAL, NO COMMON SAMPLE
# =============================================================================

def load_eye_panel() -> pd.DataFrame:
    df = pd.read_csv(EYE_PATH)
    df["participant_id"] = df["participant_id"].apply(pid_clean)
    df["block_label"] = df["block_label"].apply(block_clean)
    df = find_and_rename_bin(df, "eye data")

    df = df[df["block_label"].isin(REQUIRED_BLOCKS)].copy()
    df[BIN_COL] = pd.to_numeric(df[BIN_COL], errors="coerce")
    df[RAW_GAZE_ENTROPY] = pd.to_numeric(df[RAW_GAZE_ENTROPY], errors="coerce")

    df = df.dropna(subset=["participant_id", "block_label", BIN_COL, RAW_GAZE_ENTROPY]).copy()

    df = add_condition(df)
    df[ATTENTIONAL_ORG] = -df[RAW_GAZE_ENTROPY]

    print_sample(df, "EYE PANEL: no manual removals, no common-sample restriction")
    return df


def load_log_panel() -> pd.DataFrame:
    log = pd.read_excel(LOG_PATH, sheet_name=LOG_SHEET)
    log["participant_id"] = log["participant"].apply(pid_clean)
    log["block_label"] = log["phase"].apply(block_clean)

    log = log[log["block_label"].isin(REQUIRED_BLOCKS)].copy()
    log["ts"] = pd.to_numeric(log["ts"], errors="coerce")
    log = log.dropna(subset=["participant_id", "block_label", "ts"]).copy()
    log = add_condition(log)

    payload = pd.json_normalize(log["payload"].apply(parse_payload))
    log = pd.concat([log.reset_index(drop=True), payload.reset_index(drop=True)], axis=1)

    for col in ["x", "y", "targetId", "targetClass", "targetTag", "viewportWidth", "viewportHeight"]:
        if col not in log.columns:
            log[col] = np.nan

    for col in ["x", "y", "viewportWidth", "viewportHeight"]:
        log[col] = pd.to_numeric(log[col], errors="coerce")

    start_ts = (
        log.groupby(["participant_id", "block_label"])["ts"]
        .min().rename("start_ts").reset_index()
    )
    log = log.merge(start_ts, on=["participant_id", "block_label"], how="left")
    log["rel_sec"] = (log["ts"] - log["start_ts"]) / 1000
    log = log[log["rel_sec"] >= 0].copy()
    log[BIN_COL] = np.floor(log["rel_sec"] / BIN_SECONDS).astype(int)

    click_mask = log["type"].astype(str).eq("click")
    log["action_label"] = np.nan
    log.loc[click_mask, "action_label"] = log.loc[click_mask].apply(action_label, axis=1)

    rows = []
    nan_tee = 0

    for (pid, block, bin_id), g in log.groupby(["participant_id", "block_label", BIN_COL]):
        clicks = g[g["type"].astype(str).eq("click")].sort_values("ts")
        tee = task_execution_efficiency(clicks["action_label"])
        if pd.isna(tee):
            nan_tee += 1

        rows.append({
            "participant_id": pid,
            "block_label": block,
            BIN_COL: int(bin_id),
            RAW_BEHAVIOR_ENTROPY: transition_entropy(clicks["action_label"]),
            RAW_MOTOR_INEFFICIENCY: compute_nauc(g),
            TASK_EXECUTION_EFFICIENCY: tee,
        })

    df = pd.DataFrame(rows)
    df = add_condition(df)
    df[BEHAVIORAL_ORG] = -df[RAW_BEHAVIOR_ENTROPY]
    df[MOTOR_EFFICIENCY] = -df[RAW_MOTOR_INEFFICIENCY]

    print_sample(df, "LOG PANEL: no manual removals, no common-sample restriction")
    print(f"  Task-execution efficiency NaN bins <=1 click: {nan_tee}/{len(df)} ({100 * nan_tee / max(len(df), 1):.1f}%)")
    return df

# =============================================================================
# INTEGRATED ITS TERMS
# =============================================================================

def add_integrated_its_terms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    block_order = {"baseline_task": 0, "game1": 1, "game2": 2}
    df["block_order"] = df["block_label"].map(block_order)
    df = df.sort_values(["participant_id", "block_order", BIN_COL]).copy()

    df["global_time"] = df.groupby("participant_id").cumcount().astype(float)
    df["global_time_c"] = df["global_time"] - df["global_time"].mean()

    df["u1_level"] = df["block_label"].isin(["game1", "game2"]).astype(int)
    df["u2_level"] = df["block_label"].eq("game2").astype(int)

    df["u1_time"] = 0.0
    mask_u1 = df["block_label"].isin(["game1", "game2"])
    df.loc[mask_u1, "u1_time"] = (
        df.loc[mask_u1]
        .groupby("participant_id")
        .cumcount()
        .astype(float)
    )

    df["u2_time"] = 0.0
    mask_u2 = df["block_label"].eq("game2")
    df.loc[mask_u2, "u2_time"] = (
        df.loc[mask_u2]
        .groupby("participant_id")
        .cumcount()
        .astype(float)
    )

    df["u1_time_sq"] = df["u1_time"] ** 2
    df["u2_time_sq"] = df["u2_time"] ** 2
    return df


def add_pre_update_predictors(df: pd.DataFrame, constructs: List[str]) -> pd.DataFrame:
    """
    Adds centered pre-update values for Update 1 and Update 2.

    Update 1 pre value = participant mean in baseline_task.
    Update 2 pre value = participant mean in game1.
    """
    df = df.copy()

    baseline = (
        df[df["block_label"] == "baseline_task"]
        .groupby("participant_id")[constructs]
        .mean()
        .add_prefix("u1_pre_")
        .reset_index()
    )

    game1 = (
        df[df["block_label"] == "game1"]
        .groupby("participant_id")[constructs]
        .mean()
        .add_prefix("u2_pre_")
        .reset_index()
    )

    df = df.merge(baseline, on="participant_id", how="left")
    df = df.merge(game1, on="participant_id", how="left")

    for prefix in ["u1_pre_", "u2_pre_"]:
        for c in constructs:
            raw = f"{prefix}{c}"
            centered = f"{raw}_c"
            df[centered] = df[raw] - df[raw].mean()

    return df


def add_log_automaticity_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates log-based automaticity composite for H3d without re-merging columns.

    Composite = z(behavioral organization) + z(motor efficiency), averaged.
    This preserves the maximum log sample and avoids requiring eye data.
    """
    df = df.copy()

    for prefix in ["u1_pre_", "u2_pre_"]:
        z_cols = []

        for c in [BEHAVIORAL_ORG, MOTOR_EFFICIENCY]:
            raw = f"{prefix}{c}"
            z = f"z_{raw}"

            sd = df[raw].std(ddof=1)
            df[z] = 0.0 if pd.isna(sd) or sd == 0 else (df[raw] - df[raw].mean()) / sd
            z_cols.append(z)

        comp = f"{prefix}log_automaticity_composite"
        df[comp] = df[z_cols].mean(axis=1)
        df[f"{comp}_c"] = df[comp] - df[comp].mean()

    return df

# =============================================================================
# EARLY-WINDOW CHANGE PANEL FOR H4
# =============================================================================

def summarize_last_bins(df: pd.DataFrame, block: str, y: str, n: int) -> pd.DataFrame:
    rows = []
    sub = df[df["block_label"] == block].sort_values(["participant_id", BIN_COL])

    for pid, g in sub.groupby("participant_id"):
        tail = g.tail(n)
        if tail.empty:
            continue
        rows.append({
            "participant_id": pid,
            "guidance": tail["guidance"].iloc[-1],
            f"{y}_pre": tail[y].mean(skipna=True),
            "n_pre_bins": len(tail),
        })

    return pd.DataFrame(rows)


def summarize_first_bins(df: pd.DataFrame, block: str, y: str, n: int) -> pd.DataFrame:
    rows = []
    sub = df[df["block_label"] == block].sort_values(["participant_id", BIN_COL])

    for pid, g in sub.groupby("participant_id"):
        head = g.head(n)
        if head.empty:
            continue
        rows.append({
            "participant_id": pid,
            f"{y}_post": head[y].mean(skipna=True),
            "n_post_bins": len(head),
        })

    return pd.DataFrame(rows)


def build_early_window_change_panel(df: pd.DataFrame, y: str, n: int = EARLY_WINDOW_BINS) -> pd.DataFrame:
    """
    Builds one row per participant per update.

    update1 change = first N game1 bins - last N baseline bins
    update2 change = first N game2 bins - last N game1 bins

    Higher construct values = better.
    Negative change = disruption.
    Guidance buffering is tested as guidance > 0, meaning less negative change.
    """
    panels = []

    for update, pre_block, post_block in [
        ("update1", "baseline_task", "game1"),
        ("update2", "game1", "game2"),
    ]:
        pre = summarize_last_bins(df, pre_block, y, n)
        post = summarize_first_bins(df, post_block, y, n)

        m = pre.merge(post, on="participant_id", how="inner")
        m["update"] = update
        m[f"{y}_change"] = m[f"{y}_post"] - m[f"{y}_pre"]
        panels.append(m)

    panel = pd.concat(panels, ignore_index=True)
    return panel

# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_mixedlm_or_ols(
    df: pd.DataFrame,
    formula: str,
    title: str,
    expected: str,
    re_formula: Optional[str] = "~global_time_c",
):
    model_df = df.dropna().copy()

    print("\n" + "=" * 100)
    print(title)
    print(f"  Expected  : {expected}")
    print(f"  Formula   : {formula}")
    print(f"  RE formula: {re_formula}")
    print(f"  Rows used : {len(model_df)}  (dropped missing model vars: {len(df) - len(model_df)})")
    print(f"  N         : {model_df['participant_id'].nunique() if len(model_df) else 0}")

    if len(model_df) == 0 or model_df["participant_id"].nunique() < 3:
        print("  SKIPPED: insufficient data.")
        return None, "skipped"

    attempts = []
    if re_formula:
        attempts.append(("mixedlm_random_slope", re_formula))
    attempts.append(("mixedlm_random_intercept", None))

    for label, re in attempts:
        for method in ["lbfgs", "bfgs", "nm", "powell"]:
            try:
                md = smf.mixedlm(
                    formula,
                    data=model_df,
                    groups=model_df["participant_id"],
                    re_formula=re,
                )
                res = md.fit(method=method, reml=True)

                if getattr(res, "converged", True):
                    print(f"  Converged: {method} | {label}")
                    print(res.summary())
                    return res, label

                print(f"  Tried {method} | {label}, but converged=False.")
            except Exception:
                continue

    print("  WARNING: MixedLM failed or did not converge. Falling back to OLS + clustered SE.")

    try:
        ols = smf.ols(formula, data=model_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["participant_id"]},
        )
        print(ols.summary())
        return ols, "ols_clustered_se_fallback"
    except Exception as exc:
        print(f"  ERROR: OLS also failed ({exc}).")
        return None, "failed"


def fit_h4_early_window(df: pd.DataFrame, formula: str, title: str, expected: str):
    """
    H4 uses one/two rows per participant, so we use OLS with participant-clustered SE.
    """
    model_df = df.dropna().copy()

    print("\n" + "=" * 100)
    print(title)
    print(f"  Expected  : {expected}")
    print(f"  Formula   : {formula}")
    print("  Estimator : OLS + participant-clustered SE")
    print(f"  Rows used : {len(model_df)}  (dropped missing model vars: {len(df) - len(model_df)})")
    print(f"  N         : {model_df['participant_id'].nunique() if len(model_df) else 0}")

    if len(model_df) == 0 or model_df["participant_id"].nunique() < 3:
        print("  SKIPPED: insufficient data.")
        return None, "skipped"

    try:
        res = smf.ols(formula, data=model_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["participant_id"]},
        )
        print(res.summary())
        return res, "ols_clustered_se"
    except Exception as exc:
        print(f"  ERROR: H4 OLS failed ({exc}).")
        return None, "failed"


def collect_terms(result, hypothesis: str, construct: str, model: str, model_type: str, focal_terms: List[str]) -> List[dict]:
    if result is None:
        return []

    rows = []
    for term in result.params.index:
        if term in RE_VARIANCE_TERMS:
            continue

        rows.append({
            "hypothesis": hypothesis,
            "construct": construct,
            "model": model,
            "model_type": model_type,
            "term": term,
            "estimate": result.params.get(term, np.nan),
            "std_error": result.bse.get(term, np.nan),
            "p_value": result.pvalues.get(term, np.nan),
            "n_obs": int(result.nobs),
            "is_focal": term in focal_terms,
        })

    return rows


def apply_fdr(summary_df: pd.DataFrame, family_col: str = "model") -> pd.DataFrame:
    """
    Applies BH-FDR to focal terms within each model family.
    """
    summary_df = summary_df.copy()
    summary_df["p_value_fdr"] = np.nan

    for family in summary_df[family_col].dropna().unique():
        mask = (
            summary_df[family_col].eq(family)
            & summary_df["is_focal"].eq(True)
            & summary_df["p_value"].notna()
        )
        if mask.sum() > 0:
            _, p_adj, _, _ = multipletests(summary_df.loc[mask, "p_value"], method="fdr_bh")
            summary_df.loc[mask, "p_value_fdr"] = p_adj

    summary_df["p_for_sig"] = summary_df["p_value_fdr"].fillna(summary_df["p_value"])
    summary_df["significant_05"] = summary_df["p_for_sig"] < 0.05
    return summary_df

# =============================================================================
# RUN H1-H3 INTEGRATED ITS
# =============================================================================

def run_integrated_model(
    df: pd.DataFrame,
    y: str,
    label: str,
    suffix: str,
    u1_pre: str,
    u2_pre: str,
):
    formula = (
        f"{y} ~ "
        f"global_time_c + "
        f"u1_level + u1_time + u1_time_sq + "
        f"u2_level + u2_time + u2_time_sq + "
        f"guidance + "
        f"u1_level:{u1_pre} + u2_level:{u2_pre}"
    )

    focal_terms = [
        "u1_level", "u2_level",        # H1
        "u1_time", "u2_time",          # H2
        f"u1_level:{u1_pre}",          # H3 update 1
        f"u2_level:{u2_pre}",          # H3 update 2
    ]

    cols = [
        "participant_id", y, "global_time_c",
        "u1_level", "u1_time", "u1_time_sq",
        "u2_level", "u2_time", "u2_time_sq",
        "guidance", u1_pre, u2_pre,
    ]

    res, model_type = fit_mixedlm_or_ols(
        df[cols],
        formula=formula,
        title=f"H1-H3 Integrated ITS model for {label}",
        expected=(
            "H1: u1/u2_level < 0; "
            "H2: u1/u2_time > 0; "
            "H3: level × pre-update automaticity < 0."
        ),
        re_formula="~global_time_c",
    )

    summary = collect_terms(
        res,
        hypothesis=f"H1-H3{suffix}",
        construct=label,
        model="H1_H3_integrated_ITS",
        model_type=model_type,
        focal_terms=focal_terms,
    )

    return res, summary


def run_all_h1_h3(eye: pd.DataFrame, log: pd.DataFrame):
    results = {}
    summary = []

    # Attentional organization: eye only
    eye = add_integrated_its_terms(eye)
    eye = add_pre_update_predictors(eye, [ATTENTIONAL_ORG])

    res, summ = run_integrated_model(
        eye, ATTENTIONAL_ORG, "attentional organization", "a",
        f"u1_pre_{ATTENTIONAL_ORG}_c",
        f"u2_pre_{ATTENTIONAL_ORG}_c",
    )
    results[ATTENTIONAL_ORG] = res
    summary.extend(summ)

    # Log constructs
    log = add_integrated_its_terms(log)
    log = add_pre_update_predictors(log, [BEHAVIORAL_ORG, MOTOR_EFFICIENCY])
    log = add_log_automaticity_composite(log)

    res, summ = run_integrated_model(
        log, BEHAVIORAL_ORG, "behavioral organization", "b",
        f"u1_pre_{BEHAVIORAL_ORG}_c",
        f"u2_pre_{BEHAVIORAL_ORG}_c",
    )
    results[BEHAVIORAL_ORG] = res
    summary.extend(summ)

    res, summ = run_integrated_model(
        log, MOTOR_EFFICIENCY, "motor efficiency", "c",
        f"u1_pre_{MOTOR_EFFICIENCY}_c",
        f"u2_pre_{MOTOR_EFFICIENCY}_c",
    )
    results[MOTOR_EFFICIENCY] = res
    summary.extend(summ)

    res, summ = run_integrated_model(
        log, TASK_EXECUTION_EFFICIENCY, "task-execution efficiency", "d",
        "u1_pre_log_automaticity_composite_c",
        "u2_pre_log_automaticity_composite_c",
    )
    results[TASK_EXECUTION_EFFICIENCY] = res
    summary.extend(summ)

    integrated_summary = pd.DataFrame(summary)
    integrated_summary = apply_fdr(integrated_summary, family_col="model")
    integrated_summary.to_csv("integrated_its_H1_H3_summary.csv", index=False)

    eye.to_csv("eye_panel_final.csv", index=False)
    log.to_csv("log_panel_final.csv", index=False)

    print("\n" + "=" * 100)
    print("FOCAL TERMS SUMMARY: H1-H3 INTEGRATED ITS")
    print("=" * 100)
    print(
        integrated_summary[integrated_summary["is_focal"]][[
            "hypothesis", "construct", "model_type", "term",
            "estimate", "std_error", "p_value", "p_value_fdr", "significant_05"
        ]].to_string(index=False)
    )

    return results, integrated_summary, eye, log

# =============================================================================
# RUN H4 EARLY-WINDOW GUIDANCE TESTS
# =============================================================================

def run_h4_for_construct(df: pd.DataFrame, y: str, label: str, suffix: str):
    panel = build_early_window_change_panel(df, y, n=EARLY_WINDOW_BINS)

    y_change = f"{y}_change"

    # Primary H4 test: average early-window buffering across both updates,
    # controlling for update.
    formula = f"{y_change} ~ guidance + C(update)"

    res, model_type = fit_h4_early_window(
        panel[["participant_id", "guidance", "update", y_change]],
        formula=formula,
        title=f"H4{suffix}: early-window guidance buffering for {label}",
        expected="guidance > 0 means guidance reduced the immediate disruption.",
    )

    summary = collect_terms(
        res,
        hypothesis=f"H4{suffix}",
        construct=label,
        model="H4_early_window_buffering",
        model_type=model_type,
        focal_terms=["guidance"],
    )

    # Supplementary: update-specific guidance effects.
    formula_by_update = f"{y_change} ~ guidance * C(update)"

    res2, model_type2 = fit_h4_early_window(
        panel[["participant_id", "guidance", "update", y_change]],
        formula=formula_by_update,
        title=f"H4{suffix} supplementary: update-specific guidance buffering for {label}",
        expected=(
            "guidance = Update 1 guidance effect; "
            "guidance:C(update)[T.update2] = difference in guidance effect for Update 2."
        ),
    )

    summary.extend(collect_terms(
        res2,
        hypothesis=f"H4{suffix}_supp",
        construct=label,
        model="H4_early_window_update_specific",
        model_type=model_type2,
        focal_terms=["guidance", "guidance:C(update)[T.update2]"],
    ))

    panel.to_csv(f"H4_early_window_panel_{y}.csv", index=False)
    return {f"H4{suffix}_primary": res, f"H4{suffix}_supplementary": res2}, summary


def run_all_h4(eye: pd.DataFrame, log: pd.DataFrame):
    results = {}
    summary = []

    construct_sources = {
        ATTENTIONAL_ORG: eye,
        BEHAVIORAL_ORG: log,
        MOTOR_EFFICIENCY: log,
        TASK_EXECUTION_EFFICIENCY: log,
    }

    for suffix, y, label, _source in CONSTRUCTS:
        res_dict, summ = run_h4_for_construct(
            construct_sources[y],
            y=y,
            label=label,
            suffix=suffix,
        )
        results.update(res_dict)
        summary.extend(summ)

    h4_summary = pd.DataFrame(summary)
    h4_summary = apply_fdr(h4_summary, family_col="model")
    h4_summary.to_csv("early_window_H4_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("FOCAL TERMS SUMMARY: H4 EARLY-WINDOW GUIDANCE TESTS")
    print("=" * 100)
    print(
        h4_summary[h4_summary["is_focal"]][[
            "hypothesis", "construct", "model", "model_type", "term",
            "estimate", "std_error", "p_value", "p_value_fdr", "significant_05"
        ]].to_string(index=False)
    )

    return results, h4_summary

# =============================================================================
# MAIN
# =============================================================================

def main():
    eye_raw = load_eye_panel()
    log_raw = load_log_panel()

    h1_h3_results, h1_h3_summary, eye_panel, log_panel = run_all_h1_h3(eye_raw, log_raw)
    h4_results, h4_summary = run_all_h4(eye_panel, log_panel)

    combined = pd.concat([h1_h3_summary, h4_summary], ignore_index=True)
    combined.to_csv("final_hypothesis_summary_combined.csv", index=False)

    print("\nOutput files:")
    for f in [
        "eye_panel_final.csv",
        "log_panel_final.csv",
        "integrated_its_H1_H3_summary.csv",
        "early_window_H4_summary.csv",
        "final_hypothesis_summary_combined.csv",
    ]:
        print(f"  {f}")

    return {
        "H1_H3": h1_h3_results,
        "H4": h4_results,
    }, combined


if __name__ == "__main__":
    results, summary_df = main()
