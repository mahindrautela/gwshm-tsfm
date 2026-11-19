"""Detection metrics computed from the long-format DI table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

DI_COLS = ("di_res", "di_cc", "di_z")


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_max: float = 0.05) -> float:
    m = np.isfinite(s)
    if len(np.unique(y[m])) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y[m], s[m])
    return float(np.interp(fpr_max, fpr, tpr))


def common_range(df: pd.DataFrame) -> pd.DataFrame:
    """Keep test signals inside the temperature range covered by *both* classes.

    The intact plate was heated to ~61 deg C, the damaged plate only to ~60 deg C;
    without this filter the hottest (hardest to predict) signals are all pristine
    and every method is biased towards AUC < 0.5.
    """
    hi = df.groupby("label")["T"].max().min()
    lo = df.groupby("label")["T"].min().max()
    return df[(df["T"] <= hi + 1e-9) & (df["T"] >= lo - 1e-9)]


def stratified_auc(y: np.ndarray, s: np.ndarray, T: np.ndarray, width: float = 1.0) -> float:
    """AUC computed within temperature bins and averaged (weights = pairs per bin).

    Equivalent to allowing a temperature-dependent threshold: measures how well a
    DI separates the classes at a given temperature, free of temperature trends.
    """
    b = np.floor(T / width)
    num = den = 0.0
    for v in np.unique(b):
        m = b == v
        n1, n0 = int(y[m].sum()), int((1 - y[m]).sum())
        if n1 and n0 and np.isfinite(s[m]).all():
            num += roc_auc_score(y[m], s[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else np.nan


def per_path_auc(df: pd.DataFrame, di: str = "di_res") -> pd.DataFrame:
    """AUC for every (f, Tc, method, path)."""
    g = common_range(df).groupby(["f_khz", "Tc", "method", "path"], sort=False)
    return g.apply(lambda d: stratified_auc(d["label"].to_numpy(), d[di].to_numpy(), d["T"].to_numpy()),
                   include_groups=False).rename("auc").reset_index()


def fused_scores(df: pd.DataFrame, di: str = "di_res", paths=None) -> pd.DataFrame:
    """Structure-level score per measurement: mean log-DI over paths."""
    d = common_range(df)
    d = d if paths is None else d[d["path"].isin(paths)]
    d = d.assign(logdi=np.log(np.maximum(d[di].astype(np.float64), 1e-12)))
    return (d.groupby(["f_khz", "Tc", "method", "meas"], sort=False)
             .agg(score=("logdi", "mean"), T=("T", "first"), state=("state", "first"),
                  label=("label", "first")).reset_index())


def fused_auc(df: pd.DataFrame, di: str = "di_res", paths=None, by_state: bool = False,
              fpr: float = 0.05) -> pd.DataFrame:
    fs = fused_scores(df, di, paths)
    rows = []
    for key, d in fs.groupby(["f_khz", "Tc", "method"], sort=False):
        y, s = d["label"].to_numpy(), d["score"].to_numpy()
        r = dict(zip(["f_khz", "Tc", "method"], key), auc=_auc(y, s), tpr=tpr_at_fpr(y, s, fpr),
                 auc_strat=stratified_auc(y, s, d["T"].to_numpy()),
                 n_pos=int(y.sum()), n_neg=int((1 - y).sum()))
        if by_state:
            for st in sorted(set(d["state"]) - {"udam"}):
                m = d["state"].isin(["udam", st]).to_numpy()
                r[f"auc_{st}"] = _auc(y[m], s[m])
        rows.append(r)
    return pd.DataFrame(rows)


def auc_vs_gap(df: pd.DataFrame, di: str = "di_res", paths=None,
               edges=(0, 2.5, 5, 10, 20, 40)) -> pd.DataFrame:
    """Fused, temperature-stratified AUC as a function of the extrapolation gap T - Tc."""
    fs = fused_scores(df, di, paths)
    fs["gap"] = fs["T"] - fs["Tc"]
    fs["bin"] = pd.cut(fs["gap"], list(edges))
    rows = []
    for key, d in fs.groupby(["f_khz", "Tc", "method", "bin"], sort=False, observed=True):
        y, s = d["label"].to_numpy(), d["score"].to_numpy()
        rows.append(dict(zip(["f_khz", "Tc", "method", "bin"], key),
                         auc=stratified_auc(y, s, d["T"].to_numpy()), n=len(d)))
    return pd.DataFrame(rows)


def forecast_error(df: pd.DataFrame, paths=None) -> pd.DataFrame:
    """Median relative residual energy on *pristine* test signals vs gap."""
    d = common_range(df)
    d = d[(d["label"] == 0)]
    if paths is not None:
        d = d[d["path"].isin(paths)]
    d = d.assign(gap=d["T"] - d["Tc"])
    d["bin"] = pd.cut(d["gap"], [0, 2.5, 5, 10, 20, 40])
    return (d.groupby(["f_khz", "Tc", "method", "bin"], observed=True)["di_res"]
             .median().rename("rel_err").reset_index())
