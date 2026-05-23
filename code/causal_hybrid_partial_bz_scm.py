"""
Hybrid partial-BZ causal mixture SCM for paired time series / text embeddings.

This version follows the requested hybrid design:

1) GMM separates regimes using external/data-derived regime features (for the
   accelerometer experiment: smoothed intensity/envelope features; for text this
   can be document embeddings or other data features).
2) A and B are learned by a stable latent-mixture EM stage without C competing
   with them:
       e ~ Cat(w), Z|e ~ N(mu_e, Sigma_e)
       X = A Z + eps_x,  Y = B Z + eps_y
3) After A,B,p_e(Z) are learned, C_e is learned separately, either on residuals
   or as a direct correction. Prediction combines both parts:
       y_hat_e = B E[Z|X,e] + C_e * direct_input_e.

Default mode:
    GMM regimes: intensity features
    C mode: partial_bz_direct, i.e. C_e maps X to Y - rho B E[Z|X,Y,e]. Prediction uses Y_hat = rho B E[Z|X,e] + C_e X.

The file also includes reusable preprocessing utilities for the ZIP sensor format:
Accelerometer(time, seconds_elapsed, z, y, x)
Gyroscope(time, seconds_elapsed, z, y, x)
Orientation(time, seconds_elapsed, yaw, qx, qz, roll, qw, qy, pitch)
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

Array = np.ndarray

# =============================================================================
# I/O and synchronization
# =============================================================================

def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text), sep=",")


def parse_absolute_time_seconds(value) -> float:
    """Parse a single absolute timestamp into seconds.

    We use absolute `time` only for the start time of each recording. The actual
    within-recording time scale is `seconds_elapsed`, which is already seconds.
    """
    x = float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])
    if not np.isfinite(x):
        raise ValueError(f"Cannot parse absolute timestamp: {value!r}")
    if x > 1e17:   # ns
        return x / 1e9
    if x > 1e14:   # us
        return x / 1e6
    if x > 1e11:   # ms
        return x / 1e3
    return x       # seconds or app-specific already in seconds


def find_csv_in_zip(zip_path: str | Path, sensor_keyword: str) -> str:
    zip_path = Path(zip_path)
    key = _normalize_name(sensor_keyword)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for n in names:
            if key in _normalize_name(Path(n).name):
                return n
    raise FileNotFoundError(f"Cannot find {sensor_keyword} CSV in {zip_path}")


def load_sensor_from_zip(zip_path: str | Path, sensor_keyword: str) -> pd.DataFrame:
    member = find_csv_in_zip(zip_path, sensor_keyword)
    with zipfile.ZipFile(zip_path, "r") as zf:
        df = _read_csv_bytes(zf.read(member))
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    if "time" not in lower or "seconds_elapsed" not in lower:
        raise ValueError(f"{member} must contain columns time and seconds_elapsed")
    df = df.rename(columns={lower["time"]: "time", lower["seconds_elapsed"]: "seconds_elapsed"})
    df["seconds_elapsed"] = pd.to_numeric(df["seconds_elapsed"], errors="coerce")
    t0 = parse_absolute_time_seconds(df["time"].iloc[0])
    df["time_seconds_abs"] = t0 + df["seconds_elapsed"]
    df["time_seconds_rel"] = df["seconds_elapsed"]
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_seconds_abs", "seconds_elapsed"]).sort_values("time_seconds_abs")
    df = df.drop_duplicates(subset=["time_seconds_abs"])
    return df.reset_index(drop=True)


def estimate_hz(t: Array) -> float:
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan
    return float(1.0 / np.median(dt))


def _sensor_columns(sensor: str, use_accelerometer=True, use_gyroscope=True, use_orientation=False) -> List[str]:
    if sensor == "Accelerometer":
        return ["x", "y", "z"] if use_accelerometer else []
    if sensor == "Gyroscope":
        return ["x", "y", "z"] if use_gyroscope else []
    if sensor == "Orientation":
        return ["yaw", "qx", "qz", "roll", "qw", "qy", "pitch"] if use_orientation else []
    return []


def _interp_sensor(df: pd.DataFrame, grid_abs: Array, side: str, sprefix: str, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame({"time_seconds_abs": grid_abs})
    t = df["time_seconds_abs"].to_numpy(float)
    for c in cols:
        if c not in df.columns:
            continue
        y = df[c].to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(y)
        if mask.sum() >= 2:
            out[f"{side}_{sprefix}_{c}"] = np.interp(grid_abs, t[mask], y[mask])
    return out


def add_dynamic_features_inplace(df: pd.DataFrame, rolling_window: int = 21) -> None:
    """Add norms, dynamic norms, smoothed norms and dynamic axes."""
    for side in ["hand", "belt"]:
        for sensor in ["acc", "gyro"]:
            cols = [f"{side}_{sensor}_{a}" for a in ["x", "y", "z"]]
            if not all(c in df.columns for c in cols):
                continue
            vals = df[cols].to_numpy(float)
            norm = np.linalg.norm(vals, axis=1)
            norm_smooth = pd.Series(norm).rolling(rolling_window, center=True, min_periods=1).mean().to_numpy()
            df[f"{side}_{sensor}_norm"] = norm
            df[f"{side}_{sensor}_norm_smooth"] = norm_smooth
            df[f"{side}_{sensor}_dyn_norm"] = np.abs(norm - norm_smooth)
            df[f"{side}_{sensor}_dyn_norm_smooth"] = pd.Series(df[f"{side}_{sensor}_dyn_norm"]).rolling(
                rolling_window, center=True, min_periods=1
            ).mean().to_numpy()
            for j, c in enumerate(cols):
                smooth_c = pd.Series(vals[:, j]).rolling(rolling_window, center=True, min_periods=1).mean().to_numpy()
                df[c + "_dyn"] = vals[:, j] - smooth_c


def synchronize_phone_zips(
    hand_zip: str | Path,
    belt_zip: str | Path,
    *,
    use_accelerometer: bool = True,
    use_gyroscope: bool = True,
    use_orientation: bool = False,
    target_hz: Optional[float] = None,
    target_hz_quantile: float = 0.9,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    sensors = [
        ("Accelerometer", "acc", use_accelerometer),
        ("Gyroscope", "gyro", use_gyroscope),
        ("Orientation", "ori", use_orientation),
    ]
    loaded: Dict[Tuple[str, str], pd.DataFrame] = {}
    hzs, starts, ends = [], [], []
    for side, zpath in [("hand", hand_zip), ("belt", belt_zip)]:
        for sensor, sprefix, enabled in sensors:
            if not enabled:
                continue
            try:
                df = load_sensor_from_zip(zpath, sensor)
            except FileNotFoundError:
                continue
            cols = [c for c in _sensor_columns(sensor, use_accelerometer, use_gyroscope, use_orientation) if c in df.columns]
            if not cols:
                continue
            loaded[(side, sprefix)] = df
            starts.append(float(df["time_seconds_abs"].iloc[0]))
            ends.append(float(df["time_seconds_abs"].iloc[-1]))
            hzs.append(estimate_hz(df["time_seconds_abs"].to_numpy()))
    if not loaded:
        raise ValueError("No selected sensor streams were found.")
    start, end = max(starts), min(ends)
    if end <= start:
        raise ValueError("No temporal overlap between phone streams.")
    finite_hz = [h for h in hzs if np.isfinite(h) and h > 0]
    if target_hz is None:
        target_hz = float(np.quantile(finite_hz, target_hz_quantile)) if finite_hz else 50.0
    dt = 1.0 / target_hz
    grid_abs = np.arange(start, end, dt)
    synced = pd.DataFrame({"time_seconds_abs": grid_abs, "t_rel": grid_abs - grid_abs[0]})
    for side in ["hand", "belt"]:
        for sensor, sprefix, enabled in sensors:
            if not enabled or (side, sprefix) not in loaded:
                continue
            interp = _interp_sensor(
                loaded[(side, sprefix)], grid_abs, side, sprefix,
                _sensor_columns(sensor, use_accelerometer, use_gyroscope, use_orientation),
            )
            for c in interp.columns:
                if c != "time_seconds_abs":
                    synced[c] = interp[c].to_numpy()
    add_dynamic_features_inplace(synced)
    info = {"target_hz": float(target_hz), "dt": float(dt), "overlap_seconds": float(end - start), "n_samples": int(len(synced))}
    for (side, sprefix), df in loaded.items():
        info[f"{side}_{sprefix}_hz_est"] = estimate_hz(df["time_seconds_abs"].to_numpy())
        info[f"{side}_{sprefix}_n_raw"] = int(len(df))
    return synced, info

# =============================================================================
# Feature/dataset construction
# =============================================================================

def selected_signal_columns(
    df: pd.DataFrame,
    side: str,
    *,
    use_accelerometer: bool = True,
    use_gyroscope: bool = True,
    use_orientation: bool = False,
    include_dynamic_axes: bool = False,
) -> List[str]:
    cols: List[str] = []
    if use_accelerometer:
        cols += [f"{side}_acc_{a}" for a in ["x", "y", "z"] if f"{side}_acc_{a}" in df.columns]
        if include_dynamic_axes:
            cols += [f"{side}_acc_{a}_dyn" for a in ["x", "y", "z"] if f"{side}_acc_{a}_dyn" in df.columns]
    if use_gyroscope:
        cols += [f"{side}_gyro_{a}" for a in ["x", "y", "z"] if f"{side}_gyro_{a}" in df.columns]
        if include_dynamic_axes:
            cols += [f"{side}_gyro_{a}_dyn" for a in ["x", "y", "z"] if f"{side}_gyro_{a}_dyn" in df.columns]
    if use_orientation:
        ori = ["yaw", "qx", "qz", "roll", "qw", "qy", "pitch"]
        cols += [f"{side}_ori_{a}" for a in ori if f"{side}_ori_{a}" in df.columns]
    return cols


def intensity_regime_columns(df: pd.DataFrame, side: Optional[str] = None) -> List[str]:
    sides = [side] if side in ["hand", "belt"] else ["hand", "belt"]
    cols: List[str] = []
    for s in sides:
        for sensor in ["acc", "gyro"]:
            for suffix in ["dyn_norm_smooth", "norm_smooth", "dyn_norm"]:
                c = f"{s}_{sensor}_{suffix}"
                if c in df.columns:
                    cols.append(c)
    return cols


def data_regime_columns(df: pd.DataFrame, side: str = "hand", **kwargs) -> List[str]:
    return selected_signal_columns(df, side, **kwargs)


def select_lag_by_cca(
    df: pd.DataFrame,
    x_cols: Sequence[str],
    y_cols: Sequence[str],
    *,
    min_lag: int = 1,
    max_lag: int = 60,
) -> Tuple[int, pd.DataFrame]:
    """Choose X[t-lag] -> Y[t] lag by 1-component CCA score."""
    rows = []
    for lag in range(min_lag, max_lag + 1):
        X = df[list(x_cols)].iloc[:-lag].to_numpy(float)
        Y = df[list(y_cols)].iloc[lag:].to_numpy(float)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        if mask.sum() < 10:
            score = np.nan
        else:
            Xs = StandardScaler().fit_transform(X[mask])
            Ys = StandardScaler().fit_transform(Y[mask])
            try:
                cca = CCA(n_components=1, max_iter=1000)
                u, v = cca.fit_transform(Xs, Ys)
                score = float(np.corrcoef(u[:, 0], v[:, 0])[0, 1])
            except Exception:
                score = np.nan
        rows.append({"lag": lag, "cca_corr": score})
    table = pd.DataFrame(rows)
    best = int(table.loc[table["cca_corr"].abs().idxmax(), "lag"])
    return best, table


def make_lagged_dataset(
    df: pd.DataFrame,
    x_cols: Sequence[str],
    y_cols: Sequence[str],
    *,
    causal_lag: int,
    n_lags: int = 12,
    window_tau: int = 1,
    horizon: int = 1,
    extra_regime_cols: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Build X-window -> future Y dataset.

    For each row at target index j, X-window uses indices j-causal_lag-horizon-m*window_tau.
    Y target is at j. This keeps a causal offset X earlier than Y.
    """
    x_cols = list(x_cols)
    y_cols = list(y_cols)
    extra_regime_cols = list(extra_regime_cols or [])
    min_back = causal_lag + horizon + (n_lags - 1) * window_tau
    rows_X, rows_Y, times, regime_extra = [], [], [], []
    start = min_back
    for j in range(start, len(df)):
        idxs = [j - causal_lag - horizon - m * window_tau for m in range(n_lags)]
        if min(idxs) < 0:
            continue
        xwin = df.loc[idxs, x_cols].to_numpy(float).reshape(-1)
        y = df.loc[j, y_cols].to_numpy(float)
        if not (np.isfinite(xwin).all() and np.isfinite(y).all()):
            continue
        rows_X.append(xwin)
        rows_Y.append(y)
        times.append(float(df.loc[j, "t_rel"]))
        if extra_regime_cols:
            regime_extra.append(df.loc[j, extra_regime_cols].to_numpy(float))
    X = np.asarray(rows_X, dtype=float)
    Y = np.asarray(rows_Y, dtype=float)
    t = np.asarray(times, dtype=float)
    out = {
        "X": X,
        "Y": Y,
        "t": t,
        "x_window_feature_names": [f"{c}(t-{causal_lag + horizon + m*window_tau})" for m in range(n_lags) for c in x_cols],
        "y_feature_names": y_cols,
        "regime_extra": np.asarray(regime_extra, dtype=float) if extra_regime_cols else None,
        "regime_extra_names": extra_regime_cols,
    }
    return out


def mask_from_intervals(t: Array, intervals: Sequence[Tuple[float, float]]) -> Array:
    t = np.asarray(t, dtype=float)
    mask = np.zeros_like(t, dtype=bool)
    for a, b in intervals:
        mask |= (t >= a) & (t <= b)
    return mask


def prediction_metrics(y_true: Array, y_pred: Array) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred, multioutput="variance_weighted"))
    denom = float(np.mean((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    nmse = float(mse / (denom + 1e-12))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NMSE": nmse}

# =============================================================================
# CCA projector
# =============================================================================

class CCALatentProjector:
    def __init__(self, n_components: int = 3, ridge_decode: float = 1e-4):
        self.n_components = int(n_components)
        self.ridge_decode = float(ridge_decode)

    def fit(self, X: Array, Y: Array) -> "CCALatentProjector":
        self.x_scaler_ = StandardScaler().fit(X)
        self.y_scaler_ = StandardScaler().fit(Y)
        Xs = self.x_scaler_.transform(X)
        Ys = self.y_scaler_.transform(Y)
        n_comp = min(self.n_components, Xs.shape[1], Ys.shape[1], Xs.shape[0] - 1)
        self.n_components_ = int(max(1, n_comp))
        self.cca_ = CCA(n_components=self.n_components_, max_iter=2000)
        X_lat, Y_lat = self.cca_.fit_transform(Xs, Ys)
        self.x_lat_scaler_ = StandardScaler().fit(X_lat)
        self.y_lat_scaler_ = StandardScaler().fit(Y_lat)
        X_lat = self.x_lat_scaler_.transform(X_lat)
        Y_lat = self.y_lat_scaler_.transform(Y_lat)
        # ridge decoders from latent to standardized observed spaces
        self.decode_y_ = Ridge(alpha=self.ridge_decode, fit_intercept=True).fit(Y_lat, Ys)
        self.decode_x_ = Ridge(alpha=self.ridge_decode, fit_intercept=True).fit(X_lat, Xs)
        self.train_x_lat_ = X_lat
        self.train_y_lat_ = Y_lat
        return self

    def transform(self, X: Array, Y: Optional[Array] = None):
        Xs = self.x_scaler_.transform(X)
        if Y is None:
            # sklearn CCA transform accepts X only
            X_lat = self.cca_.transform(Xs)
            return self.x_lat_scaler_.transform(X_lat)
        Ys = self.y_scaler_.transform(Y)
        X_lat, Y_lat = self.cca_.transform(Xs, Ys)
        return self.x_lat_scaler_.transform(X_lat), self.y_lat_scaler_.transform(Y_lat)

    def inverse_y(self, Y_lat: Array) -> Array:
        Ys_hat = self.decode_y_.predict(Y_lat)
        return self.y_scaler_.inverse_transform(Ys_hat)

    def inverse_x(self, X_lat: Array) -> Array:
        Xs_hat = self.decode_x_.predict(X_lat)
        return self.x_scaler_.inverse_transform(Xs_hat)

# =============================================================================
# Hybrid model
# =============================================================================

@dataclass
class HybridPartialBZConfig:
    n_regimes: int = 2
    latent_dim: int = 3
    max_iter_ab: int = 80
    tol: float = 1e-5
    ridge_a: float = 1e-2
    ridge_b: float = 1e-2
    ridge_c: float = 1e-2
    sigma_floor: float = 1e-3
    gmm_reg_covar: float = 1e-5
    random_state: int = 42
    c_mode: str = "partial_bz_direct"  # "partial_bz_direct", "residual", "direct_residual_y", "direct_y"
    bz_weight: float = 0.5  # rho in Y_hat = rho*Bz + C_e X
    sigma_y_floor: float = 5e-2
    c_norm_clip: Optional[float] = None


def _weighted_ridge_fit(X: Array, Y: Array, w: Array, alpha: float, fit_intercept: bool = False) -> Ridge:
    reg = Ridge(alpha=float(alpha), fit_intercept=fit_intercept)
    reg.fit(X, Y, sample_weight=np.asarray(w, dtype=float))
    return reg


def _cov_from_weighted(values: Array, means: Array, covs: Optional[Array], weights: Array, floor: float = 1e-6) -> Array:
    # values: posterior means m_t, covs: posterior cov S_t or None, weights: T
    W = weights.sum() + 1e-12
    mu = (weights[:, None] * means).sum(axis=0) / W
    diff = means - mu
    S = (weights[:, None, None] * (diff[:, :, None] @ diff[:, None, :])).sum(axis=0) / W
    if covs is not None:
        S += (weights[:, None, None] * covs).sum(axis=0) / W
    S = 0.5 * (S + S.T) + floor * np.eye(S.shape[0])
    return mu, S


class HybridPartialBZMixtureSCM:
    """Hybrid model: A,B from stable latent EM; C_e from separate direct stage."""

    def __init__(self, config: HybridPartialBZConfig):
        self.config = config

    def _init_regimes(self, regime_features: Array) -> None:
        cfg = self.config
        self.regime_scaler_ = StandardScaler().fit(regime_features)
        R = self.regime_scaler_.transform(regime_features)
        self.regime_gmm_ = GaussianMixture(
            n_components=cfg.n_regimes,
            covariance_type="full",
            reg_covar=cfg.gmm_reg_covar,
            random_state=cfg.random_state,
            n_init=10,
            max_iter=500,
        ).fit(R)
        self.resp_ = self.regime_gmm_.predict_proba(R)
        self.weights_ = self.resp_.mean(axis=0)

    def regime_responsibilities(self, regime_features: Array) -> Array:
        R = self.regime_scaler_.transform(regime_features)
        return self.regime_gmm_.predict_proba(R)

    def _posterior_z_xy(self, X: Array, Y: Array, resp: Array) -> Tuple[Array, Array, Array]:
        cfg = self.config
        T, k = X.shape[0], cfg.latent_dim
        E = cfg.n_regimes
        m = np.zeros((T, E, k))
        S = np.zeros((T, E, k, k))
        eye = np.eye(k)
        inv_sx = 1.0 / max(self.sigma_x_, cfg.sigma_floor) ** 2
        inv_sy = 1.0 / max(self.sigma_y_, cfg.sigma_floor) ** 2
        AtA = self.A_.T @ self.A_ * inv_sx
        BtB = self.B_.T @ self.B_ * inv_sy
        for e in range(E):
            Se_inv = np.linalg.pinv(self.Sigma_z_[e])
            Prec = Se_inv + AtA + BtB + cfg.sigma_floor * eye
            Cov = np.linalg.pinv(Prec)
            rhs_prior = Se_inv @ self.mu_z_[e]
            rhs = rhs_prior[None, :] + (X @ self.A_) * inv_sx + (Y @ self.B_) * inv_sy
            m[:, e, :] = rhs @ Cov.T
            S[:, e, :, :] = Cov
        z_mean = (resp[:, :, None] * m).sum(axis=1)
        return m, S, z_mean

    def _posterior_z_x(self, X: Array, resp: Array) -> Tuple[Array, Array]:
        cfg = self.config
        T, k = X.shape[0], cfg.latent_dim
        E = cfg.n_regimes
        m = np.zeros((T, E, k))
        eye = np.eye(k)
        inv_sx = 1.0 / max(self.sigma_x_, cfg.sigma_floor) ** 2
        AtA = self.A_.T @ self.A_ * inv_sx
        for e in range(E):
            Se_inv = np.linalg.pinv(self.Sigma_z_[e])
            Prec = Se_inv + AtA + cfg.sigma_floor * eye
            Cov = np.linalg.pinv(Prec)
            rhs_prior = Se_inv @ self.mu_z_[e]
            rhs = rhs_prior[None, :] + (X @ self.A_) * inv_sx
            m[:, e, :] = rhs @ Cov.T
        z_mean = (resp[:, :, None] * m).sum(axis=1)
        return m, z_mean

    def fit(self, X_lat: Array, Y_lat: Array, regime_features: Array) -> "HybridTwoStageMixtureSCM":
        cfg = self.config
        X_lat = np.asarray(X_lat, dtype=float)
        Y_lat = np.asarray(Y_lat, dtype=float)
        T, kx = X_lat.shape
        ky = Y_lat.shape[1]
        if kx != ky:
            raise ValueError("This implementation expects X_lat and Y_lat to have the same latent dimension.")
        k = kx
        cfg.latent_dim = k
        self._init_regimes(regime_features)
        resp = self.resp_
        E = cfg.n_regimes

        # Initialize Z as average CCA latent, then regime priors from responsibilities.
        Z = 0.5 * (X_lat + Y_lat)
        self.mu_z_ = np.zeros((E, k))
        self.Sigma_z_ = np.zeros((E, k, k))
        for e in range(E):
            mu, S = _cov_from_weighted(Z, Z, None, resp[:, e], floor=cfg.sigma_floor)
            self.mu_z_[e], self.Sigma_z_[e] = mu, S
        self.A_ = np.eye(k)
        self.B_ = np.eye(k)
        self.sigma_x_ = 0.1
        self.sigma_y_ = 0.1
        self.history_ = []

        prev_obj = -np.inf
        for it in range(cfg.max_iter_ab):
            # E-step: posterior Z given X,Y and fixed regime responsibilities.
            m, S, Z_hat = self._posterior_z_xy(X_lat, Y_lat, resp)

            # M-step A,B via shared weighted ridge using posterior averaged Z.
            # Since sum_e r_te = 1, ordinary ridge on Z_hat is sufficient, but we
            # keep the interpretation that Z_hat is responsibility-weighted.
            regA = Ridge(alpha=cfg.ridge_a, fit_intercept=False).fit(Z_hat, X_lat)
            regB = Ridge(alpha=cfg.ridge_b, fit_intercept=False).fit(Z_hat, Y_lat)
            self.A_ = regA.coef_.T  # X ~= Z @ A.T, so A maps Z->X as X = Z A^T
            self.B_ = regB.coef_.T

            # Update regime priors p_e(Z) using per-regime posterior moments.
            for e in range(E):
                W = resp[:, e].sum() + 1e-12
                mu = (resp[:, e, None] * m[:, e, :]).sum(axis=0) / W
                diff = m[:, e, :] - mu
                cov = (resp[:, e, None, None] * (S[:, e] + diff[:, :, None] @ diff[:, None, :])).sum(axis=0) / W
                self.mu_z_[e] = mu
                self.Sigma_z_[e] = 0.5 * (cov + cov.T) + cfg.sigma_floor * np.eye(k)

            # Update isotropic noises with floors.
            X_res = X_lat - Z_hat @ self.A_.T
            Y_res = Y_lat - Z_hat @ self.B_.T
            self.sigma_x_ = float(max(np.sqrt(np.mean(X_res ** 2)), cfg.sigma_floor))
            self.sigma_y_ = float(max(np.sqrt(np.mean(Y_res ** 2)), cfg.sigma_y_floor))

            # Pseudo objective: expected complete reconstruction score + prior score.
            rec = -0.5 * (np.mean(X_res ** 2) / (self.sigma_x_ ** 2) + np.mean(Y_res ** 2) / (self.sigma_y_ ** 2))
            reg = -cfg.ridge_a * float(np.sum(self.A_ ** 2)) - cfg.ridge_b * float(np.sum(self.B_ ** 2))
            obj = rec + reg
            self.history_.append({
                "iter": it,
                "objective": float(obj),
                "reconstruction_score": float(rec),
                "x_mse": float(np.mean(X_res ** 2)),
                "y_mse": float(np.mean(Y_res ** 2)),
                "A_norm": float(np.linalg.norm(self.A_)),
                "B_norm": float(np.linalg.norm(self.B_)),
                "sigma_x": float(self.sigma_x_),
                "sigma_y": float(self.sigma_y_),
                "resp_entropy": float(-np.mean(np.sum(resp * np.log(resp + 1e-12), axis=1))),
            })
            if it > 2 and abs(obj - prev_obj) < cfg.tol * (1 + abs(prev_obj)):
                break
            prev_obj = obj

        # Store train posterior and fit C_e separately.
        self.m_train_e_, self.S_train_e_, self.z_train_xy_ = self._posterior_z_xy(X_lat, Y_lat, resp)
        self._fit_direct_channels(X_lat, Y_lat, resp, self.m_train_e_)
        self.n_iter_done_ = len(self.history_)
        return self

    def _fit_direct_channels(self, X_lat: Array, Y_lat: Array, resp: Array, m_e: Array) -> None:
        cfg = self.config
        E = cfg.n_regimes
        k = X_lat.shape[1]
        self.C_ = np.zeros((E, k, k))
        self.C_intercept_ = np.zeros((E, k))
        for e in range(E):
            Z_e = m_e[:, e, :]
            if cfg.c_mode == "partial_bz_direct":
                # Main v5 mode: direct channel explains Y after only a partial shared-latent contribution.
                # This prevents BZ from eating the whole Y signal while still retaining the common regime term.
                X_in = X_lat
                Y_tar = Y_lat - cfg.bz_weight * (Z_e @ self.B_.T)
            elif cfg.c_mode == "residual":
                X_in = X_lat - Z_e @ self.A_.T
                Y_tar = Y_lat - Z_e @ self.B_.T
            elif cfg.c_mode == "direct_residual_y":
                X_in = X_lat
                Y_tar = Y_lat - Z_e @ self.B_.T
            elif cfg.c_mode == "direct_y":
                X_in = X_lat
                Y_tar = Y_lat
            else:
                raise ValueError(f"Unknown c_mode={cfg.c_mode!r}")
            reg = _weighted_ridge_fit(X_in, Y_tar, resp[:, e], alpha=cfg.ridge_c, fit_intercept=True)
            C = reg.coef_  # Y = X @ C.T + intercept
            if cfg.c_norm_clip is not None:
                nrm = np.linalg.norm(C)
                if nrm > cfg.c_norm_clip:
                    C = C * (cfg.c_norm_clip / (nrm + 1e-12))
            self.C_[e] = C
            self.C_intercept_[e] = reg.intercept_

    def predict_latent(self, X_lat: Array, regime_features: Array, return_details: bool = False):
        resp = self.regime_responsibilities(regime_features)
        m_x_e, z_x = self._posterior_z_x(np.asarray(X_lat, dtype=float), resp)
        T, E = resp.shape
        k = X_lat.shape[1]
        Y_e = np.zeros((T, E, k))
        for e in range(E):
            Z_e = m_x_e[:, e, :]
            base_full = Z_e @ self.B_.T
            if self.config.c_mode == "partial_bz_direct":
                base = self.config.bz_weight * base_full
                corr = X_lat @ self.C_[e].T + self.C_intercept_[e]
                Y_e[:, e, :] = base + corr
            elif self.config.c_mode == "residual":
                base = base_full
                X_in = X_lat - Z_e @ self.A_.T
                corr = X_in @ self.C_[e].T + self.C_intercept_[e]
                Y_e[:, e, :] = base + corr
            elif self.config.c_mode == "direct_residual_y":
                base = base_full
                corr = X_lat @ self.C_[e].T + self.C_intercept_[e]
                Y_e[:, e, :] = base + corr
            elif self.config.c_mode == "direct_y":
                base = base_full
                corr = X_lat @ self.C_[e].T + self.C_intercept_[e]
                Y_e[:, e, :] = base + corr
        Y_hat = (resp[:, :, None] * Y_e).sum(axis=1)
        if return_details:
            return Y_hat, {"resp": resp, "m_x_e": m_x_e, "Y_e": Y_e}
        return Y_hat

    def direct_channel_norms(self) -> Array:
        return np.linalg.norm(self.C_, axis=(1, 2))

    def effective_latent_operators(self) -> Array:
        """Return simple effective operators B A^+ + C_e in latent coordinates."""
        A_pinv = np.linalg.pinv(self.A_)
        shared = self.config.bz_weight * (self.B_ @ A_pinv)
        return np.asarray([shared + self.C_[e] for e in range(self.config.n_regimes)])

    def history_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)

# =============================================================================
# Baselines and graph utilities
# =============================================================================

def fit_predict_baselines(X_train: Array, Y_train: Array, X_val: Array, *, alpha: float = 1.0) -> Dict[str, Array]:
    ridge = Ridge(alpha=alpha).fit(X_train, Y_train)
    return {"Raw Ridge X->Y": ridge.predict(X_val)}


def top_operator_edges(
    op: Array,
    x_names: Sequence[str],
    y_names: Sequence[str],
    *,
    top_k: int = 20,
    min_abs_weight: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for j, yname in enumerate(y_names):
        for i, xname in enumerate(x_names):
            w = float(op[j, i]) if j < op.shape[0] and i < op.shape[1] else 0.0
            if abs(w) >= min_abs_weight:
                rows.append({"source": xname, "target": yname, "weight": w, "abs_weight": abs(w)})
    out = pd.DataFrame(rows).sort_values("abs_weight", ascending=False).head(top_k)
    return out.reset_index(drop=True)


# Backward-compatible aliases
HybridSCMConfig = HybridPartialBZConfig
HybridTwoStageMixtureSCM = HybridPartialBZMixtureSCM

def c_difference_norm(model) -> float:
    if getattr(model, "C_", None) is None or model.C_.shape[0] < 2:
        return float("nan")
    return float(np.linalg.norm(model.C_[0] - model.C_[1]))

def c_singular_values(model) -> pd.DataFrame:
    rows=[]
    for e, C in enumerate(model.C_):
        s=np.linalg.svd(C, compute_uv=False)
        for i, val in enumerate(s):
            rows.append({"regime": e, "index": i, "singular_value": float(val)})
    return pd.DataFrame(rows)
