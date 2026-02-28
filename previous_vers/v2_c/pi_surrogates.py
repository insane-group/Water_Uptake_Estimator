# pi_surrogates.py — Linear Alpha with Mass-Based Features
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

def set_global_seed(seed: int = 0):
    try: import random; random.seed(seed)
    except: pass
    np.random.seed(seed)

def assert_sorted_time(t_h: np.ndarray):
    if not np.all(np.diff(np.asarray(t_h, float)) >= 0):
        raise ValueError("time array must be sorted ascending.")

def mae_on_val(y_true, y_pred, idx_train, n_total):
    mask = np.ones(n_total, dtype=bool)
    mask[np.asarray(idx_train, int)] = False
    if not np.any(mask): return mean_absolute_error(y_true, y_pred)
    return mean_absolute_error(y_true[mask], y_pred[mask])

def _clean_train_idx(idx, n_total):
    return np.unique(np.asarray(idx, int)[(np.asarray(idx, int) >= 0) & (np.asarray(idx, int) < n_total)])

# --- Physics Functions ---
def fickian_fraction(t_h: np.ndarray, tau_h: float) -> np.ndarray:
    t = np.asarray(t_h).reshape(1, -1)
    n = np.arange(60).reshape(-1, 1)
    lam2 = (2*n + 1)**2
    exps = np.exp(-lam2 * (t / tau_h)) / lam2
    return 1.0 - (8.0/np.pi**2) * np.sum(exps, axis=0)

def fickian_mass_curve(t_h, M0, Minf, tau_h):
    return M0 + (Minf - M0) * fickian_fraction(t_h, tau_h)

def fit_fickian_params(t_train_h, M_train, M0_full):
    M_tr = np.asarray(M_train, float); t_tr = np.asarray(t_train_h, float)
    M0 = float(M0_full)
    tau_grid = 10.0 ** np.linspace(0.3, 4.0, 60)
    best = {"tau_h": 100.0, "Minf": np.max(M_train), "sse": np.inf}
    for tau in tau_grid:
        F = fickian_fraction(t_tr, tau)
        denom = float(np.dot(F, F)) + 1e-12
        alpha_f = max(0.0, float(np.dot(F, (M_tr - M0)) / denom))
        Minf_hat = max(M0 + alpha_f, float(np.max(M_tr)))
        y_hat = M0 + (Minf_hat - M0) * F
        sse = float(np.sum((M_tr - y_hat)**2))
        if sse < best["sse"]: best = {"tau_h": float(tau), "Minf": float(Minf_hat), "sse": sse}
    return best

# --- LINEAR SWELLING PHYSICS ---
def thickness_physics_linear(M_curve: np.ndarray, h_dry: float, alpha: float, M_dry: float) -> np.ndarray:
    """
    Linear swelling rule: h(t) = h_dry * (1 + alpha * dM/M_dry)
    Since M_curve saturates, this function saturates automatically.
    """
    if M_dry < 1e-9: return np.full_like(M_curve, h_dry)
    dm = (M_curve - M_dry) / M_dry
    return h_dry * (1.0 + alpha * dm)

def rom_density(M_curve, h_curve, M0, rho_dry=1150.0, rho_fluid=1000.0):
    w = np.maximum(0.0, (M_curve - M0) / np.maximum(M_curve, 1e-12))
    return 1.0 / ((w / rho_fluid) + ((1 - w) / rho_dry))

def fit_alpha_from_cases(cases: List[Dict]) -> float:
    # Aggregates data from multiple cases to find a global alpha
    dm_all, dh_all = [], []
    for d in cases:
        M = d["mass"]; h = d["thick"]
        if len(M) < 1: continue
        M0, h0 = float(M[0]), float(h[0])
        if M0 < 1e-9: continue
        dm_all.append((M - M0)/M0)
        dh_all.append((h - h0)/h0)
    if not dm_all: return 0.5
    dm, dh = np.concatenate(dm_all), np.concatenate(dh_all)
    denom = np.dot(dm, dm)
    return float(np.dot(dm, dh)/denom) if denom > 1e-9 else 0.5

# --- Surrogates ---
@dataclass
class PISpec1D: extra_hours: Optional[np.ndarray] = None; pi_percent: float = 1.0
@dataclass
class MassResult:
    t_h: np.ndarray; M_phys: np.ndarray; M_pure_rf: np.ndarray; M_pred: np.ndarray
    mae: float; lambda_star: float; idx_train: np.ndarray; params: Dict[str, float]
    pi_hours: Optional[np.ndarray] = None; pi_values: Optional[np.ndarray] = None

class MassSurrogate:
    def __init__(self, rf_depth=3, seed=0): self.rf_depth=rf_depth; self.seed=seed
    def train(self, t_h, M_true, train_idx, lambdas, pi=None, pi_weight=1.0) -> MassResult:
        set_global_seed(self.seed); idx = _clean_train_idx(train_idx, len(t_h))
        M0 = float(M_true[0]); t_tr, M_tr = t_h[idx], M_true[idx]
        fit = fit_fickian_params(t_tr, M_tr, M0)
        M_phys = fickian_mass_curve(t_h, M0, fit["Minf"], fit["tau_h"])
        
        rf_pure = RandomForestRegressor(n_estimators=300, max_depth=self.rf_depth, random_state=self.seed)
        rf_pure.fit(t_tr.reshape(-1, 1), M_tr); M_pure = rf_pure.predict(t_h.reshape(-1, 1))

        X_tr, y_tr, w_tr = t_tr.reshape(-1, 1), M_tr.copy(), np.ones(len(M_tr))
        pi_h, pi_v = None, None
        if pi and pi.extra_hours is not None:
            eh = np.asarray(pi.extra_hours, float)
            k = int(np.ceil(len(eh)*pi.pi_percent))
            if k>0:
                eh = eh[:k]; mp = np.interp(eh, t_h, M_phys)
                X_tr = np.vstack([X_tr, eh.reshape(-1, 1)])
                y_tr = np.concatenate([y_tr, mp])
                w_tr = np.concatenate([w_tr, np.full(len(mp), pi_weight)])
                pi_h, pi_v = eh, mp
        
        rf_hyb = RandomForestRegressor(n_estimators=300, max_depth=self.rf_depth, random_state=self.seed)
        rf_hyb.fit(X_tr, y_tr, sample_weight=w_tr); M_hyb = rf_hyb.predict(t_h.reshape(-1, 1))
        
        best = {"mae": np.inf, "lam": 0.0, "pred": M_hyb}
        for lam in lambdas:
            blend = (1-lam)*M_hyb + lam*M_phys
            mae = mae_on_val(M_true, blend, idx, len(t_h))
            if mae < best["mae"]: best = {"mae": mae, "lam": lam, "pred": blend}
        return MassResult(t_h, M_phys, M_pure, best["pred"], best["mae"], best["lam"], idx, fit, pi_h, pi_v)

@dataclass
class ThickResult:
    t_h: np.ndarray; h_phys: np.ndarray; h_pure_rf: np.ndarray; h_pred: np.ndarray
    mae: float; lambda_star: float; idx_train: np.ndarray; alpha: float
    pi_hours: Optional[np.ndarray] = None; pi_values: Optional[np.ndarray] = None

class ThicknessSurrogate:
    def __init__(self, rf_depth=3, seed=0): self.rf_depth=rf_depth; self.seed=seed
    
    def train(self, t_h, h_true, M_curve, M_dry, h_dry, train_idx, lambdas,
              augment=False, pi=None, pi_weight=1.0, feature_mode="mass", fixed_alpha=None) -> ThickResult:
        
        set_global_seed(self.seed); idx = _clean_train_idx(train_idx, len(t_h))
        t_tr, h_tr = t_h[idx], h_true[idx]

        # 1. Determine Alpha
        if fixed_alpha is not None:
            alpha = float(fixed_alpha)
        else:
            # Fallback: Local calculation
            M_at_tr = np.interp(t_tr, t_h, M_curve)
            if M_dry < 1e-9: dm = M_at_tr
            else: dm = (M_at_tr - M_dry) / M_dry
            dh = (h_tr - h_dry) / max(h_dry, 1e-12)
            
            valid = np.abs(dm) > 0.001
            if np.any(valid):
                alpha = float(np.mean(dh[valid] / dm[valid]))
            else: alpha = 0.5

        # 2. Physics Baseline: Linear Law (Saturates because M_curve saturates)
        h_phys_full = thickness_physics_linear(M_curve, h_dry, alpha, M_dry)

        # 3. Features: Mass or Time
        if feature_mode == "mass":
            # Input is Mass (which captures the saturation shape)
            X_full = M_curve.reshape(-1, 1)
            X_tr = M_curve[idx].reshape(-1, 1)
        else:
            X_full = t_h.reshape(-1, 1)
            X_tr = t_tr.reshape(-1, 1)

        # 4. Pure RF
        rf_pure = RandomForestRegressor(n_estimators=300, max_depth=self.rf_depth, random_state=self.seed)
        rf_pure.fit(X_tr, h_tr); h_pure = rf_pure.predict(X_full)

        # 5. Hybrid RF (Augmented)
        X_aug, h_aug, w_aug = X_tr.copy(), h_tr.copy(), np.ones(len(h_tr))
        if augment and len(idx) >= 2:
            resid = h_tr - h_phys_full[idx]
            feat_aug, val_aug = [], []
            for i in range(len(idx)-1):
                t0, t1 = t_h[idx[i]], t_h[idx[i+1]]
                r0, r1 = resid[i], resid[i+1]
                for k in range(1, 4):
                    tau = k/4.0
                    tm = (1-tau)*t0 + tau*t1
                    rm = (1-tau)*r0 + tau*r1
                    
                    M_tm = np.interp(tm, t_h, M_curve)
                    h_phys_tm = thickness_physics_linear(np.array([M_tm]), h_dry, alpha, M_dry)[0]
                    
                    feat_aug.append(M_tm if feature_mode=="mass" else tm)
                    val_aug.append(h_phys_tm + rm)
            if len(feat_aug) > 0:
                X_aug = np.vstack([X_aug, np.array(feat_aug).reshape(-1, 1)])
                h_aug = np.concatenate([h_aug, np.array(val_aug)])
                w_aug = np.concatenate([w_aug, np.ones(len(val_aug))])

        pi_h, pi_v = None, None
        if pi and pi.extra_hours is not None:
            eh = np.asarray(pi.extra_hours, float)
            k = int(np.ceil(len(eh)*pi.pi_percent))
            if k>0:
                eh = eh[:k]
                hp = np.interp(eh, t_h, h_phys_full) 
                mp = np.interp(eh, t_h, M_curve)
                fp = mp.reshape(-1, 1) if feature_mode=="mass" else eh.reshape(-1, 1)
                
                X_aug = np.vstack([X_aug, fp])
                h_aug = np.concatenate([h_aug, hp])
                w_aug = np.concatenate([w_aug, np.full(len(hp), pi_weight)])
                pi_h, pi_v = eh, hp

        rf_hyb = RandomForestRegressor(n_estimators=300, max_depth=self.rf_depth, random_state=self.seed)
        rf_hyb.fit(X_aug, h_aug, sample_weight=w_aug); h_hyb = rf_hyb.predict(X_full)

        best = {"mae": np.inf, "lam": 0.0, "pred": h_hyb}
        for lam in lambdas:
            blend = h_hyb + lam*h_phys_full
            mae = mae_on_val(h_true, blend, idx, len(t_h))
            if mae < best["mae"]: best = {"mae": mae, "lam": lam, "pred": blend}
        return ThickResult(t_h, h_phys_full, h_pure, best["pred"], best["mae"], best["lam"], idx, alpha, pi_h, pi_v)

@dataclass
class DensityResult:
    t_h: np.ndarray; rho_phys: np.ndarray; rho_pure_rf: np.ndarray; rho_pred: np.ndarray
    mae: float; lambda_star: float; idx_train: np.ndarray

class DensitySurrogate:
    def __init__(self, rf_depth=3, seed=0): self.rf_depth=rf_depth; self.seed=seed
    def train(self, t_h, M_curve, h_curve, M_dry, train_idx, lambdas, rho_dry=1150.0, rho_fluid=1000.0):
        set_global_seed(self.seed); idx = _clean_train_idx(train_idx, len(t_h))
        rho_p = rom_density(M_curve, h_curve, M_dry, rho_dry, rho_fluid)
        X = np.column_stack([t_h, M_curve, h_curve])
        rf = RandomForestRegressor(n_estimators=300, max_depth=self.rf_depth, random_state=self.seed)
        rf.fit(X[idx], rho_p[idx]); rho_rf = rf.predict(X)
        best = {"mae": np.inf, "lam": 0.0, "pred": rho_rf}
        for lam in lambdas:
            blend = (1-lam)*rho_rf + lam*rho_p
            mae = mae_on_val(rho_p, blend, idx, len(t_h))
            if mae < best["mae"]: best = {"mae": mae, "lam": lam, "pred": blend}
        return DensityResult(t_h, rho_p, rho_rf, best["pred"], best["mae"], best["lam"], idx)

@dataclass
class CGResult:
    hours: np.ndarray; gv_true_ms: np.ndarray; gv_pred10_ms: np.ndarray; mae10_ms: float; n_extra_used: int; pi_desc: str; pi_hours: Optional[np.ndarray]=None; pi_values_ms: Optional[np.ndarray]=None

def train_cg_with_pi(t_mass, rho_curve, h_curve, meas_hours, gv_kms, pi_spec, seed=0, pi_weight=1.0):
    set_global_seed(seed)
    rho_m = np.interp(meas_hours, t_mass, rho_curve)
    h_m = np.interp(meas_hours, t_mass, h_curve)/1000.0
    X = np.column_stack([rho_m, h_m]); y = gv_kms*1000.0; w = np.ones(len(y))
    pi_pt = None; k=0
    if pi_spec and pi_spec.extra_hours is not None:
        eh = np.asarray(pi_spec.extra_hours, float)
        eg = np.asarray(pi_spec.extra_gv_ms, float)
        k = int(np.ceil(len(eh)*pi_spec.pi_percent))
        if k>0:
            eh = eh[:k]
            rp = np.interp(eh, t_mass, rho_curve); hp = np.interp(eh, t_mass, h_curve)/1000.0
            X = np.vstack([X, np.column_stack([rp, hp])])
            y = np.concatenate([y, eg[:k]]); w = np.concatenate([w, np.full(k, pi_weight)])
            pi_pt = (eh, eg[:k])
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=seed)
    rf.fit(X, y, sample_weight=w); pred = rf.predict(np.column_stack([rho_m, h_m]))
    return CGResult(meas_hours, gv_kms*1000.0, pred, mean_absolute_error(gv_kms*1000.0, pred), k, pi_spec.desc, pi_pt[0] if pi_pt else None, pi_pt[1] if pi_pt else None)

@dataclass
class EResult:
    hours: np.ndarray; E_true_GPa: Optional[np.ndarray]; E_phys_GPa: np.ndarray; E_ridge_GPa: np.ndarray; E_hybrid_GPa: np.ndarray
    best_w: float; loocv_mae_GPa: Optional[float]; pi_hours: Optional[np.ndarray]=None; pi_values_GPa: Optional[np.ndarray]=None; idx_train: Optional[np.ndarray]=None

@dataclass
class PISpecCG: extra_hours: Optional[np.ndarray]=None; extra_gv_ms: Optional[np.ndarray]=None; pi_percent: float=1.0; desc: str=""
@dataclass
class PISpecE: extra_hours: Optional[np.ndarray]=None; pi_percent: float=1.0; gv_source: str=""; desc: str=""

class YoungsSurrogate:
    def __init__(self, alpha=1.0, seed=0): self.alpha=alpha; self.seed=seed
    def train(self, hours, rho, gv_kms, E_true, gv_infer, pi, train_idx, pi_weight=1.0):
        set_global_seed(self.seed); idx = _clean_train_idx(train_idx, len(hours)) if train_idx is not None else None
        gv_ms = gv_kms*1000.0; E_ph = rho*(gv_infer**2)/1e9
        X = np.column_stack([rho, gv_ms]); y = np.asarray(E_true, float)
        X_tr, y_tr, w_tr = X[idx], y[idx], np.ones(len(idx))
        pi_h, pi_v = None, None
        if pi and pi.extra_hours is not None:
            eh = np.asarray(pi.extra_hours, float)
            r_pi = np.interp(eh, hours, rho); g_pi = np.interp(eh, hours, gv_ms)
            ep = r_pi*(g_pi**2)/1e9
            X_tr = np.vstack([X_tr, np.column_stack([r_pi, g_pi])])
            y_tr = np.concatenate([y_tr, ep]); w_tr = np.concatenate([w_tr, np.full(len(ep), pi_weight)])
            pi_h, pi_v = eh, ep
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        ridge.fit(X_tr, y_tr, ridge__sample_weight=w_tr); E_r = ridge.predict(X)
        mask = np.ones(len(hours), bool); mask[idx] = False; val = np.arange(len(hours))[mask]
        bw, berr = 0.0, np.inf
        if len(val)>0:
            for w in np.linspace(0, 1, 101):
                w=0.75
                eh = w*E_r + (1-w)*E_ph
                err = np.mean(np.abs(eh[val]-y[val]))
                if err < berr: berr=err; bw=w
        return EResult(hours, y, E_ph, E_r, bw*E_r + (1-bw)*E_ph, bw, berr, pi_h, pi_v, idx)