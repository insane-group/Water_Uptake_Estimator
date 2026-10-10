# pi_surrogates.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor

# --- Utilities ---
def set_global_seed(seed: int = 42):
    try: import random; random.seed(seed)
    except: pass
    np.random.seed(seed)

def assert_sorted_time(t: np.ndarray):
    if not np.all(np.diff(np.asarray(t, float)) >= 0):
        raise ValueError("Time must be sorted.")

def complement_idx(n, idx):
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(idx, int)] = False
    return np.where(mask)[0]

def resolve_train_idx(n_total, mode="all", n_train=None, manual_idx=None):
    if mode == "all": return np.arange(n_total)
    if mode == "manual": return np.unique(np.asarray(manual_idx, int))
    if mode == "even":
        if n_train is None: n_train = 3
        return np.unique(np.linspace(0, n_total-1, n_train).round().astype(int))
    return np.arange(n_total)

# --- Physics Specs ---
@dataclass
class PISpec1D:
    hours: np.ndarray
    pi_percent: float = 1.0
    values: Optional[np.ndarray] = None

# --- Results Classes ---
@dataclass
class ResultBase:
    t: np.ndarray
    idx_train: np.ndarray
    pred_pure: np.ndarray
    pred_phys: np.ndarray
    pred_hybrid: np.ndarray
    mae_val: float
    mae_pure: float = 0.0
    lambda_star: float = 0.0
    pi_hours: Optional[np.ndarray] = None
    pi_values: Optional[np.ndarray] = None
    idx_train_pure: Optional[np.ndarray] = None # Tracks pure ML training points

# --- PHYSICS FUNCTIONS ---
def fickian_fraction(t_h: np.ndarray, tau_h: float) -> np.ndarray:
    t = np.asarray(t_h).reshape(1, -1)
    n = np.arange(60).reshape(-1, 1)
    lam2 = (2*n + 1)**2
    tau_safe = max(tau_h, 1e-9)
    exps = np.exp(-lam2 * (t / tau_safe)) / lam2
    return 1.0 - (8.0/np.pi**2) * np.sum(exps, axis=0)

def fickian_model(t, M0, Minf, tau):
    return M0 + (Minf - M0) * fickian_fraction(t, tau)

def thickness_physics_linear(M_curve: np.ndarray, h_dry: float, alpha: float, M_dry: float) -> np.ndarray:
    if M_dry < 1e-9: return np.full_like(M_curve, h_dry)
    dm = (M_curve - M_dry) / M_dry
    return h_dry * (1.0 + alpha * dm)

def rom_density(M_curve, h_curve, M0, rho_dry=1150.0, rho_fluid=1000.0, *, h0=None):
    """Original effective arithmetic blend using absorbed-water mass fraction.

    This reproduces the released model exactly for positive mass inputs. It is
    an empirical density approximation, not an additive-volume or measured-
    swelling identity. h_curve and h0 are retained for caller compatibility
    and do not affect the output. Water uptake below zero is clipped as in
    the original implementation.
    """
    M_curve = np.asarray(M_curve, float)
    if (not np.isfinite(M0) or M0 <= 0 or np.any(~np.isfinite(M_curve))
            or np.any(M_curve <= 0) or min(rho_dry, rho_fluid) <= 0
            or not np.all(np.isfinite([rho_dry, rho_fluid]))):
        raise ValueError("Mass and constituent densities must be finite and positive")
    w = np.maximum(0.0, (M_curve - M0) / np.maximum(M_curve, 1e-12))
    return (1 - w) * rho_dry + w * rho_fluid

def fit_alpha_from_cases(cases: List[Dict]) -> float:
    dm_all, dh_all = [], []
    for d in cases:
        M, h = d["mass"], d["thick"]
        if len(M) < 2: continue
        M0, h0 = float(M[0]), float(h[0])
        if M0 < 1e-9: continue
        dm_all.append((M - M0)/M0)
        dh_all.append((h - h0)/h0)
    if not dm_all: return 0.5
    dm, dh = np.concatenate(dm_all), np.concatenate(dh_all)
    denom = np.dot(dm, dm)
    return float(np.dot(dm, dh)/denom) if denom > 1e-9 else 0.5

# --- 1. MASS SURROGATE ---
class MassSurrogate:
    def __init__(self, seed=42): self.seed = seed
    
    def train(self, t, M, idx_train, pi: Optional[PISpec1D], pi_weight=1.0, lambdas=None, idx_train_pure=None) -> ResultBase:
        set_global_seed(self.seed)
        if idx_train_pure is None: idx_train_pure = idx_train
        
        t_tr, M_tr = t[idx_train], M[idx_train]
        M0 = float(M[0])
        
        # Physics Fit (on Sparse Hybrid Data)
        best = (100.0, float(np.max(M_tr)), np.inf)
        for tau in [10, 20, 50, 100, 200, 500, 800]:
            term = fickian_fraction(t_tr, tau)
            dM = np.median((M_tr - M0) / (term + 1e-6))
            pred = fickian_model(t_tr, M0, M0+dM, tau)
            err = np.mean((pred - M_tr)**2)
            if err < best[2]: best = (tau, M0+dM, err)
        M_phys = fickian_model(t, M0, best[1], best[0])

        # Pure ML (Trained on Denser Data)
        t_tr_pure, M_tr_pure = t[idx_train_pure], M[idx_train_pure]
        rf_pure = RandomForestRegressor(n_estimators=300, random_state=self.seed)
        rf_pure.fit(t_tr_pure.reshape(-1,1), M_tr_pure)
        M_pure = rf_pure.predict(t.reshape(-1,1))

        # Hybrid (Exp Sparse + PI)
        X_aug, y_aug, w_aug = t_tr.reshape(-1,1), M_tr.copy(), np.ones(len(M_tr))
        pi_h, pi_v = None, None
        if pi and len(pi.hours) > 0:
            full_pi_h = np.asarray(pi.hours)
            k = int(np.ceil(len(full_pi_h) * pi.pi_percent))
            if k > 0:
                pi_h = full_pi_h[:k]
                pi_v = fickian_model(pi_h, M0, best[1], best[0])
                X_aug = np.vstack([X_aug, pi_h.reshape(-1,1)])
                y_aug = np.concatenate([y_aug, pi_v])
                w_aug = np.concatenate([w_aug, np.full(len(pi_h), pi_weight)])
        
        rf_hyb = RandomForestRegressor(n_estimators=300, random_state=self.seed)
        rf_hyb.fit(X_aug, y_aug, sample_weight=w_aug)
        M_rf_hyb = rf_hyb.predict(t.reshape(-1,1))

        # Eval MAE
        idx_val = complement_idx(len(t), idx_train)
        eval_idx = idx_val if len(idx_val) > 0 else idx_train
        idx_val_pure = complement_idx(len(t), idx_train_pure)
        eval_idx_pure = idx_val_pure if len(idx_val_pure) > 0 else idx_train_pure
        
        mae_pure = np.mean(np.abs(M_pure[eval_idx_pure] - M[eval_idx_pure]))

        best_lam, min_mae, M_final = 0.0, np.inf, M_rf_hyb
        l_grid = lambdas if lambdas else np.linspace(0, 1, 21)
        for lam in l_grid:
            blend = (1-lam)*M_rf_hyb + lam*M_phys
            mae = np.mean(np.abs(blend[eval_idx] - M[eval_idx]))
            if mae < min_mae: min_mae, best_lam, M_final = mae, lam, blend

        res = ResultBase(t, idx_train, M_pure, M_phys, M_final, min_mae, mae_pure, best_lam, pi_h, pi_v, idx_train_pure)
        res.params = {"Minf": best[1], "tau": best[0]}
        return res

# --- 2. THICKNESS SURROGATE ---
@dataclass
class ThickResult(ResultBase):
    alpha: float = 0.0

class ThicknessSurrogate:
    def __init__(self, seed=42): self.seed = seed
    
    def train(self, t, h_true, M_pred, M_pred_pure, idx_train, alpha, pi: Optional[PISpec1D], pi_weight=1.0, lambdas=None, idx_train_pure=None) -> ThickResult:
        set_global_seed(self.seed)
        if idx_train_pure is None: idx_train_pure = idx_train
        if M_pred_pure is None: M_pred_pure = M_pred
        
        h0 = float(h_true[0])
        M0 = float(M_pred[0])
        h_phys = thickness_physics_linear(M_pred, h0, alpha, M0)

        # Pure ML (Trained on Pure Mass predictions & Denser Data)
        t_tr_pure, h_tr_pure = t[idx_train_pure], h_true[idx_train_pure]
        M_tr_pure = M_pred_pure[idx_train_pure]
        rf_pure = RandomForestRegressor(n_estimators=300, random_state=self.seed)
        rf_pure.fit(M_tr_pure.reshape(-1,1), h_tr_pure)
        h_pure = rf_pure.predict(M_pred_pure.reshape(-1,1))

        # Hybrid
        t_tr, h_tr = t[idx_train], h_true[idx_train]
        M_tr = M_pred[idx_train]
        X_aug, y_aug, w_aug = M_tr.reshape(-1,1), h_tr.copy(), np.ones(len(h_tr))
        pi_h, pi_v = None, None
        if pi and len(pi.hours) > 0:
            full_pi_h = np.asarray(pi.hours)
            k = int(np.ceil(len(full_pi_h) * pi.pi_percent))
            if k > 0:
                pi_h = full_pi_h[:k]
                M_at_pi = np.interp(pi_h, t, M_pred)
                pi_v = thickness_physics_linear(np.array([M_at_pi]), h0, alpha, M0)[0]
                X_aug = np.vstack([X_aug, M_at_pi.reshape(-1,1)])
                y_aug = np.concatenate([y_aug, pi_v])
                w_aug = np.concatenate([w_aug, np.full(len(pi_h), pi_weight)])

        rf_hyb = RandomForestRegressor(n_estimators=300, random_state=self.seed)
        rf_hyb.fit(X_aug, y_aug, sample_weight=w_aug)
        h_rf_hyb = rf_hyb.predict(M_pred.reshape(-1,1))

        # Eval
        idx_val = complement_idx(len(t), idx_train)
        eval_idx = idx_val if len(idx_val) > 0 else idx_train
        idx_val_pure = complement_idx(len(t), idx_train_pure)
        eval_idx_pure = idx_val_pure if len(idx_val_pure) > 0 else idx_train_pure
        
        mae_pure = np.mean(np.abs(h_pure[eval_idx_pure] - h_true[eval_idx_pure]))

        best_lam, min_mae, h_final = 0.0, np.inf, h_rf_hyb
        l_grid = lambdas if lambdas else np.linspace(0, 1, 21)
        for lam in l_grid:
            blend = (1-lam)*h_rf_hyb + lam*h_phys
            mae = np.mean(np.abs(blend[eval_idx] - h_true[eval_idx]))
            if mae < min_mae: min_mae, best_lam, h_final = mae, lam, blend

        return ThickResult(t, idx_train, h_pure, h_phys, h_final, min_mae, mae_pure, best_lam, pi_h, pi_v, idx_train_pure, alpha)

# --- 3. DENSITY SURROGATE ---
class DensitySurrogate:
    def __init__(self, seed=42): self.seed = seed
    
    def train(self, t, M_hybrid, h_hybrid, M_pure, h_pure, M0, h0, rho0, idx_train, rho_exp=None, rho_fluid=1000.0, idx_train_pure=None) -> ResultBase:
        if idx_train_pure is None: idx_train_pure = idx_train
        
        rho_hybrid_phys = rom_density(M_hybrid, h_hybrid, M0, rho0, rho_fluid, h0=h0)
        rho_pure_phys = rom_density(M_pure, h_pure, M0, rho0, rho_fluid, h0=h0)
        
        mae_pure, mae_hyb = 0.0, 0.0
        if rho_exp is not None:
            eval_idx = complement_idx(len(t), idx_train)
            if len(eval_idx) == 0: eval_idx = idx_train
            eval_idx_pure = complement_idx(len(t), idx_train_pure)
            if len(eval_idx_pure) == 0: eval_idx_pure = idx_train_pure
            
            mae_pure = np.mean(np.abs(rho_pure_phys[eval_idx_pure] - rho_exp[eval_idx_pure]))
            mae_hyb = np.mean(np.abs(rho_hybrid_phys[eval_idx] - rho_exp[eval_idx]))
            
        return ResultBase(t, idx_train, rho_pure_phys, rho_hybrid_phys, rho_hybrid_phys, mae_hyb, mae_pure, 0.0, None, None, idx_train_pure)

# --- 4. CG SURROGATE ---
class CGSurrogate:
    def __init__(self, seed=42): self.seed = seed
    def train(self, t_full, t_exp, gv_exp, idx_train, pi, pi_weight=1.0, idx_train_pure=None) -> ResultBase:
        set_global_seed(self.seed)
        if idx_train_pure is None: idx_train_pure = idx_train
        
        # Pure ML
        t_tr_pure, y_tr_pure = t_exp[idx_train_pure], gv_exp[idx_train_pure]
        rf_pure = RandomForestRegressor(n_estimators=300, random_state=self.seed)
        rf_pure.fit(t_tr_pure.reshape(-1,1), y_tr_pure)
        pred_pure = rf_pure.predict(t_full.reshape(-1,1))
        
        # Physics (PI only)
        pi_h, pi_v = None, None
        if pi and len(pi.hours)>0:
            full_pi_h = np.asarray(pi.hours)
            full_pi_v = np.asarray(pi.values) if pi.values is not None else np.zeros_like(full_pi_h)
            k = int(np.ceil(len(full_pi_h)*pi.pi_percent))
            if k>0:
                pi_h, pi_v = full_pi_h[:k], full_pi_v[:k]
                rf_phys = RandomForestRegressor(n_estimators=300, random_state=self.seed)
                rf_phys.fit(pi_h.reshape(-1,1), pi_v)
                pred_phys = rf_phys.predict(t_full.reshape(-1,1))
            else: pred_phys = np.full_like(t_full, np.mean(gv_exp[idx_train]))
        else: pred_phys = np.full_like(t_full, np.mean(gv_exp[idx_train]))

        # Hybrid
        t_tr, y_tr = t_exp[idx_train], gv_exp[idx_train]
        X_aug, y_aug, w_aug = t_tr.reshape(-1,1), y_tr.copy(), np.ones(len(y_tr))
        if pi_h is not None:
            X_aug = np.vstack([X_aug, pi_h.reshape(-1,1)])
            y_aug = np.concatenate([y_aug, pi_v])
            w_aug = np.concatenate([w_aug, np.full(len(pi_h), pi_weight)])
        
        rf_hyb = RandomForestRegressor(n_estimators=300, random_state=self.seed)
        rf_hyb.fit(X_aug, y_aug, sample_weight=w_aug)
        pred_hybrid = rf_hyb.predict(t_full.reshape(-1,1))
        
        eval_idx = complement_idx(len(t_exp), idx_train) if len(t_exp)>len(idx_train) else idx_train
        eval_idx_pure = complement_idx(len(t_exp), idx_train_pure) if len(t_exp)>len(idx_train_pure) else idx_train_pure
        mae_pure = np.mean(np.abs(rf_pure.predict(t_exp[eval_idx_pure].reshape(-1,1)) - gv_exp[eval_idx_pure]))
        mae_hyb = np.mean(np.abs(rf_hyb.predict(t_exp[eval_idx].reshape(-1,1)) - gv_exp[eval_idx]))
        
        return ResultBase(t_full, idx_train, pred_pure, pred_phys, pred_hybrid, mae_hyb, mae_pure, 0.0, pi_h, pi_v, idx_train_pure)

# --- 5. YOUNGS SURROGATE ---
class YoungsSurrogate:
    def __init__(self, seed=42): self.seed = seed
    def train(self, t_full, t_exp, E_exp, idx_train, rho_hyb, gv_hyb, rho_pure, gv_pure, pi, pi_weight=1.0, idx_train_pure=None) -> ResultBase:
        set_global_seed(self.seed)
        if idx_train_pure is None: idx_train_pure = idx_train
        # Group-velocity stiffness proxy for augmentation; not a general
        # inversion of anisotropic guided-wave dispersion to tensile modulus.
        pred_phys = (rho_hyb * (gv_hyb**2)) / 1e9
        
        # Pure ML
        t_tr_pure, y_tr_pure = t_exp[idx_train_pure], E_exp[idx_train_pure]
        rf_pure = RandomForestRegressor(n_estimators=300, random_state=self.seed)
        rf_pure.fit(t_tr_pure.reshape(-1,1), y_tr_pure)
        pred_pure = rf_pure.predict(t_full.reshape(-1,1))
        
        # Hybrid
        t_tr, y_tr = t_exp[idx_train], E_exp[idx_train]
        X_aug, y_aug, w_aug = t_tr.reshape(-1,1), y_tr.copy(), np.ones(len(y_tr))
        pi_h, pi_v = None, None
        if pi and len(pi.hours) > 0:
            full_pi_h = np.asarray(pi.hours)
            k = int(np.ceil(len(full_pi_h)*pi.pi_percent))
            if k > 0:
                pi_h = full_pi_h[:k]
                pi_v = np.interp(pi_h, t_full, pred_phys)
                X_aug = np.vstack([X_aug, pi_h.reshape(-1,1)])
                y_aug = np.concatenate([y_aug, pi_v])
                w_aug = np.concatenate([w_aug, np.full(len(pi_h), pi_weight)])
        
        rf_hyb = RandomForestRegressor(n_estimators=300, random_state=self.seed)
        rf_hyb.fit(X_aug, y_aug, sample_weight=w_aug)
        pred_hybrid = rf_hyb.predict(t_full.reshape(-1,1))
        
        eval_idx = complement_idx(len(t_exp), idx_train) if len(t_exp)>len(idx_train) else idx_train
        eval_idx_pure = complement_idx(len(t_exp), idx_train_pure) if len(t_exp)>len(idx_train_pure) else idx_train_pure
        mae_pure = np.mean(np.abs(rf_pure.predict(t_exp[eval_idx_pure].reshape(-1,1)) - E_exp[eval_idx_pure]))
        mae_hyb = np.mean(np.abs(rf_hyb.predict(t_exp[eval_idx].reshape(-1,1)) - E_exp[eval_idx]))
        
        return ResultBase(t_full, idx_train, pred_pure, pred_phys, pred_hybrid, mae_hyb, mae_pure, 0.0, pi_h, pi_v, idx_train_pure)
