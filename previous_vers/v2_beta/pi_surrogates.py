# pi_surrogates.py — All surrogate classes and shared physics (Smart 2025)
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut


# ---------------------------
# Utilities
# ---------------------------
def set_global_seed(seed: int = 0):
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)


def assert_sorted_time(t_h: np.ndarray):
    t_h = np.asarray(t_h, float)
    if not np.all(np.diff(t_h) >= 0):
        raise ValueError("time array must be sorted ascending.")


def mae_on_val(y_true: np.ndarray, y_pred: np.ndarray, idx_train: np.ndarray, n_total: int) -> float:
    idx_train = np.asarray(idx_train, int)
    mask = np.ones(n_total, dtype=bool)
    mask[idx_train] = False
    if not np.any(mask):
        return mean_absolute_error(y_true, y_pred)
    return mean_absolute_error(y_true[mask], y_pred[mask])


def pick_even(n_total: int, n_pick: int) -> np.ndarray:
    n_pick = max(1, min(n_pick, n_total))
    return np.unique(np.round(np.linspace(0, n_total - 1, n_pick)).astype(int))


def pick_early(n_total: int, n_pick: int) -> np.ndarray:
    n_pick = max(1, min(n_pick, n_total))
    qs = np.linspace(0, 1, n_pick)
    pos = np.round((qs**2) * (n_total - 1)).astype(int)
    return np.unique(pos)


def pick_firstk(n_total: int, n_pick: int) -> np.ndarray:
    n_pick = max(1, min(n_pick, n_total))
    return np.arange(n_pick, dtype=int)


def make_kfold_subsets(n_total: int, n_pick: int, k_folds: int,
                       picker: Callable[[int, int], np.ndarray]) -> List[np.ndarray]:
    if k_folds <= 1:
        return [picker(n_total, n_pick)]
    sets = []
    base = np.linspace(0, n_total - 1, n_pick)
    shift = (n_total / max(3 * n_pick, 1))
    for k in range(k_folds):
        grid = base + k * shift / max(k_folds - 1, 1)
        idx = np.clip(np.round(grid).astype(int), 0, n_total - 1)
        sets.append(np.unique(idx))
    return sets


def _clean_train_idx(idx: np.ndarray, n_total: int) -> np.ndarray:
    idx = np.asarray(idx, int).ravel()
    idx = idx[(idx >= 0) & (idx < n_total)]
    idx = np.unique(idx)
    if len(idx) < 1:
        raise ValueError("Manual train_idx produced an empty set.")
    return idx


# ---------------------------
# Physics: mass (Fickian slab)
# ---------------------------
def fickian_fraction(t_h: np.ndarray, tau_h: float, n_terms: int = 60) -> np.ndarray:
    """Fractional uptake F(t; tau)."""
    t = np.asarray(t_h).reshape(1, -1)
    n = np.arange(n_terms).reshape(-1, 1)
    lam2 = (2*n + 1)**2
    exps = np.exp(-lam2 * (t / tau_h)) / lam2
    F = 1.0 - (8.0/np.pi**2) * np.sum(exps, axis=0)
    return F.ravel()


def fickian_mass_curve(t_h: np.ndarray, M0: float, Minf: float, tau_h: float) -> np.ndarray:
    F = fickian_fraction(t_h, tau_h, n_terms=60)
    return M0 + (Minf - M0) * F


def fit_fickian_params(t_train_h: np.ndarray, M_train: np.ndarray, M0_full: float) -> Dict[str, float]:
    """Fit tau_h and Minf from sparse points by scanning tau."""
    t_train_h = np.asarray(t_train_h, float)
    M_train = np.asarray(M_train, float)
    M0 = float(M0_full)

    tau_grid = 10.0 ** np.linspace(0.3, 4.0, 60)  # ~2h .. 1e4h
    best = {"tau_h": None, "Minf": None, "sse": np.inf}

    for tau in tau_grid:
        F = fickian_fraction(t_train_h, tau, 60)
        denom = float(np.dot(F, F)) + 1e-12
        alpha = float(np.dot(F, (M_train - M0)) / denom)  # alpha = Minf - M0
        alpha = max(alpha, 0.0)
        Minf_hat = max(M0 + alpha, float(np.max(M_train)))  # not below observed
        y_hat = M0 + (Minf_hat - M0) * F
        sse = float(np.sum((M_train - y_hat)**2))
        if sse < best["sse"]:
            best = {"tau_h": float(tau), "Minf": float(Minf_hat), "sse": sse}
    return best


# ---------------------------
# PI specs
# ---------------------------
@dataclass
class PISpec1D:
    extra_hours: Optional[np.ndarray] = None
    pi_percent: float = 1.0
    desc: str = "Physics-augmented"


@dataclass
class PISpecCG:
    extra_hours: Optional[np.ndarray] = None
    extra_gv_ms: Optional[np.ndarray] = None
    pi_percent: float = 1.0
    desc: str = "ROM-augmented CG"


@dataclass
class PISpecE:
    extra_hours: Optional[np.ndarray] = None
    pi_percent: float = 1.0
    gv_source: str = "interp"   # "interp" or "cg_pred"
    desc: str = "Physics-augmented E"


# ---------------------------
# Mass surrogate
# ---------------------------
@dataclass
class MassResult:
    t_h: np.ndarray
    M_phys: np.ndarray
    M_rf: np.ndarray
    M_pred: np.ndarray
    mae: float
    lambda_star: float
    idx_train: np.ndarray
    N_train: int
    strategy: str
    params: Dict[str, float]
    pi_hours: Optional[np.ndarray] = None
    pi_values: Optional[np.ndarray] = None


class MassSurrogate:
    def __init__(self, rf_depth: int = 3, seed: int = 0):
        self.rf_depth = rf_depth
        self.seed = seed

    def train(self, t_h: np.ndarray, M_true: np.ndarray,
              n_train: int = 5, k_folds: int = 3,
              lambdas: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8],
              strategy: str = "even",
              pi: Optional[PISpec1D] = None,
              train_idx: Optional[np.ndarray] = None) -> MassResult:
        """
        If train_idx is provided, it overrides strategy/n_train/k_folds and uses those experimental points.
        """
        set_global_seed(self.seed)
        t_h = np.asarray(t_h, float)
        M_true = np.asarray(M_true, float)
        assert_sorted_time(t_h)

        if train_idx is not None:
            folds = [_clean_train_idx(train_idx, len(t_h))]
            strategy_used = "manual"
        else:
            strategy_used = strategy
            if strategy == "firstk":
                folds = [pick_firstk(len(t_h), n_train)]
            else:
                picker = pick_even if strategy == "even" else pick_early
                folds = make_kfold_subsets(len(t_h), n_train, k_folds, picker)

        best = None
        best_pi_hours = None
        best_pi_vals = None
        M0_full = float(M_true[0])

        for idx in folds:
            t_tr, M_tr = t_h[idx], M_true[idx]

            fit = fit_fickian_params(t_tr, M_tr, M0_full=M0_full)
            M_phys_full = fickian_mass_curve(t_h, M0=M0_full, Minf=fit["Minf"], tau_h=fit["tau_h"])

            X_tr = t_tr.reshape(-1, 1)
            y_tr = M_tr.copy()

            pi_hours_used, pi_vals_used = None, None
            if pi is not None and pi.extra_hours is not None and len(pi.extra_hours) > 0:
                k = int(np.ceil(len(pi.extra_hours) * max(0.0, min(1.0, float(pi.pi_percent)))))
                if k > 0:
                    eh = np.asarray(pi.extra_hours, float)[:k]
                    m_pi = np.interp(eh, t_h, M_phys_full)
                    X_tr = np.vstack([X_tr, eh.reshape(-1, 1)])
                    y_tr = np.concatenate([y_tr, m_pi])
                    pi_hours_used, pi_vals_used = eh, m_pi

            rf = RandomForestRegressor(n_estimators=300, max_depth=self.rf_depth, random_state=self.seed)
            rf.fit(X_tr, y_tr)
            M_rf = rf.predict(t_h.reshape(-1, 1))

            for lam in lambdas:
                M_blend = (1 - lam) * M_rf + lam * M_phys_full
                mae = mae_on_val(M_true, M_blend, idx_train=idx, n_total=len(t_h))
                if (best is None) or (mae < best["mae"]):
                    best = {"mae": mae, "lam": lam, "M_pred": M_blend,
                            "M_rf": M_rf, "M_phys": M_phys_full, "idx": idx, "fit": fit}
                    best_pi_hours, best_pi_vals = pi_hours_used, pi_vals_used

        return MassResult(
            t_h=t_h,
            M_phys=best["M_phys"], M_rf=best["M_rf"], M_pred=best["M_pred"],
            mae=best["mae"], lambda_star=best["lam"],
            idx_train=best["idx"], N_train=len(best["idx"]), strategy=strategy_used,
            params=best["fit"], pi_hours=best_pi_hours, pi_values=best_pi_vals
        )


# ---------------------------
# Thickness surrogate
# ---------------------------
def fit_alpha_from_cases(cases: List[Dict[str, np.ndarray]]) -> float:
    dm_all, dh_all = [], []
    for d in cases:
        M = np.asarray(d["mass"], float)
        h = np.asarray(d["thick"], float)
        M0 = float(M[0]); h0 = float(h[0])
        dm = (M - M0) / max(M0, 1e-12)
        dh = (h - h0) / max(h0, 1e-12)
        dm_all.append(dm); dh_all.append(dh)
    dm_all = np.concatenate(dm_all)
    dh_all = np.concatenate(dh_all)
    denom = float(np.dot(dm_all, dm_all)) + 1e-12
    return float(np.dot(dm_all, dh_all) / denom)


def thickness_physics(M_curve: np.ndarray, h0: float, alpha: float) -> np.ndarray:
    M_curve = np.asarray(M_curve, float)
    M0 = float(M_curve[0])
    dm = (M_curve - M0) / max(M0, 1e-12)
    return h0 * (1.0 + alpha * dm)


@dataclass
class ThickResult:
    t_h: np.ndarray
    h_phys: np.ndarray
    h_rf: np.ndarray
    h_pred: np.ndarray
    mae: float
    lambda_star: float
    idx_train: np.ndarray
    N_train: int
    strategy: str
    augmented: bool
    pi_hours: Optional[np.ndarray] = None
    pi_values: Optional[np.ndarray] = None


class ThicknessSurrogate:
    def __init__(self, rf_depth: int = 3, seed: int = 0):
        self.rf_depth = rf_depth
        self.seed = seed

    def train(self, t_h: np.ndarray, h_true: np.ndarray, M_curve: np.ndarray,
              alpha: float,
              n_train: int = 7, k_folds: int = 3,
              lambdas: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8],
              strategy: str = "even",
              augment: bool = True, augment_between: int = 3,
              pi: Optional[PISpec1D] = None,
              train_idx: Optional[np.ndarray] = None) -> ThickResult:
        """
        If train_idx is provided, it overrides strategy/n_train/k_folds and uses those experimental points.
        """
        set_global_seed(self.seed)
        t_h = np.asarray(t_h, float)
        h_true = np.asarray(h_true, float)
        M_curve = np.asarray(M_curve, float)
        assert_sorted_time(t_h)

        if train_idx is not None:
            folds = [_clean_train_idx(train_idx, len(t_h))]
            strategy_used = "manual"
        else:
            strategy_used = strategy
            if strategy == "firstk":
                folds = [pick_firstk(len(t_h), n_train)]
            else:
                picker = pick_even if strategy == "even" else pick_early
                folds = make_kfold_subsets(len(t_h), n_train, k_folds, picker)

        best = None
        best_pi_hours, best_pi_vals = None, None

        h0_full = float(h_true[0])
        h_phys_full = thickness_physics(M_curve, h0_full, alpha)

        for idx in folds:
            t_tr, h_tr = t_h[idx], h_true[idx]

            # optional augmentation between experimental training points
            t_aug, h_aug = t_tr.copy(), h_tr.copy()
            if augment and len(t_tr) >= 2:
                T, H = [t_tr[0]], [h_tr[0]]
                for i in range(len(t_tr) - 1):
                    t0, t1 = t_tr[i], t_tr[i + 1]
                    for k in range(1, augment_between + 1):
                        tau = k / (augment_between + 1.0)
                        tm = (1 - tau) * t0 + tau * t1
                        h_lin = np.interp(tm, [t0, t1], [h_tr[i], h_tr[i + 1]])
                        h_phy = np.interp(tm, t_h, h_phys_full)
                        H.append(0.7 * h_lin + 0.3 * h_phy)
                        T.append(tm)
                    T.append(t1); H.append(h_tr[i + 1])
                t_aug = np.array(T, float)
                h_aug = np.array(H, float)

            # PI extras
            pi_hours_used, pi_vals_used = None, None
            if pi is not None and pi.extra_hours is not None and len(pi.extra_hours) > 0:
                kk = int(np.ceil(len(pi.extra_hours) * max(0.0, min(1.0, float(pi.pi_percent)))))
                if kk > 0:
                    eh = np.asarray(pi.extra_hours, float)[:kk]
                    h_pi = np.interp(eh, t_h, h_phys_full)
                    t_aug = np.concatenate([t_aug, eh])
                    h_aug = np.concatenate([h_aug, h_pi])
                    pi_hours_used, pi_vals_used = eh, h_pi

            rf = RandomForestRegressor(n_estimators=300, max_depth=self.rf_depth, random_state=self.seed)
            rf.fit(t_aug.reshape(-1, 1), h_aug)
            h_rf_full = rf.predict(t_h.reshape(-1, 1))

            for lam in lambdas:
                h_blend = h_rf_full + lam * h_phys_full #(1 - lam) * h_rf_full + lam * h_phys_full
                mae = mae_on_val(h_true, h_blend, idx_train=idx, n_total=len(t_h))
                if (best is None) or (mae < best["mae"]):
                    best = {"mae": mae, "lam": lam, "h_pred": h_blend,
                            "h_rf": h_rf_full, "h_phys": h_phys_full,
                            "idx": idx, "aug": augment}
                    best_pi_hours, best_pi_vals = pi_hours_used, pi_vals_used

        return ThickResult(
            t_h=t_h,
            h_phys=best["h_phys"], h_rf=best["h_rf"], h_pred=best["h_pred"],
            mae=best["mae"], lambda_star=best["lam"],
            idx_train=best["idx"], N_train=len(best["idx"]),
            strategy=strategy_used, augmented=best["aug"],
            pi_hours=best_pi_hours, pi_values=best_pi_vals
        )


# ---------------------------
# Density surrogate (your ROM rule) + RF + blend
# ---------------------------
def rom_density(M_curve: np.ndarray, h_curve_unused: np.ndarray,
                rho_dry: float = 1150.0, rho_fluid: float = 1000.0) -> np.ndarray:
    """
    ROM using fluid mass fraction w = (M - M0)/M.
    This is the PHYSICS baseline for density.
    """
    M_curve = np.asarray(M_curve, float)
    M0 = float(M_curve[0])
    w = np.maximum(0.0, (M_curve - M0) / np.maximum(M_curve, 1e-12))
    return (1 - w) * rho_dry + w * rho_fluid


@dataclass
class DensityResult:
    t_h: np.ndarray
    rho_phys: np.ndarray
    rho_rf: np.ndarray
    rho_pred: np.ndarray
    y_target: np.ndarray
    mae: float
    lambda_star: float
    idx_train: np.ndarray
    N_train: int
    strategy: str


class DensitySurrogate:
    def __init__(self, rf_depth: int = 3, seed: int = 0):
        self.rf_depth = rf_depth
        self.seed = seed

    def train(self, t_h: np.ndarray, M_curve: np.ndarray, h_curve: np.ndarray,
              n_train: int = 7, k_folds: int = 3,
              lambdas: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8],
              strategy: str = "even",
              y_true: Optional[np.ndarray] = None,
              rho_dry: float = 1150.0, rho_fluid: float = 1000.0,
              train_idx: Optional[np.ndarray] = None) -> DensityResult:
        """
        If train_idx is provided, it overrides strategy/n_train/k_folds and uses those experimental points.
        """
        set_global_seed(self.seed)
        t_h = np.asarray(t_h, float)
        M_curve = np.asarray(M_curve, float)
        h_curve = np.asarray(h_curve, float)
        assert_sorted_time(t_h)

        rho_phys_full = rom_density(M_curve, h_curve, rho_dry, rho_fluid)
        y_target_full = np.asarray(y_true, float) if y_true is not None else rho_phys_full.copy()

        if train_idx is not None:
            folds = [_clean_train_idx(train_idx, len(t_h))]
            strategy_used = "manual"
        else:
            strategy_used = strategy
            if strategy == "firstk":
                folds = [pick_firstk(len(t_h), n_train)]
            else:
                picker = pick_even if strategy == "even" else pick_early
                folds = make_kfold_subsets(len(t_h), n_train, k_folds, picker)

        best = None
        X_full = np.column_stack([t_h, M_curve, h_curve])

        for idx in folds:
            rf = RandomForestRegressor(n_estimators=300, max_depth=self.rf_depth, random_state=self.seed)
            rf.fit(X_full[idx], y_target_full[idx])
            rho_rf_full = rf.predict(X_full)

            for lam in lambdas:
                rho_blend = (1 - lam) * rho_rf_full + lam * rho_phys_full
                mae = mae_on_val(y_target_full, rho_blend, idx_train=idx, n_total=len(t_h))
                if (best is None) or (mae < best["mae"]):
                    best = {"mae": mae, "lam": lam, "rho_pred": rho_blend,
                            "rho_rf": rho_rf_full, "idx": idx}

        return DensityResult(
            t_h=t_h,
            rho_phys=rho_phys_full, rho_rf=best["rho_rf"], rho_pred=best["rho_pred"],
            y_target=y_target_full,
            mae=best["mae"], lambda_star=best["lam"],
            idx_train=best["idx"], N_train=len(best["idx"]),
            strategy=strategy_used
        )


# ---------------------------
# CG surrogate (keep as-is: 4/7/10 + PI points)
# ---------------------------
def _interp_rho_h_at_hours_from_curves(t_mass: np.ndarray,
                                      rho_curve: np.ndarray,
                                      h_curve_mm: np.ndarray,
                                      hours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rho_at = np.interp(hours, t_mass, np.asarray(rho_curve, float))
    h_at_m = np.interp(hours, t_mass, np.asarray(h_curve_mm, float)) / 1000.0
    return rho_at, h_at_m


@dataclass
class CGResult:
    hours: np.ndarray
    gv_true_ms: np.ndarray
    gv_pred4_ms: np.ndarray
    gv_pred7_ms: np.ndarray
    gv_pred10_ms: np.ndarray
    mae4_ms: float
    mae7_ms: float
    mae10_ms: float
    n_extra_used: int
    pi_desc: str
    pi_hours: Optional[np.ndarray] = None
    pi_values_ms: Optional[np.ndarray] = None


class PISurrogateCG:
    def __init__(self, n_estimators=50, max_depth=5, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

    def _train_predict(self, X_tr, y_tr, X_eval):
        rf = RandomForestRegressor(n_estimators=self.n_estimators,
                                   max_depth=self.max_depth,
                                   random_state=self.seed)
        rf.fit(X_tr, y_tr)
        return rf.predict(X_eval)

    def fit_eval(self, t_mass: np.ndarray,
                 rho_curve: np.ndarray,
                 h_curve_mm: np.ndarray,
                 meas_hours: np.ndarray,
                 gv_meas_kms: np.ndarray,
                 pi: Optional[PISpecCG],
                 four_pt_exclude: Tuple[float, float, float] = (0.0, 24.0, 528.0)) -> Dict[str, np.ndarray]:

        rho_meas, h_meas_m = _interp_rho_h_at_hours_from_curves(t_mass, rho_curve, h_curve_mm, meas_hours)
        X_meas = np.column_stack([rho_meas, h_meas_m])
        y_meas = np.asarray(gv_meas_kms, float) * 1000.0  # m/s

        # 4-pt: exclude specified hours
        excl = set([float(x) for x in four_pt_exclude])
        mask4 = ~np.isin(meas_hours.astype(float), np.array(list(excl)))
        X4, y4 = X_meas[mask4], y_meas[mask4]
        pred4 = self._train_predict(X4, y4, X_meas)
        mae4 = mean_absolute_error(y_meas, pred4)

        # 7-pt: all measured
        pred7 = self._train_predict(X_meas, y_meas, X_meas)
        mae7 = mean_absolute_error(y_meas, pred7)

        # 10-pt: add PI points
        pred10 = pred7.copy()
        mae10 = mae7
        n_extra_used = 0
        pi_pts = None

        if pi is not None and pi.extra_hours is not None and pi.extra_gv_ms is not None:
            eh = np.asarray(pi.extra_hours, float)
            eg = np.asarray(pi.extra_gv_ms, float)
            k = int(np.ceil(len(eh) * max(0.0, min(1.0, float(pi.pi_percent)))))
            if k > 0:
                idx = np.arange(len(eh))[:k]
                rho_ex, h_ex = _interp_rho_h_at_hours_from_curves(t_mass, rho_curve, h_curve_mm, eh[idx])
                Xex = np.column_stack([rho_ex, h_ex])
                yex = eg[idx]
                n_extra_used = len(idx)
                X10 = np.vstack([X_meas, Xex])
                y10 = np.concatenate([y_meas, yex])
                pred10 = self._train_predict(X10, y10, X_meas)
                mae10 = mean_absolute_error(y_meas, pred10)
                pi_pts = (eh[idx], yex)

        return {
            "hours": meas_hours, "y_meas": y_meas,
            "pred4": pred4, "mae4": mae4,
            "pred7": pred7, "mae7": mae7,
            "pred10": pred10, "mae10": mae10,
            "n_extra_used": n_extra_used,
            "pi_desc": (pi.desc if pi is not None else "None"),
            "pi_points": pi_pts
        }


def train_cg_with_pi(t_mass: np.ndarray,
                     rho_curve_pred: np.ndarray,
                     h_curve_pred_mm: np.ndarray,
                     meas_hours: np.ndarray,
                     gv_meas_kms: np.ndarray,
                     pi_spec: Optional[PISpecCG],
                     seed: int = 0) -> CGResult:
    set_global_seed(seed)
    trainer = PISurrogateCG(n_estimators=50, max_depth=5, seed=seed)
    out = trainer.fit_eval(t_mass, rho_curve_pred, h_curve_pred_mm, meas_hours, gv_meas_kms, pi_spec)

    pi_hours, pi_vals = (None, None)
    if out["pi_points"] is not None:
        pi_hours, pi_vals = out["pi_points"]

    return CGResult(
        hours=np.asarray(out["hours"], float),
        gv_true_ms=np.asarray(out["y_meas"], float),
        gv_pred4_ms=np.asarray(out["pred4"], float),
        gv_pred7_ms=np.asarray(out["pred7"], float),
        gv_pred10_ms=np.asarray(out["pred10"], float),
        mae4_ms=float(out["mae4"]),
        mae7_ms=float(out["mae7"]),
        mae10_ms=float(out["mae10"]),
        n_extra_used=int(out["n_extra_used"]),
        pi_desc=str(out["pi_desc"]),
        pi_hours=pi_hours,
        pi_values_ms=pi_vals
    )


# ---------------------------
# Young’s modulus (manual training points for E)
# ---------------------------
@dataclass
class EResult:
    hours: np.ndarray
    rho_at_hours: np.ndarray
    gv_ms: np.ndarray
    E_true_GPa: Optional[np.ndarray]
    E_ridge_GPa: np.ndarray
    E_phys_GPa: np.ndarray
    E_hybrid_GPa: np.ndarray
    best_w: float
    loocv_mae_GPa: Optional[float]
    pi_hours: Optional[np.ndarray] = None
    pi_values_GPa: Optional[np.ndarray] = None
    idx_train: Optional[np.ndarray] = None


class YoungsSurrogate:
    def __init__(self, alpha: float = 1.0, seed: int = 0):
        self.alpha = alpha
        self.seed = seed

    def _physics_E(self, rho: np.ndarray, gv_ms: np.ndarray) -> np.ndarray:
        return rho * (gv_ms**2) / 1e9  # GPa

    def train(self,
              hours: np.ndarray,
              rho_at_hours_pred: np.ndarray,
              gv_meas_kms: np.ndarray,
              E_true_GPa: Optional[np.ndarray],
              gv_pred_ms_for_inference: Optional[np.ndarray] = None,
              pi: Optional[PISpecE] = None,
              train_idx: Optional[np.ndarray] = None,
              train_strategy: str = "auto",
              n_train_E: Optional[int] = None) -> EResult:

        set_global_seed(self.seed)

        hours = np.asarray(hours, float)
        rho = np.asarray(rho_at_hours_pred, float)

        gv_ms_meas = np.asarray(gv_meas_kms, float) * 1000.0
        gv_ms_infer = np.asarray(gv_pred_ms_for_inference, float) if gv_pred_ms_for_inference is not None else gv_ms_meas

        # manual or firstk training indices (for split-based validation & plotting)
        if train_idx is None and train_strategy == "firstk" and n_train_E is not None:
            k = max(1, min(int(n_train_E), len(hours)))
            train_idx = np.arange(k, dtype=int)
        if train_idx is not None:
            train_idx = _clean_train_idx(train_idx, len(hours))

        # physics baseline uses inference GV
        E_phys = self._physics_E(rho, gv_ms_infer)

        # Ridge uses measured GV as feature
        X = np.column_stack([rho, gv_ms_meas])
        ridge = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                              Ridge(alpha=self.alpha, random_state=self.seed))

        # PI extras for E (physics-generated)
        pi_hours_used, pi_E_used = None, None
        if pi is not None and pi.extra_hours is not None and len(pi.extra_hours) > 0:
            kk = int(np.ceil(len(pi.extra_hours) * max(0.0, min(1.0, float(pi.pi_percent)))))
            if kk > 0:
                eh = np.asarray(pi.extra_hours, float)[:kk]
                rho_pi = np.interp(eh, hours, rho)
                if pi.gv_source == "cg_pred":
                    gv_pi = np.interp(eh, hours, gv_ms_infer)
                else:
                    gv_pi = np.interp(eh, hours, gv_ms_meas)
                E_pi = self._physics_E(rho_pi, gv_pi)
                pi_hours_used, pi_E_used = eh, E_pi

        if E_true_GPa is None:
            # fall back (rare): fit ridge to physics proxy
            ridge.fit(X, E_phys)
            E_ridge = ridge.predict(X)
            return EResult(hours=hours, rho_at_hours=rho, gv_ms=gv_ms_infer,
                           E_true_GPa=None, E_ridge_GPa=E_ridge, E_phys_GPa=E_phys,
                           E_hybrid_GPa=E_phys.copy(), best_w=0.0, loocv_mae_GPa=None,
                           pi_hours=pi_hours_used, pi_values_GPa=pi_E_used, idx_train=train_idx)

        y = np.asarray(E_true_GPa, float)

        # If manual train_idx exists: evaluate w on held-out points (rest)
        if train_idx is not None:
            n = len(hours)
            all_idx = np.arange(n)
            val_mask = np.ones(n, dtype=bool)
            val_mask[train_idx] = False
            val_idx = all_idx[val_mask]

            X_tr, y_tr = X[train_idx], y[train_idx]
            if pi_hours_used is not None:
                gv_pi_feat = np.interp(pi_hours_used, hours, gv_ms_meas)
                X_pi = np.column_stack([np.interp(pi_hours_used, hours, rho), gv_pi_feat])
                X_tr = np.vstack([X_tr, X_pi])
                y_tr = np.concatenate([y_tr, pi_E_used])

            ridge.fit(X_tr, y_tr)
            E_ridge_all = ridge.predict(X)

            weights = np.linspace(0, 1, 101)
            maes = []
            for w in weights:
                #w=0.5
                E_h = E_ridge_all + w * E_phys
                maes.append(np.mean(np.abs(y[val_idx] - E_h[val_idx])) if len(val_idx) else np.mean(np.abs(y - E_h)))
            best_w = float(weights[int(np.argmin(maes))])

            E_hybrid = E_ridge_all + best_w * E_phys
            return EResult(hours=hours, rho_at_hours=rho, gv_ms=gv_ms_infer,
                           E_true_GPa=y, E_ridge_GPa=E_ridge_all, E_phys_GPa=E_phys,
                           E_hybrid_GPa=E_hybrid, best_w=best_w, loocv_mae_GPa=float(np.min(maes)),
                           pi_hours=pi_hours_used, pi_values_GPa=pi_E_used, idx_train=train_idx)

        # Else: LOOCV to pick w
        loo = LeaveOneOut()
        weights = np.linspace(0, 1, 101)
        mae_per_w = np.zeros_like(weights, dtype=float)

        for wi, w in enumerate(weights):
            errs = []
            for tr, te in loo.split(X):
                ridge.fit(X[tr], y[tr])
                e_r_te = ridge.predict(X[te])
                e_p_te = E_phys[te]
                e_h_te = w * e_r_te + (1 - w) * e_p_te
                errs.append(np.abs(y[te] - e_h_te)[0])
            mae_per_w[wi] = np.mean(errs)

        best_w = float(weights[int(np.argmin(mae_per_w))])

        # final fit with all + optional PI
        X_fit, y_fit = X, y
        if pi_hours_used is not None:
            gv_pi_feat = np.interp(pi_hours_used, hours, gv_ms_meas)
            X_pi = np.column_stack([np.interp(pi_hours_used, hours, rho), gv_pi_feat])
            X_fit = np.vstack([X_fit, X_pi])
            y_fit = np.concatenate([y_fit, pi_E_used])

        ridge.fit(X_fit, y_fit)
        E_ridge = ridge.predict(X)
        E_hybrid = best_w * E_ridge + (1 - best_w) * E_phys

        return EResult(hours=hours, rho_at_hours=rho, gv_ms=gv_ms_infer,
                       E_true_GPa=y, E_ridge_GPa=E_ridge, E_phys_GPa=E_phys,
                       E_hybrid_GPa=E_hybrid, best_w=best_w, loocv_mae_GPa=float(np.min(mae_per_w)),
                       pi_hours=pi_hours_used, pi_values_GPa=pi_E_used, idx_train=None)
