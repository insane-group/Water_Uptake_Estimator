# pi_pipeline.py — Sequential pipeline (time-only inference after training)
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from pi_surrogates import (
    set_global_seed, assert_sorted_time,
    MassSurrogate, ThicknessSurrogate, DensitySurrogate,
    fit_alpha_from_cases, rom_density,
    train_cg_with_pi, YoungsSurrogate,
    PISpec1D, PISpecCG, PISpecE,
    MassResult, ThickResult, DensityResult, CGResult, EResult
)


@dataclass
class PipelineOutput:
    mass: MassResult
    thick: ThickResult
    dens: DensityResult
    alpha: float
    cg: Optional[CGResult] = None
    youngs: Optional[EResult] = None


def _idx_from_hours(t: np.ndarray, hours: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if hours is None:
        return None
    t = np.asarray(t, float)
    hrs = np.asarray(list(hours), float).ravel()
    if hrs.size == 0:
        return None
    idx = [int(np.argmin(np.abs(t - h))) for h in hrs]
    return np.unique(np.asarray(idx, int))


def run_pipeline(case: Dict[str, np.ndarray],
                 all_cases_for_alpha: List[Dict[str, np.ndarray]],
                 # sampling strategies (used if manual not provided)
                 mass_strategy: str = "even",
                 thick_strategy: str = "even",
                 dens_strategy: str = "even",
                 # manual training overrides (indices OR hours)
                 mass_train_idx: Optional[Sequence[int]] = None,
                 thick_train_idx: Optional[Sequence[int]] = None,
                 dens_train_idx: Optional[Sequence[int]] = None,
                 mass_train_hours: Optional[Sequence[float]] = None,
                 thick_train_hours: Optional[Sequence[float]] = None,
                 dens_train_hours: Optional[Sequence[float]] = None,
                 # core knobs
                 n_mass: int = 5, n_thick: int = 9, n_dens: int = 7,
                 kfolds: int = 3,
                 use_aug_for_thickness: bool = True,
                 rf_depth: int = 3, seed: int = 0,
                 density_y_true: Optional[np.ndarray] = None,
                 rho_dry: float = 1150.0, rho_fluid: float = 1000.0,
                 # optional elastic block
                 elastic: Optional[Dict[str, np.ndarray]] = None,
                 # PI knobs
                 pi_mass: Optional[PISpec1D] = None,
                 pi_thick: Optional[PISpec1D] = None,
                 pi_cg: Optional[PISpecCG] = None,
                 pi_E: Optional[PISpecE] = None,
                 # E training config
                 E_train_strategy: str = "auto",   # "auto" or "firstk"
                 E_n_train: Optional[int] = None,
                 ridge_alpha: float = 1.0) -> PipelineOutput:

    set_global_seed(seed)

    t = np.asarray(case["time"], float)
    M = np.asarray(case["mass"], float)
    h = np.asarray(case["thick"], float)
    assert_sorted_time(t)

    # resolve manual idx from hours if provided
    mass_idx = np.asarray(mass_train_idx, int) if mass_train_idx is not None else _idx_from_hours(t, mass_train_hours)
    thick_idx = np.asarray(thick_train_idx, int) if thick_train_idx is not None else _idx_from_hours(t, thick_train_hours)
    dens_idx = np.asarray(dens_train_idx, int) if dens_train_idx is not None else _idx_from_hours(t, dens_train_hours)

    # Stage 1: Mass
    mass_model = MassSurrogate(rf_depth=rf_depth, seed=seed)
    mass_res = mass_model.train(
        t_h=t, M_true=M,
        n_train=n_mass, k_folds=kfolds,
        strategy=mass_strategy, pi=pi_mass,
        train_idx=mass_idx
    )

    # Stage 2: Thickness (uses predicted mass)
    alpha = fit_alpha_from_cases(all_cases_for_alpha)
    thick_model = ThicknessSurrogate(rf_depth=rf_depth, seed=seed)
    thick_res = thick_model.train(
        t_h=t, h_true=h, M_curve=mass_res.M_pred, alpha=alpha,
        n_train=n_thick, k_folds=kfolds,
        strategy=thick_strategy,
        augment=use_aug_for_thickness, pi=pi_thick,
        train_idx=thick_idx
    )

    # Stage 3: Density (uses predicted mass and thickness)
    dens_model = DensitySurrogate(rf_depth=rf_depth, seed=seed)
    dens_res = dens_model.train(
        t_h=t, M_curve=mass_res.M_pred, h_curve=thick_res.h_pred,
        n_train=n_dens, k_folds=kfolds,
        strategy=dens_strategy,
        y_true=density_y_true, rho_dry=rho_dry, rho_fluid=rho_fluid,
        train_idx=dens_idx
    )

    cg_res = None
    e_res = None

    if elastic is not None and "hour" in elastic and "GV_meas" in elastic:
        el_hours = np.asarray(elastic["hour"], float)
        gv_meas = np.asarray(elastic["GV_meas"], float)

        # Stage 4: CG (keep as-is; uses rho_pred + h_pred)
        cg_res = train_cg_with_pi(
            t_mass=t,
            rho_curve_pred=dens_res.rho_pred,
            h_curve_pred_mm=thick_res.h_pred,
            meas_hours=el_hours,
            gv_meas_kms=gv_meas,
            pi_spec=pi_cg,
            seed=seed
        )

        # Stage 5: E
        rho_at_hours_pred = np.interp(el_hours, t, dens_res.rho_pred)
        E_true = np.asarray(elastic["E_true"], float) if "E_true" in elastic else None

        e_model = YoungsSurrogate(alpha=ridge_alpha, seed=seed)

        # Optional: manual E training indices passed inside elastic dict
        e_train_idx = np.asarray(elastic["E_train_idx"], int) if "E_train_idx" in elastic else None

        gv_pred_ms_for_inference = cg_res.gv_pred10_ms if cg_res is not None else None

        e_res = e_model.train(
            hours=el_hours,
            rho_at_hours_pred=rho_at_hours_pred,
            gv_meas_kms=gv_meas,
            E_true_GPa=E_true,
            gv_pred_ms_for_inference=gv_pred_ms_for_inference,
            pi=pi_E,
            train_idx=e_train_idx,
            train_strategy=E_train_strategy,
            n_train_E=E_n_train
        )

    return PipelineOutput(mass=mass_res, thick=thick_res, dens=dens_res,
                          alpha=alpha, cg=cg_res, youngs=e_res)
