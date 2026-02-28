# pi_pipeline.py — Orchestrates Mass → Thickness → Density → CG → E
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from pi_surrogates import (
    set_global_seed, assert_sorted_time,
    MassSurrogate, ThicknessSurrogate, DensitySurrogate,
    fit_alpha_from_cases, rom_density,
    train_cg_with_pi, PISpecCG, YoungsSurrogate, PISpec1D, PISpecE,
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

def run_pipeline(case: Dict[str, np.ndarray],
                 all_cases_for_alpha: List[Dict[str, np.ndarray]],
                 # per-stage sampling strategies
                 mass_strategy: str = "even",
                 thick_strategy: str = "even",
                 dens_strategy: str = "even",
                 E_train_strategy: str = "auto",   # "auto" | "firstk"
                 E_n_train: Optional[int] = None,
                 # core knobs
                 n_mass: int = 5, n_thick: int = 9, n_dens: int = 7,
                 kfolds: int = 3,
                 use_aug_for_thickness: bool = True,
                 rf_depth: int = 3, seed: int = 0,
                 density_y_true: Optional[np.ndarray] = None,
                 rho_dry: float = 1150.0, rho_fluid: float = 1000.0,
                 # Elastic data (optional)
                 elastic: Optional[Dict[str, np.ndarray]] = None,
                 # PI knobs (optional)
                 pi_mass: Optional[PISpec1D] = None,
                 pi_thick: Optional[PISpec1D] = None,
                 pi_cg: Optional[PISpecCG] = None,
                 pi_E: Optional[PISpecE] = None,
                 ridge_alpha: float = 1.0) -> PipelineOutput:

    set_global_seed(seed)

    t = np.asarray(case["time"], float)
    M = np.asarray(case["mass"], float)
    h = np.asarray(case["thick"], float)
    assert_sorted_time(t)

    # Stage 1: Mass
    mass_model = MassSurrogate(rf_depth=rf_depth, seed=seed)
    mass_res = mass_model.train(t, M, n_train=n_mass, k_folds=kfolds,
                                strategy=mass_strategy, pi=pi_mass)

    # Stage 2: Thickness
    alpha = fit_alpha_from_cases(all_cases_for_alpha)
    thick_model = ThicknessSurrogate(rf_depth=rf_depth, seed=seed)
    thick_res = thick_model.train(t, h, mass_res.M_pred, alpha,
                                  n_train=n_thick, k_folds=kfolds,
                                  strategy=thick_strategy,
                                  augment=use_aug_for_thickness, pi=pi_thick)

    # Stage 3: Density
    dens_model = DensitySurrogate(rf_depth=rf_depth, seed=seed)
    dens_res = dens_model.train(t, M, h, n_train=n_dens, k_folds=kfolds,
                                strategy=dens_strategy,
                                y_true=density_y_true, rho_dry=rho_dry, rho_fluid=rho_fluid)

    cg_res = None
    e_res = None
    if elastic is not None and "hour" in elastic and "GV_meas" in elastic:
        el_hours = np.asarray(elastic["hour"], float)
        gv_meas = np.asarray(elastic["GV_meas"], float)

        # Stage 4: CG (UNCHANGED baseline with PI)
        cg_res = train_cg_with_pi(t_mass=t, M_curve=M, h_curve_mm=h,
                                  meas_hours=el_hours, gv_meas_kms=gv_meas,
                                  pi_spec=pi_cg, rho_dry=rho_dry, rho_fluid=rho_fluid, seed=seed)

        # Stage 5: E (Ridge + Physics Hybrid) with firstk training option
        rho_at = np.interp(el_hours, t, rom_density(M, h, rho_dry, rho_fluid))
        E_true = np.asarray(elastic["E_true"], float) if "E_true" in elastic else None
        e_model = YoungsSurrogate(alpha=ridge_alpha, seed=seed)

        # explicit indices override strategy if provided in elastic
        e_train_idx = None
        if "E_train_idx" in (elastic or {}):
            e_train_idx = np.asarray(elastic["E_train_idx"], int)

        cg_pred_for_pi = None
        if pi_E is not None and pi_E.gv_source == "cg_pred" and cg_res is not None:
            cg_pred_for_pi = cg_res.gv_pred10_ms  # over measured hours; used with interp to PI hours

        e_res = e_model.train(hours=el_hours, rho_at_hours=rho_at,
                              gv_meas_kms=gv_meas, E_true_GPa=E_true,
                              pi=pi_E, cg_pred_ms_for_pi=cg_pred_for_pi,
                              train_idx=e_train_idx,
                              train_strategy=E_train_strategy,
                              n_train_E=E_n_train)

    return PipelineOutput(mass=mass_res, thick=thick_res, dens=dens_res,
                          alpha=alpha, cg=cg_res, youngs=e_res)
