# pi_pipeline.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from pi_surrogates import *

@dataclass
class PipelineOutput:
    mass: MassResult; thick: ThickResult; dens: DensityResult; alpha: float
    cg: Optional[CGResult] = None; youngs: Optional[EResult] = None

def _idx_from_hours(t, hours):
    if hours is None: return None
    idx = [int(np.argmin(np.abs(t - h))) for h in np.asarray(hours).ravel()]
    return np.unique(np.asarray(idx, int))

def run_pipeline(case, all_cases_for_alpha, mass_train_idx=None, thick_train_idx=None, dens_train_idx=None,
                 pi_mass=None, pi_thick=None, pi_cg=None, pi_E=None,
                 weight_pi_mass=1.0, weight_pi_thick=1.0, weight_pi_cg=1.0, weight_pi_E=1.0,
                 fixed_alpha=None,
                 # Unused args
                 mass_strategy="even", thick_strategy="even", dens_strategy="even",
                 mass_train_hours=None, thick_train_hours=None, dens_train_hours=None,
                 n_mass=5, n_thick=9, n_dens=7, kfolds=3, use_aug_for_thickness=True,
                 rf_depth=3, seed=0, lambdas_mass=None, lambdas_thick=None, lambdas_dens=None,
                 density_y_true=None, rho_dry=1150.0, rho_fluid=1000.0,
                 elastic=None, use_cg_pred_for_E_phys=True,
                 E_train_strategy="auto", E_n_train=None, ridge_alpha=1.0):

    set_global_seed(seed)
    t = np.asarray(case["time"], float)
    M = np.asarray(case["mass"], float)
    h = np.asarray(case["thick"], float)
    assert_sorted_time(t)
    M_dry, h_dry = float(M[0]), float(h[0])

    m_idx = np.asarray(mass_train_idx, int) if mass_train_idx else _idx_from_hours(t, mass_train_hours)
    th_idx = np.asarray(thick_train_idx, int) if thick_train_idx else _idx_from_hours(t, thick_train_hours)
    d_idx = np.asarray(dens_train_idx, int) if dens_train_idx else _idx_from_hours(t, dens_train_hours)

    # 1. Mass
    mass_model = MassSurrogate(rf_depth=rf_depth, seed=seed)
    mass_res = mass_model.train(t, M, m_idx, lambdas_mass or [0,0.2,0.4,0.6,0.8,1.0], pi_mass, weight_pi_mass)

    # 2. Thickness
    thick_model = ThicknessSurrogate(rf_depth=rf_depth, seed=seed)
    thick_res = thick_model.train(
        t, h, mass_res.M_pred, 
        M_dry=M_dry, h_dry=h_dry, 
        train_idx=th_idx, 
        lambdas=lambdas_thick or [0,0.2,0.4,0.6,0.8,1.0],
        augment=use_aug_for_thickness, pi=pi_thick, pi_weight=weight_pi_thick,
        feature_mode="mass", 
        fixed_alpha=fixed_alpha # Passing the LINEAR ALPHA directly
    )

    # 3. Density
    dens_model = DensitySurrogate(rf_depth=rf_depth, seed=seed)
    dens_res = dens_model.train(t, mass_res.M_pred, thick_res.h_pred, M_dry, d_idx, 
                                lambdas_dens or [0,0.2,0.4,0.6,0.8,1.0], rho_dry, rho_fluid)

    cg_res, e_res = None, None
    if elastic:
        el_h = np.asarray(elastic["hour"], float)
        gv = np.asarray(elastic["GV_meas"], float)
        cg_res = train_cg_with_pi(t, dens_res.rho_pred, thick_res.h_pred, el_h, gv, pi_cg, seed, weight_pi_cg)
        gv_inf = cg_res.gv_pred10_ms if (use_cg_pred_for_E_phys and cg_res) else None
        e_model = YoungsSurrogate(alpha=ridge_alpha, seed=seed)
        e_idx = np.asarray(elastic["E_train_idx"], int) if "E_train_idx" in elastic else None
        e_res = e_model.train(el_h, np.interp(el_h, t, dens_res.rho_pred), gv, 
                              elastic.get("E_true"), gv_inf, pi_E, e_idx, weight_pi_E)

    return PipelineOutput(mass_res, thick_res, dens_res, thick_res.alpha, cg_res, e_res)