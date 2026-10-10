# pi_pipeline.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from pi_surrogates import *

@dataclass
class PipelineOutput:
    mass: ResultBase; thick: ThickResult; dens: ResultBase
    cg: Optional[ResultBase]; youngs: Optional[ResultBase]; alpha: float
    rho_exp: Optional[np.ndarray] = None

def resolve_idx(n, mode, n_req, manual):
    if mode == "all": return np.arange(n)
    if mode == "manual": return np.array(manual)
    if mode == "even": return np.linspace(0, n-1, n_req).round().astype(int)
    return np.arange(n)

def run_pipeline(
    case, all_cases_for_alpha,
    mass_mode="all", mass_n=None, mass_idx=None, mass_pure_mode=None, mass_pure_n=None, mass_pure_idx=None, pi_mass=None, pi_w_mass=1.0,
    thick_mode="all", thick_n=None, thick_idx=None, thick_pure_mode=None, thick_pure_n=None, thick_pure_idx=None, pi_thick=None, pi_w_thick=1.0, fixed_alpha=None,
    cg_case=None, cg_mode="all", cg_n=None, cg_idx=None, cg_pure_mode=None, cg_pure_n=None, cg_pure_idx=None, pi_cg=None, pi_w_cg=1.0,
    E_case=None, E_mode="all", E_n=None, E_idx=None, E_pure_mode=None, E_pure_n=None, E_pure_idx=None, pi_E=None, pi_w_E=1.0,
    rho_0=1150.0, rho_fluid=1000.0
):
    set_global_seed(42)
    t = case["time"]
    M = case["mass"]
    h = case["thick"]
    
    # 1. Mass
    idx_m = resolve_idx(len(t), mass_mode, mass_n, mass_idx)
    idx_m_pure = resolve_idx(len(t), mass_pure_mode, mass_pure_n, mass_pure_idx) if mass_pure_mode else idx_m
    mass_res = MassSurrogate().train(t, M, idx_m, pi_mass, pi_w_mass, idx_train_pure=idx_m_pure)
    
    # 2. Thickness
    alpha = fixed_alpha if fixed_alpha is not None else fit_alpha_from_cases(all_cases_for_alpha)
    idx_th = resolve_idx(len(t), thick_mode, thick_n, thick_idx)
    idx_th_pure = resolve_idx(len(t), thick_pure_mode, thick_pure_n, thick_pure_idx) if thick_pure_mode else idx_th
    thick_res = ThicknessSurrogate().train(
        t, h, mass_res.pred_hybrid, mass_res.pred_pure,
        idx_th, alpha, pi_thick, pi_w_thick, idx_train_pure=idx_th_pure
    )
    
    # 3. Density
    # Effective reference derived from measured mass, not an independent wet
    # density measurement. Thickness does not enter the retained mixture law.
    rho_exp = rom_density(M, h, M[0], rho_0, rho_fluid, h0=h[0])
    dens_model = DensitySurrogate()
    dens_res = dens_model.train(
        t, mass_res.pred_hybrid, thick_res.pred_hybrid, mass_res.pred_pure, thick_res.pred_pure, 
        M[0], h[0], rho_0, idx_m, rho_exp=rho_exp, rho_fluid=rho_fluid, idx_train_pure=idx_m_pure
    )
    
    # 4. Group Velocity
    cg_res = None
    if cg_case:
        t_cg, gv_cg = cg_case["hour"], cg_case["gv_ms"]
        idx_cg = resolve_idx(len(t_cg), cg_mode, cg_n, cg_idx)
        idx_cg_pure = resolve_idx(len(t_cg), cg_pure_mode, cg_pure_n, cg_pure_idx) if cg_pure_mode else idx_cg
        cg_res = CGSurrogate().train(t, t_cg, gv_cg, idx_cg, pi_cg, pi_w_cg, idx_train_pure=idx_cg_pure)
        
    # 5. Youngs Modulus
    youngs_res = None
    if E_case and cg_res:
        t_E, val_E = E_case["hour"], E_case["E_GPa"]
        idx_E = resolve_idx(len(t_E), E_mode, E_n, E_idx)
        idx_E_pure = resolve_idx(len(t_E), E_pure_mode, E_pure_n, E_pure_idx) if E_pure_mode else idx_E
        youngs_res = YoungsSurrogate().train(
            t, t_E, val_E, idx_E, 
            dens_res.pred_hybrid, cg_res.pred_hybrid, 
            dens_res.pred_pure, cg_res.pred_pure,
            pi_E, pi_w_E, idx_train_pure=idx_E_pure
        )
        
    return PipelineOutput(mass_res, thick_res, dens_res, cg_res, youngs_res, alpha, rho_exp)
