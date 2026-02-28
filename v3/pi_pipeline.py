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
    mass_mode="all", mass_n=None, mass_idx=None, pi_mass=None, pi_w_mass=1.0,
    thick_mode="all", thick_n=None, thick_idx=None, pi_thick=None, pi_w_thick=1.0, fixed_alpha=None,
    cg_case=None, cg_mode="all", cg_n=None, cg_idx=None, pi_cg=None, pi_w_cg=1.0,
    E_case=None, E_mode="all", E_n=None, E_idx=None, pi_E=None, pi_w_E=1.0,
    rho_0=1200.0, rho_fluid=1000.0
):
    set_global_seed(42)
    t = case["time"]
    M = case["mass"]
    h = case["thick"]
    
    # 1. Mass
    idx_m = resolve_idx(len(t), mass_mode, mass_n, mass_idx)
    mass_res = MassSurrogate().train(t, M, idx_m, pi_mass, pi_w_mass)
    
    # 2. Thickness
    alpha = fixed_alpha if fixed_alpha else fit_alpha_from_cases(all_cases_for_alpha)
    idx_th = resolve_idx(len(t), thick_mode, thick_n, thick_idx)
    
    thick_res = ThicknessSurrogate().train(
        t, h, mass_res.pred_hybrid, 
        idx_th, alpha, pi_thick, pi_w_thick
    )
    
    # 3. Density
    rho_exp = rom_density(M, h, M[0], rho_0, rho_fluid)
    dens_model = DensitySurrogate()
    dens_res = dens_model.train(
        t, 
        mass_res.pred_hybrid, thick_res.pred_hybrid, 
        mass_res.pred_pure, thick_res.pred_pure, 
        M[0], h[0], rho_0, idx_m, rho_fluid=rho_fluid
    )
    
    # 4. Group Velocity
    cg_res = None
    if cg_case:
        t_cg, gv_cg = cg_case["hour"], cg_case["gv_ms"]
        idx_cg = resolve_idx(len(t_cg), cg_mode, cg_n, cg_idx)
        cg_res = CGSurrogate().train(t, t_cg, gv_cg, idx_cg, pi_cg, pi_w_cg)
        
    # 5. Youngs Modulus
    youngs_res = None
    if E_case and cg_res:
        t_E, val_E = E_case["hour"], E_case["E_GPa"]
        idx_E = resolve_idx(len(t_E), E_mode, E_n, E_idx)
        youngs_res = YoungsSurrogate().train(
            t, t_E, val_E, idx_E, 
            dens_res.pred_hybrid, cg_res.pred_hybrid, 
            dens_res.pred_pure, cg_res.pred_pure,
            pi_E, pi_w_E
        )
        
    return PipelineOutput(mass_res, thick_res, dens_res, cg_res, youngs_res, alpha, rho_exp)