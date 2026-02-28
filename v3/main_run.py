# main_run.py
import numpy as np
from pi_pipeline import run_pipeline
from pi_plots import plot_all
from pi_surrogates import PISpec1D, fit_alpha_from_cases

# main_run.py
import numpy as np
from pi_pipeline import run_pipeline
from pi_plots import plot_all
from pi_surrogates import PISpec1D, fit_alpha_from_cases

# --- HISTORICAL EXPERIMENTAL DATA (For STD calculation) ---
all_cases = [
    { # Case 1
        "time": np.array([0,18,42,66,90], float),
        "mass": np.array([112.51,118.8,120.52,122.17,124.27], float),
        "thick": np.array([3.42,3.61,3.67,3.74,3.84], float)
    },
    { # Case 2
        "time": np.array([0,13,37,61,85,109,133,158,180,205], float),
        "mass": np.array([101.97,106.46,108.55,109.92,110.96,112.18,113.27,114.48,113.8,113.2], float),
        "thick": np.array([3.05,3.23,3.29,3.39,3.36,3.37,3.39,3.4,3.44,3.45], float)
    },
    { # Case 3
        "time": np.array([0,23,47,71,96,120,147,170,194,218,242,266], float),
        "mass": np.array([119.38,124.67,126.37,128.07,129.84,131.03,132.13,131.8,133.62,134.32,135.01,135.71], float),
        "thick": np.array([3.72,3.85,3.87,3.93,3.99,4.02,3.99,4.02,4.02,4.03,4.05,4.06], float)
    },
    { # Case 4
        "time": np.array([0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360], float),
        "mass": np.array([111.64,116.305,120.97,122.41,123.69,124.97,126.25,125.96,125.84,125.94,126.04,126.79,127.54,128.29,127.01,126], float),
        "thick": np.array([3.3,3.42,3.54,3.66,3.7,3.73,3.77,3.82,3.81,3.83,3.85,3.83,3.82,3.8,3.79,3.78], float)
    },
    { # Case 5 (Target)
        "time":  np.array([0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528], float),
        "mass":  np.array([121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,135.61,135.02,135.29,136.3,136.95,137.61,138.29,138.75,139.05], float),
        "thick": np.array([3.45,3.52,3.58,3.6,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.1,4.07,4.073,4.077,4.08,4.09,4.07], float)
    },
]

case5 = all_cases[4]
cg_data = {"hour": np.array([0, 24, 90, 180, 270, 360, 528], float), "gv_ms": np.array([2360, 1890, 1610, 1500, 1340, 1300, 1320], float)}
E_data = {"hour": np.array([0, 24, 90, 180, 270, 360, 528], float), "E_GPa": np.array([6.479, 4.219, 3.269, 2.757, 2.773, 2.185, 2.187], float)}

# --- CALCULATE EXPERIMENTAL UNCERTAINTY (STD) MARGINS ---
def compute_std(t_eval, cases, key):
    diffs = []
    for c in cases:
        val = np.array(c[key])
        diff = val - val[0]
        diffs.append(np.interp(t_eval, c["time"], diff, right=np.nan))
    with np.errstate(invalid='ignore'): std_val = np.nanstd(np.array(diffs), axis=0)
    valid_mask = ~np.isnan(std_val) & (std_val > 0)
    if np.any(valid_mask): std_val[np.where(valid_mask)[0][-1]+1:] = std_val[np.where(valid_mask)[0][-1]]
    std_val[np.isnan(std_val)] = 0.0
    return std_val

def compute_rho_std(t_eval, cases, rho_dry=1150.0, rho_fluid=1000.0):
    diffs = []
    for c in cases:
        M = np.array(c["mass"])
        w = np.maximum(0.0, (M - M[0]) / np.maximum(M, 1e-12))
        rho = (1 - w) * rho_dry + w * rho_fluid
        diffs.append(np.interp(t_eval, c["time"], rho - rho[0], right=np.nan))
    with np.errstate(invalid='ignore'): std_val = np.nanstd(np.array(diffs), axis=0)
    valid_mask = ~np.isnan(std_val) & (std_val > 0)
    if np.any(valid_mask): std_val[np.where(valid_mask)[0][-1]+1:] = std_val[np.where(valid_mask)[0][-1]]
    std_val[np.isnan(std_val)] = 0.0
    return std_val

exp_stds = {
    "mass": compute_std(case5["time"], all_cases, "mass"),
    "thick": compute_std(case5["time"], all_cases, "thick"),
    "dens": compute_rho_std(case5["time"], all_cases)
}

# --- PIPELINE CONFIGURATION ---

cg_comp_hours = np.array([9.47, 18.94, 47.36]) #np.array([50, 150, 300, 450])
cg_comp_vals = np.array([2208, 1996, 1776]) #np.array([2000, 1550, 1350, 1310]) 

#cg_comp_hours = np.array([270, 360, 528]) #np.array([50, 150, 300, 450])
#cg_comp_vals = np.array([1340, 1300, 1320]) #np.array([2000, 1550, 1350, 1310]) 


pi_mass  = PISpec1D(np.array([150, 300]), pi_percent=1.0)
pi_thick = PISpec1D(np.array([50, 150]), pi_percent=1.0)
pi_cg    = PISpec1D(cg_comp_hours, values=cg_comp_vals, pi_percent=1.0) 
pi_E     = PISpec1D(np.array([ 10, 50, 200, 400]), pi_percent=1.0)

#even - manual
mass_cfg  = dict(mass_mode="even", mass_n=3)
thick_cfg = dict(thick_mode="even", thick_n=3) #thick_idx=[0, 5, 11]
cg_cfg    = dict(cg_mode="even", cg_n=4) #cg_idx=[1, 2, 4, 6]
E_cfg     = dict(E_mode="even", E_n=3) #E_idx=[0, 2, 6]

"""

pi_mass  = PISpec1D(np.array([150, 300]), pi_percent=1.0)
pi_thick = PISpec1D(np.array([250, 300]), pi_percent=1.0)
pi_cg    = PISpec1D(cg_comp_hours, values=cg_comp_vals, pi_percent=1.0) 
pi_E     = PISpec1D(np.array([ 10, 20, 200, 400]), pi_percent=1.0)

#even - manual
mass_cfg  = dict(mass_mode="manual", mass_idx=[0, 4, 8])
thick_cfg = dict(thick_mode="manual", thick_idx=[0, 4, 8]) #thick_idx=[0, 5, 11]
cg_cfg    = dict(cg_mode="manual", cg_idx=[0, 1, 2]) #cg_idx=[1, 2, 4, 6]
E_cfg     = dict(E_mode="manual", E_idx=[0, 1, 2]) #E_idx=[0, 2, 6]
"""

alpha_global = fit_alpha_from_cases([case5])

print("Running Pipeline...")
out = run_pipeline(
    case=case5, all_cases_for_alpha=[case5], fixed_alpha=alpha_global, #all_cases_for_alpha=all_cases,
    **mass_cfg, pi_mass=pi_mass, pi_w_mass=2.0,
    **thick_cfg, pi_thick=pi_thick, pi_w_thick=2.0,
    cg_case=cg_data, **cg_cfg, pi_cg=pi_cg, pi_w_cg=2.0,
    E_case=E_data, **E_cfg, pi_E=pi_E, pi_w_E=2.0,
    rho_0=1150.0, rho_fluid=1000.0
)

print(f"\nRESULTS (Pure vs Hybrid MAE):")
print(f"Mass:   Pure={out.mass.mae_pure:.4f} | Hybrid={out.mass.mae_val:.4f}")
print(f"Thick:  Pure={out.thick.mae_pure:.4f} | Hybrid={out.thick.mae_val:.4f}")
print(f"CG:     Pure={out.cg.mae_pure:.4f} | Hybrid={out.cg.mae_val:.4f}")
print(f"Youngs: Pure={out.youngs.mae_pure:.4f} | Hybrid={out.youngs.mae_val:.4f}")

plot_all(out, case5, cg_data, E_data, prefix="Case 5", exp_stds=exp_stds)