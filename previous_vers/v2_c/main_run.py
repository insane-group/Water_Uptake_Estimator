# main_run.py
import numpy as np
from pi_pipeline import run_pipeline
from pi_plots import plot_all
from pi_surrogates import PISpec1D, PISpecCG, PISpecE, fit_alpha_from_cases

# 1. EARLY CASES
cases = {}
cases[1] = {"time":np.array([0,18,42,66,90], float), "mass":np.array([112.51,118.8,120.52,122.17,124.27], float), "thick":np.array([3.42,3.61,3.67,3.74,3.84], float)}
cases[2] = {"time":np.array([0,13,37,61,85,109,133,158,180,205], float), "mass":np.array([101.97,106.46,108.55,109.92,110.96,112.18,113.27,114.48,113.8,113.2], float), "thick":np.array([3.05,3.23,3.29,3.39,3.36,3.37,3.39,3.4,3.44,3.45], float)}
cases[3] = {"time":np.array([0,23,47,71,96,120,147,170,194,218,242,266], float), "mass":np.array([119.38,124.67,126.37,128.07,129.84,131.03,132.13,131.8,133.62,134.32,135.01,135.71], float), "thick":np.array([3.72,3.85,3.87,3.93,3.99,4.02,3.99,4.02,4.02,4.03,4.05,4.06], float)}
cases[4] = {
    "time":  np.array([0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528], float),
    "mass":  np.array([121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05], float),
    "thick": np.array([3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085], float),
}
print("--- Discrete Alphas ---")
aa={}
for i in [1, 2, 3, 4]:
    a = fit_alpha_from_cases([cases[i]])
    aa[i]=a
    print(f"Case {i}: {a:.4f}")

# Calculate Global Alpha from ALL cases
#global_alpha = fit_alpha_from_cases([cases[1], cases[2], cases[3]])
global_alpha=max(aa.values())
print(f"Global Alpha: {global_alpha:.4f}")

# 2. CASE 5
case5 = {
    "time":  np.array([0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528], float),
    "mass":  np.array([121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05], float),
    "thick": np.array([3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085], float),
}
elastic_data = {"hour": np.array([0,24,90,180,270,360,528], float), 
                "GV_meas": np.array([2.36,1.89,1.61,1.50,1.34,1.30,1.32], float), 
                "E_true": np.array([6.479,4.219,3.269,2.757,2.773,2.185,2.187], float), 
                "E_train_idx": [0, 1, 2, 6]}

# 3. RUN
out = run_pipeline(
    case=case5, all_cases_for_alpha=[],
    mass_train_idx=[0, 1, 2, 5, 7], thick_train_idx=[5, 9, 11], dens_train_idx=[0, 5, 11, 14],
    pi_mass=PISpec1D(extra_hours=[216, 288, 360], pi_percent=1.0),
    pi_thick=PISpec1D(extra_hours=[0, 5, 10, 528], pi_percent=1.0),
    pi_cg=PISpecCG(extra_hours=[9.47, 18.94, 47.36], extra_gv_ms=[2208, 1996, 1776], pi_percent=1.0),
    pi_E=PISpecE(extra_hours=[45, 150, 270], pi_percent=1.0),
    elastic=elastic_data, fixed_alpha=global_alpha
)

print(f"Mass MAE: {out.mass.mae:.4f} | Thick MAE: {out.thick.mae:.4f} (Alpha: {out.alpha:.4f})")
plot_all(out, case5["mass"], case5["thick"], case_name="Case 5 Hybrid")