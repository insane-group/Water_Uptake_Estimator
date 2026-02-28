# main_run_comparison.py
import numpy as np
import matplotlib.pyplot as plt
from pi_pipeline import run_pipeline
from pi_surrogates import PISpec1D, PISpecCG, PISpecE, fit_alpha_from_cases

# --- SETUP DATA ---
# Early cases for Alpha calculation
cases = {}
cases[3] = {"time":np.array([0,23,47,71,96,120,147,170,194,218,242,266], float), 
            "mass":np.array([119.38,124.67,126.37,128.07,129.84,131.03,132.13,131.8,133.62,134.32,135.01,135.71], float), 
            "thick":np.array([3.72,3.85,3.87,3.93,3.99,4.02,3.99,4.02,4.02,4.03,4.05,4.06], float)}

# Target Case 5
case5 = {
    "time":  np.array([0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528], float),
    "mass":  np.array([121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05], float),
    "thick": np.array([3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085], float),
}

# 1. Calculate Alpha
alpha_early = fit_alpha_from_cases([case5])
print(f"Global Alpha: {alpha_early:.4f}")

# --- CONFIG FOR BOTH RUNS ---
common_cfg = dict(
    case=case5, all_cases_for_alpha=[],
    mass_train_idx=[0, 1, 2, 5, 7], 
    thick_train_idx=[5, 9, 11], # Sparse Training
    dens_train_idx=[0, 5, 11, 14],
    pi_mass=PISpec1D(extra_hours=[216, 288, 360], pi_percent=1.0),
    pi_thick=PISpec1D(extra_hours=[0, 5, 10, 528], pi_percent=1.0),
    pi_cg=None, pi_E=None, elastic=None,
    fixed_alpha=alpha_early
)

# --- RUN 1: WITH Augmentation (Standard) ---
print("Running WITH Augmentation...")
out_aug = run_pipeline(**common_cfg, use_aug_for_thickness=True)

# --- RUN 2: WITHOUT Augmentation (Pure Sparse RF) ---
print("Running WITHOUT Augmentation...")
out_no_aug = run_pipeline(**common_cfg, use_aug_for_thickness=False)

# --- PLOTTING COMPARISON ---
t = out_aug.thick.t_h
h_exp = case5["thick"]

plt.figure(figsize=(10, 6))

# 1. Ground Truth
plt.plot(t, h_exp, 'o', color='gray', alpha=0.5, label="Experimental Truth")
plt.scatter(t[out_aug.thick.idx_train], h_exp[out_aug.thick.idx_train], 
            marker='s', s=100, color='black', label="Training Points (Used by both)")

# 2. Physics Baseline (The linear law)
plt.plot(t, out_aug.thick.h_phys, 'k:', alpha=0.6, label=f"Physics Baseline (Linear Alpha={alpha_early:.3f})")

# 3. NO Augmentation Result (The "Staircase")
plt.plot(t, out_no_aug.thick.h_pred, 'r--', linewidth=2, label="Result WITHOUT Augmentation")

# 4. WITH Augmentation Result (The Smooth Curve)
plt.plot(t, out_aug.thick.h_pred, 'b-', linewidth=2, label="Result WITH Augmentation")

plt.title("Impact of Physics Augmentation on Thickness Prediction")
plt.xlabel("Time (h)")
plt.ylabel("Thickness (mm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()