import numpy as np
from pi_pipeline import run_pipeline
from pi_plots import plot_all
from pi_surrogates import PISpec1D, PISpecCG, PISpecE

case5 = {
    "time":  np.array([0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528], float),
    "mass":  np.array([121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05], float),
    "thick": np.array([3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085], float),
}
all_cases = [case5]

elastic = {
    "hour":    np.array([0,24,90,180,270,360,528], float),
    "GV_meas": np.array([2.36,1.89,1.61,1.50,1.34,1.30,1.32], float),   # km/s
    "E_true":  np.array([6.479,4.219,3.269,2.757,2.773,2.185,2.187], float),  # GPa

    # manual E training indices on this 7-pt grid:
    "E_train_idx": [0, 1, 2, 6],
}

# PI points
cg_pi_hours = np.array([9.47368421, 18.94736842, 47.36842105], float)
cg_pi_ms    = np.array([2208, 1996, 1776], float)

# Optional PI hours for mass/thickness/E
mass_pi_hours  = np.array([216, 288, 360], float)
thick_pi_hours = np.array([0, 5,10, 528], float)
E_pi_hours     = np.array([45, 150, 270], float)

# Manual experimental training point selection (indices on the 23-pt time grid)
mass_train_idx  = [0, 1, 2, 5, 7]   # (0,24,48)
thick_train_idx = [5, 9, 11]   # (0,24,72)
dens_train_idx  = [0, 5, 11, 14]   # (0,48,120)

out = run_pipeline(
    case=case5,
    all_cases_for_alpha=all_cases,

    # strategies used only if manual not provided
    mass_strategy="even",
    thick_strategy="even",
    dens_strategy="even",

    # manual overrides
    mass_train_idx=mass_train_idx,
    thick_train_idx=thick_train_idx,
    dens_train_idx=dens_train_idx,

    # model knobs
    n_mass=3, n_thick=3, n_dens=3,
    kfolds=3,
    use_aug_for_thickness=False,
    rf_depth=5,
    seed=0,
    rho_dry=1150.0,
    rho_fluid=1000.0,

    elastic=elastic,

    # PI controls (pi_percent 0..1)
    pi_mass=PISpec1D(extra_hours=mass_pi_hours,  pi_percent=1.0, desc="Mass PI"),
    pi_thick=PISpec1D(extra_hours=thick_pi_hours, pi_percent=1, desc="Thickness PI"),
    pi_cg=PISpecCG(extra_hours=cg_pi_hours, extra_gv_ms=cg_pi_ms, pi_percent=1.0, desc="CG PI"),
    pi_E=PISpecE(extra_hours=E_pi_hours, pi_percent=1.0, gv_source="interp", desc="E PI"),

    # E: if you want firstk instead of E_train_idx, set E_train_strategy="firstk", E_n_train=3
    E_train_strategy="auto",
    ridge_alpha=1.0
)

print("Mass   -> Val MAE:", out.mass.mae,  "| λ*:", out.mass.lambda_star, "| train idx:", out.mass.idx_train)
print("Thick  -> Val MAE:", out.thick.mae, "| λ*:", out.thick.lambda_star, "| train idx:", out.thick.idx_train, "| α:", out.alpha)
print("Density-> Val MAE:", out.dens.mae,  "| λ*:", out.dens.lambda_star, "| train idx:", out.dens.idx_train)
if out.cg is not None:
    print(f"CG MAE (m/s): 4-pt={out.cg.mae4_ms:.1f}, 7-pt={out.cg.mae7_ms:.1f}, 10-pt={out.cg.mae10_ms:.1f} | PI pts used: {out.cg.n_extra_used}")
if out.youngs is not None and out.youngs.E_true_GPa is not None:
    print("E      -> Val MAE:", out.youngs.loocv_mae_GPa, "GPa", "| best w =", out.youngs.best_w)

plot_all(out, M_exp=case5["mass"], h_exp=case5["thick"], case_name="Case 5", save_prefix=None)
