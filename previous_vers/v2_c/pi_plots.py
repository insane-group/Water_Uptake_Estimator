# pi_plots.py — Corrected for Linear Alpha Model
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.grid": True, "grid.alpha": 0.25,
    "lines.linewidth": 2.0, "legend.frameon": True,
})
TRAIN_KW = dict(marker='s', color='black', s=55, zorder=5, label='Train pts')
PI_KW    = dict(marker='^', color='orange', s=65, zorder=6, label='PI pts')
RF_LABEL = "Pure Data Fit"

def plot_mass(out, M_exp, case_name="Case"):
    t = out.mass.t_h
    plt.figure(figsize=(9, 5))
    plt.plot(t, M_exp, '-', color='gray', alpha=0.5, label="Ground Truth")
    plt.plot(t, out.mass.M_phys, ':', label="Physics (Fickian)")
    plt.plot(t, out.mass.M_pure_rf, '--', color='tab:red', label=RF_LABEL)
    plt.plot(t, out.mass.M_pred, '-', color='tab:blue', linewidth=2.5, label=f"Hybrid (λ={out.mass.lambda_star:.2f})")
    plt.scatter(t[out.mass.idx_train], M_exp[out.mass.idx_train], **TRAIN_KW)
    if out.mass.pi_hours is not None: plt.scatter(out.mass.pi_hours, out.mass.pi_values, **PI_KW)
    plt.title(f"{case_name} - Mass | MAE={out.mass.mae:.3f}"); plt.legend(); plt.show()

def plot_thickness(out, h_exp, case_name="Case"):
    t = out.thick.t_h
    plt.figure(figsize=(9, 5))
    plt.plot(t, h_exp, '-', color='gray', alpha=0.5, label="Ground Truth")
    plt.plot(t, out.thick.h_phys, ':', label="Physics (Linear Alpha)")
    plt.plot(t, out.thick.h_pure_rf, '--', color='tab:red', label=RF_LABEL)
    plt.plot(t, out.thick.h_pred, '-', color='tab:blue', linewidth=2.5, label=f"Hybrid (λ={out.thick.lambda_star:.2f})")
    plt.scatter(t[out.thick.idx_train], h_exp[out.thick.idx_train], **TRAIN_KW)
    if out.thick.pi_hours is not None: plt.scatter(out.thick.pi_hours, out.thick.pi_values, **PI_KW)
    # FIX: Display Alpha, not Gamma
    plt.title(f"{case_name} - Thickness | MAE={out.thick.mae:.3f} | Alpha={out.thick.alpha:.4f}"); plt.legend(); plt.show()

def plot_density(out, case_name="Case"):
    t = out.dens.t_h
    plt.figure(figsize=(9, 5))
    plt.plot(t, out.dens.rho_phys, ':', label="Physics (ROM)")
    plt.plot(t, out.dens.rho_pure_rf, '--', color='tab:red', label=RF_LABEL)
    plt.plot(t, out.dens.rho_pred, '-', color='tab:blue', linewidth=2.5, label=f"Hybrid (λ={out.dens.lambda_star:.2f})")
    plt.title(f"{case_name} - Density"); plt.legend(); plt.show()

def plot_cg(out, case_name="Case"):
    if out.cg:
        h = out.cg.hours; plt.figure(figsize=(9, 5))
        plt.plot(h, out.cg.gv_true_ms/1e3, 'o-', color='gray', alpha=0.5, label='Measured')
        plt.plot(h, out.cg.gv_pred10_ms/1e3, 'D--', label=f'Hybrid (MAE={out.cg.mae10_ms:.1f})')
        if out.cg.pi_hours is not None: plt.scatter(out.cg.pi_hours, out.cg.pi_values_ms/1e3, **PI_KW)
        plt.title(f"{case_name} - CG"); plt.legend(); plt.show()

def plot_youngs(out, case_name="Case"):
    if out.youngs:
        hrs = out.youngs.hours; plt.figure(figsize=(9, 5))
        plt.plot(hrs, out.youngs.E_true_GPa, 'o-', color='gray', alpha=0.5, label="Measured")
        if out.youngs.idx_train is not None: plt.scatter(hrs[out.youngs.idx_train], out.youngs.E_true_GPa[out.youngs.idx_train], **TRAIN_KW)
        plt.plot(hrs, out.youngs.E_phys_GPa, ':', label="Physics")
        plt.plot(hrs, out.youngs.E_ridge_GPa, '--', color='tab:red', label=RF_LABEL)
        plt.plot(hrs, out.youngs.E_hybrid_GPa, '-', color='tab:blue', linewidth=2.5, label=f"Hybrid (w={out.youngs.best_w:.2f})")
        if out.youngs.pi_hours is not None: plt.scatter(out.youngs.pi_hours, out.youngs.pi_values_GPa, **PI_KW)
        plt.title(f"{case_name} - Young's Modulus"); plt.legend(); plt.show()

def plot_all(out, M_exp, h_exp, case_name="Case"):
    plot_mass(out, M_exp, case_name)
    plot_thickness(out, h_exp, case_name)
    plot_density(out, case_name)
    plot_cg(out, case_name)
    plot_youngs(out, case_name)