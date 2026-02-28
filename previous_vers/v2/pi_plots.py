# pi_plots.py — consistent plotting for all stages with training & PI markers
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "legend.frameon": True,
})

# scatter styles (no linestyle — scatter doesn't support it)
TRAIN_KW = dict(marker='s', color='black', s=55, zorder=5, label='Train pts')
PI_KW    = dict(marker='^', color='orange', s=65, zorder=6, label='PI pts')

def plot_mass(out, M_exp, case_name="Case", savepath=None):
    t = out.mass.t_h
    plt.figure(figsize=(9,5))
    plt.plot(t, M_exp, '-', label="Experimental mass")
    plt.plot(t, out.mass.M_phys, ':', label="Physics (Fickian)")
    plt.plot(t, out.mass.M_rf, '--', label="RF")
    plt.plot(t, out.mass.M_pred, '-', label=f"Blend λ={out.mass.lambda_star:.2f}")
    plt.scatter(t[out.mass.idx_train], M_exp[out.mass.idx_train], **TRAIN_KW)
    if out.mass.pi_hours is not None and out.mass.pi_values is not None:
        plt.scatter(out.mass.pi_hours, out.mass.pi_values, **PI_KW)
    plt.xlabel("Time (h)"); plt.ylabel("Mass")
    plt.title(f"{case_name} — Mass  |  Val MAE={out.mass.mae:.3f}")
    plt.legend(); plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200)
    plt.show()

def plot_thickness(out, h_exp, case_name="Case", savepath=None):
    t = out.thick.t_h
    plt.figure(figsize=(9,5))
    plt.plot(t, h_exp, '-', label="Experimental thickness")
    plt.plot(t, out.thick.h_phys, ':', label="Physics (swelling)")
    plt.plot(t, out.thick.h_rf, '--', label="RF")
    plt.plot(t, out.thick.h_pred, '-', label=f"Blend λ={out.thick.lambda_star:.2f}")
    plt.scatter(t[out.thick.idx_train], h_exp[out.thick.idx_train], **TRAIN_KW)
    if out.thick.pi_hours is not None and out.thick.pi_values is not None:
        plt.scatter(out.thick.pi_hours, out.thick.pi_values, **PI_KW)
    plt.xlabel("Time (h)"); plt.ylabel("Thickness (mm)")
    plt.title(f"{case_name} — Thickness  |  Val MAE={out.thick.mae:.3f}  (α≈{out.alpha:.3f})")
    plt.legend(); plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200)
    plt.show()

def plot_density(out, case_name="Case", savepath=None):
    t = out.dens.t_h
    plt.figure(figsize=(9,5))
    plt.plot(t, out.dens.rho_phys, ':', label="Physics (ROM)")
    plt.plot(t, out.dens.rho_rf, '--', label="RF")
    plt.plot(t, out.dens.rho_pred, '-', label=f"Blend λ={out.dens.lambda_star:.2f}")
    plt.scatter(t[out.dens.idx_train], out.dens.y_target[out.dens.idx_train], **TRAIN_KW)
    plt.xlabel("Time (h)"); plt.ylabel("Density (kg/m³)")
    plt.title(f"{case_name} — Density (PI)  |  Val MAE={out.dens.mae:.3f}")
    plt.legend(); plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200)
    plt.show()

def plot_cg(out, case_name="Case", savepath=None):
    if out.cg is None:
        return
    h = out.cg.hours
    plt.figure(figsize=(9,5))
    plt.plot(h, out.cg.gv_true_ms/1000.0, '-', label='Measured CG')
    plt.plot(h, out.cg.gv_pred4_ms/1000.0, 's--', label=f'4-pt (MAE={out.cg.mae4_ms:.1f} m/s)')
    plt.plot(h, out.cg.gv_pred7_ms/1000.0, 'x--', label=f'7-pt (MAE={out.cg.mae7_ms:.1f} m/s)')
    extra_note = f" +{out.cg.n_extra_used} PI pts"
    plt.plot(h, out.cg.gv_pred10_ms/1000.0, 'D--', label=f'7-pt{extra_note} (MAE={out.cg.mae10_ms:.1f} m/s)')
    if out.cg.pi_hours is not None and out.cg.pi_values_ms is not None:
        plt.scatter(out.cg.pi_hours, out.cg.pi_values_ms/1000.0, **PI_KW)
    plt.xlabel("Hour"); plt.ylabel("CG (km/s)")
    plt.title(f"{case_name} — CG Surrogates (PI: {out.cg.pi_desc})")
    plt.legend(); plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200)
    plt.show()

def plot_youngs(out, case_name="Case", savepath=None):
    if out.youngs is None:
        return
    hrs = out.youngs.hours
    plt.figure(figsize=(9,5))
    if out.youngs.E_true_GPa is not None:
        plt.plot(hrs, out.youngs.E_true_GPa, '-', label='Measured E (GPa)')
        if out.youngs.idx_train is not None and len(out.youngs.idx_train) > 0:
            tri = np.asarray(out.youngs.idx_train, int)
            plt.scatter(hrs[tri], out.youngs.E_true_GPa[tri], **TRAIN_KW)
    plt.plot(hrs, out.youngs.E_ridge_GPa, 'x--', label='Ridge-only')
    plt.plot(hrs, out.youngs.E_phys_GPa,  ':', label='Physics-only')
    plt.plot(hrs, out.youngs.E_hybrid_GPa, '-', label=f'Hybrid (w={out.youngs.best_w:.2f})')
    if out.youngs.pi_hours is not None and out.youngs.pi_values_GPa is not None:
        plt.scatter(out.youngs.pi_hours, out.youngs.pi_values_GPa, **PI_KW)
    title = f"{case_name} — Young's Modulus"
    if out.youngs.loocv_mae_GPa is not None:
        title += f" | Val MAE={out.youngs.loocv_mae_GPa:.3f} GPa"
    plt.title(title)
    plt.xlabel("Hour"); plt.ylabel("E (GPa)")
    plt.legend(); plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200)
    plt.show()

def plot_all(out, M_exp, h_exp, case_name="Case", save_prefix=None):
    mp = f"{save_prefix}_mass.png" if save_prefix else None
    tp = f"{save_prefix}_thickness.png" if save_prefix else None
    dp = f"{save_prefix}_density.png" if save_prefix else None
    cp = f"{save_prefix}_cg.png" if save_prefix else None
    ep = f"{save_prefix}_E.png" if save_prefix else None

    plot_mass(out, M_exp, case_name, mp)
    plot_thickness(out, h_exp, case_name, tp)
    plot_density(out, case_name, dp)
    plot_cg(out, case_name, cp)
    plot_youngs(out, case_name, ep)
