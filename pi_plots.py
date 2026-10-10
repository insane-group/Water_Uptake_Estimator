# pi_plots.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- INCREASE FONT SIZES FOR PUBLICATION ---
plt.rcParams.update({
    'font.size': 14,             
    'axes.titlesize': 16,        
    'axes.labelsize': 16,        
    'xtick.labelsize': 14,       
    'ytick.labelsize': 14,       
    'legend.fontsize': 16,
    'lines.linewidth': 2.5,      
    'lines.markersize': 8        
})

def plot_prop(t, res, title, ylabel, exp_data=None, train_t=None, phys_label="Physics Baseline", exp_std=None, output_path=None, show=True, reference_label="Measured observations"):
    if res is None: return
    plt.figure(figsize=(9, 10))
    
    # 1. Observed or derived reference
    if exp_data is not None:
        x_axis = train_t if train_t is not None else t
        if exp_std is not None:
            plt.fill_between(x_axis, exp_data - exp_std, exp_data + exp_std, 
                             color='gray', alpha=0.15, label="±1 SD across case histories", zorder=0)
        plt.plot(x_axis, exp_data, 'o-', color='gray', alpha=0.4, label=reference_label, zorder=1)

    # Detect points used
    pure_pts = len(res.idx_train_pure) if res.idx_train_pure is not None else len(res.idx_train)
    hyb_pts = len(res.idx_train)

    # 2. Predictions
    plt.plot(t, res.pred_pure, 'r--', label=f"Pure ML ({pure_pts} training records)", alpha=0.8, zorder=2)
    plt.plot(t, res.pred_phys, 'k:', label=phys_label, linewidth=2.0, zorder=3)
    plt.plot(t, res.pred_hybrid, 'b-', label=f"Hybrid PI ({hyb_pts} training records)", linewidth=2.5, zorder=4)
    
    # 3. Training Points
    if train_t is not None and exp_data is not None:
        valid_idx = res.idx_train[res.idx_train < len(train_t)]
        if len(valid_idx) > 0:
            plt.scatter(train_t[valid_idx], exp_data[valid_idx], 
                        c='white', edgecolors='k', s=100, linewidth=1.5, zorder=5, label="Hybrid training observations")
        
    # 4. PI anchors
    if res.pi_hours is not None:
        plt.scatter(res.pi_hours, res.pi_values, marker='^', c='orange', s=100, zorder=5, label="PI anchors")
        
    pure_scope = "training" if exp_data is not None and pure_pts == len(exp_data) else "development"
    hybrid_scope = "training" if exp_data is not None and hyb_pts == len(exp_data) else "development"
    plt.title(f"{title}\nRF MAE ({pure_scope}): {res.mae_pure:.4f}\nHybrid MAE ({hybrid_scope}): {res.mae_val:.4f}")
    plt.xlabel("Immersion Time (H)")
    plt.ylabel(ylabel)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finish_plot(output_path, show)

def plot_all(out, case_data, cg_data, E_data, prefix="Case", exp_stds=None, output_dir=None, show=True):
    t = out.mass.t
    exp_stds = exp_stds or {}
    def opts(name):
        return {"show":show, "output_path":None if output_dir is None else Path(output_dir)/f"{name}.png"}
    plot_prop(t, out.mass, "Mass", "Mass (g)", exp_data=case_data["mass"], train_t=t, phys_label="Physics (Fickian)", exp_std=exp_stds.get("mass"), **opts("mass"))
    plot_prop(t, out.thick, f"Thickness | Alpha={out.thick.alpha:.4f}", "Thickness (mm)", exp_data=case_data["thick"], train_t=t, phys_label="Physics (Linear Alpha)", exp_std=exp_stds.get("thick"), **opts("thick"))
    plot_prop(t, out.dens, "Density", "Density (kg/m$^3$)", exp_data=out.rho_exp, train_t=t, phys_label="Effective density mixture", exp_std=exp_stds.get("dens"), reference_label="Derived effective-density reference", **opts("dens"))
    if out.cg: plot_prop(t, out.cg, "Group Velocity", "GV (m/s)", exp_data=cg_data["gv_ms"], train_t=cg_data["hour"], phys_label="Forest from supplied anchors", **opts("cg"))
    if out.youngs: plot_prop(t, out.youngs, "Young's Modulus", "E (GPa)", exp_data=E_data["E_GPa"], train_t=E_data["hour"], phys_label="Group-velocity stiffness proxy", **opts("youngs"))

# --- PARITY (SCATTER) PLOTS FOR TEST DATA ---
def plot_parity(true_vals, pure_vals, phys_vals, hyb_vals, title, unit, n_pure, n_hyb, output_path=None, show=True):
    plt.figure(figsize=(8, 9))
    
    all_vals = np.concatenate([true_vals, pure_vals, phys_vals, hyb_vals])
    min_v, max_v = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (max_v - min_v) * 0.1
    lims = [min_v - margin, max_v + margin]
    
    plt.plot(lims, lims, 'k-', alpha=0.3, linewidth=2, label='Perfect Prediction (y=x)')
    plt.scatter(true_vals, pure_vals, c='red', marker='s', s=100, alpha=0.7, label=f'Pure ML ({n_pure} Pts)')
    plt.scatter(true_vals, phys_vals, c='black', marker='^', s=100, alpha=0.7, label='Physics Baseline')
    plt.scatter(true_vals, hyb_vals, c='blue', marker='o', s=100, alpha=0.8, label=f'Hybrid ({n_hyb} Pts)')
    
    mae_pure = np.mean(np.abs(pure_vals - true_vals))
    mae_phys = np.mean(np.abs(phys_vals - true_vals))
    mae_hyb = np.mean(np.abs(hyb_vals - true_vals))
    
    plt.title(f"{title} (Test Set)\nTest MAE -> Pure: {mae_pure:.3f} | Hyb: {mae_hyb:.3f}")
    plt.xlabel(f"Measured value ({unit})")
    plt.ylabel(f"Predicted Value ({unit})")
    plt.xlim(lims); plt.ylim(lims)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finish_plot(output_path, show)

def plot_test_evaluations(out, test_data, output_dir=None, show=True):
    """Interpolates continuous model predictions to match discrete test data timestamps."""
    def opts(name):
        return {"show":show, "output_path":None if output_dir is None else Path(output_dir)/f"test_{name}.png"}
    t_test = test_data["time"]
    t_model = out.mass.t 
    
    def get_preds(res):
        pure = np.interp(t_test, t_model, res.pred_pure)
        phys = np.interp(t_test, t_model, res.pred_phys)
        hyb  = np.interp(t_test, t_model, res.pred_hybrid)
        n_pure = len(res.idx_train_pure) if res.idx_train_pure is not None else len(res.idx_train)
        n_hyb = len(res.idx_train)
        return pure, phys, hyb, n_pure, n_hyb
        
    if "mass" in test_data:
        pure, phys, hyb, nP, nH = get_preds(out.mass)
        plot_parity(test_data["mass"], pure, phys, hyb, "Mass Evaluation", "g", nP, nH, **opts("mass"))
    if "thick" in test_data:
        pure, phys, hyb, nP, nH = get_preds(out.thick)
        plot_parity(test_data["thick"], pure, phys, hyb, "Thickness Evaluation", "mm", nP, nH, **opts("thick"))
    if "gv_ms" in test_data and out.cg is not None:
        pure, phys, hyb, nP, nH = get_preds(out.cg)
        plot_parity(test_data["gv_ms"], pure, phys, hyb, "Group Velocity Evaluation", "m/s", nP, nH, **opts("gv_ms"))
    if "E_GPa" in test_data and out.youngs is not None:
        pure, phys, hyb, nP, nH = get_preds(out.youngs)
        plot_parity(test_data["E_GPa"], pure, phys, hyb, "Young's Modulus Evaluation", "GPa", nP, nH, **opts("E_GPa"))


def finish_plot(output_path, show):
    fig=plt.gcf()
    if output_path is not None:
        path=Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
