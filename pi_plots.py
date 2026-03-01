# pi_plots.py
import numpy as np
import matplotlib.pyplot as plt

# --- INCREASE FONT SIZES FOR PUBLICATION ---
plt.rcParams.update({
    'font.size': 14,             # Global font size
    'axes.titlesize': 16,        # Title size
    'axes.labelsize': 16,        # X and Y label size
    'xtick.labelsize': 14,       # X tick labels
    'ytick.labelsize': 14,       # Y tick labels
    'legend.fontsize': 13,       # Legend font size
    'lines.linewidth': 2.5,      # Thicker lines
    'lines.markersize': 8        # Larger markers
})

def plot_prop(t, res, title, ylabel, exp_data=None, train_t=None, phys_label="Physics Baseline", exp_std=None):
    if res is None: return
    plt.figure(figsize=(7, 6))
    
    # 1. Experimental Truth (Full Dataset available for this property)
    if exp_data is not None:
        x_axis = train_t if train_t is not None else t
        
        # Plot the Shaded STD Margin if provided
        if exp_std is not None:
            plt.fill_between(x_axis, exp_data - exp_std, exp_data + exp_std, 
                             color='gray', alpha=0.15, label="±1 STD (Exp. Margin)", zorder=0)
            
        plt.plot(x_axis, exp_data, 'o-', color='gray', alpha=0.4, label="Experimental Truth", zorder=1)

    # 2. Predictions (plotted against full time t)
    plt.plot(t, res.pred_pure, 'r--', label="Pure ML (Exp Only)", alpha=0.8, zorder=2)
    plt.plot(t, res.pred_phys, 'k:', label=phys_label, linewidth=2.0, zorder=3)
    plt.plot(t, res.pred_hybrid, 'b-', label="Hybrid (Proposed)", linewidth=2.5, zorder=4)
    
    # 3. Training Points (Highlight subset used for training)
    if train_t is not None and exp_data is not None:
        # Filter to show only indices marked as training
        valid_idx = res.idx_train[res.idx_train < len(train_t)]
        if len(valid_idx) > 0:
            plt.scatter(train_t[valid_idx], exp_data[valid_idx], 
                        c='white', edgecolors='k', s=100, linewidth=1.5, zorder=5, label="Train Data")
        
    # 4. PI Points
    if res.pi_hours is not None:
        plt.scatter(res.pi_hours, res.pi_values, marker='^', c='orange', s=100, zorder=5, label="PI Points")
        
    plt.title(f"{title}\nPure ML MAE: {res.mae_pure:.4f} | Hybrid MAE: {res.mae_val:.4f}")
    plt.xlabel("Immersion Time (H)")
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_all(out, case_data, cg_data, E_data, prefix="Case", exp_stds=None):
    t = out.mass.t
    exp_stds = exp_stds or {}
    
    plot_prop(t, out.mass, "Mass", "Mass (g)", 
              exp_data=case_data["mass"], train_t=t, phys_label="Physics (Fickian)", exp_std=exp_stds.get("mass"))
              
    plot_prop(t, out.thick, f"Thickness | Alpha={out.thick.alpha:.4f}", "Thickness (mm)", 
              exp_data=case_data["thick"], train_t=t, phys_label="Physics (Linear Alpha)", exp_std=exp_stds.get("thick"))
              
    plot_prop(t, out.dens, "Density", "Density (kg/m$^3$)", 
              exp_data=out.rho_exp, train_t=t, phys_label="Physics (Mass Fraction)", exp_std=exp_stds.get("dens"))
    
    if out.cg: 
        plot_prop(t, out.cg, "Group Velocity", "GV (m/s)", 
                  exp_data=cg_data["gv_ms"], train_t=cg_data["hour"], phys_label="Physics (Comp. Model)")
                  
    if out.youngs: 
        plot_prop(t, out.youngs, "Young's Modulus", "E (GPa)", 
                  exp_data=E_data["E_GPa"], train_t=E_data["hour"], phys_label="Physics (Wave Eq)")