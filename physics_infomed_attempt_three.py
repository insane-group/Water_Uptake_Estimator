import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Global font size
plt.rcParams.update({'font.size': 14})

# ------------------------
# 1) Load data
# ------------------------
data_mt = {
    "hour":      [0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528],
    "mass":      [121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05],
    "thickness":[3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085]
}
data_el = {
    "hour":    [0,24,90,180,270,360,528],
    "E":       [6.479,4.219,3.269,2.757,2.773536366,2.185,2.187],
    "GV_meas": [2.36,1.89,1.61,1.50,1.34,1.30,np.nan]  # km/s
}

mt = pd.DataFrame(data_mt)
el = pd.DataFrame(data_el)

# ------------------------
# 2) Compute true density at 7 points via rule-of-mixtures
# ------------------------
rho_dry, rho_water = 1150.0, 1000.0
mt['mass_kg']   = mt['mass'] / 1000.0
m0              = mt.loc[mt['hour']==0, 'mass_kg'].iloc[0]
V0              = m0 / rho_dry
mt['thick_m']   = mt['thickness'] / 1000.0
mt['m_water']   = mt['mass_kg'] - m0
mt['V_water']   = mt['m_water'] / rho_water
mt['V_total']   = V0 + mt['V_water']
mt['density_true'] = mt['mass_kg'] / mt['V_total']

el['mass']        = np.interp(el['hour'], mt['hour'], mt['mass'])
el['thick_m']     = np.interp(el['hour'], mt['hour'], mt['thick_m'])
el['density_true']= np.interp(el['hour'], mt['hour'], mt['density_true'])

# physics-informed GV at hours 0,24,528; measured elsewhere
phy_hours = {0,24,528}
def compute_gv(r):
    if r['hour'] in phy_hours:
        return np.sqrt(r['E']*1e9 / r['density_true'])
    else:
        return r['GV_meas'] * 1000.0

el['GV_phys'] = el.apply(compute_gv, axis=1)

# ------------------------
# 3) LOOCV helper & models
# ------------------------
models = {
    'Ridge': Ridge(),
    'XGBoost': XGBRegressor(n_estimators=5, max_depth=3, random_state=0, verbosity=0)
}

def loocv(df, feats, targ):
    loo = LeaveOneOut()
    scores = {name:0.0 for name in models}
    for tr, te in loo.split(df):
        Xtr = df.iloc[tr][feats].values
        Xte = df.iloc[te][feats].values
        ytr = df.iloc[tr][targ].values
        yte = df.iloc[te][targ].values
        for name, mdl in models.items():
            mdl.fit(Xtr, ytr)
            scores[name] += mean_absolute_error(yte, mdl.predict(Xte))
    return {name: scores[name]/len(df) for name in scores}

# ------------------------
# 4) Density CV & prediction
# ------------------------
feats_den = {
    'time': ['hour'],
    'mass+thickness': ['mass','thick_m'],
    'mass+thick+time': ['mass','thick_m','hour']
}
cv_den = {name: loocv(el, feats, 'density_true') for name,feats in feats_den.items()}
best_den_feat  = min(cv_den, key=lambda fs: min(cv_den[fs].values()))
best_den_model = min(cv_den[best_den_feat], key=lambda m: cv_den[best_den_feat][m])
den_mod = models[best_den_model]
den_mod.fit(el[feats_den[best_den_feat]], el['density_true'])
el['density_pred'] = den_mod.predict(el[feats_den[best_den_feat]])

# ------------------------
# 5) GV CV & prediction (measured-only vs physics-informed)
# ------------------------
feats_gv = {
    'density_pred': ['density_pred'],
    'density_pred+thickness': ['density_pred','thick_m']
}
# measured-only subset (exclude physics hours)
df_meas = el[~el['hour'].isin(phy_hours)].copy()
df_meas['GV_meas_m_s'] = df_meas['GV_meas'] * 1000.0

cv_gv_meas = {name: loocv(df_meas, feats, 'GV_meas_m_s') for name,feats in feats_gv.items()}
best_meas_feat  = min(cv_gv_meas, key=lambda fs: min(cv_gv_meas[fs].values()))
best_meas_model = min(cv_gv_meas[best_meas_feat], key=lambda m: cv_gv_meas[best_meas_feat][m])
meas_mod = models[best_meas_model]
meas_mod.fit(df_meas[feats_gv[best_meas_feat]], df_meas['GV_meas_m_s'])
el['GV_pred_meas'] = meas_mod.predict(el[feats_gv[best_meas_feat]])

cv_gv_phys = {name: loocv(el, feats, 'GV_phys') for name,feats in feats_gv.items()}
best_phys_feat  = min(cv_gv_phys, key=lambda fs: min(cv_gv_phys[fs].values()))
best_phys_model = min(cv_gv_phys[best_phys_feat], key=lambda m: cv_gv_phys[best_phys_feat][m])
phys_mod = models[best_phys_model]
phys_mod.fit(el[feats_gv[best_phys_feat]], el['GV_phys'])
el['GV_pred_phys'] = phys_mod.predict(el[feats_gv[best_phys_feat]])

# ------------------------
# 6) Extended E CV & prediction
# ------------------------
feats_E2 = {
    'time': ['hour'],
    'density': ['density_pred'],
    'GV (phys)': ['GV_phys'],
    'density+GV': ['density_pred','GV_phys']
}
cv_E2 = {name: loocv(el, feats, 'E') for name,feats in feats_E2.items()}
best_E_feat  = min(cv_E2, key=lambda fs: min(cv_E2[fs].values()))
best_E_model = min(cv_E2[best_E_feat], key=lambda m: cv_E2[best_E_feat][m])
E_mod = models[best_E_model]
E_mod.fit(el[feats_E2[best_E_feat]], el['E'])
el['E_pred_full'] = E_mod.predict(el[feats_E2[best_E_feat]])

# ------------------------
# 7) Bar charts
# ------------------------
def plot_bar(cv_dict, title):
    pd.DataFrame(cv_dict).T.plot(kind='bar', figsize=(7,4))
    plt.ylabel('CV MAE'); plt.title(title)
    plt.xticks(rotation=45); plt.legend(title='Model')
    plt.tight_layout(); plt.show()

plot_bar(cv_den,     'Density CV MAE by Feature Set')
plot_bar(cv_gv_meas, 'GV CV MAE (measured only)')
plot_bar(cv_gv_phys, 'GV CV MAE (physics-informed)')
plot_bar(cv_E2,      'E CV MAE: time / density / GV / density+GV')

# ------------------------
# 8) Line plots (separate figs)
# ------------------------
# A) Density
plt.figure(figsize=(7,4))
plt.plot(el['hour'], el['density_true'], 'o-', label='Actual')
plt.plot(el['hour'], el['density_pred'], 'x--', label=f'Pred ({best_den_model})')
plt.xlabel('Hour'); plt.ylabel('Density (kg/m³)')
plt.title('Density: Actual vs Predicted'); plt.legend(); plt.tight_layout(); plt.show()

# B) GV combined
plt.figure(figsize=(8,4))
plt.plot(el['hour'], el['GV_phys']/1000, 'o-', label='Physics GV')
plt.plot(el['hour'], el['GV_pred_meas']/1000, 's--', label='ML from measured')
plt.plot(el['hour'], el['GV_pred_phys']/1000, 'x-.', label='ML from phys-informed')
plt.xlabel('Hour'); plt.ylabel('GV (km/s)')
plt.title('GV: Physics vs ML (meas & phys-informed)'); plt.legend(); plt.tight_layout(); plt.show()

# C) Young’s Modulus vs Predictions
plt.figure(figsize=(8,4))
plt.plot(el['hour'], el['E'], 'o-', label='Actual E')
plt.plot(el['hour'], el['E_pred_full'], 'x--', label=f'Pred E ({best_E_feat})')
plt.xlabel('Hour'); plt.ylabel('E (GPa)')
plt.title('Young’s Modulus: Actual vs Predicted'); plt.legend(); plt.tight_layout(); plt.show()

"""
# D) Overlay E & GV_phys
fig, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(el['hour'], el['E'], 'o-', color='tab:blue', label='E (GPa)')
ax1.set_xlabel('Hour'); ax1.set_ylabel('E (GPa)', color='tab:blue')
ax2 = ax1.twinx()
ax2.plot(el['hour'], el['GV_phys']/1000, 's--', color='tab:green', label='GV phys (km/s)')
ax2.set_ylabel('GV (km/s)', color='tab:green')
fig.legend(loc='upper right'); plt.title('E & Physics-Informed GV Over Time'); plt.tight_layout(); plt.show()

"""
# ------------------------
# 9) Print tables
# ------------------------
print("=== Density Predictions ===")
print(el[['hour','density_true','density_pred']].to_string(index=False))
print("\n=== GV Predictions (measured only) ===")
print(df_meas[['hour','GV_meas_m_s']].assign(pred=el.loc[df_meas.index,'GV_pred_meas']).to_string(index=False))
print("\n=== GV Predictions (physics-informed) ===")
print(el[['hour','GV_phys','GV_pred_phys']].to_string(index=False))
print("\n=== E Predictions ===")
print(el[['hour','E','E_pred_full']].to_string(index=False))
