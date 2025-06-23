import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

plt.rcParams.update({'font.size': 14})

# -----------------------------
# 1) Load raw data
# -----------------------------
data_mt = {
    "hour":      [0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528],
    "mass":      [121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05],
    "thickness":[3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085]
}
data_el = {
    "hour":    [0,24,90,180,270,360,528],
    "E":       [6.479,4.219,3.269,2.757,2.773536366,2.185,2.187],
    "GV_meas":[2.36,1.89,1.61,1.50,1.34,1.30,1.32]  # km/s, includes physics-informed at idx 0,1,6
}

mt = pd.DataFrame(data_mt)
el = pd.DataFrame(data_el)

# -----------------------------
# 2) Interpolate mass & thickness into elasticity data
# -----------------------------
el['mass']      = np.interp(el['hour'], mt['hour'], mt['mass'])
el['thickness'] = np.interp(el['hour'], mt['hour'], mt['thickness'])

# -----------------------------
# 3) Compute true density via rule-of-mixtures
# -----------------------------
rho_dry, rho_water = 1150.0, 1000.0
mt['mass_kg']    = mt['mass'] / 1000.0
m0               = mt.loc[mt['hour']==0, 'mass_kg'].iloc[0]
V0               = m0 / rho_dry
mt['thick_m']    = mt['thickness'] / 1000.0
mt['m_water']    = mt['mass_kg'] - m0
mt['V_water']    = mt['m_water'] / rho_water
mt['V_total']    = V0 + mt['V_water']
mt['density_true']= mt['mass_kg'] / mt['V_total']

el['thick_m']      = np.interp(el['hour'], mt['hour'], mt['thick_m'])
el['density_true'] = np.interp(el['hour'], mt['hour'], mt['density_true'])

# -----------------------------
# 4) Use provided GV_meas as physics-informed values: convert km/s to m/s
# -----------------------------
el['GV_phys'] = el['GV_meas'] * 1000.0

# -----------------------------
# 5) Train models on full data using numpy arrays
# -----------------------------
dens_full = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
dens_full.fit(el[['hour','mass','thick_m']].values, el['density_true'].values)

gv_mod = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
gv_mod.fit(el[['density_true']].values, el['GV_phys'].values)

E_mod = Ridge()
E_mod.fit(el[['GV_phys']].values, el['E'].values)

# -----------------------------
# 6) Generate training predictions
# -----------------------------
den_pred_tr = dens_full.predict(el[['hour','mass','thick_m']].values)
gv_pred_tr  = gv_mod.predict(el[['density_true']].values)
E_pred_tr   = E_mod.predict(el[['GV_phys']].values)

# -----------------------------
# 7) Compute absolute errors
# -----------------------------
y_den_true = el['density_true'].values
y_gv_true  = el['GV_phys'].values / 1000.0
y_E_true   = el['E'].values

den_err = np.abs(y_den_true - den_pred_tr)
gv_err  = np.abs(y_gv_true - gv_pred_tr/1000.0)
E_err   = np.abs(y_E_true - E_pred_tr)

# -----------------------------
# 8) Inference function
# -----------------------------
def infer(df):
    df = df.copy()
    df['thick_m']      = df['thickness'] / 1000.0
    df['density_pred'] = dens_full.predict(df[['hour','mass','thick_m']].values)
    df['gv_pred']      = gv_mod.predict(df[['density_pred']].values)
    df['E_pred_ml']    = E_mod.predict(df[['gv_pred']].values)
    df['E_phys_calc']  = (df['density_pred'] * df['gv_pred']**2) / 1e9
    return df

# Example inference on unseen times
df_new = pd.DataFrame({'hour': [12, 50, 120]})
df_new['mass']      = np.interp(df_new['hour'], mt['hour'], mt['mass'])
df_new['thickness'] = np.interp(df_new['hour'], mt['hour'], mt['thickness'])
out = infer(df_new)

x = el['hour'].values

# -----------------------------
# 9) Plot Density with ±Error
# -----------------------------
plt.figure(figsize=(7,4))
plt.fill_between(x, y_den_true - den_err, y_den_true + den_err,
                 color='gray', alpha=0.3, label='Measured ± abs(error)')
plt.plot(x, y_den_true,    'o-',  label='Measured')
plt.plot(x, den_pred_tr,   'x--', label='ML Predicted')
plt.scatter(out['hour'], out['density_pred'], s=100, c='red', marker='D', label='Inferred')
plt.xlabel('Immersion time [h]'); plt.ylabel('Density (kg/m³)')
plt.title('Density over Time')
plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------
# 10) Plot Group Velocity with ±Error
# -----------------------------
plt.figure(figsize=(7,4))
plt.fill_between(x, y_gv_true - gv_err, y_gv_true + gv_err,
                 color='gray', alpha=0.3, label='Measured ± abs(error)')
plt.plot(x, y_gv_true, 'o-', label='Provided CG (physics & experimental)')
plt.plot(x, gv_pred_tr/1000.0, 'x--', label='ML Predicted CG')
plt.scatter(out['hour'], out['gv_pred']/1000.0, s=100, c='magenta', marker='D', label='Inferred GV')
plt.xlabel('Immersion time [h]'); plt.ylabel('CG (km/s)')
plt.title('Group Velocity over Time')
plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------
# 11) Plot Young's Modulus with ±Error
# -----------------------------
plt.figure(figsize=(7,4))
plt.fill_between(x, y_E_true - E_err, y_E_true + E_err,
                 color='gray', alpha=0.3, label='Measured ± abs(error)')
plt.plot(x, y_E_true,            'o-',  label='Measured E')
plt.plot(x, E_pred_tr,           'x--', label='ML Predicted E')
plt.scatter(out['hour'], out['E_pred_ml'], s=100, c='black', marker='D', label='Inferred ML E')
plt.xlabel('Immersion time [h]'); plt.ylabel('E [GPa]')
plt.title("Young's Modulus over Time")
plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------
# 12) Print inference results
# -----------------------------
print("=== Inferred Properties ===")
print(out[['hour','density_pred','gv_pred','E_pred_ml','E_phys_calc']].to_string(index=False))
