import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

plt.rcParams.update({'font.size': 14})

# 1) Prepare base data
data_mt = {
    "hour": list(range(0, 529, 24)),
    "mass": [121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05],
    "thickness": [3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085]
}
data_el = {
    "hour": [0,24,90,180,270,360,528],
    "GV_meas": [2.36,1.89,1.61,1.50,1.34,1.30,1.32]  # km/s
}

mt = pd.DataFrame(data_mt)
el = pd.DataFrame(data_el)

# 2) Compute density_true via rule-of-mixtures
rho_dry, rho_water = 1150.0, 1000.0
mt['mass_kg'] = mt['mass'] / 1000.0
m0_kg = mt.loc[mt['hour']==0, 'mass_kg'].iloc[0]
V0 = m0_kg / rho_dry
mt['thick_m'] = mt['thickness'] / 1000.0
mt['m_water'] = mt['mass_kg'] - m0_kg
mt['V_water'] = mt['m_water'] / rho_water
mt['V_total'] = V0 + mt['V_water']
mt['density_true'] = mt['mass_kg'] / mt['V_total']

# 3) Merge into el and compute GV_phys in m/s
el['thick_m'] = np.interp(el['hour'], mt['hour'], mt['thick_m'])
el['density_true'] = np.interp(el['hour'], mt['hour'], mt['density_true'])
el['GV_phys'] = el['GV_meas'] * 1000.0  # m/s

features = ['density_true', 'thick_m']
# 4-pt: exclude 0,24,528
phy_h = {0,24,528}
df4 = el[~el['hour'].isin(phy_h)].copy()
df4['target'] = df4['GV_phys']
# 7-pt: original
df7 = el.copy()
df7['target'] = df7['GV_phys']
# 10-pt: add 3 new physics-informed hours
new_hours = [9.473684210526315, 18.94736842105263, 47.368421052631575]
new_gv = [2208, 1996, 1776]  # m/s
new_df = pd.DataFrame({
    'hour': new_hours,
    'density_true': np.interp(new_hours, mt['hour'], mt['density_true']),
    'thick_m': np.interp(new_hours, mt['hour'], mt['thick_m']),
    'target': new_gv
})
df10 = pd.concat([df7[['hour','density_true','thick_m','target']], new_df], ignore_index=True)

# Train and evaluate
def rf_mae(df_train):
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
    rf.fit(df_train[features], df_train['target'])
    pred = rf.predict(df7[features])
    return mean_absolute_error(df7['GV_phys'], pred), pred

mae4, pred4 = rf_mae(df4)
mae7, pred7 = rf_mae(df7)
mae10, pred10 = rf_mae(df10)

hours7 = df7['hour']

# 1) Bar chart of MAE
plt.figure(figsize=(6,4))
plt.bar(['4-pt','7-pt','10-pt'], [mae4, mae7, mae10], color=['tab:blue','tab:green','tab:red'])
plt.ylabel('MAE (m/s)')
plt.title('CG Prediction MAE for Different Training Sets')
plt.tight_layout()
plt.show()

# 2) Line plot comparison
plt.figure(figsize=(8,4))
plt.plot(hours7, df7['GV_phys']/1000, 'o-', label='Actual CG', color='black')
plt.plot(hours7, pred4/1000, 's--', label=f'RF4-pt (MAE={mae4:.1f})', color='tab:blue')
plt.plot(hours7, pred7/1000, 'x--', label=f'RF7-pt (MAE={mae7:.1f})', color='tab:green')
plt.plot(hours7, pred10/1000, 'D--', label=f'RF10-pt (MAE={mae10:.1f})', color='tab:red')
plt.xlabel('Hour')
plt.ylabel('GV (km/s)')
plt.title('CG Predictions: 4‑pt vs 7‑pt vs 10‑pt RF')
plt.legend()
plt.tight_layout()
plt.show()
