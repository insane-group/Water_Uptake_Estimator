import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

plt.rcParams.update({'font.size': 14})

# -----------------------------
# 1) Load raw data
# -----------------------------
data_mt = {
    "hour":      list(range(0, 529, 24)),
    "mass":      [121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,
                  134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,
                  138.29,138.75,139.05],
    "thickness":[3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,
                 4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,
                 4.08,4.09,4.085]
}
data_el = {
    "hour":    [0,24,90,180,270,360,528],
    "E":       [6.479,3.819,3.269,2.757,2.773,2.185,2.187],
    "GV_meas":[2.36,1.89,1.61,1.50,1.34,1.30,1.32]  # km/s (idx 0,1,6 are physics)
}

new_hours = [9.473684210526315, 18.94736842105263, 47.368421052631575]
new_gv = [2208, 1996, 1776]  # m/s

mt = pd.DataFrame(data_mt)
el = pd.DataFrame(data_el)

# -----------------------------
# 2) Interpolate mass & thickness into elasticity data
# -----------------------------
el['mass']      = np.interp(el['hour'], mt['hour'], mt['mass'])
el['thickness']= np.interp(el['hour'], mt['hour'], mt['thickness'])

# -----------------------------
# 3) Compute “true” density via rule‐of‐mixtures
# -----------------------------
rho_dry, rho_water = 1150.0, 1000.0
mt['mass_kg']       = mt['mass']/1000.0
m0                  = mt.loc[mt['hour']==0,'mass_kg'].iloc[0]
V0                  = m0/rho_dry
mt['thick_m']       = mt['thickness']/1000.0
mt['m_water']       = mt['mass_kg'] - m0
mt['V_water']       = mt['m_water']/rho_water
mt['V_total']       = V0 + mt['V_water']
mt['density_true']  = mt['mass_kg']/mt['V_total']

el['thick_m']       = np.interp(el['hour'], mt['hour'], mt['thick_m'])
el['density_true']  = np.interp(el['hour'], mt['hour'], mt['density_true'])

# -----------------------------
# 4) Provided physics‐informed CG → convert km/s to m/s
# -----------------------------
el['GV_phys'] = el['GV_meas']*1000.0

# -----------------------------
# 5) Train thickness models
# -----------------------------
th_time = Ridge().fit(mt[['hour']].values, mt['thickness'].values)
th_full = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
th_full.fit(mt[['hour','mass']].values, mt['thickness'].values)

# -----------------------------
# 6) Train density, CG & E models
# -----------------------------
dens_full = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
dens_full.fit(el[['hour','mass','thick_m']].values, el['density_true'].values)

#gv_mod = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
gv_mod= XGBRegressor(n_estimators=10, max_depth=15, random_state=0, verbosity=0)
gv_mod.fit(el[['hour', 'thick_m', 'density_true']].values, el['GV_phys'].values)

#E_mod = Ridge().fit(el[['GV_phys']].values, el['E'].values)
#E_mod = RandomForestRegressor(n_estimators=20, max_depth=100, random_state=0)
E_mod= XGBRegressor(n_estimators=15, max_depth=20, random_state=0, verbosity=0)
E_mod.fit(el[['hour', 'thick_m', 'density_true', 'GV_phys']].values, el['E'].values)

# -----------------------------
# 7) Predictions on training set
# -----------------------------
# thickness
th_pred_time = th_time.predict(mt[['hour']].values)
th_pred_full = th_full.predict(mt[['hour','mass']].values)
# density
den_pred_tr  = dens_full.predict(el[['hour','mass','thick_m']].values)
# cg
gv_pred_tr   = gv_mod.predict(el[['hour', 'thick_m', 'density_true']].values)
# E
E_pred_tr    = E_mod.predict(el[['hour', 'thick_m', 'density_true', 'GV_phys']].values)
#E_pred_tr    = E_mod.predict(el[[ 'GV_phys']].values)

# absolute errors
den_err = np.abs(el['density_true'].values - den_pred_tr)
gv_err  = np.abs(el['GV_phys'].values/1000.0 - gv_pred_tr/1000.0)
E_err   = np.abs(el['E'].values - E_pred_tr)

# -----------------------------
# 8) Inference function
# -----------------------------
def infer(df, full_thick=True):
    df = df.copy()
    # thickness
    if full_thick:
        df['thickness_pred'] = th_full.predict(df[['hour','mass']].values)
    else:
        df['thickness_pred'] = th_time.predict(df[['hour']].values)
    df['thick_m_pred'] = df['thickness_pred']/1000.0
    # density
    df['density_pred'] = dens_full.predict(df[['hour','mass','thick_m_pred']].values)
    # cg
    df['gv_pred']      = gv_mod.predict(df[['hour', 'thick_m_pred', 'density_pred']].values)
    # E
    df['E_pred_ml']    = E_mod.predict(df[['hour', 'thick_m_pred', 'density_pred', 'gv_pred']].values)
    #df['E_pred_ml']    = E_mod.predict(df[[ 'gv_pred']].values)
    df['E_phys_calc']  = (df['density_pred'] * df['gv_pred']**2)/1e9
    return df

df_new = pd.DataFrame({'hour':[12,40]})
df_new['mass']=np.interp(df_new['hour'], mt['hour'], mt['mass'])
out = infer(df_new, full_thick=True)

# -----------------------------
# 9) Plots
# -----------------------------
hrs_mt = mt['hour'].to_numpy()
hrs_el = el['hour'].to_numpy()

# A) Thickness
plt.figure(figsize=(7,4))
plt.plot(hrs_mt, mt['thickness'], 'k-o', label='Actual')
# plt.plot(hrs_mt, th_pred_time, 'b--x', label='Ridge(time)')
plt.plot(hrs_mt, th_pred_full, 'g--s', label='RF(time+mass)')
plt.scatter(out['hour'], out['thickness_pred'], c='red', s=100, marker='D', label='Inferred')
plt.xlabel('Hour'); plt.ylabel('Thickness (mm)')
plt.title('Thickness over Time')
plt.legend(); plt.tight_layout(); plt.show()

# B) Density
plt.figure(figsize=(7,4))
plt.fill_between(hrs_el,
                 el['density_true']-den_err, el['density_true']+den_err,
                 color='gray', alpha=0.3, label='± abs(error)')
plt.plot(hrs_el, el['density_true'], 'k-o', label='Actual')
plt.plot(hrs_el, den_pred_tr, 'b--x', label='ML Pred')
plt.scatter(out['hour'], out['density_pred'], c='red', s=100, marker='D', label='Inferred')
plt.xlabel('Hour'); plt.ylabel('Density (kg/m³)')
plt.title('Density over Time')
plt.legend(); plt.tight_layout(); plt.show()

# C) Group Velocity
plt.figure(figsize=(7,4))
plt.fill_between(hrs_el,
                 el['GV_phys']/1000 - gv_err, el['GV_phys']/1000 + gv_err,
                 color='gray', alpha=0.3, label='± abs(error)')
plt.plot(hrs_el, el['GV_phys']/1000, 'k-o', label='Provided CG')
plt.plot(hrs_el, gv_pred_tr/1000, 'b--x', label='ML Pred')
plt.scatter(out['hour'], out['gv_pred']/1000, c='magenta', s=100, marker='D', label='Inferred')
plt.xlabel('Hour'); plt.ylabel('GV (km/s)')
plt.title('Group Velocity over Time')
plt.legend(); plt.tight_layout(); plt.show()

# D) Young’s Modulus
plt.figure(figsize=(7,4))
plt.fill_between(hrs_el,
                 el['E']-E_err, el['E']+E_err,
                 color='gray', alpha=0.3, label='± abs(error)')
plt.plot(hrs_el, el['E'], 'k-o', label='Actual E')
plt.plot(hrs_el, E_pred_tr, 'b--x', label='ML Pred E')
plt.scatter([12, 40], [4.86, 3.82], c='black', s=100, marker='D', label='Inferred ML E')
plt.scatter([12, 40], [5.005, 4.031], c='red', s=100, marker='s', label='Measered data for validation')
#plt.scatter(out['hour'], out['E_phys_calc'], c='gray',  s=100, marker='P', label='Inferred Phys E')
plt.xlabel('Hour'); plt.ylabel('E (GPa)')
plt.title("Young’s Modulus over Time")
plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------
# 10) Print inference
# -----------------------------
print("=== Inferred Properties ===")
print(out[['hour','thickness_pred','density_pred','gv_pred','E_pred_ml','E_phys_calc']].to_string(index=False))
