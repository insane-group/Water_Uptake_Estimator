import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

plt.rcParams.update({'font.size': 14})

# 1) Original data
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
    "E":       [6.479,4.219,3.269,2.757,2.773,2.185,2.187],
    "GV_meas":[2.36,1.89,1.61,1.50,1.34,1.30,1.32]
}
mt = pd.DataFrame(data_mt)
el = pd.DataFrame(data_el)

# 2) Compute “true” density via rule-of-mixtures
rho_dry, rho_water = 1150.0, 1000.0
mt['mass_kg']      = mt['mass']/1000.0
m0                 = mt.loc[mt['hour']==0,'mass_kg'].iloc[0]
V0                 = m0/rho_dry
mt['thick_m']      = mt['thickness']/1000.0
mt['m_water']      = mt['mass_kg'] - m0
mt['V_water']      = mt['m_water']/rho_water
mt['V_total']      = V0 + mt['V_water']
mt['density_true'] = mt['mass_kg']/mt['V_total']

# 3) Merge into el and convert GV → m/s
el['mass']          = np.interp(el['hour'], mt['hour'], mt['mass'])
el['thickness']     = np.interp(el['hour'], mt['hour'], mt['thickness'])
el['thick_m']       = np.interp(el['hour'], mt['hour'], mt['thick_m'])
el['density_true']  = np.interp(el['hour'], mt['hour'], mt['density_true'])
el['GV_phys']       = el['GV_meas'] * 1000.0

# 4) Add the 3 extra physics‐informed CG points
new_hours = [9.473684210526315, 18.94736842105263, 47.368421052631575]
new_gv = [2208, 1996, 1776]
extra = pd.DataFrame({'hour': new_hours, 'GV_phys': new_gv})
extra['thick_m']     = np.interp(extra['hour'], mt['hour'], mt['thick_m'])
extra['density_true']= np.interp(extra['hour'], mt['hour'], mt['density_true'])
extra['E']           = np.nan  # no E

el_extended = pd.concat([el, extra], ignore_index=True)

# LOOCV helper
def loocv_mae(X, y, model):
    loo, errs = LeaveOneOut(), []
    for tr, te in loo.split(X):
        model.fit(X[tr], y[tr])
        errs.append(mean_absolute_error(y[te], model.predict(X[te])))
    return np.mean(errs)

# CG CV
X_gv_orig = el[['hour','thick_m','density_true']].values
y_gv_orig = el['GV_phys'].values
X_gv_ext  = el_extended[['hour','thick_m','density_true']].values
y_gv_ext  = el_extended['GV_phys'].values

gv_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
mae_gv_orig = loocv_mae(X_gv_orig, y_gv_orig, gv_model)
gv_model.fit(X_gv_ext, y_gv_ext)
mae_gv_ext  = loocv_mae(X_gv_ext, y_gv_ext, gv_model)

# E CV
ext_known = el_extended.dropna(subset=['E'])
X_E_orig = el[['GV_phys']].values
y_E_orig = el['E'].values
X_E_ext  = ext_known[['GV_phys']].values
y_E_ext  = ext_known['E'].values

E_model = Ridge()
mae_E_orig = loocv_mae(X_E_orig, y_E_orig, E_model)
E_model.fit(X_E_ext, y_E_ext)
mae_E_ext  = loocv_mae(X_E_ext, y_E_ext, E_model)

# Analytical E
E_phys_train = el['density_true'] * (el['GV_phys']**2) / 1e9
mae_E_phys   = mean_absolute_error(el['E'], E_phys_train)

# Print results
print(f"CG CV-MAE (orig 7 pts): {mae_gv_orig:.3f} m/s")
print(f"CG CV-MAE (with 10 pts): {mae_gv_ext:.3f} m/s")
print(f"Young's E CV-MAE (orig 7): {mae_E_orig:.3f} GPa")
print(f"Young's E CV-MAE (with 10): {mae_E_ext:.3f} GPa")
print(f"Analytical E CV-MAE: {mae_E_phys:.3f} GPa")

# 9) Plots
hrs = el['hour'].values

plt.figure(figsize=(7,4))
plt.plot(hrs, y_gv_orig/1000,      'o-', label='Actual CG')
plt.plot(hrs, gv_model.predict(X_gv_orig)/1000,'x--', label='ML CG Pred')
plt.fill_between(hrs,
    (y_gv_orig - mae_gv_ext)/1000,
    (y_gv_orig + mae_gv_ext)/1000,
    color='gray', alpha=0.3,
    label='± MAE')
plt.title("Group Velocity: Orig vs Extended")
plt.ylabel("CG (km/s)")
plt.xlabel("Hour")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.plot(hrs, el['E'],       'o-', label='Actual E')
plt.plot(hrs, E_model.predict(X_E_orig), 'x--', label='ML Predicted E')
plt.scatter([12, 40], [4.86, 3.82], c='black', s=100, marker='D', label='ML Inferred E')
plt.scatter([12, 40], [5.005, 4.031], c='red', s=100, marker='s', label='Measured data for validation')
#plt.plot(hrs, E_phys_train,  's-.', label='Analytical E')
plt.fill_between(hrs,
    el['E']-mae_E_ext,
    el['E']+mae_E_ext,
    color='gray', alpha=0.3,
    label='± MAE')
plt.title("Young's Modulus: ML vs Analytical")
plt.ylabel("E (GPa)")
plt.xlabel("Immertion time [h]")
plt.legend()
plt.tight_layout()
plt.show()
