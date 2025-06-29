import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

plt.rcParams.update({'font.size': 14})

# 1) Load data
data_mt = {
    "hour":      list(range(0, 529, 24)),
    "mass":      [121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05],
    "thickness":[3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085]
}
data_el = {
    "hour":    [0,24,90,180,270,360,528],
    "E":       [6.479,3.819,3.269,2.757,2.773,2.185,2.187],
    "GV_meas": [2.36,1.89,1.61,1.50,1.34,1.30,1.32]  # km/s
}

mt = pd.DataFrame(data_mt)
el = pd.DataFrame(data_el)

# 2) Compute true density via rule-of-mixtures
rho_dry, rho_water = 1150.0, 1000.0
mt['mass_kg']      = mt['mass'] / 1000.0
m0                 = mt.loc[mt['hour']==0, 'mass_kg'].iloc[0]
V0                 = m0 / rho_dry
mt['thick_m']      = mt['thickness'] / 1000.0
mt['m_water']      = mt['mass_kg'] - m0
mt['V_water']      = mt['m_water'] / rho_water
mt['V_total']      = V0 + mt['V_water']
mt['density_true'] = mt['mass_kg'] / mt['V_total']

# interpolate into el
el['thick_m']      = np.interp(el['hour'], mt['hour'], mt['thick_m'])
el['density_true'] = np.interp(el['hour'], mt['hour'], mt['density_true'])
el['GV_phys']      = el['GV_meas'] * 1000.0  # m/s

# models dictionary
models = {
    'Ridge':   Ridge(),
    'RF':      RandomForestRegressor(n_estimators=50, max_depth=25, random_state=0),
    'XGBoost': XGBRegressor(n_estimators=5, max_depth=15, random_state=0, verbosity=0)
}

# LOOCV function
def loocv_mae(df, features, target, model):
    loo = LeaveOneOut()
    errs = []
    for tr, te in loo.split(df):
        Xtr, Xte = df.iloc[tr][features].values, df.iloc[te][features].values
        ytr, yte = df.iloc[tr][target].values, df.iloc[te][target].values
        model.fit(Xtr, ytr)
        errs.append(mean_absolute_error(yte, model.predict(Xte)))
    return np.mean(errs)

# — Density CV
den_feats = {
    'time': ['hour'],
    'time,thickness': ['hour','thick_m'],
    'mass,thick,time': ['mass','thick_m','hour']
}
cv_den = {
    feat: {name: loocv_mae(mt, cols, 'density_true', mdl) for name,mdl in models.items()}
    for feat,cols in den_feats.items()
}
# bar chart density
pd.DataFrame(cv_den).T.plot.bar(figsize=(7,4))
plt.ylabel('MAE (kg/m³)')
plt.title('Density CV MAE by Feature Set')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# final density model: RF on best RF scenario
best_den_feat = min(cv_den, key=lambda f: cv_den[f]['RF'])
rf_den = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
rf_den.fit(mt[den_feats[best_den_feat]].values, mt['density_true'].values)
mt['density_pred'] = rf_den.predict(mt[den_feats[best_den_feat]].values)
# density line plot
plt.figure(figsize=(7,4))
plt.plot(mt['hour'], mt['density_true'], 'o-', label='Actual')
#plt.plot(mt['hour'], mt['density_pred'], 'x--', label=f'RF Pred ({best_den_feat})')
plt.plot(mt['hour'], mt['density_pred'], 'x--', label=f'ML Prediction')
plt.xlabel('Hour')
plt.ylabel('Density (kg/m³)')
plt.title('True vs Predicted Density')
plt.legend()
plt.tight_layout()
plt.show()

# — CG CV
phy_hours = {0,24,528}

#df4 = el[~el['hour'].isin(phy_hours)].copy()
#df4['GV_meas_m_s'] = df4['GV_phys']
#gv_feats = ['density_true','thick_m']
#gv_feats1 = ['density_true']
#cv4 = {name: loocv_mae(df4, gv_feats, 'GV_meas_m_s', mdl) for name,mdl in models.items()}
#cv7 = {name: loocv_mae(el, gv_feats, 'GV_phys', mdl) for name,mdl in models.items()}
#df_cv = pd.DataFrame([cv4, cv7], index=['4-pt','7-pt'])

df4 = el[~el['hour'].isin(phy_hours)].copy()
df4['GV_meas_m_s'] = df4['GV_phys']
gv_feats1 = ['density_true']
gv_feats2 = ['density_true','thick_m']
cv4 = {name: loocv_mae(df4, gv_feats1, 'GV_meas_m_s', mdl) for name,mdl in models.items()}
cv7 = {name: loocv_mae(df4, gv_feats2, 'GV_meas_m_s', mdl) for name,mdl in models.items()}
df_cv = pd.DataFrame([cv4, cv7], index=['density','density, thickness'])
df_cv.plot.bar(figsize=(7,4))
plt.ylabel('MAE (m/s)')
plt.title('CG CV MAE: 4-pt vs 7-pt')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# CG final predictions
rf4 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
rf4.fit(df4[gv_feats2].values, df4['GV_meas_m_s'].values)
el['GV_pred4_all'] = rf4.predict(el[gv_feats2].values)
rf7 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
#rf7 = Ridge(alpha=0.5)
rf7.fit(el[gv_feats2].values, el['GV_phys'].values)
el['GV_pred7_all'] = rf7.predict(el[gv_feats2].values)
# CG line plot
plt.figure(figsize=(7,4))
plt.plot(el['hour'], el['GV_phys']/1000, 'o-', label='Actual CG', color='black')
plt.plot(el['hour'], el['GV_pred7_all']/1000, 's--', label='ML Prediction', color='tab:green')
#plt.plot(el['hour'], el['GV_pred4_all']/1000, 'x--', label='RF 4-pt Pred', color='tab:blue')
plt.xlabel('Hour')
plt.ylabel('CG (km/s)')
plt.title('CG Predictions: 4-pt vs 7-pt on All 7 Points')
plt.legend()
plt.tight_layout()
plt.show()

# — YM CV
E_feats = {
    'time': ['hour'],
    'density': ['density_true'],
    'CG': ['GV_phys'],
    'density,CG': ['density_true','GV_phys']
}
cv_E = {
    feat: {name: loocv_mae(el, cols, 'E', mdl) for name,mdl in models.items()}
    for feat,cols in E_feats.items()
}
pd.DataFrame(cv_E).T.plot.bar(figsize=(7,4))
plt.ylabel('MAE (GPa)')
plt.title("Young's Modulus CV MAE by Feature Set")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# YM final with RF on best scenario
best_E_feat = min(cv_E, key=lambda f: cv_E[f]['RF'])
rf_E = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
rf_E.fit(el[E_feats[best_E_feat]].values, el['E'].values)
#rf_E = Ridge(alpha=0.5)
rf_E.fit(el[E_feats[best_E_feat]].values, el['E'].values)
el['E_pred'] = rf_E.predict(el[E_feats[best_E_feat]].values)
# YM line plot
plt.figure(figsize=(7,4))
plt.plot(el['hour'], el['E'], 'o-', label='Actual E')
#plt.plot(el['hour'], el['E_pred'], 'x--', label=f'RF Pred ({best_E_feat})')
plt.plot(el['hour'], el['E_pred'], 'x--', label=f'ML Prediction')
plt.xlabel('Hour')
plt.ylabel('E (GPa)')
plt.title("Young's Modulus: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# — Thickness CV & prediction
# Feature sets: time only, and time+mass
th_feats = {
    'time':       ['hour'],
    #'time, mass': ['hour','mass']
}

# Compute LOOCV MAE for each regressor
cv_th = {
    feat: {
        name: loocv_mae(mt, cols, 'thickness', mdl)
        for name, mdl in models.items()
    }
    for feat, cols in th_feats.items()
}

# Bar chart of Thickness CV-MAE
pd.DataFrame(cv_th).T.plot.bar(figsize=(7,4))
plt.ylabel('MAE (mm)')
plt.title('Thickness CV MAE by Feature Set')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Train best RF on full data and predict
best_th_feat = min(cv_th, key=lambda f: cv_th[f]['RF'])
rf_th = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
rf_th.fit(mt[th_feats[best_th_feat]].values, mt['thickness'].values)
mt['thickness_pred'] = rf_th.predict(mt[th_feats[best_th_feat]].values)

# Line plot: Actual vs Predicted Thickness
plt.figure(figsize=(7,4))
plt.plot(mt['hour'], mt['thickness'],    'o-',  label='Actual Thickness', color='tab:orange')
#plt.plot(mt['hour'], mt['thickness_pred'],'x--', label=f'RF Pred ({best_th_feat})', color='tab:blue')
plt.plot(mt['hour'], mt['thickness_pred'],'x--', label='ML Prediction', color='tab:blue')
plt.xlabel('Hour')
plt.ylabel('Thickness (mm)')
plt.title('Thickness: Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.show()

# Summary printout
print("\n=== Thickness CV MAE ===")
print(pd.DataFrame(cv_th).T.round(3))
print(f"Best thickness features for RF: {best_th_feat}")

# Summary tables
print("\n=== Density CV MAE ===")
print(pd.DataFrame(cv_den).T.round(3))
print(f"\nBest density features for RF: {best_den_feat}")
print("\n=== CG CV MAE ===")
print(df_cv.round(3))
print("\n=== Young's Modulus CV MAE ===")
print(pd.DataFrame(cv_E).T.round(3))
print(f"\nBest E features for RF: {best_E_feat}")

print("\n=== CG Predictions on All 7 Points (km/s) ===")
print(pd.DataFrame({
    'hour': el['hour'],
    'actual': np.round(el['GV_phys']/1000,3),
    'pred7': np.round(el['GV_pred7_all']/1000,3),
    'pred4': np.round(el['GV_pred4_all']/1000,3)
}).to_string(index=False))

# -----------------------------
# 12) Predict Young’s Modulus at new times (12 h, 40 h)
# -----------------------------
df_new = pd.DataFrame({'hour': [12, 40]})

# interpolate the “true” density at those times
df_new['density_true'] = np.interp(df_new['hour'], mt['hour'], mt['density_true'])

# if your best_E_feat includes GV_phys, you’ll need to interpolate or predict it:
# here we’ll linearly interpolate the provided physics‐informed GV (in m/s)
df_new['GV_phys'] = np.interp(df_new['hour'], el['hour'], el['GV_phys'])

# now pull out exactly the columns your model expects:
E_features = E_feats[best_E_feat]   # e.g. ['density_true'] or ['density_true','GV_phys'] etc.

# predict
df_new['E_pred'] = rf_E.predict(df_new[E_features].values)

print("\n=== Young’s Modulus Predictions at 12 h and 40 h ===")
print(df_new[['hour'] + E_features + ['E_pred']].to_string(index=False))
