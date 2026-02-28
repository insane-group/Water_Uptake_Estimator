import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

# === Data Preparation ===
hours = list(range(0, 529, 24))
mass = np.array([121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,
                 134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,
                 138.29,138.75,139.05])
thickness = np.array([3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,
                      4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,
                      4.08,4.09,4.085])
rho_dry = 1.5  # g/cm³
rho_water = 1.0  # g/cm³
initial_mass = mass[0]
V0 = initial_mass / rho_dry  # cm³
delta_mass = mass - initial_mass
volume_total = V0 + delta_mass / rho_water  # cm³
density = mass / volume_total  # g/cm³

df = pd.DataFrame({'hour': hours, 'thickness': thickness, 'mass': mass, 'density': density})

# === ML Models ===
models = {
    'Ridge': Ridge(),
    'RF': RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0),
    'XGB': XGBRegressor(n_estimators=100, max_depth=4, verbosity=0, random_state=0)
}
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# === Evaluation Utility ===
def evaluate_models(X, y):
    best_mae = float('inf')
    best_model = None
    best_preds = None
    mae_dict = {}
    for name, model in models.items():
        mae = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error').mean()
        model.fit(X, y)
        preds = model.predict(X)
        mae_dict[name] = mae
        if mae < best_mae:
            best_mae = mae
            best_model = name
            best_preds = preds
    return best_model, best_preds, best_mae, mae_dict

# === Feature Sets and Evaluations ===
scaler_t = StandardScaler().fit(df[['hour']])
X_thick = scaler_t.transform(df[['hour']])
y_thick = df['thickness'].values
best_model_t, preds_thick, mae_t, maes_t = evaluate_models(X_thick, y_thick)

scaler_d = StandardScaler().fit(df[['hour', 'thickness']])
X_dens = scaler_d.transform(df[['hour', 'thickness']])
y_dens = df['density'].values
best_model_d, preds_dens, mae_d, maes_d = evaluate_models(X_dens, y_dens)

# === Bar Chart with Dual Y-Axis ===
import seaborn as sns
import matplotlib.ticker as ticker

mae_df = pd.DataFrame({
    'Model': list(maes_t.keys()) + list(maes_d.keys()),
    'MAE': list(maes_t.values()) + list(maes_d.values()),
    'Target': ['Thickness (mm)']*3 + ['Density (kg/m³)']*3
})

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

thickness_mae = mae_df[mae_df['Target'] == 'Thickness (mm)']
density_mae = mae_df[mae_df['Target'] == 'Density (kg/m³)']

ax1.bar(thickness_mae['Model'], thickness_mae['MAE'], color='skyblue', label='Thickness MAE (mm)', width=0.4, align='center')
ax2.bar(np.array(range(3)) + 0.4, np.array(density_mae['MAE'])*1000, color='salmon', label='Density MAE (kg/m³)', width=0.4, align='center')

ax1.set_ylabel("Thickness MAE (mm)", color='skyblue')
ax2.set_ylabel("Density MAE (kg/m³)", color='salmon')
ax1.set_xticks(np.arange(3))
ax1.set_xticklabels(thickness_mae['Model'])
ax1.set_title("Model MAE Comparison")
ax1.grid(True)

fig.tight_layout()
plt.show()

# === Predictive Plotting Based on User Inputs ===
import seaborn as sns
from sklearn.base import clone

# Retrain the best thickness model
best_model_instance_t = clone(models[best_model_t])
best_model_instance_t.fit(X_thick, y_thick)

# Predict using user input hours
user_hours = np.array(hours).reshape(-1, 1)
user_hours_scaled = scaler_t.transform(user_hours)
thickness_predicted = best_model_instance_t.predict(user_hours_scaled)

# Use predicted thickness to compute density (rule of mixtures)
mass_interp = np.interp(hours, df['hour'], df['mass'])  # real measured mass
delta_mass = mass_interp - mass_interp[0]
volume_total = V0 + delta_mass / rho_water
density_predicted = mass_interp / volume_total

# Retrain best density model using predicted thickness
df_pred = pd.DataFrame({'hour': hours, 'thickness': thickness_predicted})
X_dens_pred = scaler_d.transform(df_pred[['hour', 'thickness']])
best_model_instance_d = clone(models[best_model_d])
best_model_instance_d.fit(X_dens, y_dens)
density_predicted_ml = best_model_instance_d.predict(X_dens_pred)

import seaborn as sns
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot predicted thickness
axs[0].plot(hours, thickness, 'o-', label='True Thickness', color='black')
axs[0].plot(hours, thickness_predicted, '--', label=f'Predicted Thickness ({best_model_t})', color='blue')
axs[0].set_ylabel('Thickness (mm)')
axs[0].legend()
axs[0].grid(True)
axs[0].set_title("Thickness Prediction Based on Time")

# Plot predicted density
axs[1].plot(hours, density, 'o-', label='True Density', color='black')
axs[1].plot(hours, density_predicted_ml, '--', label=f'Predicted Density ({best_model_d})', color='green')
axs[1].set_ylabel('Density (g/cm³)')
axs[1].set_xlabel('Time (hour)')
axs[1].legend()
axs[1].grid(True)
axs[1].set_title("Density Prediction Based on Time and Predicted Thickness")

plt.tight_layout()
plt.show()
