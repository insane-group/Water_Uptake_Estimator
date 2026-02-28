import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# === Data Preparation ===
# Concatenated PF6 data (full)
hours = np.array(list(range(0, 529, 24)))
mass = np.array([121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,
                 134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,
                 138.29,138.75,139.05])
thickness = np.array([3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,
                      4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,
                      4.08,4.09,4.085])

# Constants for density calculation
rho_dry = 1.5
rho_water = 1.0
initial_mass = mass[0]
V0 = initial_mass / rho_dry
delta_mass = mass - initial_mass
volume_total = V0 + delta_mass / rho_water
density = mass / volume_total

# DataFrame
df = pd.DataFrame({'hour': hours, 'thickness': thickness, 'mass': mass, 'density': density})
df['roc_mass'] = (df['mass'] - df['mass'].iloc[0]) / df['hour'].replace(0, 1)
df['roc_thickness'] = (df['thickness'] - df['thickness'].iloc[0]) / df['hour'].replace(0, 1)

# Models and KFold
models = {
    'Ridge': Ridge(),
    'RF': RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0),
    'XGB': XGBRegressor(n_estimators=100, max_depth=4, verbosity=0, random_state=0)
}
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Utility function
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

# Features and targets
y_thickness = df['thickness'].values
y_density = df['density'].values

# Thickness: (1) hour only, (2) hour + roc_mass
X_t1 = StandardScaler().fit_transform(df[['hour']])
X_t2 = StandardScaler().fit_transform(df[['hour', 'roc_mass']])
best_model_t1, preds_t1, mae_t1, maes_t1 = evaluate_models(X_t1, y_thickness)
best_model_t2, preds_t2, mae_t2, maes_t2 = evaluate_models(X_t2, y_thickness)

# Density: (1) hour, (2) hour + roc_thickness, (3) hour + roc_mass + roc_thickness
X_d1 = StandardScaler().fit_transform(df[['hour']])
X_d2 = StandardScaler().fit_transform(df[['hour', 'roc_thickness']])
X_d3 = StandardScaler().fit_transform(df[['hour', 'roc_mass', 'roc_thickness']])
best_model_d1, preds_d1, mae_d1, maes_d1 = evaluate_models(X_d1, y_density)
best_model_d2, preds_d2, mae_d2, maes_d2 = evaluate_models(X_d2, y_density)
best_model_d3, preds_d3, mae_d3, maes_d3 = evaluate_models(X_d3, y_density)

# Prepare bar chart data
mae_df = pd.DataFrame({
    'Model': list(maes_t1.keys()) * 2 + list(maes_d1.keys()) * 3,
    'MAE': list(maes_t1.values()) + list(maes_t2.values()) +
           list(maes_d1.values()) + list(maes_d2.values()) + list(maes_d3.values()),
    'Target': ['Thickness (h)']*3 + ['Thickness (h + roc_m)']*3 +
              ['Density (h)']*3 + ['Density (h + roc_t)']*3 + ['Density (h + roc_m + roc_t)']*3
})

import matplotlib.pyplot as plt

# Plot bar chart with dual y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

targets_thickness = mae_df[mae_df['Target'].str.contains("Thickness")]
targets_density = mae_df[mae_df['Target'].str.contains("Density")]

xticks = np.arange(len(targets_thickness['Target'].unique()))
width = 0.35

ax1.bar(xticks - width/2, targets_thickness.groupby('Target')['MAE'].mean(), width, label='Thickness MAE (mm)', color='skyblue')
ax2.bar(xticks + width/2, targets_density.groupby('Target')['MAE'].mean().values[:len(xticks)], width, label='Density MAE (g/cm³)', color='salmon')

ax1.set_ylabel('Thickness MAE (mm)')
ax2.set_ylabel('Density MAE (g/cm³)')
ax1.set_xticks(xticks)
ax1.set_xticklabels(targets_thickness['Target'].unique(), rotation=30)
ax1.set_title('Model MAE Comparison with Dual Y-Axis')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.show()

# Best Thickness Plot
plt.figure(figsize=(10, 5))
plt.plot(df['hour'], y_thick, 'o-', label='True Thickness', color='black')
if mae_t1 < mae_t2:
    plt.plot(df['hour'], preds_t1, '--', label=f'{best_model_t1} (hour)', color='blue')
else:
    plt.plot(df['hour'], preds_t2, '--', label=f'{best_model_t2} (hour + mass)', color='blue')
plt.xlabel("Time (hour)")
plt.ylabel("Thickness (mm)")
plt.title("Best Thickness Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Best Density Plot
plt.figure(figsize=(10, 5))
plt.plot(df['hour'], y_dens, 'o-', label='True Density', color='black')
min_mae_d = min(mae_d1, mae_d2, mae_d3)
if min_mae_d == mae_d1:
    plt.plot(df['hour'], preds_d1, '--', label=f'{best_model_d1} (hour)', color='green')
elif min_mae_d == mae_d2:
    plt.plot(df['hour'], preds_d2, '--', label=f'{best_model_d2} (hour + thickness)', color='green')
else:
    plt.plot(df['hour'], preds_d3, '--', label=f'{best_model_d3} (hour + mass + thickness)', color='green')
plt.xlabel("Time (hour)")
plt.ylabel("Density (g/cm³)")
plt.title("Best Density Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

