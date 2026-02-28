import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

# === Data Preparation ===
# Concatenated PF6 data (full)
hours = list(range(0, 529, 24))
mass = np.array([121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,
                 134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,
                 138.29,138.75,139.05])
thickness = np.array([3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,
                      4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,
                      4.08,4.09,4.085])
rho_dry = 1.5
rho_water = 1.0
initial_mass = mass[0]
V0 = initial_mass / rho_dry
delta_mass = mass - initial_mass
volume_total = V0 + delta_mass / rho_water
density = mass / volume_total

df_pf6 = pd.DataFrame({'hour': hours, 'thickness': thickness, 'mass': mass, 'density': density})

# === Models ===
models = {
    'Ridge': Ridge(),
    'RF': RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0),
    'XGB': XGBRegressor(n_estimators=100, max_depth=4, verbosity=0, random_state=0)
}

kf = KFold(n_splits=5, shuffle=True, random_state=0)

# === Utility function for model evaluation ===
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

# === Thickness Prediction: (1) hour only, (2) hour + mass ===
X_thick1 = StandardScaler().fit_transform(df_pf6[['hour']])
X_thick2 = StandardScaler().fit_transform(df_pf6[['hour', 'mass']])
y_thick = df_pf6['thickness'].values

best_model_t1, preds_t1, mae_t1, maes_t1 = evaluate_models(X_thick1, y_thick)
best_model_t2, preds_t2, mae_t2, maes_t2 = evaluate_models(X_thick2, y_thick)

# === Density Prediction: (1) hour, (2) hour + thickness, (3) hour + mass + thickness ===
X_dens1 = StandardScaler().fit_transform(df_pf6[['hour']])
X_dens2 = StandardScaler().fit_transform(df_pf6[['hour', 'thickness']])
X_dens3 = StandardScaler().fit_transform(df_pf6[['hour', 'mass', 'thickness']])
y_dens = df_pf6['density'].values

best_model_d1, preds_d1, mae_d1, maes_d1 = evaluate_models(X_dens1, y_dens)
best_model_d2, preds_d2, mae_d2, maes_d2 = evaluate_models(X_dens2, y_dens)
best_model_d3, preds_d3, mae_d3, maes_d3 = evaluate_models(X_dens3, y_dens)

# === Plotting Bar Charts of MAEs ===
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

mae_df = pd.DataFrame({
    'Model': list(maes_t1.keys()) * 2 + list(maes_d1.keys()) * 3,
    'MAE': list(maes_t1.values()) + list(maes_t2.values()) +
           list(maes_d1.values()) + list(maes_d2.values()) + list(maes_d3.values()),
    'Target': ['Thickness (h)']*3 + ['Thickness (h + m)']*3 +
              ['Density (h)']*3 + ['Density (h + t)']*3 + ['Density (h + m + t)']*3
})

plt.figure(figsize=(12, 6))
sns.barplot(data=mae_df, x='Target', y='MAE', hue='Model')
plt.title('Model MAE Comparison for Thickness and Density Prediction')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# === Plot Best Thickness Prediction ===
plt.figure(figsize=(10, 5))
plt.plot(df_pf6['hour'], y_thick, 'o-', label='True Thickness', color='black')
if mae_t1 < mae_t2:
    plt.plot(df_pf6['hour'], preds_t1, '--', label=f'{best_model_t1} Prediction (hour)', color='blue')
else:
    plt.plot(df_pf6['hour'], preds_t2, '--', label=f'{best_model_t2} Prediction (hour + mass)', color='blue')
plt.xlabel("Time (hour)")
plt.ylabel("Thickness (mm)")
plt.title("Best Thickness Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot Best Density Prediction ===
plt.figure(figsize=(10, 5))
plt.plot(df_pf6['hour'], y_dens, 'o-', label='True Density', color='black')
min_mae_d = min(mae_d1, mae_d2, mae_d3)
if min_mae_d == mae_d1:
    plt.plot(df_pf6['hour'], preds_d1, '--', label=f'{best_model_d1} Prediction (hour)', color='green')
elif min_mae_d == mae_d2:
    plt.plot(df_pf6['hour'], preds_d2, '--', label=f'{best_model_d2} Prediction (hour + thickness)', color='green')
else:
    plt.plot(df_pf6['hour'], preds_d3, '--', label=f'{best_model_d3} Prediction (hour + mass + thickness)', color='green')
plt.xlabel("Time (hour)")
plt.ylabel("Density (g/cm³)")
plt.title("Best Density Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
