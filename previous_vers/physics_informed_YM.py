import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

plt.rcParams.update({'font.size': 14})

# Measurement hours and E
hours = np.array([0,24,90,180,270,360,528])
E_true = np.array([6.479,4.219,3.269,2.757,2.773,2.185,2.187])
GV_phys = np.array([2.36,1.89,1.61,1.50,1.34,1.30,1.32]) * 1000  # m/s

# Recompute density_true via rule-of-mixtures
data_mt = {
    "hour":      list(range(0, 529, 24)),
    "mass":      [121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,
                  134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,
                  138.29,138.75,139.05],
    "thickness":[3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,
                 4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,
                 4.08,4.09,4.085]
}
mt = pd.DataFrame(data_mt)
rho_dry, rho_water = 1150.0, 1000.0
mt['mass_kg'] = mt['mass'] / 1000
m0 = mt.loc[mt['hour']==0, 'mass_kg'].iloc[0]
V0 = m0 / rho_dry
mt['V_water'] = (mt['mass_kg'] - m0) / rho_water
mt['density_true'] = mt['mass_kg'] / (V0 + mt['V_water'])
# Get densities at measurement hours
rho_true = np.interp(hours, mt['hour'], mt['density_true'])

# Features for ridge: density_true and GV_phys
X = np.column_stack([rho_true, GV_phys])
y = E_true

# Train ridge
ridge = Ridge()
ridge.fit(X, y)
E_ridge = ridge.predict(X)

# Physics-only E = rho * gv^2 / E_pa conversion
E_phys = rho_true * (GV_phys**2) / 1e9

# Hybrid weight
weights = np.linspace(0,1,101)
maes = [np.mean(np.abs(y - (w*E_ridge + (1-w)*E_phys))) for w in weights]
best_w = weights[np.argmin(maes)]
E_hybrid = best_w * E_ridge + (1-best_w) * E_phys

# Plot results
plt.figure(figsize=(8,4))
plt.plot(hours, y, 'ko-', label='Measured E')
plt.plot(hours, E_ridge, 'bx--', label='Ridge-only')
plt.plot(hours, E_phys, 'g*-.' , label='Physics-only')
plt.plot(hours, E_hybrid, 'md', markersize=8, label=f'Hybrid (w={best_w:.2f})')
plt.xlabel('Hour')
plt.ylabel('E (GPa)')
plt.title("Young's Modulus: Ridge vs Physics vs Hybrid")
plt.legend()
plt.tight_layout()
plt.show()

# Print MAEs
print(f"Ridge MAE: {np.mean(np.abs(y - E_ridge)):.3f} GPa")
print(f"Physics MAE: {np.mean(np.abs(y - E_phys)):.3f} GPa")
print(f"Hybrid MAE: {np.mean(np.abs(y - E_hybrid)):.3f} GPa (w={best_w:.2f})")
