import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

# 1) Load raw data
data_mt = {
    "hour":      [0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528],
    "mass":      [121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05],
    "thickness":[3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085]
}
data_el = {
    "hour":    [0,24,90,180,270,360,528],
    "E":       [6.479,4.219,3.269,2.757,2.773,2.185,2.187],
    "GV_meas":[2.34, np.nan,1.61,1.50,1.34,1.31,1.3]  # km/s, nan where missing
}

mt = pd.DataFrame(data_mt)
el = pd.DataFrame(data_el)

# 2) Compute density via rule-of-mixtures
rho_dry, rho_water = 1150.0, 1000.0
mt['mass_kg'] = mt['mass']/1000.0
m0 = mt.loc[mt['hour']==0,'mass_kg'].iloc[0]
V0 = m0/rho_dry
mt['thick_m'] = mt['thickness']/1000.0
mt['m_water'] = mt['mass_kg'] - m0
mt['V_water'] = mt['m_water']/rho_water
mt['V_total'] = V0 + mt['V_water']
mt['density_true'] = mt['mass_kg']/mt['V_total']

# 3) Merge into elasticity DataFrame
el['density'] = np.interp(el['hour'], mt['hour'], mt['density_true'])
el['E_pa']    = el['E']*1e9

# 4) Physics‐informed GV fill (m/s)
el['GV_phys'] = el.apply(
    lambda r: np.sqrt(r['E_pa']/r['density']) if np.isnan(r['GV_meas'])
              else r['GV_meas']*1000.0,
    axis=1
)

# 5) Plot Mass & Thickness on dual axes
fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = ax1.twinx()
ax1.plot(mt['hour'], mt['mass'], 'o-', color='tab:blue', label='Mass (g)')
ax2.plot(mt['hour'], mt['thickness'], 's--', color='tab:orange', label='Thickness (mm)')
ax1.set_xlabel('Immersion time [h]')
ax1.set_ylabel('Mass [g]', color='tab:blue')
ax2.set_ylabel('Thickness [mm]', color='tab:orange')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.title('Mass and Thickness over Time')
plt.legend(h1+h2, l1+l2, loc='upper left')
plt.tight_layout()
plt.show()

# 6) CG bar chart (km/s), highlight filled at 24 h
hours = el['hour']
gv_plot = el['GV_phys']/1000.0
colors = ['tab:blue' if not np.isnan(g) and h!=24 else 'tab:red'
          for g,h in zip(el['GV_meas'], el['hour'])]
plt.figure(figsize=(6,4))
plt.bar(hours, gv_plot, color=colors, width=20)
plt.xlabel('Immersion time [h]')
plt.ylabel('Group Velocity [km/s]')
plt.title('Group Velocity: measured (blue) vs filled (red)')
plt.xticks(hours)
plt.tight_layout()
plt.show()

# 7) Young’s modulus line plot
plt.figure(figsize=(6,4))
plt.plot(el['hour'], el['E'], 'o-', color='tab:green')
plt.xlabel('Immersion time [h]')
plt.ylabel("Young's Modulus [GPa]")
plt.title("Young's Modulus over Time")
plt.tight_layout()
plt.show()

# 8) Density line plot
plt.figure(figsize=(6,4))
plt.plot(mt['hour'], mt['density_true'], 'o-', color='tab:purple')
plt.xlabel('Immersion time [h]')
plt.ylabel('Density [kg/m³]')
plt.title('Density via Rule-of-Mixtures')
plt.tight_layout()
plt.show()

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

# Data
hours = ["0H", "90H", "180H", "270H", "360H", "528H"]
exp = [2.3, 1.61, 1.5, 1.34, 1.31, 1.3]
sim = [2.4, 1.59, 1.48, 1.36, 1.33, 1.32]

x = np.arange(len(hours))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, exp, width, label='Measurement', color='tab:blue')
plt.bar(x + width/2, sim, width, label='Numerical Simulation', color='tab:orange')
plt.xticks(x, hours)
plt.xlabel('Immersion Time')
plt.ylabel('Group Velocity (km/s)')
plt.title('Experiment vs Numerical Simulation CG')
plt.legend()
plt.tight_layout()
plt.show()
"""