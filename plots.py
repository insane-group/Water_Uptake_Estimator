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
