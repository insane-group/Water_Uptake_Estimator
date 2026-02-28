# Water Uptake Estimator

This project combines **physics-driven** formulas with **data-driven** machine-learning models to predict the evolving material properties of a specimen immersed in water over time.  

---

## 1. Data & Preprocessing

- **Manual measurements**  
  - Mass (g), thickness (mm) recorded every 24 h for 0–528 h  
  - Young’s modulus E (GPa) and group velocity GV (km/s) measured at 0, 24, 90, 180, 270, 360, 528 h  

- **Rule-of-Mixtures Density**  
  \[  
    \rho_{\rm rm}(t)  
    = \frac{m(t)/1000}{V_0 + \bigl(m(t)/1000 - m_0/1000\bigr)/\rho_{\rm water}}  
    ,\quad  
    V_0 = \frac{m_0/1000}{\rho_{\rm dry}}  
  \]  
  - ρ_dry = 1150 kg/m³, ρ_water = 1000 kg/m³  
  - Interpolated to the 7 E/GV timepoints  

- **Physics-informed GV**  
  - Provided GV_meas at 0, 24, 528 h  
  - GV_phys = √(E·1e9 / ρ_rm) at those points; elsewhere use measured GV·1000  

---

## 2. Machine-Learning Pipeline

1. **Thickness prediction**  
   - Ridge on `hour`  
   - RandomForest on `[hour, mass]`  

2. **Density prediction**  
   - RandomForest on `[hour, mass, thickness]`  
   - Compared featuresets: `hour` vs `mass+thickness` vs `hour+mass+thickness` by LOOCV MAE  

3. **Group-velocity prediction**  
   - XGBoost (or RF) on `[hour, thickness, density]`  
   - Compared data-driven (4 points) vs physics-informed (7 points)  

4. **Young’s-modulus prediction**  
   - XGBoost on `[hour, thickness, density, GV_phys]` (best)  
   - Also compared simpler featuresets: `hour`, `density`, `GV_phys`, `density+GV_phys`  

5. **Inference function**  
   - Given new immersion times, sequentially predict  
     1. thickness → 2. density → 3. GV → 4. E  
   - Outputs both ML predictions and the “physics” estimate E_phys = ρ·CG²/1e9  

---

## 3. Visualization & Evaluation

- **Bar-charts** of LOOCV MAE for every model & feature set, in the order:  
  1. Ridge  
  2. RandomForest  
  3. XGBoost  

- **Line-plots** over time (0–528 h) showing:  
  - Actual vs ML–predicted **thickness**  
  - Actual vs ML–predicted **density**, with ±|error| shading  
  - Actual vs ML–predicted **CG**, highlighting physics-informed points  
  - Actual vs ML–predicted **E**, plus E_phys  

- **“Blind” inference** at unsampled times (e.g. 12 h, 40 h), plotted as distinct markers  

---

## 4. Physics Corrections

- **Void-bound correction** to the density–CG relation (improves E_phys)  
- Compared against Reuss bound  

---

## 5. Future Directions

- **Physics-informed loss** (PINN-style) within RandomForest or Ridge  
- Incorporate **poro-elastic** expansion and hydrothermal stresses  
- Extend to **synthetic** augmentation for better generalization  

---

> **Author**: Christoforos Rekatsinas  
> **Contact**: crek@iit.demokritos.gr 
