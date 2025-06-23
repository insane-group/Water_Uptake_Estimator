import logging
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Set global font size
plt.rcParams.update({'font.size': 14})

# --------------------------------------------------------------------
# Logger stub
# --------------------------------------------------------------------
class Logger:
    def __init__(self, name, level=logging.DEBUG, output="filestream", fileName=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if output=="filestream" and fileName:
            fh = logging.FileHandler(fileName)
        else:
            fh = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)
    def info(self, prefix, msg):
        self.logger.info(f"[{prefix}] {msg}")
    def error(self, prefix, msg):
        self.logger.error(f"[{prefix}] {msg}")

# --------------------------------------------------------------------
# LOOCV helper
# --------------------------------------------------------------------
def loocv(df, feats, targ, models):
    loo = LeaveOneOut()
    scores = {name: 0.0 for name in models}
    n = 0
    for tr, te in loo.split(df):
        Xtr = df.iloc[tr][feats].values
        Xte = df.iloc[te][feats].values
        ytr = df.iloc[tr][targ].to_numpy()
        yte = df.iloc[te][targ].to_numpy()
        for name, mdl in models.items():
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            scores[name] += mean_absolute_error(yte, pred)
        n += 1
    return {name: scores[name]/n for name in scores}

# --------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--target",default="all",choices=["all","density","GV","E"])
    args = parser.parse_args()
    
    logger = Logger("main", output="filestream", fileName=None)
    
    # 1) Load raw data
    data_mt = {
        "hour":      [0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528],
        "mass":      [121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,136.72,136.82,137.29,137.3,137.95,138.61,138.29,138.75,139.05],
        "thickness":[3.45,3.52,3.58,3.60,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.09,4.05,4.073,4.077,4.08,4.09,4.085]
    }
    data_el = {
        "hour":    [0,24,90,180,270,360,528],
        "E":       [6.479,4.219,3.269,2.757,2.773536366,2.185,2.187],
        "GV_meas": [2.36,1.89,1.61,1.50,1.34,1.30,np.nan]  # km/s
    }
    mt = pd.DataFrame(data_mt)
    el = pd.DataFrame(data_el)
    
    # 2) Density via rule-of-mixtures for mt
    rho_dry, rho_water = 1150.0, 1000.0
    mt['mass_kg']   = mt['mass']/1000.0
    m0 = mt.loc[mt['hour']==0,'mass_kg'].iloc[0]
    V0 = m0/rho_dry
    mt['m_water']  = mt['mass_kg'] - m0
    mt['V_water']  = mt['m_water']/rho_water
    mt['V_total']  = V0 + mt['V_water']
    mt['density_true'] = mt['mass_kg']/mt['V_total']
    
    # 3) Build 7-point elasticity table
    el['mass']      = np.interp(el['hour'], mt['hour'], mt['mass'])
    el['thickness']= np.interp(el['hour'], mt['hour'], mt['thickness'])
    el['density_true'] = np.interp(el['hour'], mt['hour'], mt['density_true'])
    el['GV_phys'] = el.apply(lambda r:
        r['GV_meas']*1000 if not np.isnan(r['GV_meas'])
        else np.sqrt(r['E']*1e9 / r['density_true']), axis=1)
    
    # 4) Models dict (no SVR)
    models = {
        'Ridge': Ridge(),
        'XGBoost': XGBRegressor(n_estimators=5, max_depth=3,
                                 random_state=0, verbosity=0)
    }
    
    # 5) Density CV on el with three feature sets
    feats_den = {
        'time': ['hour'],
        'mass+thickness': ['mass','thickness'],
        'mass+thickness+time': ['mass','thickness','hour']
    }
    cv_den = {k: loocv(el, feats, 'density_true', models)
              for k,feats in feats_den.items()}
    
    # pick best feature‐set and model for density
    best_den_feat = min(cv_den, key=lambda fs: min(cv_den[fs].values()))
    best_den_model = min(cv_den[best_den_feat],
                         key=lambda m: cv_den[best_den_feat][m])
    den_mod = models[best_den_model]
    den_mod.fit(el[feats_den[best_den_feat]], el['density_true'])
    el['density_pred'] = den_mod.predict(el[feats_den[best_den_feat]])
    logger.info("Density",
        f"best={best_den_feat} via {best_den_model}, MAE={cv_den[best_den_feat][best_den_model]:.3f}")
    
    # 6) GV CV using predicted density only
    feats_gv = {'density_pred': ['density_pred']}
    cv_gv = {k: loocv(el, feats, 'GV_phys', models)
             for k,feats in feats_gv.items()}
    best_gv_feat = 'density_pred'
    best_gv_model = min(cv_gv[best_gv_feat],
                       key=lambda m: cv_gv[best_gv_feat][m])
    gv_mod = models[best_gv_model]
    gv_mod.fit(el[['density_pred']], el['GV_phys'])
    el['GV_pred'] = gv_mod.predict(el[['density_pred']])
    logger.info("GV",
        f"best via {best_gv_model}, MAE={cv_gv[best_gv_feat][best_gv_model]:.3f}")
    
    # 7) E CV with five feature sets
    feats_E = {
        'time': ['hour'],
        'mass': ['mass'],
        'mass+thickness': ['mass','thickness'],
        'mass+thickness+GV': ['mass','thickness','GV_pred'],
        'GV': ['GV_pred']
    }
    cv_E = {k: loocv(el, feats, 'E', models)
            for k,feats in feats_E.items()}
    best_E_feat = min(cv_E, key=lambda fs: min(cv_E[fs].values()))
    best_E_model = min(cv_E[best_E_feat],
                       key=lambda m: cv_E[best_E_feat][m])
    E_mod = models[best_E_model]
    E_mod.fit(el[feats_E[best_E_feat]], el['E'])
    el['E_pred'] = E_mod.predict(el[feats_E[best_E_feat]])
    logger.info("E",
        f"best={best_E_feat} via {best_E_model}, MAE={cv_E[best_E_feat][best_E_model]:.3f}")
    
    # 8) Bar charts of CV MAEs
    def plot_bar(cvd, title):
        df = pd.DataFrame(cvd).T
        df.plot(kind='bar', figsize=(7,4))
        plt.ylabel("CV MAE"); plt.title(title)
        plt.xticks(rotation=45); plt.legend(title="Model")
        plt.tight_layout(); plt.show()
    
    plot_bar(cv_den, "Density CV MAE by Feature Set")
    plot_bar(cv_gv,  "GV CV MAE (density_pred)")
    plot_bar(cv_E,   "E CV MAE by Feature Set")
    
    # 9) Line plots of actual vs predicted
    def plot_act_pred(x, y_act, y_pred, xlabel, ylabel, title, label_pred):
        plt.figure(figsize=(7,4))
        plt.plot(x, y_act, 'o-', label="Actual")
        plt.plot(x, y_pred, 'x--', label=label_pred)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.title(title); plt.legend(); plt.tight_layout(); plt.show()
    
    plot_act_pred(el['hour'], el['density_true'], el['density_pred'],
                  "Hour", "Density (kg/m³)", "Density: Actual vs Predicted",
                  f"{best_den_model}")
    plot_act_pred(el['hour'], el['GV_phys'], el['GV_pred'],
                  "Hour", "GV (m/s)", "GV: Actual vs Predicted",
                  f"{best_gv_model}")
    plot_act_pred(el['hour'], el['E'], el['E_pred'],
                  "Hour", "E (GPa)", "Young’s Modulus: Actual vs Predicted",
                  f"{best_E_model}")
    
    # 10) Print tables
    print("\n=== Density Predictions ===")
    print(el[['hour','density_true','density_pred']].to_string(index=False))
    print("\n=== GV Predictions ===")
    print(el[['hour','GV_phys','GV_pred']].to_string(index=False))
    print("\n=== E Predictions ===")
    print(el[['hour','E','E_pred']].to_string(index=False))

