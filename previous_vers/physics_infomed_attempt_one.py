import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# Logger (light stub)
# -------------------------------------------------------
class Logger:
    def __init__(self, name, level=logging.DEBUG, output="filestream", fileName=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        fh = logging.FileHandler(fileName) if output == "filestream" else logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def info(self, prefix, message):
        self.logger.info(f"[{prefix}] {message}")

    def error(self, prefix, message):
        self.logger.error(f"[{prefix}] {message}")

# -------------------------------------------------------
# DataHandler: read data, compute density (rule-of-mixtures), fill GV
# -------------------------------------------------------
class DataHandler:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.logPrefix = "DataHandler"
        self.rho_dry = 1150.0    # initial dry density (kg/m³)
        self.rho_water = 1000.0  # water density (kg/m³)

    def read_manual_data(self):
        """
        Returns:
          - mass_thick: DataFrame with [hour, mass(g), thickness(mm)]
          - elasticity: DataFrame with [hour, E(GPa), GV(km/s)]   (GV has NaNs)
        """
        self.logger.info(self.logPrefix, "Reading manual data.")
        data_1 = {
            "hour":      [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 481, 504, 528],
            "mass":      [121.5, 124.18, 126.86, 128.1, 129.35, 130.59, 131.84, 132.66, 134.01, 134.27, 134.48, 135.2, 135.92, 136.64, 136.72, 136.82, 137.29, 137.3, 137.95, 138.61, 138.29, 138.75, 139.05],
            "thickness":[3.45, 3.52, 3.58, 3.60, 3.63, 3.72, 3.75, 3.82, 3.95, 4.03, 4.05, 4.06, 4.06, 4.06, 4.08, 4.08, 4.09, 4.05, 4.073, 4.077, 4.08, 4.09, 4.085]
        }
        data_2 = {
            "hour": [0, 24, 90, 180, 270, 360, 528],
            "E":    [6.479, 4.219, 3.269, 2.757, 2.773536366, 2.185, 2.187],
            "GV":   [2.36,   1.89, 1.61,  1.5,   1.34, 1.30,  np.nan]  # in km/s
        }          # calc   calc                             calc   
        return pd.DataFrame(data_1), pd.DataFrame(data_2)

    def compute_density_rule_of_mixtures(self, mass_thick: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rule-of-mixtures density for each row:
          - m0 = mass at hour=0 (kg)
          - V0 = m0 / rho_dry
          - for each hour: m_total = mass/1000 (kg)
                          m_water = m_total - m0
                          V_water = m_water / rho_water
                          V_total = V0 + V_water
                          density_rm = m_total / V_total
        Returns mass_thick with columns added: 
          ['mass_kg','thick_m','m_water_kg','V_water_m3','V_total_m3','density_rm']
        """
        self.logger.info(self.logPrefix, "Computing density via rule-of-mixtures.")
        df = mass_thick.copy()
        df['thick_m']   = df['thickness'] / 1000.0
        df['mass_kg']   = df['mass'] / 1000.0
        m0_kg           = df.loc[df['hour'] == 0, 'mass_kg'].values[0]
        V0              = m0_kg / self.rho_dry
        df['m_water_kg']= df['mass_kg'] - m0_kg
        df['V_water_m3'] = df['m_water_kg'] / self.rho_water
        df['V_total_m3'] = V0 + df['V_water_m3']
        df['density_rm'] = df['mass_kg'] / df['V_total_m3']
        return df

    def fill_physics_informed_gv(self, mass_thick_rm: pd.DataFrame, elasticity: pd.DataFrame) -> pd.DataFrame:
        """
        1) Converts provided GV (in km/s) to m/s.
        2) Fills NaN GV using GV = sqrt(E_pa / rho_rm) in m/s,
           where rho_rm is interpolated from mass_thick_rm['density_rm'].
        """
        self.logger.info(self.logPrefix, "Converting GV to m/s and filling NaNs via physics.")
        df = elasticity.copy()
        # Convert any non-NaN GV from km/s → m/s
        df['GV'] = df['GV'].apply(lambda x: x * 1000 if not np.isnan(x) else np.nan)

        hours_rm = mass_thick_rm['hour'].values
        rho_vals = mass_thick_rm['density_rm'].values
        for idx, row in df.iterrows():
            if np.isnan(row['GV']):
                h = row['hour']
                rho_interp = np.interp(h, hours_rm, rho_vals)
                E_pa = row['E'] * 1e9
                gv_phys = np.sqrt(E_pa / rho_interp)
                df.at[idx, 'GV'] = gv_phys
                self.logger.info(self.logPrefix, f"Hour={h}: rho={rho_interp:.1f} → GV_filled={gv_phys:.3f} m/s")
        return df

# -------------------------------------------------------
# MLPredictor: Evaluate models via LOOCV, plot results
# -------------------------------------------------------
class MLPredictor:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.logPrefix = "MLModelPredictor"

    def evaluate_models_loocv(self, data: pd.DataFrame, feature_cols: list, target: str) -> dict:
        """
        Runs Leave-One-Out CV for:
          - Ridge
          - XGBoost (5 trees, max_depth=3)
          - SVR (with refined hyperparameters + scaling)
        Returns: { 'ridge': MAE, 'xgboost': MAE, 'svr': MAE }
        """
        loo = LeaveOneOut()
        model_names = ["ridge", "xgboost", "svr"]
        scores = {name: 0.0 for name in model_names}
        count = 0

        for train_idx, test_idx in loo.split(data):
            X_train = data[feature_cols].iloc[train_idx].values
            X_test  = data[feature_cols].iloc[test_idx].values
            y_train = data[target].iloc[train_idx].values.ravel()
            y_test  = data[target].iloc[test_idx].values.ravel()

            # Ridge
            ridge = Ridge()
            ridge.fit(X_train, y_train)
            scores['ridge'] += mean_absolute_error(y_test, ridge.predict(X_test))

            # XGBoost (5 trees, max_depth=3 to reduce overfitting)
            xgb = XGBRegressor(n_estimators=15, max_depth=3, random_state=0, verbosity=0)
            xgb.fit(X_train, y_train)
            scores['xgboost'] += mean_absolute_error(y_test, xgb.predict(X_test))

            # SVR with refined hyperparameters (C=10, epsilon=0.05) and scaling
            svr = make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05))
            svr.fit(X_train, y_train)
            scores['svr'] += mean_absolute_error(y_test, svr.predict(X_test))

            count += 1

        # Average MAE
        for name in model_names:
            scores[name] /= count

        return scores

    def train_all_and_plot(self, data: pd.DataFrame, feature_cols: list, target: str, hours: np.ndarray):
        """
        1) Runs LOOCV and prints CV MAE per model (bar chart).
        2) Trains Ridge, XGBoost, SVR on full data and plots their predictions vs. actual.
        3) Warns if any model’s CV MAE < 1e-6 (possible overfitting).
        """
        cv_scores = self.evaluate_models_loocv(data, feature_cols, target)
        self.logger.info(self.logPrefix, f"CV scores for {target}: {cv_scores}")

        # 1) Bar chart of CV MAEs
        plt.figure(figsize=(6, 3))
        plt.bar(cv_scores.keys(), cv_scores.values(), color=['tab:blue', 'tab:green', 'tab:orange'])
        plt.ylabel('MAE')
        plt.title(f'CV MAE Comparison for {target}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        for mname, mae_val in cv_scores.items():
            if mae_val < 1e-6:
                print(f"⚠️ Warning: {target} MAE for {mname} is {mae_val:.6f} (near zero). Possible overfitting.")

        # 2) Train each model on full data
        X_full = data[feature_cols].values
        y_full = data[target].values

        models = {
            'ridge': Ridge(),
            'xgboost': XGBRegressor(n_estimators=5, max_depth=3, random_state=0, verbosity=0),
            'svr': make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05))
        }

        preds = {}
        for name, mdl in models.items():
            mdl.fit(X_full, y_full)
            preds[name] = mdl.predict(X_full)

        # 3) Plot actual vs predicted for all three models on one plot
        plt.figure(figsize=(8, 4))
        plt.plot(hours, y_full, 'o-', label='Actual', color='black')
        for name, y_pred in preds.items():
            plt.plot(hours, y_pred, marker='x', linestyle='--', label=name.replace('_', ' ').title())
        plt.xlabel('Hour')
        plt.ylabel(target)
        plt.title(f'Actual vs Predicted {target} (Ridge, XGBoost, SVR)')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 4) Print table of Actual vs Predicted for each model
        table_dict = {'hour': hours, 'actual': np.round(y_full, 3)}
        for name, y_pred in preds.items():
            table_dict[name + '_pred'] = np.round(y_pred, 3)
        table_df = pd.DataFrame(table_dict)
        print(f"\n=== {target} Predictions (Ridge, XGBoost, SVR) ===")
        print(table_df.to_string(index=False))

        return cv_scores

# -------------------------------------------------------
# Main script
# -------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', default='all', help='Target: one of [E, density, GV, all]')
    parsed_args = parser.parse_args()
    target_name = parsed_args.target

    logger = Logger(
        name='estimator_logger',
        level=logging.DEBUG,
        output="filestream",
        fileName=datetime.now().strftime('logfile_%d-%m-%Y-%H-%M-%S.%f')[:-3] + ".log"
    )

    data_handler = DataHandler(logger)
    mass_thick, elasticity = data_handler.read_manual_data()

    # 1) Compute density via rule-of-mixtures
    mass_thick_rm = data_handler.compute_density_rule_of_mixtures(mass_thick)

    # 2) Fill missing GV (convert km/s → m/s, then fill via sqrt(E_pa/rho_rm))
    elasticity_filled = data_handler.fill_physics_informed_gv(mass_thick_rm, elasticity)

    # 3) Interpolate mass → compute density_rm for elasticity_filled
    elasticity_filled['mass'] = np.interp(
        elasticity_filled['hour'],
        mass_thick_rm['hour'],
        mass_thick_rm['mass']
    )
    m0_kg = mass_thick_rm.loc[mass_thick_rm['hour'] == 0, 'mass_kg'].values[0]
    V0 = m0_kg / data_handler.rho_dry
    elasticity_filled['mass_kg']    = elasticity_filled['mass'] / 1000.0
    elasticity_filled['m_water_kg'] = elasticity_filled['mass_kg'] - m0_kg
    elasticity_filled['V_water_m3'] = elasticity_filled['m_water_kg'] / data_handler.rho_water
    elasticity_filled['V_total_m3'] = V0 + elasticity_filled['V_water_m3']
    elasticity_filled['density']    = elasticity_filled['mass_kg'] / elasticity_filled['V_total_m3']

    # 4) Prepare DataFrames for each target
    df_E     = elasticity_filled[['hour', 'mass', 'GV', 'density', 'E']].copy()
    df_dense = mass_thick_rm[['hour', 'mass', 'density_rm']].rename(columns={'density_rm': 'density'}).copy()
    df_GV    = elasticity_filled[['hour', 'mass', 'GV']].copy()

    predictor = MLPredictor(logger)

    # 5) Decide which targets to run
    if target_name in ['E', 'density', 'GV']:
        targets_to_run = [target_name]
    else:
        targets_to_run = ['E', 'density', 'GV']

    # 6) Loop over each requested target
    for tgt in targets_to_run:
        if tgt == 'E':
            data_df = df_E
            feature_cols = ['mass', 'GV', 'density']
            hours = data_df['hour'].values
        elif tgt == 'density':
            data_df = df_dense
            feature_cols = ['hour', 'mass']
            hours = data_df['hour'].values
        else:  # 'GV'
            data_df = df_GV
            feature_cols = ['hour', 'mass']
            hours = data_df['hour'].values

        logger.info("Main", f"Evaluating target: {tgt}")
        predictor.train_all_and_plot(data_df, feature_cols, tgt, hours)
