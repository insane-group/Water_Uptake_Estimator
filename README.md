# Water Uptake Estimator

Code and numerical records for the revised flax/epoxy moisture-ageing study. The model estimates mass, thickness, effective density, S0 guided-wave group velocity and tensile Young's modulus over 0–528 hours. The repository includes baseline, ablation and conditional uncertainty analyses, plus the editable [Figure 7 draw.io source](figures/Figure_7.drawio).

## Install and run

Use Python 3.12. [requirements.txt](requirements.txt) pins the revision environment: NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.8.0 and Matplotlib 3.11.2. Other versions can change predictions or optimization results.

```bash
git clone https://github.com/insane-group/Water_Uptake_Estimator.git
cd Water_Uptake_Estimator
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install -r requirements.txt
python main_run.py --no-show
python -m unittest -v test_reproduction
```

The default run compares the early three-record hybrid with the seven-record data-driven forest. It saves numerical results and nine figures in `results/active_three_vs_seven/`. Legends use 16-point text. Omit `--no-show` to display figures; use `--no-plots` for numerical output only. Importing `main_run` does not fit models or open figures.

The three manuscript configurations and additional analyses run as follows:

```bash
python main_run.py --configuration even_three --no-show
python main_run.py --configuration early_three --no-show
python main_run.py --configuration active_three_vs_seven --no-show
python revision_baselines.py
python revision_ablations.py
python revision_uncertainty.py
```

Generated files go to `results/`, which Git ignores. Each command accepts `--output-dir PATH`; archived reference files remain unchanged. The uncertainty run uses 512 trials at each of three error levels (1536 total) and can take several minutes. `--trials 8 --output-dir results/smoke` checks execution but does not reproduce the paper's uncertainty results. With Bash, `bash run_evaluations.sh` runs the default model, all analyses and regression checks.

## Data and evaluation

[main_run.py](main_run.py) contains five mass/thickness histories, the target history (Case 5), seven velocity/modulus observations and five reserved records. Units are hours, grams, millimetres, metres per second and GPa.

The campaign comprised seven parent panels: dry PF0 and six exposure conditions PF24, PF90, PF180, PF270, PF360 and PF528. Five tensile coupons per condition give 35 coupons, not 35 independent panels. The five stored gravimetric histories are not a seven-panel specimen manifest. Test modulus values came from additional tensile measurements. All development and test records share the same batch/parent-panel material; individual specimen-to-record assignments are unavailable.

| Configuration | Hybrid mass/thickness training times (h) | Hybrid velocity/modulus training times (h) | Data-driven forest |
| --- | --- | --- | --- |
| `even_three` | 0, 264, 528 | 0, 180, 528 | Same three records |
| `early_three` | 0, 96, 192 | 0, 24, 90 | Same three records |
| `active_three_vs_seven` | 0, 96, 192 | 0, 24, 90 | Mass/thickness: 0, 96, 168, 264, 360, 432, 528; velocity/modulus: all seven records |

The reserved times are **13, 40, 72, 120 and 240 h**. These records were kept aside during original model development. Predictions are linearly interpolated from the common 23-point target grid before scoring. This is record-level evaluation within the same batch, not independent-panel validation.

The hybrid has broader development information: its swelling coefficient uses the full target history, and wave-anchor generation used training and validation information. Mass/thickness development scores also use the observations that select blending weights, so they are tuning scores. The seven-record velocity/modulus forest development errors are training errors. Added analyses are retrospective on the existing test records and do not establish prospective accuracy or extrapolation beyond 528 h.

## Model and settings

1. **Mass:** time-based Random Forest plus a 60-term Fickian series, with a selected physics blending weight.
2. **Thickness:** mass-based forest plus linear swelling, `h = h0 * (1 + alpha * (M - M0) / M0)`, with a selected physics weight.
3. **Effective density:** deterministic mixture from each mass-prediction path.
4. **Group velocity:** time-based forest augmented with supplied wave-anchor labels. Density/thickness are not direct regressor inputs; there is no final velocity blending.
5. **Modulus:** time-based forest augmented with `rho * cg**2 / 1e9` labels. This group-velocity stiffness proxy is approximate in the anisotropic guided-wave setting, not an exact inversion to tensile modulus. The augmented forest gives the final prediction.

Forests use 300 trees and seed 42; remaining settings are in [R1e_settings.json](reference/R1e_settings.json). Measured records have weight 1 and PI anchors weight 2; all supplied anchors are used. Mass fitting searches `tau = [10, 20, 50, 100, 200, 500, 800]` h on sparse training error. Mass/thickness blending searches physics weights from 0 to 1 in steps of 0.05 on the remaining development observations. Active weights are 0.8 and 1.0; full-history `alpha = 1.3972938599464269`. Ridge is an added baseline, not a component of the original cascade.

### Retained density mixture

```text
w = max(0, (M - M0) / M)
rho = (1 - w) * 1150 + w * 1000   [kg/m³]
```

Net mass gain represents absorbed water, assuming negligible dry-material mass loss. Dry density was physically measured at 1150 kg/m³; water density is fixed at 1000 kg/m³. The arithmetic blend uses water mass fraction as an effective coordinate. It does not enforce additive constituent volumes or measured swelling. `rom_density` retains thickness arguments for compatibility; they do not affect the output. Density references use measured mass in the same law, not independently measured wet density.

At 528 h this mixture gives **1131.0680 kg/m³**. The constant-area bulk estimate `rho0 * (M/M0) * (h0/h)` gives **1115.6224 kg/m³**, a 1.3845% difference. Length/width changes below 0.1% support that comparison, not the mixture approximation. The additive-volume mass-fraction mixture would instead be harmonic. These limitations remain in the paper and supplement.

### Supplied wave anchors

The experiment identifies S0 propagation at 0° to the fibres and 185 kHz. The [SMART 2025 study](https://doi.org/10.7712/150125.13726.306002) supplies the prior dispersion/modelling context; its nominal 3 mm laminate and 130 kHz experiment differ from the journal conditions.

| Time (h) | Group velocity (m/s) | Configuration |
| --- | --- | --- |
| 9.47 | 2208 | All three |
| 18.94 | 1996 | Even three |
| 47.36 | 1776 | All three |
| 270 | 1360 | Early three and active |
| 360 | 1300 | Even three |
| 360 | 1330 | Early three and active |
| 528 | 1320 | Early three and active |

The first three listed pairs come from dispersion calculations with ML-derived inputs; later pairs combine measurements and dispersion calculations. The 2208 m/s value at 9.47 h specifically comes from the dispersion code. Regression uses these fixed labels. The separate time-domain spectral-element/dispersion code can be provided by the authors on request. Complete journal-state inputs and raw-output-to-anchor mapping are not included; this repository does not fully regenerate the physical anchors.

## Reference results

Common five-record test MAEs, matching the revised manuscript:

| Model | Mass (g) | Thickness (mm) | Velocity (m/s) | Modulus (GPa) |
| --- | --- | --- | --- | --- |
| Random Forest 3 | 1.8401 | 0.0523 | 131.2539 | 0.3890 |
| Ridge 3 | 1.0969 | 0.0413 | 196.1634 | 0.5991 |
| Gaussian process 3 | 0.7325 | 0.0786 | 185.4361 | 0.8177 |
| Constrained saturation 3 | 0.5993 | 0.0411 | 44.8616 | 0.3109 |
| Random Forest 7 | 1.5251 | 0.0703 | 67.8033 | 0.1860 |
| Hybrid | 0.9297 | 0.0244 | 58.5354 | 0.1793 |

The constrained baseline has lower mass/velocity errors; the hybrid has lower thickness/modulus errors among these methods. Information access differs as described above. No overall superiority or statistical significance is established. The legacy JSON key `Published hybrid` refers to the Hybrid row and is retained for compatibility; it does not indicate acceptance/publication status.

| Files in `reference/` | Contents |
| --- | --- |
| `R1c_corrected_metrics.json`, `R1e_settings.json` | Three core configurations, trajectories, scores, settings and anchors |
| `R2_7_baseline_results.json`, `R2_7_test_predictions.csv` | Six-model comparison; 120 test predictions |
| `R2_8_ablation_results.json`, `R2_8_test_predictions.csv` | Full hybrid and seven interventions; 160 test predictions |
| `R2_9_uncertainty_results.json`, `R2_9_conditional_draws.npz` | Three conditional scenarios, all draws and 20 sensitivity interventions |
| `R2_9_test_envelopes.csv`, `R2_9_sensitivity.csv` | Reserved-time bands and sensitivity summaries |

These archived manuscript records retain historical source hashes. New run reports record current source hashes; see [reference/README.md](reference/README.md).

## Added analyses and verification

**Baselines:** Ridge uses training-only leave-one-out penalty selection (`1e-4` through `1e4` by decades); the GP uses training marginal likelihood; positive saturation/swelling fits use training-only least squares. Reports include parameters and optimizer warnings. GP predictions are not clipped.

**Ablations:** replace the thickness forest's mass input with time, replace modulus proxy labels with training-only RF pseudo labels, combine both, remove PI anchors, substitute measured upstream development trajectories, or fix blending weights at 0.5 or 0. The measured-upstream case is a diagnostic unavailable for forecasting. Full-history swelling calibration and wave-anchor information are retained. Effects are property dependent; variants are not selected from test scores.

**Conditional uncertainty:** seed 20261005; assumed relative input standard deviations 0.5%, 1% and 2%; independent mean-one lognormal multipliers per development record and supplied wave anchor, plus a shared dry-density draw per trial. Dry mass/thickness draws are reused in normalization. Transport and full-history swelling fits are recalculated; blending weights stay fixed. Pointwise 2.5–97.5% bands describe assumed input scenarios, not calibrated population prediction intervals. Panel variability, unknown correlations and physical-model discrepancy are not estimated; the wave solver is not rerun.

**Sensitivity:** 20 interventions perturb wet mass, wet thickness, swelling coefficient, dry density and velocity by ±1% and ±5%, recomputing affected outputs. Thickness/swelling perturbations leave density/modulus unchanged because geometry is absent from density. A ±5% velocity change gives −9.75%/+10.25% change in the stiffness proxy.

`test_reproduction.py` checks all three core configurations and scores, density behavior, the nominal uncertainty model and one full conditional refit per noise level, every archived grid/test band, and all 20 sensitivity interventions against the manuscript records (absolute tolerance `1e-10`). Optional `--case-variability` figure shading describes case-history variation, not predictive uncertainty.

The evidence is an offline study of one material batch. Real-time monitoring, digital-twin operation, independent-material generalization and long-term prediction require further validation.
