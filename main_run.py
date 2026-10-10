"""Recorded observations and entry point for the three manuscript configurations."""
import argparse
import json
import numpy as np
from pi_pipeline import run_pipeline
from pi_surrogates import PISpec1D, fit_alpha_from_cases, rom_density
from repo_io import output_directory

# Recorded case histories; Case 5 is the target history.
all_cases = [
    { # Case 1
        "time": np.array([0,18,42,66,90], float),
        "mass": np.array([112.51,118.8,120.52,122.17,124.27], float),
        "thick": np.array([3.42,3.61,3.67,3.74,3.84], float)
    },
    { # Case 2
        "time": np.array([0,13,37,61,85,109,133,158,180,205], float),
        "mass": np.array([101.97,106.46,108.55,109.92,110.96,112.18,113.27,114.48,113.8,113.2], float),
        "thick": np.array([3.05,3.23,3.29,3.39,3.36,3.37,3.39,3.4,3.44,3.45], float)
    },
    { # Case 3
        "time": np.array([0,23,47,71,96,120,147,170,194,218,242,266], float),
        "mass": np.array([119.38,124.67,126.37,128.07,129.84,131.03,132.13,131.8,133.62,134.32,135.01,135.71], float),
        "thick": np.array([3.72,3.85,3.87,3.93,3.99,4.02,3.99,4.02,4.02,4.03,4.05,4.06], float)
    },
    { # Case 4
        "time": np.array([0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360], float),
        "mass": np.array([111.64,116.305,120.97,122.41,123.69,124.97,126.25,125.96,125.84,125.94,126.04,126.79,127.54,128.29,127.01,126], float),
        "thick": np.array([3.3,3.42,3.54,3.66,3.7,3.73,3.77,3.82,3.81,3.83,3.85,3.83,3.82,3.8,3.79,3.78], float)
    },
    { # Case 5 (Target)
        "time":  np.array([0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,481,504,528], float),
        "mass":  np.array([121.5,124.18,126.86,128.1,129.35,130.59,131.84,132.66,134.01,134.27,134.48,135.2,135.92,136.64,135.61,135.02,135.29,136.3,136.95,137.61,138.29,138.75,139.05], float),
        "thick": np.array([3.45,3.52,3.58,3.6,3.63,3.72,3.75,3.82,3.95,4.03,4.05,4.06,4.06,4.06,4.08,4.08,4.1,4.07,4.073,4.077,4.08,4.09,4.07], float)
    },
]

case5 = all_cases[4]
cg_data = {"hour": np.array([0, 24, 90, 180, 270, 360, 528], float), "gv_ms": np.array([2360, 1890, 1610, 1500, 1340, 1310, 1300], float)}
E_data = {"hour": np.array([0, 24, 90, 180, 270, 360, 528], float), "E_GPa": np.array([6.479, 4.219, 3.269, 2.757, 2.773, 2.185, 2.187], float)}


# Five reserved observations from the same batch/parent panels.
# These are record-level tests, not independent-panel validation.
test_data = {
    "time": np.array([13, 40, 72, 120, 240], float),
    "mass": np.array([123.21, 126.47, 128.72, 130.7, 136.6], float),
    "thick": np.array([3.49, 3.54, 3.68, 3.77, 3.9], float),
    "gv_ms": np.array([2115, 1760, 1650, 1580, 1410], float),
    "E_GPa": np.array([5.01, 4.031, 3.561, 3.36, 2.55], float)
}

# Optional descriptive variability across case histories; not measurement uncertainty.
def compute_std(t_eval, cases, key):
    diffs = []
    for c in cases:
        val = np.array(c[key])
        diff = val - val[0]
        diffs.append(np.interp(t_eval, c["time"], diff, right=np.nan))
    with np.errstate(invalid='ignore'): std_val = np.nanstd(np.array(diffs), axis=0)
    valid_mask = ~np.isnan(std_val) & (std_val > 0)
    if np.any(valid_mask): std_val[np.where(valid_mask)[0][-1]+1:] = std_val[np.where(valid_mask)[0][-1]]
    std_val[np.isnan(std_val)] = 0.0
    return std_val

def compute_rho_std(t_eval, cases, rho_dry=1150.0, rho_fluid=1000.0):
    diffs = []
    for c in cases:
        M = np.array(c["mass"])
        h = np.asarray(c["thick"], float)
        rho = rom_density(M, h, M[0], rho_dry, rho_fluid, h0=h[0])
        diffs.append(np.interp(t_eval, c["time"], rho - rho[0], right=np.nan))
    with np.errstate(invalid='ignore'): std_val = np.nanstd(np.array(diffs), axis=0)
    valid_mask = ~np.isnan(std_val) & (std_val > 0)
    if np.any(valid_mask): std_val[np.where(valid_mask)[0][-1]+1:] = std_val[np.where(valid_mask)[0][-1]]
    std_val[np.isnan(std_val)] = 0.0
    return std_val


CONFIGURATIONS = ('even_three', 'early_three', 'active_three_vs_seven')


def run_configuration(name='active_three_vs_seven'):
    """Reproduce the original fit; reserved test labels are not passed to fitting.

    Full-history alpha and supplied wave anchors retain their original broader
    development-data access. The main README explains this evaluation scope.
    """
    if name not in CONFIGURATIONS:
        raise ValueError(f'Unknown configuration: {name}')
    even = name == 'even_three'
    cfg = dict(mass_mode='even', mass_n=3, thick_mode='even', thick_n=3,
               cg_mode='even', cg_n=3, E_mode='even', E_n=3) if even else dict(
        mass_mode='manual', mass_idx=[0,4,8], thick_mode='manual', thick_idx=[0,4,8],
        cg_mode='manual', cg_idx=[0,1,2], E_mode='manual', E_idx=[0,1,2])
    if name == 'active_three_vs_seven':
        for prop in ['mass', 'thick', 'cg', 'E']:
            cfg.update({f'{prop}_pure_mode':'even', f'{prop}_pure_n':7})
    return run_pipeline(case=case5, all_cases_for_alpha=[case5],
        fixed_alpha=fit_alpha_from_cases([case5]), **cfg,
        pi_mass=PISpec1D(np.array([150,300])), pi_w_mass=2.,
        pi_thick=PISpec1D(np.array([50,150] if even else [250,300])), pi_w_thick=2.,
        cg_case=cg_data, pi_cg=PISpec1D(
            np.array([9.47,18.94,47.36,360] if even else [9.47,47.36,270,360,528]),
            values=np.array([2208,1996,1776,1300] if even else [2208,1776,1360,1330,1320])), pi_w_cg=2.,
        E_case=E_data, pi_E=PISpec1D(np.array([10,50,200,400] if even else [10,20,200,400])), pi_w_E=2.,
        rho_0=1150., rho_fluid=1000.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', choices=CONFIGURATIONS, default='active_three_vs_seven')
    parser.add_argument('--output-dir', help='Generated files (default: results/<configuration>)')
    parser.add_argument('--no-show', action='store_true', help='Save figures without opening windows')
    parser.add_argument('--no-plots', action='store_true', help='Write numerical results only')
    parser.add_argument('--case-variability', action='store_true', help='Add descriptive case-history SD shading')
    args=parser.parse_args()
    out=run_configuration(args.configuration)
    folder=output_directory(args.output_dir or output_directory()/args.configuration)
    props=['mass','thick','dens','cg','youngs']
    record={'configuration':args.configuration, 'alpha_full_history':out.alpha,
            'blending':{'mass':out.mass.lambda_star, 'thick':out.thick.lambda_star},
            'time_h':out.mass.t.tolist(), 'predictions':{}, 'development_mae':{}, 'test_mae':{}}
    for prop in props:
        res=getattr(out,prop)
        record['predictions'][prop]={k:getattr(res,'pred_'+v).tolist() for k,v in [('pure','pure'),('physics','phys'),('hybrid','hybrid')]}
        record['development_mae'][prop]={'pure':res.mae_pure,'hybrid':res.mae_val}
    for prop,key in [('mass','mass'),('thick','thick'),('cg','gv_ms'),('youngs','E_GPa')]:
        res=getattr(out,prop)
        record['test_mae'][prop]={k:float(np.mean(np.abs(np.interp(test_data['time'],res.t,getattr(res,'pred_'+k))-test_data[key]))) for k in ['pure','hybrid']}
    record['score_scope']='Development scores include tuning scores and (for seven-point wave/modulus RF) training errors; tests share batch/parent panels.'
    (folder/'core_results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Five-record test MAEs:',json.dumps(record['test_mae'],indent=2))
    print(f'Results: {folder}')
    if not args.no_plots:
        if args.no_show:
            import matplotlib
            matplotlib.use('Agg')
        from pi_plots import plot_all, plot_test_evaluations
        stds=None
        if args.case_variability:
            stds={'mass':compute_std(case5['time'],all_cases,'mass'),
                  'thick':compute_std(case5['time'],all_cases,'thick'),
                  'dens':compute_rho_std(case5['time'],all_cases)}
        plot_all(out,case5,cg_data,E_data,exp_stds=stds,output_dir=folder,show=not args.no_show)
        plot_test_evaluations(out,test_data,output_dir=folder,show=not args.no_show)


if __name__ == '__main__':
    main()
