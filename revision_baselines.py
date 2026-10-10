"""Reviewer 2 comment 7: retrospective early-data baseline comparison.

Run with NumPy, SciPy and scikit-learn available. Fits receive only the selected
training arrays. Test labels are used solely for the final scoring functions.
The original hybrid is a frozen reference with broader development-data access.
"""
from pathlib import Path
import ast,csv,hashlib,json,warnings
import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct,Matern,WhiteKernel
import scipy,sklearn

from repo_io import ROOT, REFERENCE, output_directory, analysis_arguments
ALPHAS=[1e-4,1e-3,1e-2,1e-1,1.,10.,100.,1000.,10000.]
TAUS=[10.,20.,50.,100.,200.,500.,800.]
PROPERTIES=['mass','thick','cg','youngs']

def load_records():
    source=ROOT/'main_run.py'
    env={'np':np}
    for node in ast.parse(source.read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'all_cases','cg_data','E_data','test_data'} for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'data_records','exec'),env)
    return env,source

def ridge_fit(t,y):
    x=np.asarray(t,float).reshape(-1,1);y=np.asarray(y,float)
    errors=[]
    for alpha in ALPHAS:
        preds=[]
        for j in range(len(y)):
            keep=np.arange(len(y))!=j
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='svd'))
            model.fit(x[keep],y[keep]);preds.append(model.predict(x[j:j+1])[0])
        errors.append(float(np.mean(np.abs(np.array(preds)-y))))
    index=int(np.argmin(errors))
    model=make_pipeline(StandardScaler(),Ridge(alpha=ALPHAS[index],solver='svd')).fit(x,y)
    return lambda q:model.predict(np.asarray(q,float).reshape(-1,1)),{'alpha':ALPHAS[index],'training_loo_mae':errors[index],'all_training_loo_mae':errors,'feature_scaling':'StandardScaler fitted inside every training LOO fold and refitted on all 3 records'}

def gp_fit(t,y):
    scaler=StandardScaler().fit(np.asarray(t,float).reshape(-1,1))
    kernel=(DotProduct(sigma_0=1.,sigma_0_bounds='fixed')+
            Matern(length_scale=1.,length_scale_bounds=(.1,10.),nu=1.5)+
            WhiteKernel(noise_level=.01,noise_level_bounds='fixed'))
    model=GaussianProcessRegressor(kernel=kernel,alpha=1e-8,normalize_y=True,
        n_restarts_optimizer=0,random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(scaler.transform(np.asarray(t,float).reshape(-1,1)),y)
    return lambda q:model.predict(scaler.transform(np.asarray(q,float).reshape(-1,1))),{
        'kernel':str(model.kernel_),'matern_length_scale':float(model.kernel_.k1.k2.length_scale),
        'normalized_target_noise_variance_fixed':.01,'diagonal_jitter':1e-8,
        'optimizer':'fmin_l_bfgs_b; training marginal likelihood only; zero restarts',
        'warnings':[str(w.message) for w in caught]}

def fraction(t,tau):
    t=np.asarray(t,float).reshape(1,-1);odd=(2*np.arange(60)+1).reshape(-1,1)
    f=1.-8./np.pi**2*np.sum(np.exp(-odd**2*t/tau)/odd**2,axis=0)
    f0=1.-8./np.pi**2*np.sum(1./odd[:,0]**2)
    normalized=(f-f0)/(1.-f0)
    normalized[t[0]==0.]=0.  # avoid summation-roundoff drift at the measured dry state
    return normalized

def constrained_mass(t,y):
    y0=float(y[0]);best=None
    for tau in TAUS:
        f=fraction(t,tau);amp=max(0.,float(np.dot(f,y-y0)/np.dot(f,f)))
        mse=float(np.mean((y0+amp*f-y)**2))
        if best is None or mse<best[0]:best=(mse,tau,amp)
    mse,tau,amp=best
    return lambda q:y0+amp*fraction(q,tau),{'tau_h':tau,'saturation_mass_g':y0+amp,'training_mse':mse,'dry_state_normalized_60_term_series':True}

def constrained_decay(t,y):
    t=np.asarray(t,float);y=np.asarray(y,float);y0=float(y[0]);minimum=float(np.min(y))
    best=None;successes=0
    for ratio in [.25,.75]:
        for tau in TAUS:
            fit=least_squares(lambda p:(p[0]+(y0-p[0])*np.exp(-t/p[1])-y)/y0,
                x0=[ratio*minimum,tau],bounds=([1e-9,.1],[minimum,1e5]),
                x_scale=[y0,100.],ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=10000)
            successes+=int(fit.success)
            sse=float(np.sum((fit.x[0]+(y0-fit.x[0])*np.exp(-t/fit.x[1])-y)**2))
            if best is None or sse<best[0]:best=(sse,fit)
    sse,fit=best;floor,tau=map(float,fit.x)
    assert fit.success and np.isfinite(sse)
    return lambda q:floor+(y0-floor)*np.exp(-np.asarray(q,float)/tau),{
        'positive_asymptote':floor,'tau_h':tau,'training_sse':sse,'dry_value_fixed':y0,
        'bounds':{'asymptote':[1e-9,minimum],'tau_h':[.1,1e5]},
        'successful_starts':successes,'starts':14,'selected_by':'minimum training SSE',
        'interpretation':'empirical positive monotone saturation baseline; not a guided-wave or tensile constitutive inversion'}

def run(output_dir=None):
    env,source=load_records();case=env['all_cases'][-1]
    grid=case['time'];test=env['test_data'];tt=test['time']
    dataset={'mass':(case['time'],case['mass'],[0,4,8],'mass','g'),
             'thick':(case['time'],case['thick'],[0,4,8],'thick','mm'),
             'cg':(env['cg_data']['hour'],env['cg_data']['gv_ms'],[0,1,2],'gv_ms','m/s'),
             'youngs':(env['E_data']['hour'],env['E_data']['E_GPa'],[0,1,2],'E_GPa','GPa')}
    stored=json.loads((REFERENCE/'R1c_corrected_metrics.json').read_text(encoding='utf-8'))
    models={};settings={}
    for name,fit in [('Ridge 3',ridge_fit),('Gaussian process 3',gp_fit)]:
        models[name]={};settings[name]={}
        for prop,(t,y,idx,key,unit) in dataset.items():
            pred,detail=fit(t[idx],y[idx]);models[name][prop]=pred(grid);settings[name][prop]=detail
    mass_pred,mass_info=constrained_mass(case['time'][[0,4,8]],case['mass'][[0,4,8]])
    idx=np.array([0,4,8]);dm=case['mass'][idx]/case['mass'][0]-1.;dh=case['thick'][idx]/case['thick'][0]-1.
    alpha=max(0.,float(np.dot(dm,dh)/np.dot(dm,dm)))
    models['Constrained saturation 3']={'mass':mass_pred(grid),
        'thick':case['thick'][0]*(1.+alpha*(mass_pred(grid)/case['mass'][0]-1.))}
    settings['Constrained saturation 3']={'mass':mass_info,'thick':{'training_only_swelling_alpha':alpha,'features':'predicted mass from constrained mass model; dry mass/thickness measured'}}
    for prop in ['cg','youngs']:
        t,y,idx,key,unit=dataset[prop];pred,detail=constrained_decay(t[idx],y[idx])
        models['Constrained saturation 3'][prop]=pred(grid);settings['Constrained saturation 3'][prop]=detail
    for name,config,path in [('Random Forest 3','early_three','pure'),('Random Forest 7','active_three_vs_seven','pure'),('Published hybrid','active_three_vs_seven','hybrid')]:
        models[name]={p:np.array(stored['configurations'][config]['predictions'][p][path]) for p in PROPERTIES}
        settings[name]={'source':'frozen R1c_corrected_metrics.json; original predictions unchanged','configuration':config,'path':path}
    for name,props in models.items():
        assert all(len(v)==len(grid) and np.all(np.isfinite(v)) for v in props.values())
    c=models['Constrained saturation 3']
    assert np.all(np.diff(c['mass'])>=-1e-9) and np.all(np.diff(c['thick'])>=-1e-9)
    assert np.all(np.diff(c['cg'])<=1e-9) and np.all(np.diff(c['youngs'])<=1e-9)
    assert all(np.all(v>0) for v in c.values())
    assert c['mass'][0]==case['mass'][0] and c['thick'][0]==case['thick'][0]
    scores={};prediction_rows=[]
    for name,props in models.items():
        scores[name]={}
        for prop,(obs_t,obs_y,idx,key,unit) in dataset.items():
            horizon=float(obs_t[idx][-1]);prediction=np.interp(tt,grid,props[prop]);truth=test[key]
            later=tt>horizon;dev_later=obs_t>horizon
            later_pred=np.interp(obs_t[dev_later],grid,props[prop])
            scores[name][prop]={'unit':unit,'test_mae_all_5':float(np.mean(np.abs(prediction-truth))),
                'test_mae_later_than_early_training':float(np.mean(np.abs(prediction[later]-truth[later]))),
                'test_later_count':int(later.sum()),'test_later_times_h':tt[later].tolist(),
                'later_development_grid_mae':float(np.mean(np.abs(later_pred-obs_y[dev_later]))),
                'later_development_count':int(dev_later.sum()),'early_training_horizon_h':horizon,
                'test_predictions':prediction.tolist(),'nonpositive_grid_predictions_count':int((props[prop]<=0).sum())}
            for i,t in enumerate(tt):
                prediction_rows.append({'model':name,'property':prop,'time_h':float(t),'truth':float(truth[i]),'prediction':float(prediction[i]),'absolute_error':float(abs(prediction[i]-truth[i])),'unit':unit})
    # Audit exact agreement with previously reported test metrics for frozen references.
    for name,path in [('Random Forest 7','pure'),('Published hybrid','hybrid')]:
        for prop in PROPERTIES:
            assert abs(scores[name][prop]['test_mae_all_5']-stored['configurations']['active_three_vs_seven']['test'][prop][path])<1e-10
    report={'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'protocol':'Retrospective added baseline analysis; no prospective or independent-panel validation claimed',
        'ridge_alpha_grid':ALPHAS,'mass_tau_grid_h':TAUS,'settings':settings,'scores':scores,
        'time_grid_h':grid.tolist(),'predictions':{name:{p:v.tolist() for p,v in props.items()} for name,props in models.items()},
        'training_times_h':{p:obs_t[idx].tolist() for p,(obs_t,obs_y,idx,key,unit) in dataset.items()},
        'test_times_h':tt.tolist(),'new_baselines_fit_validation_or_test_labels':False,
        'published_hybrid_uses_broader_development_information':True,
        'new_baseline_score_based_selection_or_revisions':False,
        'same_interpolated_grid_evaluation_for_all_models':True,
        'checks':{'finite_predictions':True,'constrained_positivity_monotonicity_and_dry_state':True,'frozen_reference_test_maes_reproduced':True},
        'source_hashes':{source.name:hashlib.sha256(source.read_bytes()).hexdigest(),
            'frozen_metrics':hashlib.sha256((REFERENCE/'R1c_corrected_metrics.json').read_bytes()).hexdigest(),
            'analysis_script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    out=output_directory(output_dir)
    (out/'R2_7_baseline_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (out/'R2_7_test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=prediction_rows[0]);writer.writeheader();writer.writerows(prediction_rows)
    for name,s in scores.items():
        print(name,'test MAEs:',{p:round(v['test_mae_all_5'],6) for p,v in s.items()})
    print('Constrained model checks passed; original frozen test metrics exactly reproduced. New settings are not changed after viewing test scores.')

if __name__=='__main__':
    args=analysis_arguments(__doc__).parse_args()
    run(args.output_dir)
