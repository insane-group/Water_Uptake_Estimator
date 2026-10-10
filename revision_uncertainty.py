"""R2.9 conditional input uncertainty and cascade sensitivity.

Assumed relative input errors are scenarios, not measured instrument precision.
No specimen bootstrap or calibrated population prediction interval is claimed.
Original settings and supplied wave anchors are retained; no test-driven tuning.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import scipy, sklearn
from sklearn.ensemble import RandomForestRegressor
from revision_baselines import load_records
from pi_surrogates import fickian_fraction, fickian_model, fit_alpha_from_cases, rom_density, thickness_physics_linear

from repo_io import ROOT, REFERENCE, output_directory, analysis_arguments
SEED=20261005
TRIALS=512
LEVELS=[.005,.01,.02]
PROPS=['mass','thick','dens','cg','proxy','youngs']
ANCHOR_H=np.array([9.47,47.36,270.,360.,528.])
ANCHOR_V=np.array([2208.,1776.,1360.,1330.,1320.])
E_ANCHOR_H=np.array([10.,20.,200.,400.])
IM=np.array([0,4,8]); IW=np.array([0,1,2])

def rf(x,y,q,weights):
    f=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=1)
    f.fit(np.asarray(x).reshape(-1,1),y,sample_weight=weights)
    return f.predict(np.asarray(q).reshape(-1,1))

def modulus(t,E,rho,cg):
    proxy=rho*cg**2/1e9
    y=np.r_[E['E_GPa'][IW],np.interp(E_ANCHOR_H,t,proxy)]
    pred=rf(np.r_[E['hour'][IW],E_ANCHOR_H],y,t,np.r_[np.ones(3),np.full(4,2.)])
    return proxy,pred

def fit(case,cg,E,wave_anchors=ANCHOR_V,rho0=1150.):
    """Exact active hybrid at fixed original lambda_M=.8, lambda_h=1.

    Omit unused pure/PI-only forests and the thickness forest (zero final weight).
    Full-history alpha recalibration is retained and explicitly disclosed.
    """
    t=case['time']; M=case['mass']; h=case['thick']; M0=float(M[0]); h0=float(h[0])
    alpha=fit_alpha_from_cases([case])
    best=(100.,float(max(M[IM])),float('inf'))
    for tau in [10,20,50,100,200,500,800]:
        term=fickian_fraction(t[IM],tau)
        dM=np.median((M[IM]-M0)/(term+1e-6))
        error=np.mean((fickian_model(t[IM],M0,M0+dM,tau)-M[IM])**2)
        if error<best[2]:best=(tau,M0+dM,error)
    mass_phys=fickian_model(t,M0,best[1],best[0])
    anchors=fickian_model(np.array([150.,300.]),M0,best[1],best[0])
    mass_rf=rf(np.r_[t[IM],[150.,300.]],np.r_[M[IM],anchors],t,np.r_[np.ones(3),[2.,2.]])
    mass=.2*mass_rf+.8*mass_phys
    thick=thickness_physics_linear(mass,h0,alpha,mass[0])
    dens=rom_density(mass,thick,M0,rho0,h0=h0)
    wave=rf(np.r_[cg['hour'][IW],ANCHOR_H],np.r_[cg['gv_ms'][IW],wave_anchors],t,np.r_[np.ones(3),np.full(5,2.)])
    proxy,youngs=modulus(t,E,dens,wave)
    return dict(zip(PROPS,[mass,thick,dens,wave,proxy,youngs])),{'alpha':float(alpha),'tau_h':int(best[0]),'Minf_g':float(best[1])}

def noise(y,z,u):
    s=np.sqrt(np.log1p(u*u))
    return np.asarray(y)*np.exp(s*z-.5*s*s)

def propagated(case,E,base,settings,source,delta):
    t=case['time']; q={p:v.copy() for p,v in base.items()}
    if source=='Wet mass trajectory':
        q['mass'][t>0]*=1+delta
        q['thick']=thickness_physics_linear(q['mass'],case['thick'][0],settings['alpha'],q['mass'][0])
    elif source=='Wet thickness trajectory':q['thick'][t>0]*=1+delta
    elif source=='Swelling coefficient':
        q['thick']=thickness_physics_linear(q['mass'],case['thick'][0],settings['alpha']*(1+delta),q['mass'][0])
    elif source=='Group velocity trajectory':q['cg']*=1+delta
    elif source!='Dry density':raise ValueError(source)
    rho0=1150*(1+delta) if source=='Dry density' else 1150.
    q['dens']=rom_density(q['mass'],q['thick'],case['mass'][0],rho0,h0=case['thick'][0])
    q['proxy'],q['youngs']=modulus(t,E,q['dens'],q['cg'])
    return q

def run(output_dir=None, trials=TRIALS):
    if trials < 2:
        raise ValueError("At least two trials per scenario are required")
    env,src=load_records();case=env['all_cases'][-1];cg=env['cg_data'];E=env['E_data'];t=case['time']
    # All protocols/distributions declared above, independent of test labels.
    base,settings=fit(case,cg,E)
    ref=json.loads((REFERENCE/'R2_8_ablation_results.json').read_text())['predictions']['Full reproduced hybrid']
    for p in ['mass','thick','dens','cg','youngs']:np.testing.assert_allclose(base[p],ref[p],rtol=0,atol=1e-10)
    rho_check=1150+(1000-1150)*np.maximum(0,(base['mass']-case['mass'][0])/base['mass'])
    np.testing.assert_allclose(base['dens'],rho_check,rtol=0,atol=1e-10)
    # Shared random numbers improve comparison across assumed noise levels.
    rng=np.random.default_rng(SEED)
    Z={k:rng.standard_normal((trials,n)) for k,n in [('mass',len(t)),('thick',len(t)),('cg',len(cg['hour'])),('youngs',len(E['hour'])),('wave_anchors',5),('rho0',1)]}
    mc={};summary={}
    for u in LEVELS:
        samples={p:[] for p in PROPS}; params=[]
        for j in range(trials):
            c={**case,'mass':noise(case['mass'],Z['mass'][j],u),'thick':noise(case['thick'],Z['thick'][j],u)}
            w={**cg,'gv_ms':noise(cg['gv_ms'],Z['cg'][j],u)}
            e={**E,'E_GPa':noise(E['E_GPa'],Z['youngs'][j],u)}
            q,param=fit(c,w,e,noise(ANCHOR_V,Z['wave_anchors'][j],u),float(noise([1150.],Z['rho0'][j],u)[0]))
            assert all(np.all(np.isfinite(q[p])) and np.all(q[p]>0) for p in PROPS)
            for p in PROPS:samples[p].append(q[p])
            params.append(param)
            if (j+1)%128==0:print(f'Conditional scenario {100*u:g}%: {j+1}/{trials} trials',flush=True)
        mc[str(u)]={p:np.asarray(samples[p]) for p in PROPS}
        summary[str(u)]={'parameters':params,'grid':{}}
        for p,a in mc[str(u)].items():
            lo,median,hi=np.quantile(a,[.025,.5,.975],axis=0)
            first=np.quantile(a[:trials//2],[.025,.975],axis=0)
            width=hi-lo
            summary[str(u)]['grid'][p]={'lower':lo.tolist(),'median':median.tolist(),'upper':hi.tolist(),'standard_deviation':a.std(axis=0,ddof=1).tolist(),
                'average_width':float(width.mean()),'half_vs_full_average_width_change_percent':float(100*(np.mean(first[1]-first[0])/width.mean()-1))}
    sensitivity={};sensitivity_rows=[]
    for source in ['Wet mass trajectory','Wet thickness trajectory','Swelling coefficient','Dry density','Group velocity trajectory']:
        sensitivity[source]={}
        for delta in [-.05,-.01,.01,.05]:
            q=propagated(case,E,base,settings,source,delta)
            changes={p:{'mean_absolute_change':float(np.mean(np.abs(q[p]-base[p]))),'max_absolute_change':float(np.max(np.abs(q[p]-base[p]))),
                'mean_absolute_relative_change_percent':float(100*np.mean(np.abs(q[p]/base[p]-1)))} for p in PROPS}
            sensitivity[source][str(delta)]={'changes':changes,'predictions':{p:v.tolist() for p,v in q.items()}}
            for p in PROPS:sensitivity_rows.append({'source':source,'relative_perturbation':delta,'property':p,**changes[p]})
            if source=='Group velocity trajectory':np.testing.assert_allclose(q['proxy']/base['proxy'],(1+delta)**2,rtol=0,atol=1e-12)
            if source=='Dry density':
                w=np.maximum(0,(base['mass']-case['mass'][0])/base['mass'])
                np.testing.assert_allclose(q['dens']-base['dens'],delta*1150*(1-w),rtol=0,atol=1e-10)
    # Access the reserved tests only after fitting every conditional draw and sensitivity run.
    test=env['test_data']; tt=test['time']; key={'mass':'mass','thick':'thick','cg':'gv_ms','youngs':'E_GPa'}
    test_rows=[]
    for u in LEVELS:
        summary[str(u)]['test']={}
        for p,a in mc[str(u)].items():
            estimates=np.asarray([np.interp(tt,t,row) for row in a]); lo,med,hi=np.quantile(estimates,[.025,.5,.975],axis=0)
            item={'lower':lo.tolist(),'median':med.tolist(),'upper':hi.tolist(),'average_width':float(np.mean(hi-lo))}
            if p in key:item['observations_inside_count']=int(np.sum((test[key[p]]>=lo)&(test[key[p]]<=hi)))
            summary[str(u)]['test'][p]=item
            for i,hour in enumerate(tt):test_rows.append({'assumed_input_rsd':u,'property':p,'time_h':float(hour),'nominal_prediction':float(np.interp(hour,t,base[p])),
                'scenario_lower':float(lo[i]),'scenario_median':float(med[i]),'scenario_upper':float(hi[i]),'truth':float(test[key[p]][i]) if p in key else ''})
    out=output_directory(output_dir)
    record={'protocol':'Conditional input-error scenarios and one-at-a-time coherent cascade perturbations; not calibrated confidence/prediction intervals',
        'mc_trials_per_scenario':trials,'mc_seed':SEED,'assumed_relative_standard_deviations':LEVELS,
        'distribution':'Independent mean-one lognormal multipliers per development record and supplied wave anchor; one shared dry-density draw per trial',
        'shared_dry_mass_and_thickness_draws_reused_in_normalization':True,'no_specimen_bootstrap':True,
        'frozen_settings':{'forest_trees':300,'forest_seed':42,'mass_physics_weight':.8,'thickness_physics_weight':1.,'anchor_weight':2},
        'recalibration':'Original mass tau/Minf training fit and full-history swelling alpha recomputed for each perturbed development dataset; blend weights frozen',
        'wave_solver_not_regenerated':True,'source_correlations_and_model_discrepancy_not_estimated':True,
        'test_labels_not_used_in_fitting_or_scenario_selection':True,'no_post_test_tuning':True,'time_grid_h':t.tolist(),'test_times_h':tt.tolist(),
        'nominal_settings':settings,'nominal_predictions':{p:v.tolist() for p,v in base.items()},'conditional_mc':summary,'sensitivity':sensitivity,
        'checks':{'original_full_predictions_reproduced':True,'density_identity':True,'analytic_velocity_square_and_density_blending':True,'all_trials_finite_positive':True},
        'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,Path(__file__),ROOT/'pi_surrogates.py',REFERENCE/'R2_8_ablation_results.json']}}
    (out/'R2_9_uncertainty_results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(out/'R2_9_conditional_draws.npz',**{f'rsd_{u}_{p}':a for u,props in mc.items() for p,a in props.items()})
    for path,rows in [(out/'R2_9_test_envelopes.csv',test_rows),(out/'R2_9_sensitivity.csv',sensitivity_rows)]:
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    for u in LEVELS:print('Scenario',u,{p:{'width':round(summary[str(u)]['test'][p]['average_width'],6),'inside':summary[str(u)]['test'][p].get('observations_inside_count')} for p in PROPS},flush=True)
    print('Nominal reproduction, finite-positive trials and analytic propagation checks passed.',flush=True)

if __name__=='__main__':
    parser=analysis_arguments(__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS, help="Trials per scenario; 512 reproduces the paper")
    args=parser.parse_args()
    run(args.output_dir, args.trials)
