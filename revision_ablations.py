"""R2.8 retrospective component ablations with frozen settings and test scoring.

The oracle-upstream variant is a diagnostic using development measurements,
not a prospective forecasting model. No test label enters fitting or selection.
"""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import scipy,sklearn
from sklearn.ensemble import RandomForestRegressor
from revision_baselines import load_records
from pi_surrogates import (MassSurrogate,ThicknessSurrogate,CGSurrogate,YoungsSurrogate,
    PISpec1D,fit_alpha_from_cases,rom_density,thickness_physics_linear)

from repo_io import ROOT, REFERENCE, output_directory, analysis_arguments
VARIANTS={
    'Full reproduced hybrid':{},
    'Thickness time input':{'time_thickness':True},
    'Modulus RF pseudo labels':{'rf_proxy':True},
    'Both learning-feature substitutions':{'time_thickness':True,'rf_proxy':True},
    'No PI anchor points':{'no_anchors':True},
    'Measured upstream diagnostic':{'oracle':True},
    'Fixed half blending':{'lambdas':[.5]},
    'No blending RF outputs':{'lambdas':[0.]},
}

def forest(x,y,x_eval,weights=None):
    model=RandomForestRegressor(n_estimators=300,random_state=42)
    model.fit(np.asarray(x).reshape(-1,1),y,sample_weight=weights)
    return model.predict(np.asarray(x_eval).reshape(-1,1))

def fit_variant(case,cg,E,opts):
    # Access only development observations; reserved test data are not arguments.
    t=case['time']; M=case['mass']; h=case['thick']
    im=np.array([0,4,8]); iw=np.array([0,1,2]); vm=np.setdiff1d(np.arange(len(t)),im)
    alpha=fit_alpha_from_cases([case])  # retain original full-history calibration
    no=opts.get('no_anchors',False); lambdas=opts.get('lambdas')
    pm=None if no else PISpec1D(np.array([150.,300.]))
    ph=None if no else PISpec1D(np.array([250.,300.]))
    pw=None if no else PISpec1D(np.array([9.47,47.36,270.,360.,528.]),values=np.array([2208.,1776.,1360.,1330.,1320.]))
    pe=None if no else PISpec1D(np.array([10.,20.,200.,400.]))
    mass=MassSurrogate().train(t,M,im,pm,2.,lambdas=lambdas)
    wave=CGSurrogate().train(t,cg['hour'],cg['gv_ms'],iw,pw,2.)
    upstream_mass=M if opts.get('oracle') else mass.pred_hybrid
    if opts.get('time_thickness'):
        # Change only the thickness forest input from predicted mass to time.
        # Keep the mass-driven swelling baseline, anchor labels/count/weight,
        # validation rule and all upstream/downstream equations.
        h_phys=thickness_physics_linear(upstream_mass,h[0],alpha,upstream_mass[0])
        x=t[im]; y=h[im]; weights=np.ones(3)
        if ph is not None:
            vals=thickness_physics_linear(np.interp(ph.hours,t,upstream_mass),h[0],alpha,upstream_mass[0])
            x=np.r_[x,ph.hours]; y=np.r_[y,vals]; weights=np.r_[weights,np.full(len(vals),2.)]
        h_rf=forest(x,y,t,weights)
        grid=np.linspace(0,1,21) if lambdas is None else lambdas
        choices=[((1-lam)*h_rf+lam*h_phys,float(lam)) for lam in grid]
        h_pred,h_lambda=min(choices,key=lambda z:np.mean(np.abs(z[0][vm]-h[vm])))
    else:
        thick=ThicknessSurrogate().train(t,h,upstream_mass,upstream_mass,im,alpha,ph,2.,lambdas=lambdas)
        h_pred=thick.pred_hybrid;h_lambda=thick.lambda_star
    rho=rom_density(mass.pred_hybrid,h_pred,M[0],1150.,h0=h[0])
    rho_up=rom_density(M,h,M[0],1150.,h0=h[0]) if opts.get('oracle') else rho
    cg_up=np.interp(t,cg['hour'],cg['gv_ms']) if opts.get('oracle') else wave.pred_hybrid
    if opts.get('rf_proxy'):
        # Hold four modulus anchor times and weights constant while replacing
        # rho*cg^2 labels with labels from a training-only time forest.
        base=forest(E['hour'][iw],E['E_GPa'][iw],t)
        x=E['hour'][iw];y=E['E_GPa'][iw];weights=np.ones(3)
        if pe is not None:
            vals=np.interp(pe.hours,t,base)
            x=np.r_[x,pe.hours];y=np.r_[y,vals];weights=np.r_[weights,np.full(len(vals),2.)]
        youngs=forest(x,y,t,weights)
    else:
        er=YoungsSurrogate().train(t,E['hour'],E['E_GPa'],iw,rho_up,cg_up,rho_up,cg_up,pe,2.)
        youngs=er.pred_hybrid
    predictions={'mass':mass.pred_hybrid,'thick':h_pred,'dens':rho,'cg':wave.pred_hybrid,'youngs':youngs}
    return predictions,{'alpha_full_history':float(alpha),'mass_lambda':float(mass.lambda_star),'thick_lambda':float(h_lambda),
        'mass_physics_fit':mass.params,'mass_anchor_count':0 if pm is None else len(pm.hours),
        'thick_anchor_count':0 if ph is None else len(ph.hours),'cg_anchor_count':0 if pw is None else len(pw.hours),
        'modulus_anchor_count':0 if pe is None else len(pe.hours),'intervention':opts,
        'oracle_uses_development_measurements':bool(opts.get('oracle'))}

def run(output_dir=None):
    env,source=load_records();case=env['all_cases'][-1];cg=env['cg_data'];E=env['E_data'];grid=case['time']
    # Fit every prespecified intervention before access to test labels for scoring.
    fitted={name:fit_variant(case,cg,E,opts) for name,opts in VARIANTS.items()}
    stored=json.loads((REFERENCE/'R1c_corrected_metrics.json').read_text(encoding='utf-8'))
    frozen=stored['configurations']['active_three_vs_seven']
    for p in ['mass','thick','dens','cg','youngs']:
        np.testing.assert_allclose(fitted['Full reproduced hybrid'][0][p],frozen['predictions'][p]['hybrid'],rtol=0,atol=1e-10)
    assert all(np.array_equal(fitted[n][0]['mass'],fitted['Full reproduced hybrid'][0]['mass']) for n in ['Thickness time input','Modulus RF pseudo labels','Both learning-feature substitutions','Measured upstream diagnostic'])
    assert all(np.array_equal(fitted[n][0]['cg'],fitted['Full reproduced hybrid'][0]['cg']) for n in VARIANTS if n!='No PI anchor points')
    scores={};rows=[];test=env['test_data'];tt=test['time']; keys={'mass':'mass','thick':'thick','cg':'gv_ms','youngs':'E_GPa'}
    for name,(pred,settings) in fitted.items():
        scores[name]={}
        for p,k in keys.items():
            estimate=np.interp(tt,grid,pred[p]);truth=test[k]
            obs_t=cg['hour'] if p=='cg' else E['hour'] if p=='youngs' else grid
            obs_y=cg['gv_ms'] if p=='cg' else E['E_GPa'] if p=='youngs' else case[p]
            horizon=90. if p in ['cg','youngs'] else 192.
            later=obs_t>horizon
            scores[name][p]={'test_mae':float(np.mean(np.abs(estimate-truth))),
                'test_mae_change_from_full':float(np.mean(np.abs(estimate-truth))-frozen['test'][p]['hybrid']),
                'later_development_mae':float(np.mean(np.abs(np.interp(obs_t[later],grid,pred[p])-obs_y[later]))),
                'later_development_count':int(later.sum()),'test_predictions':estimate.tolist()}
            for i,time in enumerate(tt):rows.append({'variant':name,'property':p,'time_h':float(time),'truth':float(truth[i]),'prediction':float(estimate[i]),'absolute_error':float(abs(estimate[i]-truth[i]))})
        scores[name]['derived_density_later_mae']=float(np.mean(np.abs(pred['dens'][grid>192]-rom_density(case['mass'],case['thick'],case['mass'][0],1150.,h0=case['thick'][0])[grid>192])))
        assert all(np.all(np.isfinite(v)) for v in pred.values())
    record={'protocol':'Retrospective one-component/group interventions; test data used only after all fits; no post-score tuning',
        'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'scores':scores,'settings':{n:x[1] for n,x in fitted.items()},
        'predictions':{n:{p:v.tolist() for p,v in x[0].items()} for n,x in fitted.items()},'time_grid_h':grid.tolist(),
        'test_times_h':tt.tolist(),'training_times_h':{'mass_thick':[0,96,192],'cg_modulus':[0,24,90]},
        'training_record_count_per_property':3,'forest_trees':300,'seed':42,'anchor_weight':2,
        'validation_blend_grid':np.linspace(0,1,21).tolist(),
        'validation_reselected_for_every_variant_except_fixed_weights':True,
        'full_history_alpha_and_supplied_wave_anchor_provenance_retained':True,
        'oracle_not_deployable':True,'oracle_inputs_include_later_development_observations':True,
        'density_reference_is_derived_not_independently_measured':True,
        'no_test_labels_in_fitting_or_selection':True,'no_post_test_variant_or_parameter_selection':True,
        'checks':{'full_reference_predictions_exactly_reproduced':True,'unaffected_upstream_stages_identical':True,'finite_predictions':True},
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__),ROOT/'pi_surrogates.py',REFERENCE/'R1c_corrected_metrics.json']}}
    out=output_directory(output_dir);(out/'R2_8_ablation_results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    with (out/'R2_8_test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for n,s in scores.items():print(n,{p:round(s[p]['test_mae'],6) for p in keys})
    print('Full predictions reproduced; unaffected-stage invariants and finite predictions checked.')

if __name__=='__main__':
    args=analysis_arguments(__doc__).parse_args()
    run(args.output_dir)
