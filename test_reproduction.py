"""Regression checks against the manuscript's archived numerical records."""
import json
import unittest
import numpy as np
from main_run import CONFIGURATIONS, run_configuration, case5, cg_data, E_data, test_data
from pi_surrogates import rom_density
from repo_io import REFERENCE
from revision_uncertainty import fit, noise, propagated, SEED, LEVELS, ANCHOR_V, PROPS


def record(name):
    return json.loads((REFERENCE/name).read_text(encoding='utf-8'))


class ReproductionTests(unittest.TestCase):
    def test_three_core_configurations(self):
        reference=record('R1c_corrected_metrics.json')['configurations']
        for name in CONFIGURATIONS:
            with self.subTest(configuration=name):
                output=run_configuration(name)
                expected=reference[name]
                for prop in ['mass','thick','dens','cg','youngs']:
                    result=getattr(output,prop)
                    for field,attribute in [('pure','pred_pure'),('physics','pred_phys'),('hybrid','pred_hybrid')]:
                        np.testing.assert_allclose(getattr(result,attribute),expected['predictions'][prop][field],rtol=0,atol=1e-10)
                    self.assertAlmostEqual(result.mae_val,expected['reported_scores'][prop]['hybrid'],places=10)
                    self.assertAlmostEqual(result.mae_pure,expected['reported_scores'][prop]['pure'],places=10)
                if name=='active_three_vs_seven':
                    for prop,key in [('mass','mass'),('thick','thick'),('cg','gv_ms'),('youngs','E_GPa')]:
                        result=getattr(output,prop)
                        for path in ['pure','hybrid']:
                            prediction=np.interp(test_data['time'],result.t,getattr(result,'pred_'+path))
                            self.assertAlmostEqual(float(np.mean(abs(prediction-test_data[key]))),expected['test'][prop][path],places=10)

    def test_density_law_units_and_geometry(self):
        m=case5['mass']; h=case5['thick']; m0=m[0]
        expected=1150.-150.*np.maximum(0,(m-m0)/m)
        np.testing.assert_allclose(rom_density(m,h,m0,h0=h[0]),expected,rtol=0,atol=1e-10)
        np.testing.assert_allclose(rom_density(m/1000,h*2,m0/1000,h0=h[0]*2),expected,rtol=0,atol=1e-10)
        self.assertEqual(rom_density([m0-1],None,m0)[0],1150.)
        for invalid in [0.,-1.,np.nan,np.inf]:
            with self.assertRaises(ValueError):rom_density([invalid],None,m0)

    def test_conditional_nominal_and_first_draw_each_level(self):
        archived=record('R2_9_uncertainty_results.json')
        base,settings=fit(case5,cg_data,E_data)
        for prop in PROPS:
            np.testing.assert_allclose(base[prop],archived['nominal_predictions'][prop],rtol=0,atol=1e-10)
        rng=np.random.default_rng(SEED)
        n=archived['mc_trials_per_scenario']
        z={k:rng.standard_normal((n,size)) for k,size in [('mass',23),('thick',23),('cg',7),('youngs',7),('wave_anchors',5),('rho0',1)]}
        with np.load(REFERENCE/'R2_9_conditional_draws.npz') as saved:
            for level in LEVELS:
                case={**case5,'mass':noise(case5['mass'],z['mass'][0],level),'thick':noise(case5['thick'],z['thick'][0],level)}
                cg={**cg_data,'gv_ms':noise(cg_data['gv_ms'],z['cg'][0],level)}
                e={**E_data,'E_GPa':noise(E_data['E_GPa'],z['youngs'][0],level)}
                result,_=fit(case,cg,e,noise(ANCHOR_V,z['wave_anchors'][0],level),float(noise([1150.],z['rho0'][0],level)[0]))
                for prop in PROPS:
                    np.testing.assert_allclose(result[prop],saved[f'rsd_{level}_{prop}'][0],rtol=0,atol=1e-10)

    def test_all_archived_conditional_bands(self):
        archived=record('R2_9_uncertainty_results.json')
        with np.load(REFERENCE/'R2_9_conditional_draws.npz') as saved:
            for level in LEVELS:
                for prop in PROPS:
                    draws=saved[f'rsd_{level}_{prop}']
                    self.assertEqual(draws.shape,(512,23))
                    self.assertTrue(np.all(np.isfinite(draws)) and np.all(draws>0))
                    grid=np.quantile(draws,[.025,.5,.975],axis=0)
                    estimates=np.array([np.interp(test_data['time'],case5['time'],row) for row in draws])
                    test=np.quantile(estimates,[.025,.5,.975],axis=0)
                    for index,label in enumerate(['lower','median','upper']):
                        np.testing.assert_allclose(grid[index],archived['conditional_mc'][str(level)]['grid'][prop][label],rtol=0,atol=1e-10)
                        np.testing.assert_allclose(test[index],archived['conditional_mc'][str(level)]['test'][prop][label],rtol=0,atol=1e-10)

    def test_all_sensitivity_interventions(self):
        archived=record('R2_9_uncertainty_results.json')
        base,settings=fit(case5,cg_data,E_data)
        for source,perturbations in archived['sensitivity'].items():
            for delta,detail in perturbations.items():
                q=propagated(case5,E_data,base,settings,source,float(delta))
                for prop in PROPS:
                    np.testing.assert_allclose(q[prop],detail['predictions'][prop],rtol=0,atol=1e-10)
                if source=='Group velocity trajectory':
                    np.testing.assert_allclose(q['proxy']/base['proxy'],(1+float(delta))**2,rtol=0,atol=1e-12)


if __name__=='__main__':
    unittest.main()
