"""Read every committed unit. No simulation, tuning, or selective inclusion."""
import json
import time
from pathlib import Path
import numpy as np
import run_support as r

root=Path(__file__).resolve().parent
start=time.perf_counter()
rows=[]; intervals=[]; inventory=[]; primary=None
for di in range(3):
    for ri in range(4):
        chunks=[]
        for block in range(r.CFG['blocks']):
            name='%02d_%02d_%03d'%(di,ri,block)
            path=root/'results/raw'/(name+'.npz')
            receipt=json.loads(path.with_suffix('.json').read_text())
            r.validate(path,receipt,(di,ri,block))
            if receipt['smoke']: raise ValueError('smoke promoted')
            with np.load(path,allow_pickle=False) as d: chunks.append({k:d[k].copy() for k in d.files})
            inventory.append({'path':path.relative_to(root).as_posix(),'sha256':r.sha(path),
                              'receipt_sha256':r.sha(path.with_suffix('.json'))})
        estimates=np.concatenate([d['estimates'] for d in chunks])
        valid=np.concatenate([d['valid'] for d in chunks])
        support=np.concatenate([d['support'] for d in chunks])
        low=np.concatenate([d['interval_low'] for d in chunks])
        high=np.concatenate([d['interval_high'] for d in chunks])
        _,_,mu,_=r.base.population(di)
        target=float(r.base.functional(mu)[0])
        covered=(low<=target)&(high>=target)
        R=len(valid)
        for k in range(valid.shape[1]):
            common={'prevalence':r.CFG['prevalences'][di],'rho':r.CFG['rhos'][ri],
                    'w':r.CFG['windows'][k//2],'kernel':['window','fading'][k%2],
                    'target':target,'n_total':R,'n_defined':int(valid[:,k].sum()),
                    'undefined_fraction':float(1-valid[:,k].mean()),
                    'support_accept_fraction':float(support[:,k].mean())}
            for j,name in enumerate(chunks[0]['estimator_names']):
                values=estimates[:,k,j][valid[:,k]]
                error=values-target
                row={**common,'estimator':str(name)}
                row.update(bias=float(error.mean()) if len(error) else None,
                           rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                           standard_deviation=float(values.std(ddof=1)) if len(values)>1 else None,
                           out_of_range_fraction=float((np.abs(values)>1).mean()) if len(values) else None,
                           mse_mc_se=float(np.std(error**2,ddof=1)/np.sqrt(len(error))) if len(error)>1 else None)
                rows.append(row)
            for j,name in enumerate(chunks[0]['interval_names']):
                take=valid[:,k]
                intervals.append({**common,'interval':str(name),
                    'coverage_joint':float(covered[:,k,j].mean()),
                    'coverage_conditional':float(covered[:,k,j][take].mean()) if take.any() else None,
                    'mean_length_conditional':float((high[:,k,j]-low[:,k,j])[take].mean()) if take.any() else None,
                    'out_of_range_fraction':float(((low[:,k,j]<-1)|(high[:,k,j]>1))[take].mean()) if take.any() else None})
        if di==0 and ri==2:
            k=5
            difference=covered[:,k,2].astype(float)-covered[:,k,0].astype(float)
            rng=np.random.default_rng(r.CFG['bootstrap_seed'])
            boots=np.array([difference[rng.integers(0,R,R)].mean() for _ in range(r.CFG['bootstrap_resamples'])])
            ci=np.quantile(boots,[.025,.975])
            delta=float(difference.mean())
            primary={'prevalence':.5,'rho':.6,'w':100,'kernel':'fading','endpoint':'joint_coverage_hac1_minus_iid',
                     'difference':delta,'ci95':ci.tolist(),'paired_mc_se':float(difference.std(ddof=1)/np.sqrt(R)),
                     'iid_coverage':float(covered[:,k,0].mean()),'hac1_coverage':float(covered[:,k,2].mean()),
                     'n_total':R,'n_defined':int(valid[:,k].sum()),
                     'verdict':'SUPPORTED_COVERAGE_GAIN' if delta>=.05 and ci[0]>0 else 'NOT_SUPPORTED'}
assert primary is not None and len(inventory)==192 and len(rows)==1320 and len(intervals)==600
r.atomic_json(root/'results/analysis.json',{'status':'ANALYZED_NOT_INDEPENDENTLY_VERIFIED',
    'created_at':r.now(),'binding':r.binding(),'analysis_code_sha256':r.sha(__file__),
    'raw_inventory':inventory,'primary':primary,'estimator_rows':rows,'interval_rows':intervals,
    'elapsed_seconds':time.perf_counter()-start,'interpretation':'No general superiority or finite-memory coverage guarantee.'})
print(json.dumps({'primary':primary,'raw_units':len(inventory),'score_rows':len(rows),'interval_rows':len(intervals)}))
