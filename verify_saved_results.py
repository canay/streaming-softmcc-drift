"""Read-only checks of shipped replication-unit and aggregate results.

No simulation, training, input-data download, or result-file write occurs.
"""
from pathlib import Path
import csv, hashlib, json, math, statistics

ROOT=Path(__file__).resolve().parent
def rows(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-13)
matched=Path('studies/2026-08-26_codex_local_kais_memory_matched')
raw=rows(matched/'raw/synthetic_metrics.csv'); primary=rows(matched/'evidence/primary_paired_effects.csv')
assert len({(r['regime'],r['seed']) for r in raw})==120
effects=[]
for r in primary:
    w=int(r['w']); cells={}
    for t in raw:
        if t['regime']=='abrupt' and int(t['w'])==w and t['kernel'] in ['window','fading']:
            cells.setdefault(t['seed'],{})[t['kernel']]=float(t['iae1000'])
    assert len(cells)==30 and all(set(v)=={'window','fading'} for v in cells.values())
    delta=[v['window']-v['fading'] for v in cells.values()]
    assert close(statistics.median(delta),r['median_window_minus_fading_iae1000'])
    assert sum(x>0 for x in delta)==int(r['fading_better_count'])
    effects.append({'w':w,'median':statistics.median(delta),'fading_better':sum(x>0 for x in delta)})
ordered=rows(matched/'evidence/ordered_matched_contrasts.csv'); assert len(ordered)==33
signs=[float(r['window_minus_fading']) for r in ordered]
assert sum(x<0 for x in signs)==15 and sum(x>0 for x in signs)==18
bias=ROOT/'studies/2026-08-27_claude_local_finite_ess_bias/evidence'
s1=json.loads((bias/'stage1_raw.json').read_text());s4=json.loads((bias/'stage4_raw.json').read_text());s4b=json.loads((bias/'stage4b_raw.json').read_text())
assert len(s1['rows'])==500 and len(s4['rows'])+len(s4b['rows'])==2500
mean20=statistics.mean(r['estimators']['win20']['dev_pop'] for r in s1['rows'])
assert round(mean20,5)==-0.00696
dep=ROOT/'studies/2026-09-04_codex_vps_dependence';sup=ROOT/'studies/2026-09-05_codex_vps_support_covariance'
d=json.loads((dep/'results/analysis.json').read_text());u=json.loads((sup/'results/analysis.json').read_text())
assert len(d['moment_conditions'])==120 and len(d['score_conditions'])==600
assert len(u['estimator_rows'])==1320 and len(u['interval_rows'])==600
assert d['primary']['n_trajectories']==4096 and u['primary']['n_total']==1024
assert round(d['primary']['empirical_ratio'],3)==.878 and round(d['primary']['exact_ratio'],3)==.881
assert close(u['primary']['difference'],u['primary']['hac1_coverage']-u['primary']['iid_coverage'])
assert u['primary']['difference']>=.05 and u['primary']['ci95'][0]>0 and u['primary']['hac1_coverage']<.95
condition=lambda r:(r['prevalence'],r['rho'],r['w'],r['kernel'])
raw_rmse={condition(r):r['rmse'] for r in u['estimator_rows'] if r['estimator']=='raw' and r['undefined_fraction']<=.01+1e-12}
assert len(raw_rmse)==94
counts={}
for method,expected in [('iid',48),('hac1',16),('iid_guard',38)]:
    count=sum(r['rmse']<raw_rmse[condition(r)] for r in u['estimator_rows'] if r['estimator']==method and condition(r) in raw_rmse)
    assert count==expected;counts[method]=count
extreme={r['estimator']:r for r in u['estimator_rows'] if condition(r)==(.5,.9,20,'fading')}
assert extreme['exact_plugin']['rmse']>1e7 and extreme['exact_guard']['rmse']>extreme['raw']['rmse']
for folder,n in [(dep,384),(sup,192)]:
    identity=json.loads((folder/'original_generated_artifact_identities.json').read_text());assert len(identity['records'])==n
    cfg=hashlib.sha256((folder/'config.json').read_bytes()).hexdigest()
    assert all(r['binding']['config_sha256']==cfg for r in identity['records'])
manifest=ROOT/'SHA256SUMS.txt';checks=0
if manifest.exists():
    for line in manifest.read_text().splitlines():
        digest,relative=line.split('  ',1);p=(ROOT/relative).resolve();assert p.is_relative_to(ROOT) and p.is_file()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,relative;checks+=1
print(json.dumps({'status':'PASS','scope':'Saved-result consistency and shipped-file identity; not a full experiment rerun.',
 'matched_effects':effects,
 'ordered_direction_counts':{
   'per_kernel_averaged':{'window':15,'fading':18},
   'paired_common_anchor':{'window':9,'fading':24},
   'note':'Two estimands, not a disagreement. The archived run averages each kernel over its own valid anchors; the manuscript reports the paired form over anchors where both kernels are defined, which reverses 8 of the 33 contrasts. code/reanalysis_20260920/reanalysis.py recomputes BOTH from the shipped ordered_anchors.csv.'},
 'finite_bias_window20':mean20,
 'dependence_primary':d['primary'],'support_primary':u['primary'],'rmse_improvement_counts_of_94':counts,
 'shipped_hash_checks':checks},indent=2))
