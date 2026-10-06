"""Prospective endpoint experiment. New seeds; the predecessor is read-only.

HAC is the Bartlett lag sandwich of weighted, centered observations. This is
an applicability benchmark, not a new covariance estimator or coverage proof.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
PREDECESSOR = ROOT.parent / '2026-09-04_codex_vps_dependence' / 'probe.py'
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
spec = importlib.util.spec_from_file_location('frozen_dependence', PREDECESSOR)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.CFG = CFG.copy()  # New explicit seed/config; no file in the predecessor is edited.
sha = base.sha
atomic_json = base.atomic_json
now = base.now


def binding():
    return {'code_sha256': sha(__file__), 'config_sha256': sha(ROOT/'config.json'),
            'predecessor_code_sha256': sha(PREDECESSOR),
            'protocol_sha256': sha(PROJECT/'protocols/support_covariance_protocol_20260905.md')}


def gradient(theta):
    a, b, c = np.moveaxis(np.asarray(theta), -1, 0)
    A, B = a*(1-a), b*(1-b)
    scale = 1/np.sqrt(A*B)
    numerator = c-a*b
    return np.stack([scale*(-b+numerator*(2*a-1)/(2*A)),
                     scale*(-a+numerator*(2*b-1)/(2*B)), scale], axis=-1)


def hac(z, alpha, mean, bandwidths):
    """No lag-dependent denominator: PSD Bartlett sandwich, exact weights."""
    keep = np.flatnonzero(alpha)
    z, alpha = z[:, keep[0]:], alpha[keep[0]:]
    v = (z-mean[:, None, :])*alpha[None, :, None]
    diagonal = np.einsum('rti,rtj->rij', v, v)
    outputs = [diagonal.copy() for _ in bandwidths]
    for lag in range(1, max(bandwidths, default=0)+1):
        cross = np.einsum('rti,rtj->rij', v[:, lag:], v[:, :-lag])
        symmetric = cross + np.swapaxes(cross, -1, -2)
        for k, bandwidth in enumerate(bandwidths):
            if lag <= bandwidth:
                outputs[k] += (1-lag/(bandwidth+1))*symmetric
    return np.stack(outputs, axis=1)


def make_unit(di, ri, block, smoke):
    start = time.perf_counter()
    data = base.make_unit(di, ri, block, smoke=smoke)
    generator_seconds = time.perf_counter()-start
    _, z, mu, sigma = base.population(di)
    observations = z[data['states']]
    W = base.weights()
    means, second = data['means'], data['second']
    cov = second-means[..., :, None]*means[..., None, :]
    raw, valid = base.functional(means)
    safe = np.where(valid[..., None], means, [.5, .5, .25])
    g, H = gradient(safe), base.hessian(safe)
    all_cov = []
    timings = []
    for k in range(W.shape[1]):
        w = CFG['windows'][k//2]
        bandwidths = [0, max(1, int(w**(1/3)+1e-10)), max(1, int(np.sqrt(w)))]
        start = time.perf_counter()
        V = hac(observations, W[:, k], means[:, k], bandwidths)
        timings.append(time.perf_counter()-start)
        iid = cov[:, k]*data['weight_Q'][k]
        exact = np.broadcast_to(sigma*data['exact_K'][k], iid.shape)
        all_cov.append(np.concatenate([iid[:, None], V, exact[:, None]], axis=1))
    V = np.stack(all_cov, axis=1)  # R,K,[iid,hc0,hac1,hac2,exact],3,3
    eigen = np.linalg.eigvalsh(V)
    if eigen.min() < -1e-10:
        raise ValueError('PSD failure: '+str(eigen.min()))
    variance = np.einsum('rki,rkmij,rkj->rkm', g, V, g)
    if variance.min() < -1e-12:
        raise ValueError('negative delta variance')
    negative_residuals = np.count_nonzero(variance < 0)
    halfwidth = 1.959963984540054*np.sqrt(np.maximum(variance, 0))
    interval_low = raw[..., None]-halfwidth
    interval_high = raw[..., None]+halfwidth
    correction = .5*np.einsum('rkij,rkmji->rkm', H, V[:, :, [0, 2, 3, 4]])
    oracle_constant = .5*np.trace(base.hessian(mu) @ sigma)
    oracle = raw-oracle_constant*data['exact_K']
    unguarded = np.concatenate([raw[..., None]-correction, oracle[..., None]], axis=-1)
    a, b = means[..., 0], means[..., 1]
    support = (np.minimum.reduce([a,1-a,b,1-b]) >= CFG['support_margin']) & (
        np.minimum(b,1-b)/data['weight_Q'] >= CFG['support_count']) & valid
    guarded = np.where(support[..., None], unguarded, raw[..., None])
    estimates = np.concatenate([raw[..., None], unguarded, guarded], axis=-1)
    estimates = np.where(valid[..., None], estimates, np.nan)
    data.update(estimates=estimates, support=support, covariance_estimates=V,
                interval_low=interval_low, interval_high=interval_high,
                estimator_names=np.array(['raw','iid','hac1','hac2','exact_plugin','oracle',
                                          'iid_guard','hac1_guard','hac2_guard','exact_guard','oracle_guard']),
                interval_names=np.array(['iid','hc0','hac1','hac2','exact']),
                generator_seconds=np.array(generator_seconds), hac_seconds=np.array(timings),
                negative_variance_roundoff_count=np.array(negative_residuals))
    return data


def validate(path, receipt, identity):
    if receipt['identity'] != list(identity) or receipt['binding'] != binding() or receipt['sha256'] != sha(path):
        raise ValueError('checkpoint identity/hash mismatch')
    with np.load(path, allow_pickle=False) as d:
        R, T, K = CFG['replicates_per_block'], CFG['length'], len(CFG['windows'])*2
        assert d['states'].shape == (R,T) and np.all(d['states'] < 6)
        assert d['estimates'].shape == (R,K,11)
        mask = np.broadcast_to(d['valid'][...,None], d['estimates'].shape)
        assert np.isfinite(d['estimates'][mask]).all() and np.isnan(d['estimates'][~mask]).all()
        assert np.isfinite(d['covariance_estimates']).all()
        assert d['interval_low'].shape == (R,K,5)
        assert np.array_equal(np.isfinite(d['interval_low']), np.broadcast_to(d['valid'][...,None], (R,K,5)))


def tests():
    rng = np.random.default_rng(55501)
    theta = np.array([.4,.3,.18])
    e = np.eye(3)*1e-5
    numerical_g = np.array([(base.functional(theta+h)[0]-base.functional(theta-h)[0])/2e-5 for h in e])
    numerical_H = np.stack([(gradient(theta+h)-gradient(theta-h))/2e-5 for h in e], axis=-1)
    np.testing.assert_allclose(gradient(theta), numerical_g, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(base.hessian(theta), numerical_H, atol=1e-7, rtol=1e-7)
    z = rng.random((4,31,3))
    alpha = rng.random(31); alpha /= alpha.sum()
    mean = np.einsum('t,rti->ri', alpha, z)
    V = hac(z, alpha, mean, [0,3,9])
    v = (z-mean[:,None])*alpha[None,:,None]
    for k,L in enumerate([0,3,9]):
        B = np.maximum(1-np.abs(np.arange(31)[:,None]-np.arange(31)[None,:])/(L+1),0)
        direct = np.stack([row.T@B@row for row in v])
        np.testing.assert_allclose(V[:,k],direct,atol=1e-14,rtol=1e-12)
    assert np.linalg.eigvalsh(V).min() >= -1e-14
    for di in range(3):
        probs,z,mu,sigma = base.population(di)
        np.testing.assert_allclose(mu[:2], CFG['prevalences'][di],atol=1e-14)
    W=base.weights()
    np.testing.assert_allclose(base.covariance_factors(W,0),(W*W).sum(0),atol=1e-14)
    return {'status':'PASS','gradient_finite_difference':True,'hessian_finite_difference':True,
            'hac_dense_matrix':True,'hac_psd':True,'population_calibration':True,'iid_limit':True,
            'binding':binding(),'created_at':now()}


def run(args):
    out = ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    jobs=[(di,ri,b) for di in range(3) for ri in range(4) for b in range(CFG['blocks'])]
    if args.smoke: jobs=jobs[:args.limit]
    status={'status':'running','planned':len(jobs),'completed':0,'pid':os.getpid(),
            'binding':binding(),'command':sys.argv,'cwd':os.getcwd(),'updated_at':now(),'smoke':args.smoke}
    atomic_json(out/'environment.json',{'python':sys.version,'numpy':np.__version__,
                'os':platform.system(),'arch':platform.machine(),'hostname':platform.node(),
                'platform':platform.platform(),'binding':binding(),'created_at':now()})
    start=time.monotonic(); done=threading.Event()
    def beat():
        while not done.is_set():
            status.update(updated_at=now(),elapsed_seconds=time.monotonic()-start,
                          cpu_seconds=time.process_time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            atomic_json(out/'heartbeat.json',dict(status)); done.wait(2)
    thread=threading.Thread(target=beat,daemon=True); thread.start()
    def alarm(sig,frame): raise TimeoutError('signal '+str(sig))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]: signal.signal(sig,alarm)
    code=0
    try:
        for identity in jobs:
            name='%02d_%02d_%03d'%identity
            path=out/'raw'/(name+'.npz'); rec=path.with_suffix('.json')
            if rec.exists():
                validate(path,json.loads(rec.read_text()),identity)
                status['completed']+=1
                continue
            if path.exists(): raise ValueError('orphan uncommitted NPZ')
            if shutil.disk_usage(out).free < CFG['disk_min_gib']*2**30: raise RuntimeError('disk floor')
            if time.monotonic()-start > CFG['watchdog_seconds']: raise TimeoutError('whole-run watchdog')
            status['active_unit']=name; unit_start=time.perf_counter()
            signal.alarm(CFG['unit_timeout_seconds'])
            try: data=make_unit(*identity,args.smoke)
            finally: signal.alarm(0)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f,**data); f.flush(); os.fsync(f.fileno())
            os.replace(path.with_suffix('.tmp'),path)
            receipt={'identity':list(identity),'binding':binding(),'sha256':sha(path),
                     'created_at':now(),'seconds':time.perf_counter()-unit_start,'smoke':args.smoke}
            validate(path,receipt,identity); atomic_json(rec,receipt)
            status['completed']+=1
            print(json.dumps({'unit':name,'completed':status['completed'],'seconds':receipt['seconds']}),flush=True)
            if sum(p.stat().st_size for p in path.parent.glob('*')) > CFG['max_output_gib']*2**30: raise RuntimeError('output ceiling')
            if args.fail_after and status['completed'] >= args.fail_after: raise InterruptedError('controlled smoke interruption')
        status['status']='completed'
    except InterruptedError as e: code=75; status.update(status='controlled_interruption',error=str(e))
    except TimeoutError as e: code=124; status.update(status='timed_out',error=str(e)); traceback.print_exc()
    except Exception as e: code=1; status.update(status='failed',error=str(e)); traceback.print_exc()
    finally:
        done.set();thread.join(5)
        status.update(exit_code=code,updated_at=now(),elapsed_seconds=time.monotonic()-start,
                      cpu_seconds=time.process_time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out/'terminal_status.json',status)
        atomic_json(out/'attempts'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),status)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['test','run'])
    parser.add_argument('--output',default='results')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--limit',type=int,default=3)
    parser.add_argument('--fail-after',type=int,default=0)
    args=parser.parse_args()
    if args.mode=='test':
        report=tests();atomic_json(ROOT/'tests.json',report);print(json.dumps(report))
    else: sys.exit(run(args))
