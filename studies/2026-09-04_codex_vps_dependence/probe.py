"""Frozen refresh-chain mechanism experiment; no manuscript edits.

run writes atomic per-block states/moments/scores with a hash-bound receipt.
analyze consumes all receipts, never regenerates data. test uses fixtures only.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def binding():
    return {'code_sha256': sha(__file__), 'config_sha256': sha(ROOT / 'config.json')}


def population(di):
    p = np.asarray(CFG['scores'])
    prevalence, mid = CFG['prevalences'][di], CFG['middle_masses'][di]
    hi = (prevalence - .01 - .24 * mid) / .89
    masses = np.array([1 - mid - hi, mid, hi])
    probs = np.column_stack([masses * (1-p), masses*p]).ravel()
    ps = np.repeat(p, 2)
    ys = np.tile([0., 1.], 3)
    z = np.column_stack([ps, ys, ps*ys])
    mu = probs @ z
    sigma = (z.T * probs) @ z - np.outer(mu, mu)
    return probs, z, mu, sigma


def functional(theta):
    a, b, c = np.moveaxis(np.asarray(theta), -1, 0)
    good = (a > CFG['margin']) & (a < 1-CFG['margin']) & (b > CFG['margin']) & (b < 1-CFG['margin'])
    aa, bb = np.where(good, a, .5), np.where(good, b, .5)
    out = (c-aa*bb)/np.sqrt(aa*(1-aa)*bb*(1-bb))
    return np.where(good, out, np.nan), good


def hessian(theta):
    a, b, c = np.moveaxis(np.asarray(theta), -1, 0)
    A, B = a*(1-a), b*(1-b)
    g, n = 1/np.sqrt(A*B), c-a*b
    ra, rb = (2*a-1)/(2*A), (2*b-1)/(2*B)
    da, db = .5/a**2 + .5/(1-a)**2, .5/b**2 + .5/(1-b)**2
    h = np.zeros(np.shape(a)+(3, 3))
    h[..., 0, 0] = g*(n*(ra**2+da)-2*b*ra)
    h[..., 1, 1] = g*(n*(rb**2+db)-2*a*rb)
    h[..., 0, 1] = h[..., 1, 0] = g*(n*ra*rb-b*rb-a*ra-1)
    h[..., 0, 2] = h[..., 2, 0] = g*ra
    h[..., 1, 2] = h[..., 2, 1] = g*rb
    return h


def weights(length=None):
    length = length or CFG['length']
    cols = []
    for w in CFG['windows']:
        win = np.zeros(length)
        win[-min(w,length):] = 1/min(w,length)
        lam = (w-1)/(w+1)
        fade = lam**np.arange(length-1, -1, -1)
        fade /= fade.sum()
        cols.extend([win, fade])
    return np.stack(cols, axis=1)


def covariance_factors(W, rho):
    previous = np.zeros(W.shape[1])
    out = np.zeros(W.shape[1])
    for alpha in W:
        previous *= rho
        out += alpha*alpha + 2*alpha*previous
        previous += alpha
    return out


def make_unit(di, ri, block, smoke=False):
    rng = np.random.default_rng(np.random.SeedSequence([CFG['master_seed'], di, ri, block, int(smoke)]))
    R, T = CFG['replicates_per_block'], CFG['length']
    probs, z, mu, sigma = population(di)
    rho = CFG['rhos'][ri]
    fresh = rng.choice(6, size=(R,T), p=probs).astype(np.uint8)
    refresh = rng.random((R,T)) >= rho
    refresh[:, 0] = True
    indices = np.maximum.accumulate(np.where(refresh, np.arange(T), 0), axis=1)
    states = np.take_along_axis(fresh, indices, axis=1)
    W = weights()
    histogram = np.stack([(states == s).astype(float) @ W for s in range(6)], axis=-1)
    means = histogram @ z
    second = np.einsum('rks,si,sj->rkij', histogram, z, z)
    sample_cov = second - means[..., :, None]*means[..., None, :]
    raw, valid = functional(means)
    safe_means = np.where(valid[...,None], means, np.array([.5,.5,.25]))
    C_hat = .5*np.einsum('rkij,rkji->rk', hessian(safe_means), sample_cov)
    C = .5*np.trace(hessian(mu) @ sigma)
    K = covariance_factors(W, rho)
    Q = (W*W).sum(axis=0)
    estimators = np.stack([raw, raw-C_hat*Q, raw-C_hat*K, raw-C*K], axis=-1)
    # NPZ uses NaN with an explicit mask; JSON uses null in summaries only.
    estimators = np.where(valid[...,None], estimators, np.nan)
    return dict(states=states, means=means, second=second, estimates=estimators,
                valid=valid, exact_mu=mu, exact_sigma=sigma, exact_K=K, weight_Q=Q)


def validate_unit(path, receipt, expected, bind):
    if receipt['identity'] != expected or receipt['binding'] != bind or receipt['sha256'] != sha(path):
        raise ValueError('checkpoint identity/hash mismatch')
    with np.load(path, allow_pickle=False) as d:
        R, T, J = CFG['replicates_per_block'], CFG['length'], 2*len(CFG['windows'])
        assert d['states'].shape == (R,T) and d['states'].dtype == np.uint8
        assert np.all(d['states'] < 6)
        assert d['means'].shape == (R,J,3) and np.isfinite(d['means']).all()
        assert d['second'].shape == (R,J,3,3) and np.isfinite(d['second']).all()
        assert d['estimates'].shape == (R,J,4)
        mask = np.broadcast_to(d['valid'][...,None], d['estimates'].shape)
        assert np.isfinite(d['estimates'][mask]).all()
        assert np.isnan(d['estimates'][~mask]).all()
        assert np.max(np.abs(d['estimates'][...,0][d['valid']]), initial=0) <= 1+1e-10
        assert np.isfinite(d['exact_K']).all() and np.all(d['exact_K'] > 0)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bind = binding()
    started = time.monotonic()
    done = threading.Event()
    progress = {'status':'running','completed':0,'planned':384 if not args.smoke else args.limit,
                'pid':os.getpid(),'updated_at':now(),'binding':bind,'active_unit':None}
    def heartbeat():
        while not done.is_set():
            progress.update(updated_at=now(), elapsed_seconds=time.monotonic()-started)
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                child = resource.getrusage(resource.RUSAGE_CHILDREN)
                progress.update(process_tree_cpu_seconds=usage.ru_utime+usage.ru_stime+child.ru_utime+child.ru_stime,
                                peak_rss_kib=usage.ru_maxrss, sampled_pid_scope='python worker and reaped children; no child compute processes')
            except ImportError:
                progress['process_tree_cpu_seconds'] = time.process_time()
            atomic_json(out/'heartbeat.json', dict(progress))
            done.wait(1)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    def terminate(signum, _frame):
        raise TimeoutError('signal '+str(signum))
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, terminate)
    atomic_json(out/'environment.json', {'created_at':now(),'python':sys.version,'numpy':np.__version__,
                'platform':platform.platform(),'hostname':platform.node(),'pid':os.getpid(),'binding':bind})
    jobs = [(di,ri,b) for di in range(3) for ri in range(4) for b in range(CFG['blocks'])]
    if args.smoke:
        jobs = jobs[:args.limit]
    exit_code, status = 0, 'completed'
    try:
        for identity in jobs:
            name = '%02d_%02d_%03d' % identity
            path, recpath = out/'raw'/(name+'.npz'), out/'raw'/(name+'.json')
            if recpath.exists():
                validate_unit(path, json.loads(recpath.read_text()), list(identity), bind)
                progress['completed'] += 1
                continue
            if path.exists():
                # Preserve an uncommitted data file; never promote it by presence alone.
                raise ValueError('orphan NPZ without committed receipt: '+str(path))
            if shutil.disk_usage(out).free < CFG['disk_min_gib']*2**30:
                raise RuntimeError('disk floor')
            if time.monotonic()-started > CFG['watchdog_seconds']:
                raise TimeoutError('whole-run watchdog')
            progress['active_unit'] = name
            unit_start = time.monotonic()
            if hasattr(signal, 'alarm'):
                signal.alarm(CFG['unit_timeout_seconds'])
            try:
                data = make_unit(*identity, smoke=args.smoke)
            finally:
                if hasattr(signal, 'alarm'):
                    signal.alarm(0)
            if time.monotonic()-unit_start > CFG['unit_timeout_seconds']:
                raise TimeoutError('unit timeout')
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, **data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            receipt = {'schema':1,'identity':list(identity),'binding':bind,'sha256':sha(path),
                       'created_at':now(),'duration_seconds':time.monotonic()-unit_start,'smoke':args.smoke}
            validate_unit(path, receipt, list(identity), bind)
            atomic_json(recpath, receipt)
            progress['completed'] += 1
            if sum(p.stat().st_size for p in (out/'raw').glob('*')) > CFG['max_output_gib']*2**30:
                raise RuntimeError('declared output ceiling exceeded')
            print(json.dumps({'unit':name,'completed':progress['completed'],'seconds':receipt['duration_seconds']}), flush=True)
            if args.fail_after and progress['completed'] >= args.fail_after:
                raise InterruptedError('controlled interruption smoke')
        progress['active_unit'] = None
    except InterruptedError as e:
        exit_code, status = 75, 'controlled_interruption'
        progress['error'] = str(e)
    except TimeoutError as e:
        exit_code, status = 124, 'timeout'
        progress['error'] = str(e)
    except Exception as e:
        exit_code, status = 1, 'failed'
        progress['error'] = repr(e)
    finally:
        done.set()
        thread.join(5)
        progress.update(status=status,exit_code=exit_code,updated_at=now(),elapsed_seconds=time.monotonic()-started)
        atomic_json(out/'heartbeat.json',progress)
        # Attempt history is immutable; latest status is a pointer for resume.
        atomic_json(out/'attempts'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),progress)
        atomic_json(out/'terminal_status.json',progress)
    return exit_code


def scalar_stats(x):
    x = np.asarray(x)
    return {'n':int(len(x)), 'mean':float(x.mean()) if len(x) else None,
            'mc_se':float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None}


def analyze(args):
    out = Path(args.output)
    bind = binding()
    terminal = json.loads((out/'terminal_status.json').read_text())
    if terminal['status'] != 'completed' or terminal['completed'] != 384:
        raise ValueError('partial promotion forbidden')
    rows, moment_rows, raw_bindings = [], [], []
    primary = None
    for di in range(3):
        for ri in range(4):
            units=[]
            for block in range(CFG['blocks']):
                path=out/'raw'/('%02d_%02d_%03d.npz'%(di,ri,block))
                rec=json.loads(path.with_suffix('.json').read_text())
                if rec['smoke']: raise ValueError('smoke cannot enter scientific evidence')
                validate_unit(path,rec,[di,ri,block],bind)
                with np.load(path, allow_pickle=False) as d:
                    units.append({k:d[k].copy() for k in ['means','estimates','valid','exact_K','weight_Q']})
                raw_bindings.append({'path':str(path.relative_to(out)), 'sha256':rec['sha256']})
            means=np.concatenate([u['means'] for u in units])
            scores=np.concatenate([u['estimates'] for u in units])
            valid=np.concatenate([u['valid'] for u in units])
            _,_,mu,sigma=population(di)
            target=float(functional(mu)[0])
            for j in range(10):
                condition={'prevalence':CFG['prevalences'][di], 'rho':CFG['rhos'][ri],
                           'window':CFG['windows'][j//2], 'kernel':['window','fading'][j%2]}
                variance_samples=(means[:,j,0]-mu[0])**2/sigma[0,0]
                factor=scalar_stats(variance_samples)
                exact=float(units[0]['exact_K'][j])
                moment_rows.append({**condition, 'covariance_factor':factor,'exact_factor':exact,
                                    'weight_ess':float(1/units[0]['weight_Q'][j]),'information_ess':1/exact})
                vg=valid[:,j]
                invalid=float(1-vg.mean())
                b,rho,w=mu[1],CFG['rhos'][ri],CFG['windows'][j//2]
                invalid_exact=float((1-b)*(rho+(1-rho)*(1-b))**(w-1)+b*(rho+(1-rho)*b)**(w-1)) if j%2==0 else None
                for k,label in enumerate(['raw','iid_plugin','known_rho_plugin','population_oracle']):
                    errors=scores[vg,j,k]-target
                    estimates=scores[vg,j,k]
                    stats=scalar_stats(errors)
                    mse=scalar_stats(errors**2)
                    rows.append({**condition, 'estimator':label,'population_target':target,
                                 'bias_conditional_valid':stats,'mse_conditional_valid':mse,
                                 'rmse_conditional_valid':float(np.sqrt(mse['mean'])) if len(errors) else None,
                                 'undefined_fraction':invalid,'undefined_window_exact':invalid_exact,
                                 'score_range_excursions':int(np.count_nonzero(np.abs(estimates)>1)),
                                 'bias_expansion_admissible':invalid<=.01})
                rows.append({**condition,'estimator':'raw_zero_extension_DIAGNOSTIC',
                             'bias_unconditional_extended':scalar_stats(np.where(vg,scores[:,j,0],0)-target),
                             'undefined_fraction':invalid})
            if di==0 and ri==3:
                sq=(means[:,:2,0]-mu[0])**2/sigma[0,0]
                expected=units[0]['exact_K'][:2]
                empirical=sq.mean(axis=0)
                rng=np.random.default_rng(CFG['bootstrap_seed'])
                ratios=[]
                for start in range(0,CFG['bootstrap_resamples'],100):
                    ix=rng.integers(0,len(sq),size=(min(100,CFG['bootstrap_resamples']-start),len(sq)))
                    v=sq[ix].mean(axis=1)
                    ratios.extend((v[:,1]/v[:,0]).tolist())
                ci=np.quantile(ratios,[.025,.975])
                se=sq.std(axis=0,ddof=1)/np.sqrt(len(sq))
                z=np.abs(empirical-expected)/se
                exact_ratio=float(expected[1]/expected[0])
                supported=abs(exact_ratio-1)>=.10 and (ci[1]<1 or ci[0]>1) and np.all(z<=5)
                primary={'condition':{'prevalence':.5,'rho':.9,'window':20},'n_trajectories':len(sq),
                         'empirical_covariance_factors':empirical.tolist(),'exact_covariance_factors':expected.tolist(),
                         'mc_se':se.tolist(),'absolute_z_errors':z.tolist(),'exact_ratio':exact_ratio,
                         'empirical_ratio':float(empirical[1]/empirical[0]),'paired_bootstrap_95ci':ci.tolist(),
                         'criteria':{'min_exact_departure':.10,'ci_excludes_one':bool(ci[1]<1 or ci[0]>1),'max_z':5},
                         'verdict':'SUPPORTED_MECHANISM' if supported else 'NEGATIVE_OR_INCONCLUSIVE',
                         'claim_boundary':'refresh-chain mechanism only; not new theory or deployment superiority'}
    result={'schema':1,'created_at':now(),'binding':bind,'units':384,'primary':primary,
            'moment_conditions':moment_rows,'score_conditions':rows,'raw_inventory':raw_bindings}
    atomic_json(out/'analysis.json',result)
    print(json.dumps(primary,indent=2))
    return 0


def test(_args):
    tests=[]
    for theta in [np.array([.4,.3,.2]),population(0)[2],population(1)[2],population(2)[2]]:
        H=hessian(theta); h=1e-5; numeric=np.zeros((3,3)); eye=np.eye(3)*h
        for i in range(3):
            for j in range(3):
                numeric[i,j]=(functional(theta+eye[i]+eye[j])[0]-functional(theta+eye[i]-eye[j])[0]
                              -functional(theta-eye[i]+eye[j])[0]+functional(theta-eye[i]-eye[j])[0])/(4*h*h)
        assert np.allclose(H,numeric,rtol=2e-5,atol=2e-5),(H,numeric)
    tests.append('analytic_hessian_matches_finite_differences')
    W=weights(25)
    for rho in CFG['rhos']:
        C=rho**np.abs(np.arange(25)[:,None]-np.arange(25)[None,:])
        assert np.allclose(covariance_factors(W,rho),np.einsum('ti,tu,ui->i',W,C,W),atol=1e-13)
    assert np.allclose(covariance_factors(W,0),(W*W).sum(axis=0),atol=1e-13)
    tests.append('covariance_recursion_matches_direct_toeplitz_and_iid')
    for di in range(3):
        probs,z,mu,sigma=population(di)
        assert np.all(probs>0) and abs(probs.sum()-1)<1e-14
        assert abs(mu[1]-CFG['prevalences'][di])<1e-14
        assert abs(mu[0]-mu[1])<1e-14
        assert np.linalg.eigvalsh(sigma).min()>-1e-13
    tests.append('six_state_population_and_calibration_identities')
    assert not bool(functional([.4,0,0])[1])
    assert not bool(functional([.4,1,.4])[1])
    tests.append('undefined_margins_not_epsilon_divided')
    report={'created_at':now(),'status':'PASS','tests':tests,'binding':binding()}
    atomic_json(ROOT/'tests.json',report)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['run','analyze','test'])
    parser.add_argument('--output',default=str(ROOT/'results'))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--limit',type=int,default=3)
    parser.add_argument('--fail-after',type=int,default=0)
    args=parser.parse_args()
    sys.exit({'run':run,'analyze':analyze,'test':test}[args.mode](args))
