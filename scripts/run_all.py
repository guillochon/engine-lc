"""Run the whole light-curve pipeline for every catalog: MOSFiT libraries, post-processing,
and the variant table.  Usage (from the repository root, with the MOSFiT environment's Python):

    python scripts/run_all.py [--skip-lcs] [--workers N] [tags...]

Each catalog is an independent job (one MOSFiT model, one output library), so with
--workers N up to N catalogs run at once; the default is one job per physical core, capped
at the number of catalogs.  Progress is appended to products/run_all.log."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = ['fiducial', 'gaslimited', 'burstrelation', 'simple', 'young1', 'young0', 'KH13', 'miller',
        'fiducial_y1', 'fiducial_L1', 'fiducial_L0.3']   # '<catalog>_L<value>' reruns a catalog with the
# disk cap at <value> L_Edd, '_y1' with the dark-year viscous delay switched on


def lcs_args(t):
    # '<catalog>[_y<darkyear>][_p<eddslope>][_L<leddlim>][_s<tviscslope>][_q<fradslope>][_c<rcollmode>]
    # [_d<rcolldisk>]'
    # reruns a catalog with those emission parameters
    parts = t.split('_')
    args = ['scripts/run_lcs.py', 'catalog_%s.npz' % parts[0]]
    for tok in parts[1:]:
        if tok.startswith('y'):
            args += ['--darkyear', tok[1:]]
        elif tok.startswith('p'):
            args += ['--eddslope', tok[1:]]
        elif tok.startswith('L'):
            args += ['--leddlim', tok[1:]]
        elif tok.startswith('s'):
            args += ['--tviscslope', tok[1:]]
        elif tok.startswith('q'):
            args += ['--fradslope', tok[1:]]
        elif tok.startswith('c'):
            args += ['--rcollmode', tok[1:]]
        elif tok.startswith('d'):
            args += ['--rcolldisk', tok[1:]]
    if len(parts) > 1:
        args += ['--tag', t]
    return args
LOG = os.path.join(ROOT, 'products', 'run_all.log')


def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('%s  %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg))


def run(args, out):
    """Run one pipeline step, writing its output to `out`; return the exit code."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    with open(out, 'a', encoding='utf-8') as f:
        r = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    return r.returncode


STAGGER = 0.0    # s between job starts (MOSFiT now guards its filter cache, so concurrent loads are safe)


def stage(name, tags, args_for, workers, stagger=0.0):
    """Run one stage for every tag, `workers` at a time, and stop the pipeline on any failure."""
    def one(it):
        i, t = it
        time.sleep(stagger * min(i, workers - 1))
        log('%s %s' % (name, t))
        rc = run(args_for(t), os.path.join(ROOT, 'products', '%s_%s.txt' % (name, t)))
        log('%s %s exit %d' % (name, t, rc))
        return rc
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tags)))) as ex:
        codes = list(ex.map(one, list(enumerate(tags))))
    if any(codes):
        log('FAILED %s' % name)
        sys.exit(1)


if __name__ == '__main__':
    argv = sys.argv[1:]
    skip_lcs = '--skip-lcs' in argv
    workers = int(argv[argv.index('--workers') + 1]) if '--workers' in argv else max(1, (os.cpu_count() or 2) // 2)
    skip = {'--skip-lcs', '--workers'} | ({argv[argv.index('--workers') + 1]} if '--workers' in argv else set())
    tags = [a for a in argv if a not in skip] or TAGS
    log('start %s (workers=%d)' % (' '.join(tags), workers))
    if not skip_lcs:
        stage('run_lcs', tags, lcs_args, workers, stagger=STAGGER)
    # the constant-temperature blackbody fits Figure 2's temperature panel compares with the
    # observed sample; cheap, but it loads MOSFiT's filters, so it runs before postprocess
    stage('fit_bbtemp', tags, lambda t: ['scripts/fit_bbtemp.py', t], workers)
    stage('postprocess', tags, lambda t: ['scripts/postprocess.py', t], workers)
    log('variant_table')
    rc = run(['scripts/variant_table.py'], LOG)
    log('ALLDONE' if rc == 0 else 'FAILED variant_table')
