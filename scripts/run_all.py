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
TAGS = ['fiducial', 'prompt', 'gaslimited', 'burstrelation', 'simple', 'scaleeff', 'young1', 'young0', 'KH13', 'miller']
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


def stage(name, tags, args_for, workers):
    """Run one stage for every tag, `workers` at a time, and stop the pipeline on any failure."""
    def one(t):
        log('%s %s' % (name, t))
        rc = run(args_for(t), os.path.join(ROOT, 'products', '%s_%s.txt' % (name, t)))
        log('%s %s exit %d' % (name, t, rc))
        return rc
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tags)))) as ex:
        codes = list(ex.map(one, tags))
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
        stage('run_lcs', tags, lambda t: ['scripts/run_lcs.py', 'catalog_%s.npz' % t], workers)
    stage('postprocess', tags, lambda t: ['scripts/postprocess.py', t], workers)
    log('variant_table')
    rc = run(['scripts/variant_table.py'], LOG)
    log('ALLDONE' if rc == 0 else 'FAILED variant_table')
