"""Collect products/results_<tag>.json for the model variants into
tables/variant_rows.tex (engine population) for the manuscript."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = [
    ('fiducial', 'Fiducial (slow accretion, fed disk, growing black holes)'),
    ('prompt', 'Prompt circularization (no viscous delay)'),
    ('gaslimited', 'Gas-limited lifetime ($\\tau = t_{\\rm gas}$)'),
    ('burstrelation', 'Relation applies at the burst'),
    ('simple', 'Simple (all three of the above)'),
    ('young1', 'Burst population only'),
    ('young0', 'Old population only'),
    ('KH13', '\\citet{Kormendy:2013a} relation'),
    ('miller', '\\citet{Miller:2015a} occupation fraction'),
    ('fiducial_L1', 'Disk cap at $L_{\\rm Edd}$ (no EUV fraction)'),
    ('fiducial_L0.1', 'Disk cap at $0.1\\,L_{\\rm Edd}$'),
]


def f(v, fmt):
    try:
        return fmt % v
    except Exception:
        return r'\nodata'


rows = []
for tag, label in VARIANTS:
    p = os.path.join(ROOT, 'products', 'results_%s.json' % tag)
    if not os.path.exists(p):
        continue
    R = json.load(open(p))
    e = R['eng']
    y = R['yields_per_yr_fspark1']
    rubin = [k for k in y if k.startswith('Rubin')][0]
    ztf = [k for k in y if k.startswith('ZTF')][0]
    rows.append(' & '.join([
        label,
        f(R['ndot_eng'] * 1e9, '%.1f'),
        f(R['active_frac'], '%.3f'),
        f(R['fspark_lam_needed'], '%.1f'),
        f(e['logmh'][1], '%.2f'),
        f(e['partial_frac'], '%.2f'),
        f(e['loglpeak_full'][1], '%.2f'),
        f(e['thalf'][1], '%.0f'),
        f(e['absmag_g'][1], '%.1f'),
        f(y[ztf]['eng_AV50'], '%.1f'),
        f(y[rubin]['eng_AV50'], '%.0f'),
    ]) + r' \\')
os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
open(os.path.join(ROOT, 'tables', 'variant_rows.tex'), 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
