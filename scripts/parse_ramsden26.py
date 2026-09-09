"""Host-split rates for the Ramsden et al. (2025, 2026) plateau-TDE sample.

Ramsden et al. (2026) do not publish a luminosity function. This script takes
their 40-event 2025 table (plus the two 2026 additions if present), matches
peak g-band luminosities and photometric galaxy masses from the Mummery &
van Velzen manyTDE plateau catalog, and splits the sample 50:50 about the
TDE-only M_BH--M_* relation of Ramsden et al. (2025, Appendix A), the split
Ramsden et al. (2026) use: the undermassive half is quenched-dominated, the
overmassive half is star-forming-dominated.

Writes scripts/ramsden26_sample.json for Figure 4.
"""
import json
import os
import pickle
import re
import urllib.request

import numpy as np
from astropy.cosmology import FlatLambdaCDM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
PC = 3.086e18
C = 2.998e10
PKL_URL = 'https://raw.githubusercontent.com/sjoertvv/manyTDE/main/data/inferred_params/plateau_tdes.pkl'
# Ramsden et al. (2025) Appendix A: log M_BH = 1.11 log M_* - 4.431
SLOPE, INTERCEPT = 1.11, -4.431
MLIM = 19.5


def ramsden_names():
    txt = open(os.path.join(ROOT, 'research', 'ramsden25', 'ramsden25_arxiv2506.16155.txt'), encoding='utf-8').read()
    i0 = txt.find('Table 1:')
    i1 = txt.find('Our sample of TDE hosts', i0)
    names = re.findall(r'\| ([A-Za-z0-9-]+) \|', txt[i0:i1])
    extra = ['AT2022wtn', 'AT2023clx']
    return names + extra


def load_manytde():
    path = os.path.join(ROOT, 'research', 'ramsden26', 'plateau_tdes.pkl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        req = urllib.request.Request(PKL_URL, headers={'User-Agent': 'Mozilla/5.0'})
        open(path, 'wb').write(urllib.request.urlopen(req, timeout=60).read())
    return pickle.load(open(path, 'rb'))


def med(x):
    x = np.asarray(x, dtype=float).ravel()
    return float(x[0])


def zmax_from_Mg(Mg, mlim=MLIM):
    """Redshift at which a source of peak M_g reaches mlim, including the
    2.5 log(1+z) bandwidth term used for the model luminosity functions."""
    lo, hi = 1e-4, 2.0
    for _ in range(40):
        z = 0.5 * (lo + hi)
        dm = 5 * np.log10(cosmo.luminosity_distance(z).value * 1e5) - 2.5 * np.log10(1 + z)
        if Mg + dm < mlim:
            lo = z
        else:
            hi = z
    return 0.5 * (lo + hi)


def main():
    names = ramsden_names()
    cat = load_manytde()
    by = {str(n): i for i, n in enumerate(cat['Name'])}
    ev = {}
    for name in names:
        if name not in by:
            continue
        i = by[name]
        logLg = med(cat['Peak g-band'][i])
        logMgal = med(cat['Galaxy mass'][i])
        logMbh = med(cat['Plateau black hole mass'][i])
        z = float(np.asarray(cat['Redshift'][i]).ravel()[0])
        Lg = 10 ** logLg
        Mg = -2.5 * np.log10(Lg / (C / 4741e-8) / (4 * np.pi * (10 * PC) ** 2)) - 48.6
        zm = zmax_from_Mg(Mg)
        V = float(cosmo.comoving_volume(zm).value)
        residual = logMbh - (SLOPE * logMgal + INTERCEPT)
        ev[name] = dict(z=z, logLg=logLg, Mg=Mg, logMgal=logMgal, logMbh=logMbh,
                        zmax=zm, residual=residual, w_1overV=1.0 / V)
    res = np.array([v['residual'] for v in ev.values()])
    cut = float(np.median(res))
    n_under = 0
    for v in ev.values():
        v['undermassive'] = bool(v['residual'] < cut)
        n_under += v['undermassive']
    print('matched', len(ev), 'of', len(names), 'Ramsden names; undermassive', n_under,
          'overmassive', len(ev) - n_under, 'cut', round(cut, 3))
    out = os.path.join(HERE, 'ramsden26_sample.json')
    json.dump(ev, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
