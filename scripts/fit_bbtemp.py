"""Fit a constant-temperature blackbody to the synthetic photometry, the way
\\citet{Yao:2023a} fit theirs, so that the model temperatures can be compared with the
observed ones like for like.

Yao et al. report the blackbody parameters at maximum light, but their temperature is a
single constant per event: they fix it to its near-peak value because most TDEs show little
temperature evolution, and for the eleven events that do evolve they trim the late-time
UVOT data so the constant lands near peak.  MOSFiT's photosphere, by contrast, evolves
($R_{\\rm ph} \\propto L^{l}$), so its temperature at the epoch of peak light is not the
quantity Yao measure.  This script measures the quantity they do measure.

Method.  For each synthetic event we take the UVOT UVW2 and LSST u, g, r photometry that
MOSFiT generated (the UV--optical baseline that pins the temperature in their fits), select
the epochs within `window` mag of the g-band peak, and fit a blackbody whose temperature is
constant and whose normalisation is free at every epoch.  With the normalisation free the fit
is driven entirely by colours, and because the observed colours of a blackbody at rest-frame
temperature $T$ and redshift $z$ are the rest-frame colours of one at $T/(1+z)$, the model
side reduces to a one-parameter colour table, built here from MOSFiT's own filter curves,
integrals, and zeropoint offsets so that the two sides share a photometric convention.
Least squares in magnitude space then depends only on the window-averaged colours, which is
what makes the fit cheap:

    chi2(T) = sum_b [ nep * cm_b(T)^2 - 2 * S_b * cm_b(T) ] + const,

with cm_b(T) the table's colours and S_b the summed colours of the event's photometry.

No host-extinction correction is applied on the model side, matching Yao et al., who correct
only for Galactic extinction; the synthetic photometry carries the host columns that MOSFiT
fitted to the real sample, so a reddened event is recovered cool exactly as a real one would
be.  The script reports the size of that bias against the model's own $T_{\\rm ph}$.

Usage:  python fit_bbtemp.py [tag] [max_events]

Must be run from the paper root, so that MOSFiT picks up the local filterrules.json.
Writes products/bbfit_<tag>.npz with, per event and population, the fitted constant
temperature in each window, the single-epoch fit at peak, and the model's own T_ph there.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE)
import run_lcs  # noqa: E402  (MOSFiT model setup, so we reuse its loaded filters)

H_CGS, C_CGS, K_CGS = 6.62607015e-27, 2.99792458e10, 1.380649e-16

FIT_BANDS = [('UVW2', 'UVOT'), ('u', 'LSST'), ('g', 'LSST'), ('r', 'LSST')]
TGRID = np.logspace(np.log10(3.0e3), np.log10(4.0e5), 400)   # blackbody temperature of the redshifted SED
WINDOWS = {'fwhm': 0.753,   # epochs above half maximum in g: the peak-light window
           'wide': 2.5}     # everything within 2.5 mag of peak: a ZTF-baseline analogue


def colour_table():
    """Colours of a unit-normalised blackbody through MOSFiT's own filters, (nband, nT)."""
    m = run_lcs.make_model()
    ph = m._modules['photometry']
    tab = np.zeros((len(FIT_BANDS), len(TGRID)))
    for j, (b, inst) in enumerate(FIT_BANDS):
        bi = ph.find_band_index(b, instrument=inst)
        wav = np.asarray(ph._band_wavelengths[bi], dtype=float)          # Angstrom
        trans = np.asarray(ph._transmissions[bi], dtype=float)
        fint, off = ph._filter_integrals[bi], ph._band_offsets[bi]
        lam = wav * 1e-8
        for i, T in enumerate(TGRID):
            bb = 2 * H_CGS * C_CGS ** 2 / lam ** 5 / np.expm1(H_CGS * C_CGS / (lam * K_CGS * T))
            tab[j, i] = -off - 2.5 * np.log10(np.trapezoid(trans * bb, wav) / fint)
    return tab


def fit_population(mags, tph, zs, tab, bcols, nmax=None):
    """Constant-T fits per event, in each window, plus the single-epoch fit at the g peak."""
    cm = tab - tab.mean(axis=0, keepdims=True)                            # (nband, nT) model colours
    cm2 = (cm ** 2).sum(axis=0)                                           # (nT,)
    n = len(mags) if nmax is None else min(nmax, len(mags))
    out = {k: np.full(n, np.nan) for k in WINDOWS}
    nep = {k: np.zeros(n, dtype=np.int32) for k in WINDOWS}
    single = np.full(n, np.nan)
    true = np.full(n, np.nan)
    gj = bcols[2]
    for i in range(n):
        mg = mags[i, :, gj].astype(float)
        ok = np.isfinite(mg)
        if not ok.any():
            continue
        ipk = int(np.nanargmin(mg))
        zp1 = 1 + zs[i]
        row = mags[i, ipk, bcols].astype(float)
        if np.isfinite(row).all():
            co = row - row.mean()
            single[i] = TGRID[int(np.argmin(cm2 - 2 * co.dot(cm)))] * zp1
            true[i] = tph[i, ipk]
        for key, w in WINDOWS.items():
            sel = ok & (mg < mg[ipk] + w)
            obs = mags[i, sel][:, bcols].astype(float)
            obs = obs[np.isfinite(obs).all(axis=1)]
            if len(obs) < 2:
                continue
            co = (obs - obs.mean(axis=1, keepdims=True)).sum(axis=0)       # summed colours, (nband,)
            nep[key][i] = len(obs)
            out[key][i] = TGRID[int(np.argmin(len(obs) * cm2 - 2 * co.dot(cm)))] * zp1
    return out, nep, single, true


def wq(x, w, q=(16, 50, 84)):
    x = np.asarray(x, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w)
    x, w = x[ok], w[ok]
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return [float(np.interp(k / 100, c, x)) for k in q]


def main(tag='fiducial', nmax=None):
    tab = colour_table()
    cat_tag = tag if os.path.exists('products/catalog_%s.npz' % tag) else tag.split('_')[0]
    cat = np.load('products/catalog_%s.npz' % cat_tag, allow_pickle=True)
    lcs = np.load('products/lcs_%s.npz' % tag, allow_pickle=True)
    bands = list(lcs['bands'])
    bcols = [bands.index(b) for b, _ in FIT_BANDS]
    store = {}
    print('fitting a constant-temperature blackbody to %s in %s' % (
        ', '.join(b for b, _ in FIT_BANDS), tag))
    for p in ['eng', 'field']:
        mags, tph = lcs['%s_mags' % p], lcs['%s_tph' % p]
        zs = cat['%s_z' % p][:len(mags)].astype(float)
        out, nep, single, true = fit_population(mags, tph, zs, tab, bcols, nmax)
        w = cat['%s_w' % p][:len(single)].astype(float)
        nh = lcs['%s_nh' % p][:len(single)].astype(float)
        for k in WINDOWS:
            store['%s_%s' % (p, k)] = out[k]
            store['%s_nep_%s' % (p, k)] = nep[k]
        store['%s_single' % p] = single
        store['%s_tph_peak' % p] = true
        store['%s_w' % p] = w
        store['%s_nh' % p] = nh
        r = np.log10(single / true)
        lo = nh < 1e19
        print('\n%s (n=%d)' % (p, len(single)))
        print('  single-epoch fit / model T_ph: %+.3f dex median (low column), %+.3f dex (all, reddened)'
              % (np.nanmedian(r[lo]), np.nanmedian(r)))
        print('  model T_ph at g peak      16/50/84: %s' % ['%.2f' % v for v in wq(np.log10(true), w)])
        for k in WINDOWS:
            print('  fitted constant T (%-4s) 16/50/84: %s  [%d epochs median]'
                  % (k, ['%.2f' % v for v in wq(np.log10(out[k]), w)], np.median(nep[k][nep[k] > 0])))
    yao = json.load(open('scripts/yao2023_sample.json'))
    ev = [v for v in yao.values() if isinstance(v, dict) and 'logT' in v]
    print('\nobserved (Yao et al. 2023) 16/50/84: %s' % ['%.2f' % v for v in
          wq([v['logT'] for v in ev], [v['w_1overV'] for v in ev])])
    np.savez_compressed('products/bbfit_%s.npz' % tag, **store)
    print('wrote products/bbfit_%s.npz' % tag)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    main(args[0] if args else 'fiducial', int(args[1]) if len(args) > 1 else None)
