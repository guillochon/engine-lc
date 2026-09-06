"""Generate light curves for a population catalog with the MOSFiT TDE model.

Usage:  python run_lcs.py catalog_fiducial.npz [max_events]

Must be run from the paper root so that MOSFiT picks up the local
modules/observables/filterrules.json (Euclid and SPHEREx bands).
Writes products/lcs_<tag>.npz.
"""
import os
import sys
import time

import numpy as np
from astropy.cosmology import FlatLambdaCDM

import mosfit
from mosfit.fitter import Fitter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE)
import popsynth as ps  # noqa: E402
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

BANDS = ['UVW2', 'u', 'g', 'r', 'i', 'z', 'y', 'VIS', 'YE', 'JE', 'HE',
         'F062', 'F106', 'F146', 'F158', 'F213', 'W1', 'W2', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
INSTS = ['UVOT'] + ['LSST'] * 6 + ['VIS'] + ['NISP'] * 3 + ['WFI'] * 5 + ['WISE'] * 2 + ['SPHEREx'] * 6
# observer-frame days after first fallback
T_OBS = np.concatenate([[0.5], np.logspace(0, np.log10(36525.0), 149)])   # to 100 yr, so the rest-frame flare is followed for decades
T_REST_GRID = np.logspace(-1, np.log10(40000.0), 200)   # rest-frame days for bolometric storage (110 yr)


def make_model():
    f = Fitter()
    m = mosfit.model.Model(model='tde', fitter=f)
    data = f.generate_dummy_data('engine', max_time=T_OBS[-1], time_list=list(T_OBS),
                                 band_list=BANDS, band_instruments=INSTS)
    m.load_data(data, event_name='engine', smooth_times=0, band_list=BANDS, band_instruments=INSTS,
                user_fixed_parameters=['redshift', 0.05, 'texplosion', 0.0, 'variance', 0.1])
    return m


T_PEAK_MIN = 1.0e4   # K; observed optical TDEs have blackbody temperatures above this at peak
MAX_REDRAW = 12


def run_population(m, pop, nmax=None, scale_eff=False, darkyear=False):
    names = m.free_parameter_names()
    n = len(pop['mh']) if nmax is None else min(nmax, len(pop['mh']))
    mags = np.full((n, len(T_OBS), len(BANDS)), np.nan, dtype=np.float16)   # 0.02 mag precision suffices
    lbol = np.zeros((n, len(T_REST_GRID)), dtype=np.float64)   # float32 overflows above 3.4e38 erg/s
    tph = np.zeros((n, len(T_OBS)), dtype=np.float32)
    rph = np.zeros((n, len(T_OBS)), dtype=np.float32)
    scal = {k: np.zeros(n) for k in ['lpeak', 'tpeak', 'ledd', 'dmbound', 'beta', 'rstar', 'tfallback', 'erad',
                                     'eff', 'rph0', 'lph', 'tvisc', 'nh', 'nredraw']}
    t0 = time.time()
    gi = BANDS.index('g')
    for i in range(n):
        z = float(pop['z'][i])
        m._modules['redshift'].fix_value(z)
        m._modules['lumdist'].fix_value(float(cosmo.luminosity_distance(z).value))
        nuis = dict(efficiency=float(pop['eff'][i]), Tviscous=float(pop['tvisc'][i]), Rph0=float(pop['rph0'][i]),
                    lphoto=float(pop['lph'][i]), nhhost=float(pop['nh'][i]))
        o = None
        for attempt in range(MAX_REDRAW + 1):
            vals = dict(starmass=float(pop['mstar'][i]), b=float(pop['b'][i]), bhmass=float(pop['mh'][i]), **nuis)
            x = [m._modules[k].fraction(vals[k]) for k in names]
            try:
                o = m.run(x)
            except Exception as ex:  # noqa
                print('event %d failed: %s' % (i, ex))
                o = None
                break
            mo = np.asarray(o['model_observations'], dtype=float)
            ab = np.asarray(o['all_bands'])
            sel = ab == 'g'
            tpk = np.asarray(o['temperaturephot'])[sel][np.nanargmin(mo[sel])]
            if tpk >= T_PEAK_MIN or attempt == MAX_REDRAW:
                break
            # photosphere parameters fitted at low Eddington ratio do not transfer to this event: redraw
            eff, rph0, lph, tv, nh = ps.sample_nuisance(1, np.array([pop['mh'][i]]), scale_eff, darkyear,
                                                        np.array([pop['mstar'][i]]), np.array([pop['beta'][i]]))
            nuis = dict(efficiency=float(eff[0]), Tviscous=float(tv[0]), Rph0=float(rph0[0]), lphoto=float(lph[0]), nhhost=float(nh[0]))
        if o is None:
            continue
        scal['nredraw'][i] = attempt
        for k, kk in [('eff', 'efficiency'), ('rph0', 'Rph0'), ('lph', 'lphoto'), ('tvisc', 'Tviscous'), ('nh', 'nhhost')]:
            scal[k][i] = nuis[kk]
        at = np.asarray(o['all_times'], dtype=float)
        for j, b in enumerate(BANDS):
            sel = ab == b
            mags[i, :, j] = np.interp(T_OBS, at[sel], mo[sel])
        sel = ab == 'g'
        tph[i] = np.interp(T_OBS, at[sel], np.asarray(o['temperaturephot'])[sel])
        rph[i] = np.interp(T_OBS, at[sel], np.asarray(o['radiusphot'])[sel])
        dt = np.asarray(o['dense_times'], dtype=float)
        dl = np.asarray(o['dense_luminosities'], dtype=float)
        lbol[i] = np.interp(T_REST_GRID, dt, dl, left=0.0, right=0.0)
        scal['lpeak'][i] = dl.max()
        scal['tpeak'][i] = dt[dl.argmax()]
        scal['ledd'][i] = o['Ledd']
        scal['dmbound'][i] = np.trapezoid(np.asarray(o['dmdt'], dtype=float), dt * 86400.0) / 1.989e33
        scal['beta'][i] = o['beta']
        scal['rstar'][i] = o['Rstar']
        scal['tfallback'][i] = o['tfallback']
        scal['erad'][i] = np.trapezoid(dl, dt * 86400.0)
        if i % 500 == 0:
            print('  %d / %d  (%.0f s)' % (i, n, time.time() - t0), flush=True)
    return dict(mags=mags, lbol=lbol, tph=tph, rph=rph, **scal)


if __name__ == '__main__':
    catpath = sys.argv[1]
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cat = np.load(os.path.join(ROOT, 'products', catpath), allow_pickle=True)
    tag = str(cat['tag'])
    scale_eff = bool(cat['scale_eff'])
    darkyear = bool(cat['darkyear']) if 'darkyear' in cat.files else False
    m = make_model()
    if darkyear:
        # the tde prior caps T_viscous at 100 d; slowed circularization needs up to ~1e5 d
        m._modules['Tviscous']._max_value = np.log(1e5)
    out = {}
    for pop in ['eng', 'field']:
        d = {k[len(pop) + 1:]: cat[k] for k in cat.files if k.startswith(pop + '_')}
        print('population', pop, len(d['mh']))
        res = run_population(m, d, nmax, scale_eff, darkyear)
        for k, v in res.items():
            out[pop + '_' + k] = v
    out['bands'] = np.array(BANDS); out['t_obs'] = T_OBS; out['t_rest'] = T_REST_GRID
    np.savez_compressed(os.path.join(ROOT, 'products', 'lcs_%s.npz' % tag), **out)
    print('done')
