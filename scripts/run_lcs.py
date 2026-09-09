"""Generate light curves for a population catalog with the MOSFiT ``tde_shock`` model:
prompt collision-powered emission at pericenter (radiated fraction f_rad of the stream's
kinetic energy, so that epsilon_shock = f_rad r_g / r_p; Jiang, Guillochon & Loeb 2016)
plus accretion at epsilon_acc (log-normal about 0.03), Eddington-capped and then delayed by the viscous time of the Guillochon &
Ramirez-Ruiz (2015) dark-year map; the collision term is capped separately, so L <= 2 L_Edd.

Usage:  python run_lcs.py catalog_fiducial.npz [max_events] [--darkyear 0|1] [--eddslope P]
                          [--leddlim F] [--tviscslope S] [--fradslope Q] [--rcollmode M]
                          [--rcolldisk R]
                          [--tag NAME]

--eddslope sets the super-Eddington exponent p of the accretion term (L = L_Edd m/(1+m)^p; p = 1 is the
harmonic cap), --leddlim sets the accretion-term cap in units of L_Edd (the thermal UV/optical fraction
of an Eddington-saturated disk; the collision cap and the photosphere normalization stay at L_Edd),
--tviscslope sets d log T_visc / d log(r_p/r_g) of the dark-year map (2.1 is the Guillochon &
Ramirez-Ruiz 2015 value; 0 is what an already-magnetized stream implies, since the viscous time is then a
fixed multiple of the fallback time), --fradslope sets d log epsilon_shock / d log(r_g/r_coll), pivoted at
r_p/r_g = 22 so the normalization is unchanged (1 leaves epsilon = f_rad r_g / r_coll), --rcollmode selects
the collision radius (0 = r_p, the default; 1 = relativistic free-stream self-intersection; 2 = as 1,
capped at --rcolldisk cm where a pre-existing disk intercepts the stream), and
--darkyear turns the Guillochon & Ramirez-Ruiz (2015) viscous delay on; the fiducial model leaves
it off, because the MOSFiT fits of the optically selected sample return T_visc < t_pk for every event
(slope 0.79 +/- 0.46 against r_p/r_g, excluding the map's 2.1 at 2.9 sigma), and --tag names the output
library (default: the catalog tag).  The catalog's own `darkyear` field is provenance only; the delay is
a driver setting so that the fiducial and the dark-year variant share one population.

Must be run from the paper root so that MOSFiT picks up the local
modules/observables/filterrules.json (Euclid and SPHEREx bands).
Writes products/lcs_<tag>.npz.
"""
import os
import sys
import time
import zlib

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

MODEL = 'tde_shock'
FRAD_RANGE = (0.02, 0.07)   # radiated fraction of the collision kinetic energy (Jiang et al. 2016: 2-7%)
EFF_ACC = 0.03              # median accretion efficiency of the delayed component (fitted-sample value)
EFF_ACC_SCATTER = 0.3       # dex, log-normal event-to-event scatter, clipped to the tde_shock prior [0.01, 0.1]
TVISC_SCATTER = 0.5         # dex, event-to-event scatter about the dark-year map
PROMPT_OFFSET = -6.0        # dex offset that removes the viscous delay (prompt-circularization variant)
EDDSLOPE = 1.0              # default super-Eddington exponent of the accretion term (harmonic cap)
TVISCSLOPE = 2.1            # default d log T_visc / d log(r_p/r_g); the tde_shock model's own value
FRADSLOPE = 1.0             # default d log epsilon_shock / d log(r_g/r_coll): 1 leaves epsilon = f_rad r_g / r_coll
LEDDLIM = 0.1               # default disk cap in units of L_Edd: the thermal UV/optical share of an Eddington-limited disk
DARKYEAR = False            # default circularization: prompt, as the fits of optically selected TDEs return
RCOLLMODE = 0               # default collision radius: r_p (mode 0 reproduces current libraries)
RCOLLDISK = 1.0e13          # default disk intercept radius in cm (used only for rcollmode 2)


def make_model():
    f = Fitter()
    m = mosfit.model.Model(model=MODEL, fitter=f)
    data = f.generate_dummy_data('engine', max_time=T_OBS[-1], time_list=list(T_OBS),
                                 band_list=BANDS, band_instruments=INSTS)
    m.load_data(data, event_name='engine', smooth_times=0, band_list=BANDS, band_instruments=INSTS,
                user_fixed_parameters=['redshift', 0.05, 'texplosion', 0.0, 'variance', 0.1])
    return m


T_PEAK_MIN = 1.0e4   # K; observed optical TDEs have blackbody temperatures above this at peak
V_PHOT_MAX = 0.5     # MOSFiT tde_constraints: photosphere inside a 0.5c wind launched at first fallback
G_CGS, C_CGS, MSUN_CGS = 6.674e-8, 2.99792458e10, 1.989e33
MAX_REDRAW = 12


def run_population(m, pop, nmax=None, darkyear=True, seed=0, eddslope=EDDSLOPE, leddlim=LEDDLIM,
                   tviscslope=TVISCSLOPE, fradslope=FRADSLOPE, rcollmode=RCOLLMODE,
                   rcolldisk=RCOLLDISK):
    m._modules['eddslope'].fix_value(float(eddslope))
    m._modules['Leddlimdisk'].fix_value(float(leddlim))   # disk (accretion) cap only; shock cap and photosphere stay at L_Edd
    m._modules['tviscslope'].fix_value(float(tviscslope))
    m._modules['fradslope'].fix_value(float(fradslope))
    m._modules['rcollmode'].fix_value(float(rcollmode))
    m._modules['rcolldisk'].fix_value(float(rcolldisk))
    rng = np.random.default_rng(seed)
    names = m.free_parameter_names()
    n = len(pop['mh']) if nmax is None else min(nmax, len(pop['mh']))
    mags = np.full((n, len(T_OBS), len(BANDS)), np.nan, dtype=np.float16)   # 0.02 mag precision suffices
    lbol = np.zeros((n, len(T_REST_GRID)), dtype=np.float64)   # float32 overflows above 3.4e38 erg/s
    tph = np.zeros((n, len(T_OBS)), dtype=np.float32)
    rph = np.zeros((n, len(T_OBS)), dtype=np.float32)
    scal = {k: np.zeros(n) for k in ['lpeak', 'tpeak', 'ledd', 'dmbound', 'beta', 'rstar', 'tfallback', 'erad',
                                     'frad', 'eff', 'shock_eff', 'rp_over_rg', 'r_coll_over_rp', 'tvisc', 'tviscoffset',
                                     'rph0', 'lph', 'nh', 'nredraw', 'rphot_ratio']}
    # per-event draws for the emission model: f_rad log-uniform over the simulated range, and the
    # dex offset of the viscous time about the dark-year map (or none, for prompt circularization)
    frad_all = 10 ** rng.uniform(np.log10(FRAD_RANGE[0]), np.log10(FRAD_RANGE[1]), n)
    off_all = TVISC_SCATTER * rng.standard_normal(n) if darkyear else np.full(n, PROMPT_OFFSET)
    eff_all = np.clip(EFF_ACC * 10 ** (EFF_ACC_SCATTER * rng.standard_normal(n)), 0.01, 0.1)
    t0 = time.time()
    for i in range(n):
        z = float(pop['z'][i])
        m._modules['redshift'].fix_value(z)
        m._modules['lumdist'].fix_value(float(cosmo.luminosity_distance(z).value))
        m._modules['tviscoffset'].fix_value(float(off_all[i]))
        nuis = dict(frad=float(frad_all[i]), efficiency=float(eff_all[i]), Rph0=float(pop['rph0'][i]),
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
            # the photosphere must stay inside the envelope MOSFiT fits enforce (tde_constraints):
            # a relativistic wind from the circularization radius, 2 r_p + v_max c t + 2 a(t)
            t_rest = np.asarray(o['all_times'], dtype=float)[sel] / (1 + z) * 86400.0
            a_t = (G_CGS * float(pop['mh'][i]) * MSUN_CGS * (np.maximum(t_rest, 0.0) / np.pi) ** 2) ** (1.0 / 3.0)
            rmax = 2 * o['rp'] + V_PHOT_MAX * C_CGS * np.maximum(t_rest, 0.0) + 2 * a_t
            rratio = float(np.nanmax(np.asarray(o['radiusphot'])[sel] / rmax))
            if (tpk >= T_PEAK_MIN and rratio <= 1.0) or attempt == MAX_REDRAW:
                break
            # photosphere parameters fitted at low Eddington ratio do not transfer to this event: redraw
            _, rph0, lph, _, nh = ps.sample_nuisance(1, np.array([pop['mh'][i]]), False, False,
                                                     np.array([pop['mstar'][i]]), np.array([pop['beta'][i]]))
            nuis.update(Rph0=float(rph0[0]), lphoto=float(lph[0]), nhhost=float(nh[0]))
        if o is None:
            continue
        scal['nredraw'][i] = attempt; scal['rphot_ratio'][i] = rratio
        scal['frad'][i] = nuis['frad']; scal['eff'][i] = nuis['efficiency']
        scal['rph0'][i] = nuis['Rph0']; scal['lph'][i] = nuis['lphoto']; scal['nh'][i] = nuis['nhhost']
        scal['shock_eff'][i] = o['shock_efficiency']; scal['rp_over_rg'][i] = o['rp_over_rg']
        scal['r_coll_over_rp'][i] = o['r_coll_over_rp']
        scal['tvisc'][i] = o['Tviscous']; scal['tviscoffset'][i] = off_all[i]
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
    argv = sys.argv[1:]
    eddslope = float(argv[argv.index('--eddslope') + 1]) if '--eddslope' in argv else EDDSLOPE
    leddlim = float(argv[argv.index('--leddlim') + 1]) if '--leddlim' in argv else LEDDLIM
    tviscslope = float(argv[argv.index('--tviscslope') + 1]) if '--tviscslope' in argv else TVISCSLOPE
    fradslope = float(argv[argv.index('--fradslope') + 1]) if '--fradslope' in argv else FRADSLOPE
    rcollmode = float(argv[argv.index('--rcollmode') + 1]) if '--rcollmode' in argv else RCOLLMODE
    rcolldisk = float(argv[argv.index('--rcolldisk') + 1]) if '--rcolldisk' in argv else RCOLLDISK
    out_tag = argv[argv.index('--tag') + 1] if '--tag' in argv else None
    skip = set()
    for flag in ('--darkyear', '--eddslope', '--leddlim', '--tviscslope', '--fradslope', '--rcollmode',
                 '--rcolldisk', '--tag'):
        if flag in argv:
            skip |= {flag, argv[argv.index(flag) + 1]}
    pos = [a for a in argv if a not in skip]
    catpath = pos[0]
    nmax = int(pos[1]) if len(pos) > 1 else None
    cat = np.load(os.path.join(ROOT, 'products', catpath), allow_pickle=True)
    tag = str(cat['tag'])
    out_tag = out_tag or tag
    darkyear = bool(float(argv[argv.index('--darkyear') + 1])) if '--darkyear' in argv else DARKYEAR
    if 'scale_eff' in cat.files and bool(cat['scale_eff']):
        raise SystemExit('the efficiency-scaling variant is superseded by the tde_shock model (epsilon = f_rad r_g / r_p)')
    m = make_model()
    out = {}
    for pop in ['eng', 'field']:
        d = {k[len(pop) + 1:]: cat[k] for k in cat.files if k.startswith(pop + '_')}
        print('population', pop, len(d['mh']))
        res = run_population(m, d, nmax, darkyear, seed=zlib.crc32((tag + pop).encode()), eddslope=eddslope,
                             leddlim=leddlim, tviscslope=tviscslope, fradslope=fradslope,
                             rcollmode=rcollmode, rcolldisk=rcolldisk)
        for k, v in res.items():
            out[pop + '_' + k] = v
    out['bands'] = np.array(BANDS); out['t_obs'] = T_OBS; out['t_rest'] = T_REST_GRID
    out['model'] = MODEL; out['frad_range'] = np.array(FRAD_RANGE); out['eff_acc'] = EFF_ACC; out['eff_acc_scatter'] = EFF_ACC_SCATTER; out['darkyear'] = darkyear; out['eddslope'] = eddslope; out['leddlim'] = leddlim; out['tviscslope'] = tviscslope; out['fradslope'] = fradslope; out['rcollmode'] = rcollmode; out['rcolldisk'] = rcolldisk
    np.savez_compressed(os.path.join(ROOT, 'products', 'lcs_%s.npz' % out_tag), **out)
    print('done')
