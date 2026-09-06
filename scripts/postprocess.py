"""Post-process the MOSFiT light-curve library: apply the circumnuclear-disk
screen, compute infrared echoes, luminosity functions, survey yields, and make
the paper figures.  Writes products/results_<tag>.json and figures/*.pdf.

Usage: python postprocess.py [tag]   (default: fiducial)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import extinction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import popsynth as ps  # noqa: E402

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
PC = 3.086e18; C = 2.998e10; YR = 3.156e7; SIGMA_SB = 5.670e-5; H = 6.626e-27; KB = 1.381e-16
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 6.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 6, 'axes.titlesize': 7, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

# effective wavelengths (Angstrom) of the bands in run_lcs.BANDS
LAMBDA_EFF = {'W4': 220883, 'F2100W': 207950,   # WISE W4 and JWST/MIRI F2100W, echo photometry only
              'UVW2': 2085, 'u': 3671, 'g': 4741, 'r': 6173, 'i': 7502, 'z': 8679, 'y': 9711,
              'VIS': 6870, 'YE': 10689, 'JE': 13423, 'HE': 17411,
              'F062': 6142, 'F106': 10465, 'F146': 13050, 'F158': 15578, 'F213': 21117,
              'W1': 33526, 'W2': 46028, 'S1': 9300, 'S2': 13750, 'S3': 20300, 'S4': 31200, 'S5': 41200, 'S6': 47100}

# Survey detection thresholds on the peak apparent AB magnitude, and sky fractions.
SURVEYS = {
    'ZTF ($g < 19.5$)':            dict(band='g', mlim=19.5, fsky=0.60),
    'Rubin WFD ($g < 24.0$)':      dict(band='g', mlim=24.0, fsky=0.44),
    'Euclid Wide (VIS $< 24.5$)':  dict(band='VIS', mlim=24.5, fsky=0.34, snapshot=True),
    'Roman HLWAS (F106 $< 26.0$)': dict(band='F106', mlim=26.0, fsky=0.048, snapshot=True),
    'Roman HLTDS (F106 $< 26.0$)': dict(band='F106', mlim=26.0, fsky=4.6e-4),
    'NEOWISE (W1 $< 19.0$)':       dict(band='W1', mlim=19.0, fsky=1.0),
    'NEOWISE (W2 $< 18.5$)':       dict(band='W2', mlim=18.5, fsky=1.0),
    'SPHEREx (S3 $< 19.5$)':       dict(band='S3', mlim=19.5, fsky=1.0),
    'SPHEREx (S6 $< 18.5$)':       dict(band='S6', mlim=18.5, fsky=1.0),
}


def alam_over_av(band, rv=5.0):
    """Extinction curve: CCM89 with R_V = 5 to 3.3 micron, Indebetouw et al.
    (2005) mid-infrared ratios relative to K beyond."""
    lam = LAMBDA_EFF[band]
    ak = extinction.ccm89(np.array([21900.0]), 1.0, rv)[0]
    if lam <= 33000:
        return extinction.ccm89(np.array([float(lam)]), 1.0, rv)[0]
    ratios = {'W1': 0.56, 'W2': 0.43, 'S5': 0.47, 'S6': 0.43, 'S4': 0.60}
    return ak * ratios.get(band, 0.45)


def bb_nu(nu, T):
    x = H * nu / (KB * T)
    return 2 * H * nu ** 3 / C ** 2 / np.expm1(np.clip(x, 1e-6, 700))


def abmag_from_LT(L, T, lam_obs_A, z):
    """AB magnitude of a blackbody of bolometric luminosity L and temperature T
    observed at wavelength lam_obs at redshift z."""
    dl = cosmo.luminosity_distance(z).value * 1e6 * PC
    nu_rest = C / (lam_obs_A * 1e-8) * (1 + z)
    lnu = np.pi * bb_nu(nu_rest, T) * L / (SIGMA_SB * T ** 4)      # erg/s/Hz (rest)
    fnu = lnu * (1 + z) / (4 * np.pi * dl ** 2)
    return -2.5 * np.log10(np.maximum(fnu, 1e-99)) - 48.6


def load(tag):
    cat = np.load(os.path.join(ROOT, 'products', 'catalog_%s.npz' % tag), allow_pickle=True)
    lcs = np.load(os.path.join(ROOT, 'products', 'lcs_%s.npz' % tag), allow_pickle=True)
    pops = {}
    for p in ['eng', 'field']:
        d = {k[len(p) + 1:]: cat[k] for k in cat.files if k.startswith(p + '_')}
        d.update({k[len(p) + 1:]: lcs[k] for k in lcs.files if k.startswith(p + '_')})
        n = len(d['lpeak'])
        for k in list(d):
            if hasattr(d[k], '__len__') and len(d[k]) > n:
                d[k] = d[k][:n]
        pops[p] = d
    meta = {k: cat[k].item() if cat[k].shape == () else cat[k] for k in
            ['ndot_eng', 'ndot_field', 'active_frac', 'mean_rate_per_psb', 'V_eff', 'lifetime', 'lam']}
    bands = list(lcs['bands']); t_obs = lcs['t_obs']; t_rest = lcs['t_rest']
    return pops, meta, bands, t_obs, t_rest


def lc_shape_stats(d, t_rest):
    """Rise (10% -> peak), decline (peak -> half), and late-time slope of L_bol."""
    n = len(d['lpeak'])
    trise = np.full(n, np.nan); thalf = np.full(n, np.nan); slope = np.full(n, np.nan); tpk = np.full(n, np.nan)
    for i in range(n):
        L = d['lbol'][i].astype(float);
        if L.max() <= 0: continue
        ip = L.argmax(); tp = t_rest[ip]; tpk[i] = tp
        pre = np.where(L[:ip] < 0.1 * L[ip])[0]
        trise[i] = tp - (t_rest[pre[-1]] if len(pre) else t_rest[0])
        post = np.where(L[ip:] < 0.5 * L[ip])[0]
        thalf[i] = (t_rest[ip + post[0]] - tp) if len(post) else np.nan
        t1, t2 = tp + 200, tp + 600
        if t2 < t_rest[-1]:
            L1, L2 = np.interp(t1, t_rest, L), np.interp(t2, t_rest, L)
            if L1 > 0 and L2 > 0:
                slope[i] = np.log(L2 / L1) / np.log(t2 / t1)
    return trise, thalf, slope, tpk


def gband_thalf(mags_g, t_obs, z):
    """Rest-frame time for the g-band light curve to fade by a factor of two
    (0.753 mag) from its peak, as in Yao et al. (2023).  mags_g: (n, n_t)."""
    n = mags_g.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        m = mags_g[i].astype(float)
        if not np.isfinite(m).any():
            continue
        ip = np.nanargmin(m)
        post = np.where(m[ip:] > m[ip] + 0.753)[0]
        if len(post):
            j = ip + post[0]
            # linear interpolation in time between j-1 and j
            f = (m[ip] + 0.753 - m[j - 1]) / max(m[j] - m[j - 1], 1e-6)
            t_cross = t_obs[j - 1] + f * (t_obs[j] - t_obs[j - 1])
            out[i] = (t_cross - t_obs[ip]) / (1 + z[i])
    return out


def echo(L_rest, t_rest, mh, cosi, fomega=ps.F_OMEGA, r_in_pc=None, smooth=True, t_max_yr=None):
    """Infrared echo light curve (erg/s) from the inner face of the disk.
    Returns (t_yr, L_IR, T_barvainis, T_gray)."""
    m6 = mh / 1e6
    # inner edge r_b - R_MC with the f_* = 0.22 scalings (scripts/engine_window_exponents.py)
    r_in = (4.8 * m6 ** 1.00 - 0.71 * m6 ** 1.49 if r_in_pc is None else r_in_pc) * PC
    tau0 = r_in / C / YR
    span = max(2.2 * tau0 + 6, t_max_yr or 0.0)                 # yr; t_max_yr extends the grid past the ring's own window
    tgrid = np.linspace(0, span, int(2400 * span / (2.2 * tau0 + 6)))
    dt = tgrid[1] - tgrid[0]
    # never let the ring collapse below ~one grid cell (exactly face-on limit)
    sini = max(np.sqrt(max(1 - cosi ** 2, 0.0)), 1.5 * dt / tau0)
    # transfer function: ring of radius r_in inclined by i, normalised to fomega
    x = (1 - tgrid / tau0) / sini
    psi = np.where(np.abs(x) < 1, 1.0 / (np.pi * tau0 * sini * np.sqrt(np.clip(1 - x ** 2, 1e-12, None))), 0.0)
    if smooth:  # light-travel spread across the cloud-scale corrugation of the inner face
        w = min(max(int(round(0.71 * m6 ** 1.49 * PC / C / YR / dt)), 1), len(tgrid) // 2)
        psi = np.convolve(psi, np.ones(w) / w, mode='same')[:len(tgrid)]
    psi *= fomega / max(np.trapezoid(psi, tgrid), 1e-30)
    # flare on the same grid (yr)
    Lf = np.interp(tgrid, t_rest / 365.25, L_rest, left=0, right=0)
    Lir = np.convolve(Lf, psi, mode='full')[:len(tgrid)] * dt
    # equivalent incident luminosity and dust temperatures at r_in
    Linc = Lir / fomega
    T_barv = 1500.0 * (r_in / PC / (1.3 * np.sqrt(np.maximum(Linc, 1e30) / 1e46))) ** (-1 / 2.8)
    T_gray = (np.maximum(Linc, 1e30) / (16 * np.pi * SIGMA_SB * r_in ** 2)) ** 0.25
    return tgrid, Lir, T_barv, T_gray, tau0


def main(tag='fiducial'):
    pops, meta, bands, t_obs, t_rest = load(tag)
    R = {'tag': tag, **{k: float(v) if not isinstance(v, str) else v for k, v in meta.items()}}
    bi = {b: bands.index(b) for b in bands}
    ext = {b: alam_over_av(b) for b in bands}
    R['alam_over_av'] = ext

    # ------------------------------------------------------------ per-event derived quantities
    for p, d in pops.items():
        d['trise'], d['thalf'], d['slope'], d['tpk'] = lc_shape_stats(d, t_rest)
        d['thalf_g'] = gband_thalf(d['mags'][:, :, bi['g']], t_obs, d['z'][:len(d['lpeak'])])
        d['peakmag'] = np.nanmin(d['mags'], axis=1)                           # (n, nbands) apparent
        dm = 5 * np.log10(cosmo.luminosity_distance(d['z']).value * 1e5)
        d['absmag_g'] = d['peakmag'][:, bi['g']] - dm[:len(d['peakmag'])] + 2.5 * np.log10(1 + d['z'][:len(d['peakmag'])])
        d['eddratio'] = d['lpeak'] / d['ledd']
        d['partial'] = d['b'] < 1
        d['tph_peak'] = np.array([d['tph'][i][np.nanargmin(d['mags'][i, :, bi['g']])] for i in range(len(d['lpeak']))])
        d['w'] = d['w'][:len(d['lpeak'])]

    e, f = pops['eng'], pops['field']
    we, wf = e['w'] / e['w'].sum(), f['w'] / f['w'].sum()

    def wq(x, w, q=(16, 50, 84)):
        x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & (w > 0)
        if m.sum() < 2:
            return [float('nan')] * len(q)
        xs = np.argsort(x[m]); cw = np.cumsum(w[m][xs]); cw /= cw[-1]
        return [float(np.interp(qq / 100, cw, x[m][xs])) for qq in q]

    R['eng'] = dict(
        logmh=wq(np.log10(e['mh']), we), mstar=wq(e['mstar'], we), beta=wq(e['beta'], we),
        partial_frac=float((we * e['partial']).sum()), full_lc_frac=float((we * e['full_lc']).sum()),
        dmbound=wq(e['dmbound'], we), loglpeak=wq(np.log10(e['lpeak']), we), logedd=wq(np.log10(e['eddratio']), we),
        frac_supereddington=float((we * (e['eddratio'] > 0.9)).sum()),
        frac_supereddington_full=float(np.average(e['eddratio'][~e['partial']] > 0.9, weights=we[~e['partial']])),
        loglpeak_full=wq(np.log10(e['lpeak'][~e['partial']]), we[~e['partial']]),
        loglpeak_partial=wq(np.log10(e['lpeak'][e['partial']]), we[e['partial']]),
        thalf_full=wq(e['thalf'][~e['partial']], we[~e['partial']]), thalf_partial=wq(e['thalf'][e['partial']], we[e['partial']]),
        thalf_g=wq(e['thalf_g'], we), thalf_g_full=wq(e['thalf_g'][~e['partial']], we[~e['partial']]), thalf_g_partial=wq(e['thalf_g'][e['partial']], we[e['partial']]),
        tpeak=wq(e['tpk'], we), trise=wq(e['trise'], we), thalf=wq(e['thalf'], we), slope=wq(e['slope'], we),
        slope_partial=wq(e['slope'][e['partial']], we[e['partial']]), slope_full=wq(e['slope'][~e['partial']], we[~e['partial']]),
        absmag_g=wq(e['absmag_g'], we), tph_peak=wq(e['tph_peak'], we), logerad=wq(np.log10(e['erad']), we),
        young_frac=float(e['young'].mean()), mstar_young=wq(e['mstar'][e['young']], we[e['young']]),
        partial_frac_young=float(np.average(e['partial'][e['young']], weights=we[e['young']])) if e['young'].any() else float('nan'),
        partial_frac_old=float(np.average(e['partial'][~e['young']], weights=we[~e['young']])) if (~e['young']).any() else float('nan'))
    R['field'] = dict(
        logmh=wq(np.log10(f['mh']), wf), mstar=wq(f['mstar'], wf), beta=wq(f['beta'], wf),
        partial_frac=float((wf * f['partial']).sum()), full_lc_frac=float((wf * f['full_lc']).sum()),
        dmbound=wq(f['dmbound'], wf), loglpeak=wq(np.log10(f['lpeak']), wf), logedd=wq(np.log10(f['eddratio']), wf),
        frac_supereddington=float((wf * (f['eddratio'] > 0.9)).sum()),
        frac_supereddington_full=float(np.average(f['eddratio'][~f['partial']] > 0.9, weights=wf[~f['partial']])),
        loglpeak_full=wq(np.log10(f['lpeak'][~f['partial']]), wf[~f['partial']]),
        loglpeak_partial=wq(np.log10(f['lpeak'][f['partial']]), wf[f['partial']]),
        thalf_full=wq(f['thalf'][~f['partial']], wf[~f['partial']]), thalf_partial=wq(f['thalf'][f['partial']], wf[f['partial']]),
        thalf_g=wq(f['thalf_g'], wf), thalf_g_full=wq(f['thalf_g'][~f['partial']], wf[~f['partial']]), thalf_g_partial=wq(f['thalf_g'][f['partial']], wf[f['partial']]),
        tpeak=wq(f['tpk'], wf), trise=wq(f['trise'], wf), thalf=wq(f['thalf'], wf), slope=wq(f['slope'], wf),
        slope_partial=wq(f['slope'][f['partial']], wf[f['partial']]), slope_full=wq(f['slope'][~f['partial']], wf[~f['partial']]),
        absmag_g=wq(f['absmag_g'], wf), tph_peak=wq(f['tph_peak'], wf), logerad=wq(np.log10(f['erad']), wf))

    # ------------------------------------------------------------ screen
    obscured = e['cosi'] < ps.F_OMEGA
    R['obscured_frac'] = float(obscured.mean())
    R['screen_dimming_mag'] = {av: {b: float(av * ext[b]) for b in ['g', 'z', 'HE', 'F213', 'W1', 'W2', 'S6']} for av in [50, 400]}

    def screened_peakmag(d, av, polar_av=0.0, band='g'):
        m = d['peakmag'][:, bi[band]].copy()
        if 'cosi' in d:
            ob = d['cosi'] < ps.F_OMEGA
            m[ob] += av * ext[band]; m[~ob] += polar_av * ext[band]
        return m

    # ------------------------------------------------------------ yields
    yields = {}
    for name, s in SURVEYS.items():
        row = {}
        for p, d, in [('eng', e), ('field', f)]:
            for av in ([50, 400] if p == 'eng' else [0]):
                m = screened_peakmag(d, av, band=s['band'])
                det = m < s['mlim']
                if s.get('snapshot'):
                    # number present above the limit at any instant: rate x duration above limit
                    dur = np.zeros(len(m))
                    for i in np.where(det)[0]:
                        lc = d['mags'][i, :, bi[s['band']]] + (av * ext[s['band']] if (p == 'eng' and d['cosi'][i] < ps.F_OMEGA) else 0)
                        dur[i] = np.trapezoid((lc < s['mlim']).astype(float), t_obs) / 365.25
                    val = float(s['fsky'] * meta['V_eff'] * (d['w'] * dur).sum())
                else:
                    val = float(s['fsky'] * meta['V_eff'] * (d['w'] * det).sum())
                row['%s_AV%d' % (p, av) if p == 'eng' else p] = val
                if p == 'eng' and av == 50:
                    row['eng_zmed'] = float(np.median(d['z'][det])) if det.sum() > 2 else None
                    row['eng_partial_frac_det'] = float(np.average(d['partial'][det], weights=d['w'][det])) if det.sum() > 2 else None
                    row['eng_logmh_det'] = wq(np.log10(d['mh'][det]), d['w'][det]) if det.sum() > 2 else None
                if p == 'field':
                    row['field_zmed'] = float(np.median(d['z'][det])) if det.sum() > 2 else None
                    row['field_logmh_det'] = wq(np.log10(d['mh'][det]), d['w'][det]) if det.sum() > 2 else None
        yields[name] = row
    R['yields_per_yr_fspark1'] = yields
    R['ndot_psb_observed'] = 1.0e-7
    R['fspark_lam_needed'] = R['ndot_psb_observed'] / meta['ndot_eng']

    # ------------------------------------------------------------ echoes
    # Two limits: a thin illuminated ring at r_in (no light-travel spread beyond
    # inclination), and a face corrugated on the cloud scale (spread R_MC/c).
    zs_echo = [0.01, 0.03, 0.1]
    # rate-weighted resample (with replacement) so that unweighted statistics of the subset are rate-weighted
    idx = np.random.default_rng(1).choice(len(e['lpeak']), min(4000, len(e['lpeak'])), replace=True, p=we)
    wi = np.full(len(idx), 1.0 / len(idx))
    R['echo'] = {'z_grid': zs_echo}
    ech_store = {}
    for label, smooth in [('ring', False), ('smoothed', True)]:
        ech = {'tau0_yr': [], 'lir_peak': [], 'dur_yr': [], 'T_barv': [], 'T_gray': [], 'cosi': [],
               'W2_peak_z': [], 'S6_peak_z': [], 'F213_peak_z': [], 'W4_peak_z': [], 'F2100W_peak_z': [],
               'duty41': [], 'duty42': []}
        for i in idx:
            tg, Lir, Tb, Tg, tau0 = echo(e['lbol'][i].astype(float), t_rest, e['mh'][i], e['cosi'][i], smooth=smooth)
            ip = Lir.argmax()
            ech['tau0_yr'].append(tau0); ech['lir_peak'].append(Lir[ip]); ech['cosi'].append(e['cosi'][i])
            ech['dur_yr'].append(np.trapezoid((Lir > 0.5 * Lir[ip]).astype(float), tg))
            ech['T_barv'].append(Tb[ip]); ech['T_gray'].append(Tg[ip])
            for key, band in [('W2_peak_z', 'W2'), ('S6_peak_z', 'S6'), ('F213_peak_z', 'F213'), ('W4_peak_z', 'W4'), ('F2100W_peak_z', 'F2100W')]:
                ech[key].append([abmag_from_LT(Lir[ip], Tb[ip], LAMBDA_EFF[band], z) for z in zs_echo])
            g = ps.gamma_eq(e['mh'][i])
            ech['duty41'].append(np.trapezoid((Lir > 1e41).astype(float), tg) * g)
            ech['duty42'].append(np.trapezoid((Lir > 1e42).astype(float), tg) * g)
        for k in ech: ech[k] = np.array(ech[k])
        ech_store[label] = ech
        face = ech['cosi'] > 0.9; edge = ech['cosi'] < 0.3
        R['echo'][label] = dict(
            tau0_yr=wq(ech['tau0_yr'], wi), lir_peak_face=wq(np.log10(ech['lir_peak'][face]), wi[face]),
            lir_peak_edge=wq(np.log10(ech['lir_peak'][edge]), wi[edge]), lir_peak_all=wq(np.log10(ech['lir_peak']), wi),
            dur_face=wq(ech['dur_yr'][face], wi[face]), dur_edge=wq(ech['dur_yr'][edge], wi[edge]),
            T_barv_face=wq(ech['T_barv'][face], wi[face]), T_gray_face=wq(ech['T_gray'][face], wi[face]),
            T_barv_all=wq(ech['T_barv'], wi),
            W2_peak_face=[wq(ech['W2_peak_z'][face][:, j], wi[face]) for j in range(len(zs_echo))],
            S6_peak_face=[wq(ech['S6_peak_z'][face][:, j], wi[face]) for j in range(len(zs_echo))],
            F213_peak_face=[wq(ech['F213_peak_z'][face][:, j], wi[face]) for j in range(len(zs_echo))],
            W4_peak_face=[wq(ech['W4_peak_z'][face][:, j], wi[face]) for j in range(len(zs_echo))],
            F2100W_peak_face=[wq(ech['F2100W_peak_z'][face][:, j], wi[face]) for j in range(len(zs_echo))],
            W4_peak_all=[wq(ech['W4_peak_z'][:, j], wi) for j in range(len(zs_echo))],
            lir_over_lpeak_face=wq(ech['lir_peak'][face] / e['lpeak'][idx][face], wi[face]),
            lir_over_lpeak_exact_face=wq(ech['lir_peak'][ech['cosi'] > 0.995] / e['lpeak'][idx][ech['cosi'] > 0.995], wi[ech['cosi'] > 0.995]),
            duty_1e41=wq(ech['duty41'], wi), duty_1e42=wq(ech['duty42'], wi),
            frac_duty42_pos=float(np.average(ech['duty42'] > 0, weights=wi)))
    ech = ech_store['ring']
    R['echo']['mean_ir_lum'] = float(np.average(ps.F_OMEGA * e['erad'] * ps.gamma_eq(e['mh']) / YR, weights=we))
    R['echo']['r_in_pc'] = wq(4.8 * (e['mh'] / 1e6) ** 1.00 - 0.71 * (e['mh'] / 1e6) ** 1.49, we)
    R['echo']['rmc_over_c_yr'] = wq(0.71 * (e['mh'] / 1e6) ** 1.49 * PC / C / YR, we)

    # ------------------------------------------------------------ LaTeX table rows
    os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
    if tag == 'fiducial':
        E, Fd = R['eng'], R['field']

        def pm(q, fmt='%.2f'):
            return '$' + (fmt % q[1]) + '^{+' + (fmt % (q[2] - q[1])) + '}_{-' + (fmt % (q[1] - q[0])) + '}$'

        def pmk(q):   # kilo-Kelvin
            return pm([x / 1e4 for x in q], '%.1f')
        nd = r'\nodata'
        lc_rows = [
            r'$\log M_{\rm h}/\msun$ & %s & %s & %s & %s & %s & %s \\' % (pm(E['logmh']), nd, nd, pm(Fd['logmh']), nd, nd),
            r'$M_\ast/\msun$ & %s & %s & %s & %s & %s & %s \\' % (pm(E['mstar']), nd, nd, pm(Fd['mstar']), nd, nd),
            r'partial fraction & $%.2f$ & %s & %s & $%.2f$ & %s & %s \\' % (E['partial_frac'], nd, nd, Fd['partial_frac'], nd, nd),
            r'$\log L_{\rm peak}$ (erg s$^{-1}$) & %s & %s & %s & %s & %s & %s \\' % (
                pm(E['loglpeak']), pm(E['loglpeak_full']), pm(E['loglpeak_partial']), pm(Fd['loglpeak']), pm(Fd['loglpeak_full']), pm(Fd['loglpeak_partial'])),
            r'fraction with $L_{\rm peak} > 0.9\,L_{\rm Edd}$ & $%.2f$ & $%.2f$ & %s & $%.2f$ & $%.2f$ & %s \\' % (
                E['frac_supereddington'], E['frac_supereddington_full'], nd, Fd['frac_supereddington'], Fd['frac_supereddington_full'], nd),
            r'$M_g$ at peak & %s & %s & %s & %s & %s & %s \\' % (pm(E['absmag_g'], '%.1f'), nd, nd, pm(Fd['absmag_g'], '%.1f'), nd, nd),
            r'$t_{\rm peak}$ after first fallback (d) & %s & %s & %s & %s & %s & %s \\' % (pm(E['tpeak'], '%.0f'), nd, nd, pm(Fd['tpeak'], '%.0f'), nd, nd),
            r'$t_{1/2,\rm decline}$, bolometric (d) & %s & %s & %s & %s & %s & %s \\' % (
                pm(E['thalf'], '%.0f'), pm(E['thalf_full'], '%.0f'), pm(E['thalf_partial'], '%.0f'), pm(Fd['thalf'], '%.0f'), pm(Fd['thalf_full'], '%.0f'), pm(Fd['thalf_partial'], '%.0f')),
            r'$t_{1/2,\rm decline}$, $g$ band (d) & %s & %s & %s & %s & %s & %s \\' % (
                pm(E['thalf_g'], '%.0f'), pm(E['thalf_g_full'], '%.0f'), pm(E['thalf_g_partial'], '%.0f'), pm(Fd['thalf_g'], '%.0f'), pm(Fd['thalf_g_full'], '%.0f'), pm(Fd['thalf_g_partial'], '%.0f')),
            r'${\rm d}\ln L/{\rm d}\ln t$ (200--600 d) & %s & %s & %s & %s & %s & %s \\' % (
                pm(E['slope']), pm(E['slope_full']), pm(E['slope_partial']), pm(Fd['slope']), pm(Fd['slope_full']), pm(Fd['slope_partial'])),
            r'$T_{\rm ph}$ at peak ($10^{4}$~K) & %s & %s & %s & %s & %s & %s \\' % (pmk(E['tph_peak']), nd, nd, pmk(Fd['tph_peak']), nd, nd),
            r'$\log E_{\rm rad}$ (erg) & %s & %s & %s & %s & %s & %s \\' % (pm(E['logerad'], '%.1f'), nd, nd, pm(Fd['logerad'], '%.1f'), nd, nd),
        ]
        open(os.path.join(ROOT, 'tables', 'lcstats_rows.tex'), 'w').write('\n'.join(lc_rows) + '\n')
        rows = []
        for b, lab in [('UVW2', 'UVW2'), ('g', '$g$'), ('z', '$z$'), ('YE', r'\euclid $Y_{\rm E}$'), ('HE', r'\euclid $H_{\rm E}$'),
                       ('F213', r'\romantel F213'), ('S4', r'\spherex 2.4--3.8~$\um$'), ('W1', 'W1'), ('W2', 'W2')]:
            rows.append('%s & %.2f & %.3f & %.1f & %.0f \\\\' % (lab, LAMBDA_EFF[b] / 1e4, ext[b], 50 * ext[b], 400 * ext[b]))
        open(os.path.join(ROOT, 'tables', 'screen_rows.tex'), 'w').write('\n'.join(rows) + '\n')
        fm = R['fspark_lam_needed']; fobs = R['ndot_psb_observed'] * 3 / meta['ndot_field']   # scale field to observed 3e-7
        rows = []
        for name, y in yields.items():
            def fmt(v):
                if v is None or not np.isfinite(v): return r'\nodata'
                if v >= 100: return '%.0f' % v
                if v >= 10: return '%.0f' % v
                if v >= 1: return '%.1f' % v
                return '%.2f' % v
            lm = y.get('eng_logmh_det'); pf = y.get('eng_partial_frac_det'); zm = y.get('eng_zmed')
            rows.append('%s & %s & %s & %s & %s & %s & %s \\\\' % (
                name, fmt(y['field'] * fobs), fmt(y['eng_AV50']), fmt(y['eng_AV50'] * fm),
                ('%.2f' % zm) if zm is not None else r'\nodata',
                ('%.1f--%.1f' % (lm[0], lm[2])) if lm is not None else r'\nodata',
                ('%.2f' % pf) if pf is not None else r'\nodata'))
        open(os.path.join(ROOT, 'tables', 'yield_rows.tex'), 'w').write('\n'.join(rows) + '\n')
        R['field_rescale_to_observed'] = fobs

    json.dump(R, open(os.path.join(ROOT, 'products', 'results_%s.json' % tag), 'w'), indent=1, default=float)

    # ============================================================ figures
    if tag != 'fiducial':
        return R
    # --- Fig: BHMF and disruption mass distributions
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    lg = np.linspace(8, 12, 400); rng = np.random.default_rng(3)
    for rel, ls, lab in [('RV15', '-', 'Reines & Volonteri (2015)'), ('KH13', '--', 'Kormendy & Ho (2013)')]:
        samp = np.interp(rng.random(400000), np.cumsum(ps.gsmf(lg)) / np.cumsum(ps.gsmf(lg))[-1], lg)
        lmb = ps.logmbh_from_logmstar(samp, rel, size=len(samp))
        h, ed = np.histogram(lmb, bins=np.linspace(4, 10, 49))
        ngal = np.trapezoid(ps.gsmf(lg), lg)
        ax[0].step(0.5 * (ed[1:] + ed[:-1]), h / h.sum() * ngal / (ed[1] - ed[0]), where='mid', ls=ls, color='k', label=lab)
    ax[0].axvspan(np.log10(ps.M_FLOOR), np.log10(ps.M_CEIL), color='tab:orange', alpha=0.25, label='engine window')
    for a in ax:   # outer 1-sigma soft edges applied in the Monte Carlo (below the floor, above the ceiling)
        a.axvline(np.log10(ps.M_FLOOR) - ps.SIG_FLOOR, color='tab:orange', ls='--', lw=0.8)
        a.axvline(np.log10(ps.M_CEIL) + ps.SIG_CEIL, color='tab:orange', ls='--', lw=0.8)
    ax[0].set_yscale('log'); ax[0].set_ylim(1e-6, 1e-1); ax[0].set_xlim(4.5, 9.5)
    ax[0].set_xlabel(r'$\log_{10} M_{\rm h}/M_\odot$'); ax[0].set_ylabel(r'$\phi$ (Mpc$^{-3}$ dex$^{-1}$)'); ax[0].legend(loc='lower left')
    bins = np.linspace(5, 8.5, 36)
    ax[1].axvspan(np.log10(ps.M_FLOOR), np.log10(ps.M_CEIL), color='tab:orange', alpha=0.12)
    ax[1].hist(np.log10(f['mh']), bins=bins, weights=wf, histtype='step', color='gray', lw=1.5, label='field disruptions')
    ax[1].hist(np.log10(e['mh']), bins=bins, weights=we, histtype='step', color='tab:orange', lw=1.5, label='engine disruptions')
    # detected by Rubin
    for d, w, c in [(f, wf, 'gray'), (e, we, 'tab:orange')]:
        det = screened_peakmag(d, 50, band='g') < 24.0
        ax[1].hist(np.log10(d['mh'][det]), bins=bins, weights=w[det] / w.sum() * w.sum() / max(w[det].sum(), 1e-30) * 0.999, histtype='stepfilled', color=c, alpha=0.25)
    # Mummery & van Velzen (2025) fitted black hole mass distribution of the observed TDE population
    # (their Eq. 1, read as a density per logarithmic mass; alpha_l is unconstrained, >0), normalized to
    # the same area as the Rubin-detected field histogram, without Hills-mass suppression.
    lm = np.linspace(5.0, 8.5, 400); m = 10 ** lm
    al, ah, Mc, Mg, gam = 1.0, -0.85, 10 ** 5.8, 6.4e7, 0.49
    pmv = m ** ah / (1 + (Mc / m) ** (al - ah)) * np.exp(-(m / Mg) ** gam)
    detf = screened_peakmag(f, 50, band='g') < 24.0
    area_det = wf[detf].sum() / wf.sum() * (bins[1] - bins[0]) / (bins[1] - bins[0])   # fraction of field rate detected
    hist_det = np.histogram(np.log10(f['mh'][detf]), bins=bins, weights=wf[detf] / wf.sum() * wf.sum() / max(wf[detf].sum(), 1e-30) * 0.999)[0]
    pmv *= hist_det.sum() * (bins[1] - bins[0]) / np.trapezoid(pmv, lm)
    ax[1].plot(lm, pmv, color='tab:purple', ls=':', lw=1.4, label='observed TDE hosts (Mummery & van Velzen 2025)')
    ax[1].set_xlabel(r'$\log_{10} M_{\rm h}/M_\odot$'); ax[1].set_ylabel('rate-weighted fraction per bin'); ax[1].legend(loc='center right', fontsize=5.5)
    ax[1].text(0.97, 0.78, 'filled: Rubin $g<24$ subsample', transform=ax[1].transAxes, va='top', ha='right', fontsize=7)
    ymax = np.histogram(np.log10(e['mh']), bins=bins, weights=we)[0].max()
    if ymax > 0.3:
        ax[1].set_ylim(0, 0.3)
        ax[1].annotate(r'floor spike reaches %.2f' % ymax, xy=(np.log10(ps.M_FLOOR) + 0.05, 0.295),
                       xytext=(np.log10(ps.M_FLOOR) + 0.55, 0.27), fontsize=7, va='center', ha='left',
                       arrowprops=dict(arrowstyle='-', color='tab:orange', lw=0.8),
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.15))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'bhmf.pdf')); plt.close(fig)

    # --- Fig: light-curve property distributions
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6))
    panels = [(np.log10(e['lpeak']), np.log10(f['lpeak']), r'$\log_{10} L_{\rm peak}$ (erg s$^{-1}$)', np.linspace(41, 45.5, 40)),
              (e['absmag_g'], f['absmag_g'], r'$M_g$ at peak', np.linspace(-23, -13, 40)),
              (np.log10(e['eddratio']), np.log10(f['eddratio']), r'$\log_{10} L_{\rm peak}/L_{\rm Edd}$', np.linspace(-3, 0.5, 40)),
              (np.log10(e['thalf_g']), np.log10(f['thalf_g']), r'$\log_{10} t_{1/2,\rm decline}$ ($g$ band; d)', np.linspace(0.5, 3.2, 40)),
              (e['slope'], f['slope'], r'late-time slope d$\ln L$/d$\ln t$', np.linspace(-4, 0, 40)),
              (np.log10(e['dmbound']), np.log10(f['dmbound']), r'$\log_{10} \Delta M_{\rm bound}/M_\odot$', np.linspace(-4, 0.5, 40))]
    # observed ZTF sample (Yao et al. 2023), each event weighted by 1/V_max so the histogram is rate-weighted
    yao = json.load(open(os.path.join(HERE, 'yao2023_sample.json')))
    yv = [v for v in yao.values() if 'logMBH' in v]
    yw = np.array([v['w_1overV'] for v in yv]); yw /= yw.sum()
    y_logL = np.array([v['logLbb'] for v in yv])
    y_Lg = 10 ** np.array([v['logLg'] for v in yv])
    y_Mg = -2.5 * np.log10(y_Lg / (C / 4741e-8) / (4 * np.pi * (10 * PC) ** 2)) - 48.6
    y_ledd = np.array([v['logLbb'] for v in yv]) - np.log10(1.26e38 * 10 ** np.array([v['logMBH'] for v in yv]))
    y_thalf = np.log10(np.array([v['t_decline'] for v in yv]))
    observed = {0: y_logL, 1: y_Mg, 2: y_ledd, 3: y_thalf}
    # actual volumetric normalizations: field at its theoretical rate, engine at the rate-matched
    # normalization, observed events at their summed 1/V_max rate densities (Mpc^-3 yr^-1)
    y_rate = np.array([v['w_1overV'] for v in yv])
    wf_abs = wf * meta['ndot_field']; we_abs = we * meta['ndot_eng'] * R['fspark_lam_needed']
    for k, (a, (xe, xf, lab, bins)) in enumerate(zip(ax.ravel(), panels)):
        bw = bins[1] - bins[0]
        a.hist(xf, bins=bins, weights=wf_abs / bw, histtype='step', color='gray', lw=1.5, label='field (theoretical rate)')
        a.hist(xe, bins=bins, weights=we_abs / bw, histtype='step', color='tab:orange', lw=1.5, label=r'engine ($\dot n_{\rm eng}=10^{-7}$)')
        a.hist(xe[e['partial']], bins=bins, weights=we_abs[e['partial']] / bw, histtype='stepfilled', color='tab:orange', alpha=0.25, label='engine, partial')
        if k in observed:
            ob = bins[::3]   # coarser bins for 33 events
            a.hist(observed[k], bins=ob, weights=y_rate / (3 * bw), histtype='step', color='tab:blue', lw=1.2, ls='--', label=r'observed (ZTF, $\sum 1/V_{\rm max}$)')
        a.set_xlabel(lab); a.set_yscale('log'); a.set_ylim(3e-10, 3e-5)
    for a in ax[:, 0]:
        a.set_ylabel(r'd$\dot n$/d$x$ (Mpc$^{-3}$ yr$^{-1}$ dex$^{-1}$ or mag$^{-1}$)')
    ax[0, 0].legend(loc='upper left', fontsize=5.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'lcdist.pdf')); plt.close(fig)

    # --- Fig: luminosity functions (g and W1) with and without the screen
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    bins = np.linspace(-23, -13, 26)
    for d, w, c, lab in [(f, wf * meta['ndot_field'], 'gray', 'field'), (e, we * meta['ndot_eng'] * R['fspark_lam_needed'], 'tab:orange', r'engine ($\dot n_{\rm eng}=10^{-7}$)')]:
        dm = 5 * np.log10(cosmo.luminosity_distance(d['z']).value * 1e5) - 2.5 * np.log10(1 + d['z'])
        for band, a, avs in [('g', ax[0], [0, 50]), ('W1', ax[1], [0, 50, 400])]:
            for av, ls in zip(avs, ['-', '--', ':']):
                m = screened_peakmag(d, av, band=band) - dm
                if 'cosi' not in d and av > 0: continue
                a.hist(m, bins=bins, weights=w / (bins[1] - bins[0]), histtype='step', color=c, ls=ls, lw=1.5,
                       label=lab + (r', $A_V=%d$ in disk' % av if av else ''))
    # observed rest-frame g-band luminosity functions, converted to per-magnitude rate densities
    Mg = np.linspace(-23, -13, 300)
    Lnu = 4 * np.pi * (10 * PC) ** 2 * 10 ** (-(Mg + 48.6) / 2.5)          # erg/s/Hz
    Lg = (C / 4741e-8) * Lnu                                               # nu L_nu at g
    yao = 2.87e-7 * ((Lg / 1.36e43) ** 0.26 + (Lg / 1.36e43) ** 2.58) ** -1 * 0.4   # Yao et al. 2023, per mag
    vv = 1.9e-7 * (Lg / 1e43) ** -1.6 * 0.4                                        # van Velzen 2018, per mag
    m_yao = (Lg > 10 ** 42.5) & (Lg < 10 ** 45.0); m_vv = (Lg > 10 ** 42.3) & (Lg < 10 ** 44.8)
    ax[0].plot(Mg[m_yao], yao[m_yao], color='tab:blue', lw=1.8, label='observed: Yao et al. (2023)')
    ax[0].plot(Mg[m_vv], vv[m_vv], color='tab:blue', lw=1.2, ls='--', label='observed: van Velzen (2018)')
    for a, lab in zip(ax, [r'peak $M_g$', r'peak $M_{W1}$']):
        a.set_yscale('log'); a.set_xlabel(lab); a.set_ylabel(r'd$\dot n$/d$M$ (Mpc$^{-3}$ yr$^{-1}$ mag$^{-1}$)'); a.set_ylim(1e-10, 1e-4); a.invert_xaxis()
    # legends sit in the empty top two decades, clear of every curve
    ax[0].legend(fontsize=5.5, loc='upper left', ncol=1, frameon=False); ax[1].legend(fontsize=5.5, loc='upper left', frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'lf.pdf')); plt.close(fig)

    # --- Fig: echo properties
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    # representative event: median engine mass, full disruption of a 1 Msun star -> pick closest catalog event
    j = idx[np.argmin(np.abs(np.log10(e['mh'][idx]) - 6.0) + 3 * np.abs(e['b'][idx] - 1.0) + np.abs(e['mstar'][idx] - 1))]
    for ci, ls in [(1.0, '-'), (0.7, '--'), (0.3, ':'), (0.05, '-.')]:
        tg, Lir, Tb, Tg, tau0 = echo(e['lbol'][j].astype(float), t_rest, e['mh'][j], ci, smooth=False)
        ax[0].plot(tg, Lir, 'k', ls=ls, label=r'$\cos i=%.2f$' % ci)
        tg, Lir, Tb, Tg, tau0 = echo(e['lbol'][j].astype(float), t_rest, e['mh'][j], ci, smooth=True)
        ax[0].plot(tg, Lir, color='tab:red', ls=ls, lw=1)
    ax[0].plot(t_rest / 365.25, e['lbol'][j], color='tab:blue', lw=1, label='optical/UV flare')
    # the same flare reprocessed by the dust geometries inferred for observed echoes: the sub-parsec,
    # ~1% covering-factor shell of optically selected TDEs (van Velzen et al. 2016; Jiang et al. 2021) and
    # the parsec-scale, AGN-like torus fitted to 1eRASS J0758 by Eyles-Ferris et al. (2026)
    for rpc, fc, ci, lab in [(0.15, 0.01, 0.7, r'$0.15$ pc, $f_c=0.01$ (optical TDE echoes)'),
                             (2.4, 0.3, 0.53, r'$2.4$ pc, $f_c=0.3$ (1eRASS J0758 torus)')]:
        tg, Lir, _, _, _ = echo(e['lbol'][j].astype(float), t_rest, e['mh'][j], ci, fomega=fc, r_in_pc=rpc, smooth=False, t_max_yr=2.2 * tau0 + 2)
        ax[0].plot(tg, Lir, color='tab:gray', ls='-' if rpc < 1 else '--', lw=0.9, label=lab)
    ax[0].set_yscale('log'); ax[0].set_ylim(1e40, 3e45); ax[0].set_xlim(-0.5, 2.2 * tau0 + 2)
    ax[0].set_xlabel('years after disruption'); ax[0].set_ylabel(r'$L$ (erg s$^{-1}$)'); ax[0].legend(fontsize=4.8, loc='upper right', frameon=False, handlelength=1.8)
    def echo_hist(a, y, bins):
        """Histograms of an echo property for the two limits (all sightlines solid,
        the unobscured face-on subset cos i > 0.9 dashed)."""
        for lab, c in [('ring', 'k'), ('smoothed', 'tab:red')]:
            es = ech_store[lab]; v = y(es)
            a.hist(v, bins=bins, weights=wi, histtype='step', color=c, lw=1.5, label=lab.replace('ring', 'thin ring').replace('smoothed', 'cloud-smoothed'))
            fo = es['cosi'] > 0.9
            a.hist(v[fo], bins=bins, weights=wi[fo] / wi[fo].sum() * wi.sum(), histtype='step', color=c, lw=0.9, ls='--')
        a.set_yticks([])

    echo_hist(ax[1], lambda es: np.log10(es['lir_peak']), np.linspace(39, 45.5, 53))
    echo_hist(ax[2], lambda es: np.log10(es['dur_yr']), np.linspace(-0.5, 2, 31))
    for a in ax[1:]:   # headroom for the legend and the literature bars above the histograms
        a.set_ylim(0, a.get_ylim()[1] * 2.4)

    def ref_bars(a, bars, y0=0.66, dy=0.065):
        # horizontal bars marking ranges from the cited echo literature, in axis-fraction rows
        # below the legend; each bar is (x_lo, x_hi, label) in the panel's x units
        xmid = 0.5 * sum(a.get_xlim())
        for k, (lo, hi, lab) in enumerate(bars):
            y = y0 - k * dy
            a.plot([lo, hi], [y, y], color='tab:blue', lw=2.2, solid_capstyle='butt', transform=a.get_xaxis_transform(), zorder=5)
            right = 0.5 * (lo + hi) > xmid
            a.text(hi if right else lo, y + 0.012, lab, fontsize=4.8, color='tab:blue', transform=a.get_xaxis_transform(),
                   va='bottom', ha='right' if right else 'left')
    # peak dust luminosities: Jiang et al. (2021) detections excluding ASASSN-15lh and the AGN-like hosts;
    # Masterson et al. (2024) gold sample L_W2; Nair et al. (2026) luminous NEOWISE sample; Hinkle et al. (2024) ANTs
    ref_bars(ax[1], [(40.7, 42.3, 'optical TDE echoes (Jiang+21)'),
                     (42.0, 43.5, 'MIR-selected TDEs (Masterson+24)'),
                     (43.5, 44.2, 'luminous MIR TDEs (Nair+26)'),
                     (42.0, 45.0, 'ANT echoes (Hinkle+24)')])
    # durations: light-travel spans 2 R_dust / c of the same samples (R < 0.3 pc; 0.05-0.46 pc), the several-year
    # ANT echoes, and the 0.26-2.75 pc rings and tori fitted to individual echoes (Eyles-Ferris+26; Wu+26)
    ref_bars(ax[2], [(np.log10(0.5), np.log10(2.0), r'optical TDE echoes, $2R/c$'),
                     (np.log10(0.3), np.log10(5.0), 'MIR-selected TDEs, rise to 5 yr'),
                     (np.log10(2.0), np.log10(6.0), 'ANT echoes, several yr'),
                     (np.log10(1.7), np.log10(18.0), r'pc-scale ring/torus fits, $2R/c$')])
    ax[1].set_xlabel(r'$\log_{10} L_{\rm IR,peak}$ (erg s$^{-1}$)'); ax[1].set_ylabel('rate-weighted fraction')
    ax[1].legend(fontsize=5.5, loc='upper left', title='solid: all $i$; dashed: $\\cos i > 0.9$', title_fontsize=5.5, frameon=False)
    ax[2].set_xlabel(r'$\log_{10}$ echo duration (yr)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'echo.pdf')); plt.close(fig)

    return R


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'fiducial'
    R = main(tag)
    print(json.dumps({k: R[k] for k in ['ndot_eng', 'ndot_field', 'active_frac', 'fspark_lam_needed', 'obscured_frac']}, indent=1))
    print(json.dumps(R['eng'], indent=1)); print(json.dumps(R['field'], indent=1))
    print(json.dumps(R['yields_per_yr_fspark1'], indent=1)); print(json.dumps(R['echo'], indent=1))
    print(json.dumps(R['screen_dimming_mag'], indent=1))
