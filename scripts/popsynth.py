"""Population synthesis for tidal disruptions in self-sustaining black hole
engines (Guillochon & Loeb 2026) versus the field population.

Produces products/catalog.npz with one row per synthetic disruption for the
engine and field populations, each carrying a rate weight so that
sum(weight) equals the volumetric rate contributed by that population
(Mpc^-3 yr^-1), plus the MOSFiT parameters needed to generate its light curve.

Run with the MOSFiT venv python (numpy, scipy, astropy available).
"""
import json
import os
import sys

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
rng = np.random.default_rng(20260905)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ---------------------------------------------------------------- constants
MSUN = 1.989e33
RSUN = 6.957e10
G = 6.674e-8
C = 2.998e10

# Engine equilibrium at f_* = 0.22 (Guillochon & Loeb 2026, Eqs. 81-84)
M_FLOOR = 7.3e5          # Msun
M_CEIL = 5.0e6           # Msun; TDE momentum ceases to dominate AGN feedback
# Nucleus-to-nucleus scatter of the window edges (dex): the floor through the allowed
# flattening range (M_floor ~ f_*^1.64 over 0.18-0.25), the ceiling through the rate
# normalization and Bondi efficiency (p_TDE/p_AGN ~ M_h^-1.4).  See Section 2.2 of the paper.
SIG_FLOOR = 0.12
SIG_CEIL = 0.25
GAMMA0 = 1.4e-2          # yr^-1 at 1e6 Msun
GAMMA_SLOPE = -0.84
T_GAS0, T_GAS_SLOPE = 5.5e7, 2.31      # yr
T_CUSP0, T_CUSP_SLOPE = 2.8e8, 1.84    # yr
T_EFOLD = 8.8e8 / 1.99                 # yr; growth e-folding at f_Edd = 0.05, eps = 0.05
F_OMEGA = 0.22
R_IN_PC = 4.1            # inner edge of the disk at 1e6 Msun (r_b - R_MC)

# Field loss-cone rate (Stone & Metzger 2016)
def gamma_field(mh):
    return 2.9e-5 * (mh / 1e8) ** -0.404

def gamma_eq(mh, lam=1.0):
    return lam * GAMMA0 * (mh / 1e6) ** GAMMA_SLOPE

def t_gas(mh):
    return T_GAS0 * (mh / 1e6) ** T_GAS_SLOPE

def t_cusp(mh):
    return T_CUSP0 * (mh / 1e6) ** T_CUSP_SLOPE

# ---------------------------------------------------------------- demographics
def gsmf(logm):
    """Baldry et al. (2012) double Schechter, Mpc^-3 dex^-1 (h = 0.7)."""
    x = 10 ** (logm - 10.66)
    return np.log(10) * np.exp(-x) * (3.96e-3 * x ** (-0.35 + 1) + 0.79e-3 * x ** (-1.47 + 1))

def logmbh_from_logmstar(logm, relation='RV15', scatter=True, size=None):
    if relation == 'RV15':        # Reines & Volonteri 2015, local AGN
        mu, sig = 7.45 + 1.05 * (logm - 11.0), 0.55
    else:                          # Kormendy & Ho 2013, ellipticals/bulges (using M* as bulge mass)
        mu, sig = 8.69 + 1.17 * (logm - 11.0), 0.29
    if not scatter:
        return mu
    return mu + sig * rng.standard_normal(size if size is not None else np.shape(logm))

def f_occ(logm, mode='unity'):
    """Black hole occupation fraction of galaxies. 'miller' approximates the
    Miller et al. (2015) constraint that f_occ > 0.2 down to 1e8 Msun."""
    if mode == 'unity':
        return np.ones_like(logm)
    return np.clip(0.2 + 0.8 * (logm - 8.0) / 2.0, 0.2, 1.0)

def psb_weight(logm):
    """Relative stellar-mass distribution of post-starburst hosts: the GSMF
    times a lognormal preference for ~1e10.2 Msun (French et al. 2018)."""
    return gsmf(logm) * np.exp(-0.5 * ((logm - 10.2) / 0.45) ** 2)

N_PSB = 2.0e-5           # Mpc^-3, French et al. 2018

# ---------------------------------------------------------------- stars
def mass_radius(m):
    """ZAMS mass-radius relation (approximation to Tout et al. 1996)."""
    m = np.asarray(m, dtype=float)
    return np.where(m < 1.0, m ** 0.8, m ** 0.57)

def kroupa_pdf(m):
    m = np.asarray(m, dtype=float)
    return np.where(m < 0.5, (m / 0.5) ** -1.3, (m / 0.5) ** -2.3)

def sample_disrupted_mass(n, age_yr):
    """Present-day mass function of a population of the given age, weighted by
    the tidal cross-section r_t ~ R_* M_*^(-1/3) (full loss cone)."""
    mmax = min(20.0, (1.0e10 / age_yr) ** 0.4)
    grid = np.logspace(np.log10(0.1), np.log10(mmax), 400)
    w = kroupa_pdf(grid) * mass_radius(grid) * grid ** (-1.0 / 3.0) * grid  # * m for log spacing
    cdf = np.cumsum(w); cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, grid)

# ---------------------------------------------------------------- impact parameters
def mosfit_betas(b, mstar):
    """MOSFiT scaled impact parameter b -> physical beta for a star of mass
    mstar, reproducing the piecewise-linear map in mosfit/modules/engines/fallback.py."""
    b = np.asarray(b, dtype=float); mstar = np.asarray(mstar, dtype=float)
    lo = b < 1
    b43 = np.where(lo, 0.6 + 1.25 * b, 1.85 + 2.15 * (b - 1))
    b53 = np.where(lo, 0.5 + 0.4 * b, 0.9 + 1.6 * (b - 1))
    gfrac = np.where(mstar <= 0.3, 1.0, np.where(mstar >= 1.0, 0.0, (mstar - 1.0) / (0.3 - 1.0)))
    gfrac = np.where(mstar >= 15, np.clip((mstar - 15.0) / 7.0, 0, 1), gfrac)
    return b53 + (b43 - b53) * (1.0 - gfrac)

def b_from_beta(beta, mstar):
    bgrid = np.linspace(0, 2, 2001)
    return np.array([np.interp(be, mosfit_betas(bgrid, ms), bgrid) for be, ms in zip(beta, mstar)])

def sample_b(mh, mstar, f_full):
    """Draw MOSFiT b. With probability f_full the star arrives from the full
    loss cone (dN/dbeta ~ beta^-2 between the minimum-disruption beta and the
    capture limit); otherwise it diffuses in and grazes at beta ~ beta_min."""
    n = len(mh)
    beta_min = mosfit_betas(np.zeros(n), mstar)
    beta_b2 = mosfit_betas(2 * np.ones(n), mstar)
    rt = RSUN * mass_radius(mstar) * (mh / mstar) ** (1.0 / 3.0)
    rg = G * mh * MSUN / C ** 2
    beta_cap = rt / (4.0 * rg)            # r_p < 4 r_g: swallowed whole (Schwarzschild)
    full = rng.random(n) < f_full
    u = rng.random(n)
    beta_max = np.minimum(beta_cap, 50.0)
    inv = 1.0 / beta_min - u * (1.0 / beta_min - 1.0 / beta_max)
    beta_full = 1.0 / inv
    beta_diff = beta_min * (1.0 + rng.exponential(0.08, n))
    beta = np.where(full, beta_full, beta_diff)
    swallowed = beta_cap < beta_min
    b = b_from_beta(np.minimum(beta, beta_b2), mstar)
    return b, beta, full, swallowed

# ---------------------------------------------------------------- MOSFiT nuisance parameters
POST = json.load(open(os.path.join(HERE, 'nicholl2022_posteriors.json')))
POST_ARR = np.array([[p['logeff'], p['logRph0'], p['lph'], p['logTvisc'], p['lognH'], p['logMbh']] for p in POST])

def darkyear_tvisc(mh, mstar, beta):
    """Viscous (circularization) time following Guillochon & Ramirez-Ruiz (2015).

    The self-intersection of the debris stream, and hence the viscous time
    relative to the fallback peak, is governed by the apsidal precession per
    orbit, i.e. by r_p/r_g.  For a solar-type star at beta = 1 around 1e6 Msun
    (r_p/r_g = 47) flares are slowed by about an order of magnitude; prompt
    flares (3 t_visc < t_peak) around such holes require beta >~ 5 (r_p/r_g ~ 9);
    the transition to mostly prompt flares occurs near 1e7 Msun; and t_visc is
    capped at ~100 t_peak by the time for the stream to cover 4 pi.  We encode
    this as log10(t_visc/t_peak) ~ N(mu, 0.5) with mu = 1.0 + 2.1 log10(r_p/r_g / 47)
    clipped to [-1.5, 2].  t_peak is the fallback peak of Guillochon & Ramirez-Ruiz (2013)."""
    mh = np.asarray(mh, float); mstar = np.asarray(mstar, float); beta = np.asarray(beta, float)
    rstar = mass_radius(mstar)
    rt = RSUN * rstar * (mh / mstar) ** (1.0 / 3.0)
    rp_rg = (rt / beta) / (G * mh * MSUN / C ** 2)
    mu = np.clip(1.0 + 2.1 * np.log10(rp_rg / 47.0), -1.5, 2.0)
    logratio = mu + 0.5 * rng.standard_normal(np.shape(mh))
    t_peak = 41.0 * np.sqrt(mh / 1e6) * mstar ** -1.0 * rstar ** 1.5     # days
    return 10 ** logratio * t_peak


def sample_nuisance(n, mh, scale_eff=False, darkyear=False, mstar=None, beta=None):
    idx = rng.integers(0, len(POST_ARR), n)
    p = POST_ARR[idx].copy()
    p[:, :4] += 0.1 * rng.standard_normal((n, 4))
    if scale_eff:   # Nicholl et al. 2022: eps ~ M_bh^0.97
        p[:, 0] += 0.97 * (np.log10(mh) - p[:, 5])
    logeff = np.clip(p[:, 0], -4, np.log10(0.4))
    logrph = np.clip(p[:, 1], -4, 4)
    lph = np.clip(p[:, 2], 0, 2)          # MOSFiT tde prior range for the photosphere exponent
    logtv = np.clip(p[:, 3], -3, 3)
    lognh = np.clip(p[:, 4], 16, 23)
    tvisc = 10 ** logtv
    if darkyear:
        tvisc = np.clip(darkyear_tvisc(mh, mstar, beta), 1e-3, 1e5)
    return 10 ** logeff, 10 ** logrph, lph, tvisc, 10 ** lognh

# ---------------------------------------------------------------- redshift sampling
Z_MAX = 1.0
def dVdz_over_1pz(z):
    return cosmo.differential_comoving_volume(z).value * 4 * np.pi / (1 + z)  # Mpc^3 per unit z, all sky
_zg = np.linspace(1e-4, Z_MAX, 600)
_cdf = np.cumsum(dVdz_over_1pz(_zg)); V_EFF = _cdf[-1] * (_zg[1] - _zg[0]); _cdf /= _cdf[-1]
def sample_z(n):
    return np.interp(rng.random(n), _cdf, _zg)

# ---------------------------------------------------------------- engine occupancy
def engine_hosts(n_host, lifetime='gas', lam=1.0, relation='RV15', occ='unity', growth='burst'):
    """Monte Carlo over post-starburst nuclei. Returns per-host arrays and the
    mean disruption rate per PSB galaxy.

    growth = 'burst': the M_h-M_* relation gives the black hole mass at the
        time of the burst; the hole grows only once it accretes from the engine
        disk (after switch-on).
    growth = 'now': the relation describes the black holes as they are today,
        after accreting at f_Edd = 0.05 from the circumnuclear gas throughout
        the post-starburst phase; the seed at the burst is smaller by
        exp(-age / t_efold)."""
    lg = np.linspace(8.0, 12.0, 2000)
    w = psb_weight(lg); cdf = np.cumsum(w); cdf /= cdf[-1]
    logm = np.interp(rng.random(n_host), cdf, lg)
    has_bh = rng.random(n_host) < f_occ(logm, occ)
    mi = 10 ** logmbh_from_logmstar(logm, relation)
    age = 10 ** rng.uniform(8.0, np.log10(2e9), n_host)   # time since burst
    if growth == 'now':
        mi = mi * np.exp(-age / T_EFOLD)
    # soft window edges: each nucleus draws its own floor and ceiling
    mfloor = M_FLOOR * 10 ** (SIG_FLOOR * rng.standard_normal(n_host))
    mceil = M_CEIL * 10 ** (SIG_CEIL * rng.standard_normal(n_host))
    below = mi < mfloor
    t_on = np.where(below, T_EFOLD * np.log(mfloor / np.maximum(mi, 1)), 0.0)
    m_on = np.where(below, mfloor, mi)
    life = t_gas(m_on) if lifetime == 'gas' else np.minimum(t_gas(m_on) * 1e9, t_cusp(m_on))
    active = has_bh & (mi <= mceil) & (age > t_on) & (age < t_on + life)
    m_now = np.where(active, m_on * np.exp((age - t_on) / T_EFOLD), mi)
    active &= m_now <= mceil
    rate = np.where(active, gamma_eq(m_now, lam), 0.0)
    return dict(logmstar=logm, mi=mi, age=age, t_on=t_on, active=active, m_now=m_now, rate=rate,
                mfloor=mfloor, mceil=mceil)

# ---------------------------------------------------------------- field
def field_rate_density(relation='RV15', occ='unity', n=400000):
    """Volumetric TDE rate of the general galaxy population and a sample of
    disrupting black hole masses weighted by rate."""
    lg = np.linspace(8.0, 12.0, 2000)
    w = gsmf(lg); cdf = np.cumsum(w); cdf /= cdf[-1]
    n_gal = np.trapezoid(gsmf(lg), lg)                    # Mpc^-3
    logm = np.interp(rng.random(n), cdf, lg)
    mh = 10 ** logmbh_from_logmstar(logm, relation)
    wt = f_occ(logm, occ) * gamma_field(mh) * n_gal / n     # each sample carries this rate density
    # Restrict to the mass range over which the Stone & Metzger (2016) fit is
    # calibrated and where the loss-cone rate is meaningful (M_h >= 1e5 Msun).
    keep = mh >= 1e5
    return logm[keep], mh[keep], wt[keep]

# ---------------------------------------------------------------- build catalogs
def build(n_eng=6000, n_field=6000, lifetime='gas', lam=1.0, relation='RV15', occ='unity',
          f_young=0.5, f_full_eng=0.7, scale_eff=False, darkyear=False, growth='burst', tag='fiducial'):
    hosts = engine_hosts(400000, lifetime, lam, relation, occ, growth)
    mean_rate_per_psb = hosts['rate'].mean()               # yr^-1 per PSB galaxy, f_spark = 1
    ndot_eng = N_PSB * mean_rate_per_psb                   # Mpc^-3 yr^-1
    active_frac = hosts['active'].mean()

    act = np.where(hosts['active'])[0]
    pick = rng.choice(act, n_eng, p=hosts['rate'][act] / hosts['rate'][act].sum())
    mh_e = hosts['m_now'][pick]
    age_e = hosts['age'][pick]
    young = rng.random(n_eng) < f_young
    ms_e = np.where(young, sample_disrupted_mass(n_eng, 1.0), 0.0)
    for i in np.where(young)[0]:
        ms_e[i] = sample_disrupted_mass(1, age_e[i])[0]
    ms_e[~young] = sample_disrupted_mass((~young).sum(), 1.0e10)
    b_e, beta_e, full_e, sw_e = sample_b(mh_e, ms_e, np.full(n_eng, f_full_eng))
    eff_e, rph_e, lph_e, tv_e, nh_e = sample_nuisance(n_eng, mh_e, scale_eff, darkyear, ms_e, beta_e)
    cosi_e = rng.random(n_eng)                              # isotropic disk normals
    z_e = sample_z(n_eng)
    w_e = np.full(n_eng, ndot_eng / n_eng)

    logm_f, mh_f, wt_f = field_rate_density(relation, occ, int(n_field * 1.8))
    n_field = len(mh_f)
    ndot_field = wt_f.sum()
    ms_f = sample_disrupted_mass(n_field, 1.0e10)
    f_full_f = np.clip(0.6 - 0.3 * np.log10(mh_f / 1e6), 0.05, 0.9)
    b_f, beta_f, full_f, sw_f = sample_b(mh_f, ms_f, f_full_f)
    eff_f, rph_f, lph_f, tv_f, nh_f = sample_nuisance(n_field, mh_f, scale_eff, darkyear, ms_f, beta_f)
    z_f = sample_z(n_field)

    out = dict(
        tag=tag, lifetime=lifetime, lam=lam, relation=relation, occ=occ, f_young=f_young,
        f_full_eng=f_full_eng, scale_eff=scale_eff, darkyear=darkyear, growth=growth,
        sig_floor=SIG_FLOOR, sig_ceil=SIG_CEIL,
        ndot_eng=ndot_eng, ndot_field=ndot_field, active_frac=active_frac,
        mean_rate_per_psb=mean_rate_per_psb, V_eff=V_EFF, z_max=Z_MAX,
        host_logmstar=hosts['logmstar'], host_active=hosts['active'], host_m_now=hosts['m_now'],
        host_mi=hosts['mi'], host_age=hosts['age'], host_t_on=hosts['t_on'],
        eng=dict(mh=mh_e, mstar=ms_e, b=b_e, beta=beta_e, full_lc=full_e, swallowed=sw_e, young=young,
                 eff=eff_e, rph0=rph_e, lph=lph_e, tvisc=tv_e, nh=nh_e, cosi=cosi_e, z=z_e, w=w_e, age=age_e),
        field=dict(mh=mh_f, mstar=ms_f, b=b_f, beta=beta_f, full_lc=full_f, swallowed=sw_f,
                   eff=eff_f, rph0=rph_f, lph=lph_f, tvisc=tv_f, nh=nh_f, z=z_f, w=wt_f, logmstar=logm_f))
    return out

def save(cat, path):
    flat = {}
    for k, v in cat.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[k + '_' + kk] = vv
        else:
            flat[k] = v
    np.savez_compressed(path, **flat)

if __name__ == '__main__':
    os.makedirs(os.path.join(ROOT, 'products'), exist_ok=True)
    # Fiducial: slowed circularization (dark year), resupplied disk (cusp-limited
    # lifetime), and black holes that grow throughout the post-starburst phase
    # (the M_h-M_* relation applies today).  Each other variant changes one
    # ingredient relative to the fiducial; 'simple' drops all three.
    FID = dict(darkyear=True, lifetime='cusp', growth='now')
    variants = {
        'fiducial': dict(FID),
        'simple': dict(),
        'prompt': dict(FID, darkyear=False),
        'gaslimited': dict(FID, lifetime='gas'),
        'burstrelation': dict(FID, growth='burst'),
        'young1': dict(FID, f_young=1.0),
        'young0': dict(FID, f_young=0.0),
        'KH13': dict(FID, relation='KH13'),
        'miller': dict(FID, occ='miller'),
        'lam0.1': dict(FID, lam=0.1),
    }
    only = sys.argv[1:]
    if only:
        variants = {k: v for k, v in variants.items() if k in only}
    summary = json.load(open(os.path.join(ROOT, 'products', 'popsynth_summary.json'))) if only and os.path.exists(os.path.join(ROOT, 'products', 'popsynth_summary.json')) else {}
    for tag, kw in variants.items():
        n = 20000 if tag == 'fiducial' else 8000
        cat = build(n_eng=n, n_field=n, tag=tag, **kw)
        save(cat, os.path.join(ROOT, 'products', 'catalog_%s.npz' % tag))
        e, f = cat['eng'], cat['field']
        summary[tag] = dict(
            ndot_eng=cat['ndot_eng'], ndot_field=cat['ndot_field'], active_frac=cat['active_frac'],
            mean_rate_per_psb=cat['mean_rate_per_psb'],
            eng_logmh_p16_50_84=list(np.percentile(np.log10(e['mh']), [16, 50, 84])),
            field_logmh_p16_50_84=list(np.percentile(np.log10(f['mh']), [16, 50, 84])),
            eng_partial_frac=float(np.mean(e['b'] < 1)), field_partial_frac=float(np.mean(f['b'] < 1)),
            eng_mstar_med=float(np.median(e['mstar'])), field_mstar_med=float(np.median(f['mstar'])),
            eng_swallowed=float(e['swallowed'].mean()), field_swallowed=float(f['swallowed'].mean()))
        s = summary[tag]
        print('%-9s ndot_eng=%.2e ndot_field=%.2e active=%.3f <G>/PSB=%.2e  logMh eng %s field %s  partial eng %.2f field %.2f  M* med eng %.2f field %.2f' % (
            tag, s['ndot_eng'], s['ndot_field'], s['active_frac'], s['mean_rate_per_psb'],
            np.round(s['eng_logmh_p16_50_84'], 2), np.round(s['field_logmh_p16_50_84'], 2),
            s['eng_partial_frac'], s['field_partial_frac'], s['eng_mstar_med'], s['field_mstar_med']))
    json.dump(summary, open(os.path.join(ROOT, 'products', 'popsynth_summary.json'), 'w'), indent=1)
