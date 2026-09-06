"""Evaluate the engine equilibrium at the flattening-window midpoint
f_* = f_Omega = 0.22 and report normalizations at 1e6 Msun together with
mass exponents, using the Cohn-Kulsrud solver in the tde-engine repository
(scripts/scan_flattening.py and losscone.py).  Nothing is written into that
repository.  Output: products/engine_window_f022.json and tables/engine_rows.tex.
"""
import json
import os
import sys

import numpy as np
import sympy as sp

ENGINE = os.environ.get('TDE_ENGINE_REPO', r'C:\Users\guill\tde-engine')
os.environ['TDE_FIG_OUT'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
sys.path.insert(0, os.path.join(ENGINE, 'scripts'))
import scan_flattening as S  # noqa: E402
import losscone as L         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = 0.22
fr = sp.Rational(11, 50)

# collision-capped rate and dispersion, fitted over m6 = 0.03-300 as in the paper
(sig_p, sig_s), (gam_p, gam_s) = S.fitpow(F)
Geq = sp.Float(str(gam_p), 30) / S.yr * S.m6 ** sp.Float(str(gam_s), 30)
sig_expr = sp.Float(str(sig_p), 30) * S.km * S.m6 ** sp.Float(str(sig_s), 30)

# belt solve (virial + C1 + C2) at that rate; mirrors scan_flattening.belt
beta = 1 / fr
Nmc_ = fr * 4 * S.rb ** 2 / S.rc ** 2
eqs = [S.Mach ** 2 * S.cs ** 2 / (sp.Rational(4, 5) * sp.pi * S.G * S.rhoMC * S.rc ** 2),
       Geq * S.fUDR * S.tc * beta,
       Geq * S.E / (S.Lam0 * S.nMC ** 2 * Nmc_ * S.Vmc)]
sol = S._solve(eqs, [S.lM, S.lr, S.lb])
d = {}
for v, x in zip([S.Mach, S.rc, S.rb], sol):
    x = sp.expand(x)
    d[v] = sp.exp(x.subs(S.lm, 0)) * S.m6 ** sp.nsimplify(sp.diff(x, S.lm), rational=True)
d[S.sig] = sig_expr

vc2 = S.G * S.mh / S.rb
rhobg = S.rhoMC * S.sigcl ** 2 / vc2
nbg = rhobg / (S.muH * S.mp)
mdot = 4 * sp.pi * S.rb ** 2 * rhobg * sp.sqrt(vc2) * sp.Float('1e-3', 30)
LEdd = 4 * sp.pi * S.G * S.mh * S.mp * S.cc / S.sigT
fEdd = sp.Rational(1, 20) * mdot * S.cc ** 2 / LEdd
Mgas = S.rhoMC * S.Vmc * Nmc_
Nst = S.Nst
t_cusp = Nst / Geq
t_gas = Mgas / (S.mstar * Geq)
r_in = S.rb - S.rc


def pl(expr, unit=1):
    """Return (value at m6 = 1, d ln / d ln m6) of a monomial expression."""
    e = sp.powsimp(sp.together((expr / unit).subs(d)), force=True)
    val = float(sp.N(e.subs(S.m6, 1)))
    try:
        lg = sp.expand_log(sp.log(e.subs(S.m6, sp.exp(S.lm))), force=True)
        slope = float(sp.N(sp.diff(lg, S.lm)))
    except TypeError:
        # not a monomial (e.g. r_b - R_MC): local logarithmic slope about 1e6 Msun
        lo, hi = float(sp.N(e.subs(S.m6, 0.5))), float(sp.N(e.subs(S.m6, 2.0)))
        slope = np.log(hi / lo) / np.log(4.0)
    return val, slope


rows = [
    ('gameq', r'$\Gamma_{\rm eq}$', Geq * S.yr, 1, 'yr$^{-1}$'),
    ('sigma', r'$\sigma$', S.sig, S.km, 'km~s$^{-1}$'),
    ('a_h', r'$a_{\rm h}$', S.ah, S.pc, 'pc'),
    ('r_b', r'$\rb$', S.rb, S.pc, 'pc'),
    ('R_MC', r'$R_{\rm MC}$', S.rc, S.pc, 'pc'),
    ('n_MC', r'$n_{\rm MC}$', S.nMC, 1, 'cm$^{-3}$'),
    ('Mach', r'${\cal M}$', S.Mach, 1, ''),
    ('N_MC', r'$N_{\rm MC}$', Nmc_, 1, ''),
    ('M_disk', r'$M_{\rm disk}$', Mgas, S.msun, r'$\msun$'),
    ('A_V', r'$\Av$ (through gas)', nbg * S.rb / sp.Float('2.2e21', 30), 1, ''),
    ('L_AGN', r'$L_{\rm AGN}/L_{\rm Edd}$', fEdd, 1, r'$f_{\rm B,-3}$'),
    ('t_gas', r'$t_{\rm gas}$', t_gas, S.yr, 'yr'),
    ('t_cusp', r'$t_{\rm cusp}$', t_cusp, S.yr, 'yr'),
    ('r_in_over_c', r'$r_{\rm in}/c$', r_in / S.cc, S.yr, 'yr'),
]
out = {}
for key, lab, expr, unit, u in rows:
    v, s = pl(expr, unit)
    out[key] = dict(value=v, slope=s)
    print('%-12s %12.4g  m6^%+.3f' % (key, v, s))

# mass floor from the compression criterion (as in scan_flattening.scan)
S.G_ck = S.fit_ck_sigma_m6()
sc, scs = S.sigma_comp(fr)
floor = (sig_p / sc) ** (1 / (scs - sig_s)) * 1e6
out['M_floor'] = dict(value=floor, slope=None)
print('M_floor      %12.4g' % floor)


def fmt(v):
    if v == 0:
        return '0'
    e = int(np.floor(np.log10(abs(v))))
    if -1 <= e <= 2:
        return ('%.3g' % v)
    m = v / 10 ** e
    return r'$%.1f \times 10^{%d}$' % (m, e)


tex = []
for key, lab, expr, unit, u in rows:
    v, s = out[key]['value'], out[key]['slope']
    val = fmt(v) + (('~' + u) if u else '')
    if key == 'gameq':
        val = r'$1.4 \times 10^{-2}\,\lamc$~yr$^{-1}$' if abs(v - 0.0142) < 1e-3 else val
    tex.append('%s & %s & $%+.2f$ \\\\' % (lab, val, s))
tex.insert(0, r'$\fom = f_{\ast}$ & $0.22$ & \nodata \\')
tex.append(r'$M_{\rm floor}$ & $%s~\msun$ & \nodata \\' % fmt(floor).strip('$'))
tex.append(r'$M_{\rm ceil}$ & $5 \times 10^{6}~\msun$ & \nodata \\')
os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
open(os.path.join(ROOT, 'tables', 'engine_rows.tex'), 'w').write('\n'.join(tex) + '\n')
json.dump(out, open(os.path.join(ROOT, 'products', 'engine_window_f022.json'), 'w'), indent=1)
print('\n'.join(tex))
