"""Extract the per-event light-curve properties (Table 4) and black hole masses
(Table 5) of the 33 ZTF TDEs of Yao et al. (2023) from the arXiv HTML text in
research/y2023/, and write scripts/yao2023_sample.json with a 1/V_max rate weight."""
import json
import os
import re

import numpy as np
from astropy.cosmology import FlatLambdaCDM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
t = open(os.path.join(ROOT, 'research', 'y2023', 'y_arxiv2303.06523.txt'), encoding='utf-8').read()

num = r'(-?\d+\.\d+)'
err = r'(?:\s*−\s*\d+\.\d+\s*\+\s*\d+\.\d+\s*-?\d+\.\d+_\{-\d+\.\d+\}\^\{\+\d+\.\d+\})?'
# Table 4 rows: ID name model tpeak(+err) logT logLg logLbb logRbb trise(+err) tdecl(+err) Dmax zmax floss
row4 = re.compile(r'(\d{1,2})\s+(AT20\d{2}[a-z]+)\s+(\S+)\s+' + num + err + r'\s+' + num + r'\s+' + num + r'\s+' + num + r'\s+' + num +
                  r'\s+' + num + err + r'\s+' + num + err + r'\s+(\d+)\s+' + num + r'\s+' + num)
i4 = t.find('Table 4: Light Curve Properties'); i5 = t.find('Table 5: Host Galaxy Properties')
seg4 = t[i4:i5]
ev = {}
for m in row4.finditer(seg4):
    ID, name, model, tpk, logT, logLg, logLbb, logR, trise, tdec, Dmax, zmax, floss = m.groups()
    ev[name] = dict(id=int(ID), logT=float(logT), logLg=float(logLg), logLbb=float(logLbb), t_rise=float(trise),
                    t_decline=float(tdec), Dmax_Mpc=float(Dmax), zmax_t=float(zmax), floss=float(floss))
print('table 4 rows:', len(ev))
# Table 5: name ... log M_BH with asymmetric errors ... sigma ... r1/2 ... zmax_h ; grab log M_BH as the value preceding the sigma column
i6 = t.find('Table 6', i5)
seg5 = t[i5:i6 if i6 > 0 else i5 + 40000]
row5 = re.compile(r'(AT20\d{2}[a-z]+)\s+(\d+\.\d+)_\{-\d+\.\d+\}\^\{\+\d+\.\d+\}.*?log M_\{\\rm BH\}|(AT20\d{2}[a-z]+)')
# simpler: for each event name in seg5, take the sequence of 'x.xx_{-a}^{+b}' values after it; log M_BH is the 7th asymmetric value
for name in ev:
    j = seg5.find(name)
    if j < 0:
        continue
    chunk = seg5[j:j + 1200]
    # log M_BH is the first symmetric-error value ("7.93\pm 0.35") after the six asymmetric SED-fit columns
    pm = re.findall(r'(-?\d+\.\d+)\\pm\s*\d+\.\d+', chunk)
    if pm:
        ev[name]['logMBH'] = float(pm[0])
    asym = re.findall(r'(-?\d+\.\d+)_\{-(\d+\.\d+)\}\^\{\+(\d+\.\d+)\}', chunk)
    if asym:
        ev[name]['logMgal'] = float(asym[0][0])
        ev[name]['logMgal_err'] = 0.5 * (float(asym[0][1]) + float(asym[0][2]))
    if len(asym) > 1:   # second SED column is the rest-frame, extinction-corrected (0,0)u-r colour
        ev[name]['umr'] = float(asym[1][0])
        ev[name]['umr_err'] = 0.5 * (float(asym[1][1]) + float(asym[1][2]))
    # r_1/2 and z_max,h are the two plain numbers before the next row's ID
    mz = re.search(r'(\d+\.\d+)\s+(0\.\d+)\s+(?:\d{1,2}\s+AT20|$)', chunk)
    if mz is None:
        mz = re.search(r'(\d+\.\d+)\s+(0\.\d+)\s*$', chunk[:chunk.find('Table') if 'Table' in chunk else len(chunk)].rstrip())
    ev[name]['zmax_h'] = float(mz.group(2)) if mz else None
# Yao et al. (2023) Eqs. 22-23: the mass-corrected colour C = (0,0)u-r - 0.5 - 0.15 log Mgal, with red C > 0.1,
# green |C| <= 0.1, blue C < -0.1; membership probabilities from a Gaussian in C (their Section VI.5)
from scipy.stats import norm
for n in ev:
    e = ev[n]
    if 'umr' in e and 'logMgal' in e:
        C = e['umr'] - 0.5 - 0.15 * e['logMgal']
        sC = float(np.hypot(e.get('umr_err', 0.15), 0.15 * e.get('logMgal_err', 0.15)))
        e['C_color'] = C
        e['p_red'] = float(norm.sf((0.1 - C) / sC)); e['p_blue'] = float(norm.cdf((-0.1 - C) / sC))
        e['p_green'] = max(0.0, 1.0 - e['p_red'] - e['p_blue'])
ok = [n for n in ev if 'logMBH' in ev[n]]
print('with colour:', sum('p_green' in ev[n] for n in ev), 'expected green:', round(sum(ev[n].get('p_green', 0) for n in ev), 1))
print('with M_BH:', len(ok))
for n in ev:
    e = ev[n]
    zm = e['zmax_t'] if e.get('zmax_h') is None else min(e['zmax_t'], e['zmax_h'])
    V = cosmo.comoving_volume(zm).value              # Mpc^3 (all sky)
    e['zmax'] = zm
    e['w_1overV'] = 1.0 / (V * max(1.0 - e['floss'], 0.05))
json.dump(ev, open(os.path.join(ROOT, 'scripts', 'yao2023_sample.json'), 'w'), indent=1)
for n in list(ev)[:5]:
    print(n, ev[n])
