"""Assemble engine-lc.bib from ADS exports: the original citation keys mapped
to their validated ADS bibcodes, plus new references for the revised paper."""
import json
import os
import re

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.environ['ADS_TOKEN']
H = {'Authorization': 'Bearer ' + TOK, 'Content-Type': 'application/json'}

# Original keys -> bibcodes found by ads_validate.py
keys = json.load(open(os.path.join(ROOT, 'scripts', 'bibcodes.json')))
# Unpublished companion paper now on arXiv
keys['Guillochon:2026a'] = '2026arXiv260828947G'
# New references
keys.update({
    'Angus:2026a': '2026MNRAS.550g1285A',
    'Coughlin:2019a': '2019ApJ...883L..17C',
    'Bortolas:2023a': '2023MNRAS.524.3026B',
    'Krolik:2020a': '2020ApJ...904...68K',
    'Chen:2021a': '2021ApJ...914...69C',
    'Zhong:2022a': '2022ApJ...933...96Z',
    'Makrygianni:2025a': '2025ApJ...987L..20M',
    'Mummery:2024a': '2024MNRAS.527.2452M',
    'Mummery:2025a': '2025MNRAS.541..429M',
    'Mummery:2025b': '2025arXiv251209143M',
    'Guolo:2025a': '2025arXiv251026774G',
    'Somalwar:2025a': '2025ApJ...985..175S',
    'Lin:2024a': '2024ApJ...971L..26L',
    'Hinkle:2024a': '2024arXiv241215326H',
    'Payne:2021a': '2021ApJ...910..125P',
    'Wevers:2023a': '2023ApJ...942L..33W',
    'Liu:2025a': '2025ApJ...979...40L',
    'Bandopadhyay:2024a': '2024ApJ...974...80B',
    'Broggi:2024a': '2024OJAp....7E..48B',
    'Miles:2020a': '2020ApJ...899...36M',
    'Nixon:2021a': '2021ApJ...922..168N',
    'Ryu:2020c': '2020ApJ...904..100R',
    'Stone:2020a': '2020SSRv..216...35S',
    'Magill:2026a': '2026RASTI...5ag019M',
    'Gezari:2021a': '2021ARA&A..59...21G',
    'vanVelzen:2021a': '2021ApJ...908....4V',
    'Steinberg:2024a': '2024Natur.625..463S',
    'Metzger:2022a': '2022ApJ...937L..12M',
    'Sarin:2024a': '2024ApJ...961L..19S',
    'Mainetti:2017a': '2017A&A...600A.124M',
    'Golightly:2019a': '2019ApJ...882L..26G',
    'Cufari:2022a': '2022ApJ...929L..20C',
    'Kiroglu:2023a': '2023ApJ...948...89K',
    'Ramsden:2022a': '2022MNRAS.515.1146R',
    'Kroupa:2001a': '2001MNRAS.322..231K',
    'Speagle:2020a': '2020MNRAS.493.3132S',
    'Tout:1996a': '1996MNRAS.281..257T',
    'Guillochon:2015a': '2015ApJ...809..166G',
    'Mockler:2021a': '2021ApJ...906..101M',
    'Vishniac2025': '2025BAAS...57a.014V',   # AAS editorial on authorship (key matches the manuscript)
    'Eyles-Ferris:2026a': '2026MNRAS.550g1150E',  # TDE IR counterparts from dust rings, observing angle
    'Tuna:2025a': '2025ApJ...989...27T',         # radiation transport of IR echoes, torus geometry
    'Hinkle:2024b': '2024MNRAS.531.2603H',       # ANT mid-IR echoes, high covering fractions
    'Wu:2026a': '2026ApJ..1008...47W',           # IR echoes of precessing TDEs
    'Grotova:2025a': '2025A&A...697A.159G',      # eROSITA X-ray-selected TDE population and rate
    'Nair:2026a': '2026ApJ..1007...40N',         # suppressed rate of luminous mid-IR TDEs
    'Ramsden:2026a': '2026ApJ...998L..25R',      # undermassive SMBHs in quenched TDE hosts
    'Mummery:2026a': '2026arXiv260114483M',      # TDEFLARE optical-flare mass inference, Malmquist-Hills bias
    # software acknowledgments
    'Astropy:2013a': '2013A&A...558A..33A',
    'Astropy:2018a': '2018AJ....156..123A',
    'Harris:2020a': '2020Natur.585..357H',        # numpy
    'Virtanen:2020a': '2020NaMet..17..261V',      # scipy
    'Hunter:2007a': '2007CSE.....9...90H',        # matplotlib
    'Barbary:2016a': '2016zndo....804967B',      # extinction
})

bibcodes = list(keys.values())
r = requests.post('https://api.adsabs.harvard.edu/v1/export/bibtex', headers=H,
                  data=json.dumps({'bibcode': bibcodes, 'keyformat': '%R', 'maxauthor': 0, 'journalformat': 1}))
r.raise_for_status()
export = r.json()['export']
entries = {}
for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}\n', export, flags=re.S):
    entries[m.group(2).strip()] = (m.group(1), m.group(3))
missing = [b for b in bibcodes if b not in entries]
print('missing from export:', missing)

order = list(keys.items())
out = ['%% engine-lc.bib -- references for "Light Curves and Demographics of Tidal',
       '%% Disruptions in Self-Sustaining Black Hole Engines".',
       '%% Every entry below was exported from NASA/ADS on 2026-09-05 by scripts/build_bib.py',
       '%% and keyed to the bibcodes listed in scripts/build_bib.py.', '']
for key, bc in order:
    if bc not in entries:
        continue
    kind, body = entries[bc]
    body = body.replace('\n     keywords = {' + body.split('keywords = {')[1].split('},')[0] + '},', '') if 'keywords = {' in body else body
    body = re.sub(r'\n\s*adsnote = \{[^}]*\},?', '', body)
    out.append('@%s{%s,%s\n}\n' % (kind, key, body.rstrip(',')))
open(os.path.join(ROOT, 'engine-lc.bib'), 'w', encoding='utf-8').write('\n'.join(out))
print('wrote', len(order), 'entries')
