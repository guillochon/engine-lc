"""Check the abstract of engine-lc.tex against the arXiv 1920-character limit
and the no-custom-macro rule (.cursor/rules/abstract-*.mdc)."""
import re
import sys

MACROS = ['msun', 'mhsix', 'Sim', 'mosfit', 'gameq', 'fom', 'rb', 'Av', 'um', 'lsst', 'wise', 'neowise',
          'jwst', 'romantel', 'euclid', 'spherex', 'fspark', 'lamc', 'colr', 'citep', 'citet', 'citetalias']

t = open('engine-lc.tex', encoding='utf-8').read()
a = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S).group(1)
s = ' '.join(a.split())
found = sorted({m for m in re.findall(r'\\([A-Za-z]+)', s) if m in MACROS})
print('abstract characters (collapsed, incl. TeX):', len(s))
print('forbidden macros:', found or 'none')
print('leads with citation:', s.lstrip().startswith(('\\cite', 'Guillochon')))
sys.exit(0 if len(s) <= 1920 and not found else 1)
