"""Validate engine-lc.bib against ADS: look up each entry by DOI (or arXiv id,
or title+year), report mismatches, and write a fresh ADS-exported bibtex file
keyed by the original citation keys."""
import os, re, sys, json, requests, time
TOK = os.environ['ADS_TOKEN']; H = {'Authorization': 'Bearer ' + TOK}
API = 'https://api.adsabs.harvard.edu/v1'

def parse_bib(path):
    txt = open(path, encoding='utf-8').read()
    entries = {}
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', txt, flags=re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|"[^"]*"|\w+)', body):
            v = fm.group(2).strip()
            if v[0] in '{"': v = v[1:-1]
            fields[fm.group(1).lower()] = v
        entries[key] = (kind, fields)
    return entries

def search(q, fl='bibcode,title,author,year,pub,volume,page,doi,identifier,doctype'):
    r = requests.get(API + '/search/query', headers=H, params={'q': q, 'fl': fl, 'rows': 5})
    r.raise_for_status(); return r.json()['response']['docs']

def norm(s):
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = re.sub(r'[{}$~\\]', '', s).lower()
    return re.sub(r'[^a-z0-9 ]', '', s)

def find(key, f):
    if 'doi' in f:
        d = search('doi:"%s"' % f['doi'])
        if d: return d, 'doi'
    if 'eprint' in f:
        d = search('arXiv:%s' % f['eprint'])
        if d: return d, 'arxiv'
    au = f.get('author', '').split(' and ')[0]
    au = re.sub(r'[{}]', '', au).split(',')[0].strip()
    yr = f.get('year', '')
    ti = norm(f.get('title', ''))
    words = ' '.join(ti.split()[:8])
    d = search('author:"^%s" year:%s title:"%s"' % (au, yr, words))
    if d: return d, 'title'
    d = search('author:"^%s" year:%s-%d' % (au, int(yr) - 1 if yr else 1900, int(yr) + 1 if yr else 2100) + ' abs:"%s"' % ' '.join(ti.split()[:4]))
    return d, 'loose'

def main():
    bib = parse_bib('engine-lc.bib')
    report = []; bibcodes = {}
    for key, (kind, f) in bib.items():
        docs, how = find(key, f)
        if not docs:
            report.append((key, 'NOT FOUND', how, f.get('title', ''))); continue
        # prefer refereed article over arXiv
        docs.sort(key=lambda d: (d.get('doctype') == 'eprint', ))
        d = docs[0]
        issues = []
        if f.get('year') and str(d.get('year')) != f['year']: issues.append('year %s->%s' % (f['year'], d.get('year')))
        if f.get('volume') and str(d.get('volume')) != f['volume']: issues.append('vol %s->%s' % (f['volume'], d.get('volume')))
        p = (d.get('page') or [''])[0]
        if f.get('pages') and norm(f['pages'].split('-')[0]) != norm(p): issues.append('page %s->%s' % (f['pages'], p))
        if norm(f.get('title', ''))[:40] != norm(d['title'][0])[:40]: issues.append('TITLE: "%s"' % d['title'][0])
        fa = norm(re.sub(r'[{}]', '', f.get('author', '')).split(',')[0])
        if fa and fa not in norm(d['author'][0]): issues.append('AUTHOR %s -> %s' % (fa, d['author'][0]))
        if d.get('doi') and f.get('doi') and d['doi'][0].lower() != f['doi'].lower(): issues.append('doi -> %s' % d['doi'][0])
        report.append((key, d['bibcode'], how, '; '.join(issues) if issues else 'OK'))
        bibcodes[key] = d['bibcode']
        time.sleep(0.1)
    for r in report: print('%-22s %-22s %-6s %s' % r)
    json.dump(bibcodes, open('scripts/bibcodes.json', 'w'), indent=1)
    # export
    r = requests.post(API + '/export/bibtex', headers={**H, 'Content-Type': 'application/json'},
                      data=json.dumps({'bibcode': list(bibcodes.values()), 'keyformat': '%1H:%Y', 'maxauthor': 0, 'journalformat': 1}))
    r.raise_for_status()
    open('scripts/ads_export_raw.bib', 'w', encoding='utf-8').write(r.json()['export'])
    print('exported', len(bibcodes))

if __name__ == '__main__': main()
