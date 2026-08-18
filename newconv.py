#!/usr/bin/env python3
"""
newconv.py — fitted replacement for the additive `Conv` score, v1.0

WHAT THIS IS
    A slate-conditional logistic model (L2, C=0.005) over 31 quantitative +
    PropFinder features, isotonic-calibrated to a real HR probability, fitted on
    48 slates (2026-06-25 .. 2026-08-17, n=3,218, base 15.44%).

    It does NOT use the named qualitative grades. Four variants including them
    were tested on a 17-slate August walk-forward; every one lost lift, and an
    L1 fit given the option to use them zeroed all but 2 of 87 coefficients.
    Grades still belong in the card prose for tracking — just not in this number.

MEASURED (August walk-forward, 17 slates, base 15.1%, out-of-sample)
    NewConv top-3 per slate ......... 2.21x   (old Conv: 0.91x, Fisher p=0.034)
    within-slate AUC ................ 0.564   (old Conv: 0.513)
    CARD STRUCTURE 3-0-2 ............ 32/85 = 2.49x   (H1 2.51x / H2 2.47x)
      - 3 Score slots (NEG-screened) . 21/51 = 41.2%  (2.73x)
      - 2 NewConv slots .............. 11/34 = 32.4%  (2.14x)
    For comparison on the same slates:
      5-0-0 pure Score ............... 2.03x
      4-1-0 Score + PT reserve ....... 1.87x   <- PT slot itself went 1/16 = 0.41x
      4-0-1 Score + 1 NewConv ........ 2.18x

    CAVEATS: 17 evaluation slates. 3-0-2 was chosen from 13 structures on that
    one window; it is preferred for split-half stability, not for the point
    estimate, and 3-0-2 vs 4-0-1 is not statistically separable. The PT slot's
    0.41x rests on n=16. Re-measure at ~35 slates.

USAGE
    python3 newconv.py --outputs /path/to/outputs            # score today's slate
    python3 newconv.py --outputs ./outputs --write           # also write newconv.csv
"""
import argparse, csv, json, os, re, sys, unicodedata
import numpy as np
import pandas as pd

csv.field_size_limit(10 ** 9)
HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = json.load(open(os.path.join(HERE, 'newconv_params.json')))
DISPLAY_CEIL = 0.35          # p -> CONV100 scale


# ----------------------------------------------------------------- helpers
def norm(s):
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z ]', '', s).lower().strip())


def num(x):
    if x is None:
        return np.nan
    x = str(x).strip().replace('%', '').replace('★', '').replace('~', '')
    x = x.replace('+', '').replace(',', '')
    if x in ('', '-', '–', '—', 'N/A', 'nan'):
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def read_csv(path):
    return list(csv.reader(open(path, encoding='utf-8-sig', errors='replace')))


# ----------------------------------------------------------------- loaders
def load_rankings(path):
    rows = read_csv(path)
    hdr = next(i for i, r in enumerate(rows) if r[:3] == ['Rank', 'Batter', 'Status'])
    out = []
    for r in rows[hdr + 1:]:
        if len(r) < 19 or not r[1].strip():
            break
        out.append(dict(
            rank_raw=r[0].strip(), batter=r[1].strip(), key=norm(r[1]), status=r[2],
            game_time=r[3], team=r[4], opp=r[5], pitcher=r[6],
            odds=num(r[7]), hr_prob=num(r[8]), score=num(r[9]), power=num(r[10]),
            vuln=num(r[11]), pm=num(r[12]), park=num(r[13]), env=num(r[14]),
            est_pa=num(r[15]), edge=num(r[16]), hit_score=num(r[17]), sig=num(r[18])))
    df = pd.DataFrame(out)
    df['medal'] = df.rank_raw.map({'🥇': 1, '🥈': 2, '🥉': 3}).fillna(
        pd.to_numeric(df.rank_raw, errors='coerce'))
    return df


NINE_RE = re.compile(
    r'ISO ([\d.]+) \| Barrel% ([\d.]+) \| HH% ([\d.]+) \| Air% ([\d.]+) \| GB% ([\d.]+) \| '
    r'PullBrl% ([\d.]+) \| FB% ([\d.]+) \| Blast% ([\d.]+) \| Pull% ([\d.]+)')
NINE = ['iso', 'barrel', 'hh', 'air', 'gb', 'pullbrl', 'fb', 'blast', 'pull']


def load_detailed(path):
    """Pull the PropFinder 9-STAT line per batter."""
    rows = read_csv(path)
    cur, out = None, {}
    for r in rows:
        if not r:
            continue
        head = (r[0] or '').strip()
        m = re.match(r'^#\d+\s+(.+?)\s+\(\w+\)\s+vs\s', head)
        if m:
            cur = norm(m.group(1))
            continue
        if cur and '9-STAT' in head:
            g = NINE_RE.search(head)
            if g:
                out[cur] = {n: float(v) for n, v in zip(NINE, g.groups())}
    return out


def load_sharp(path):
    """HR-side grade table only. Returns NEG05/NEG06 flags and legacy Conv."""
    rows = read_csv(path)
    try:
        hdr = next(i for i, r in enumerate(rows)
                   if len(r) > 2 and r[0].strip() == 'Rank' and r[2].strip() == 'HR Odds')
    except StopIteration:
        return {}
    out = {}
    for r in rows[hdr + 1:]:
        if len(r) < 6 or not r[1].strip():
            break
        blob = ' '.join(r[5:9])
        out[norm(r[1])] = dict(
            neg05='NEG05' in blob, neg06='NEG06' in blob,
            old_conv=num(r[8]) if len(r) > 8 else np.nan,
            grade=r[5], rationale=r[6] if len(r) > 6 else '')
    return out


# ----------------------------------------------------------------- features
def build_features(df):
    F = pd.DataFrame(index=df.index)
    F['nlogodds'] = -np.log(df.odds)
    F['odds_le250'] = (df.odds <= 250).astype(float)
    F['odds_250_350'] = ((df.odds > 250) & (df.odds <= 350)).astype(float)
    F['odds_ge600'] = (df.odds >= 600).astype(float)
    F['power'] = df.power
    F['power_deadzone'] = ((df.power >= 77) & (df.power <= 84)).astype(float)
    F['vuln'] = df.vuln
    F['vuln_ge56'] = (df.vuln >= 56).astype(float)
    F['vuln_ge52'] = (df.vuln >= 52).astype(float)
    F['vuln_lt44'] = (df.vuln < 44).astype(float)
    F['pm'] = df.pm
    F['pm_lt100'] = (df.pm < 1.00).astype(float)
    F['env'] = df.env
    F['park'] = df.park
    F['est_pa'] = df.est_pa
    F['hit_score'] = df.hit_score
    F['sig'] = df.sig
    F['sig_ge15'] = (df.sig >= 15).astype(float)
    F['hs_lt15'] = (df.hit_score < 15).astype(float)
    F['edge'] = df.edge
    F['status_conf'] = df.status.str.contains('Conf', na=False).astype(float)
    for c in NINE:
        F[c] = df[c]
    F['gate_count'] = ((df.iso > 0.2).astype(float) + (df.barrel > 15) + (df.hh > 40) +
                       (df.air > 50) + (df.gb < 40) + (df.pullbrl > 10) + (df.fb > 35) +
                       (df.blast > 15) + (df.pull > 30))
    return F[PARAMS['cols']].replace([np.inf, -np.inf], np.nan)


def score_slate(df):
    """Adds newconv_p (calibrated HR probability) and CONV100."""
    F = build_features(df)
    med = pd.Series({c: (PARAMS['median'][c] if PARAMS['median'][c] is not None else 0.0)
                     for c in PARAMS['cols']})
    F = F.fillna(med)
    mu = pd.Series(PARAMS['mean']); sd = pd.Series(PARAMS['std'])
    Z = (F - mu) / sd
    Z = Z - Z.mean()                       # within-slate centering (conditional logit)
    coef = pd.Series(PARAMS['coef'])[PARAMS['cols']]
    lin = Z.values @ coef.values + PARAMS['intercept']
    raw = 1 / (1 + np.exp(-lin))
    df = df.copy()
    df['newconv_raw'] = raw
    df['newconv_p'] = np.interp(raw, PARAMS['iso_x'], PARAMS['iso_y'])
    df['CONV100'] = np.clip(df.newconv_p / DISPLAY_CEIL * 100, 0, 100).round(0).astype(int)
    return df


# ----------------------------------------------------------------- card
def build_card(df, n_score=3, n_new=2):
    """3-0-2: Score slots NEG-screened, then NewConv slots. Returns the 5 picks."""
    d = df.copy()
    d['neg'] = d.neg05 | d.neg06
    d['composite'] = d.score * 1000 + d.hr_prob
    eligible = d[~d.neg]
    picks = eligible.nlargest(n_score, 'composite').assign(slot='SCORE')
    pool = eligible[~eligible.key.isin(picks.key)]
    picks = pd.concat([picks, pool.nlargest(n_new, 'newconv_raw').assign(slot='NEWCONV')])
    if len(picks) < n_score + n_new:                     # backfill
        rest = eligible[~eligible.key.isin(picks.key)]
        picks = pd.concat([picks, rest.nlargest(n_score + n_new - len(picks),
                                                'composite').assign(slot='SCORE-FILL')])
    return picks.reset_index(drop=True)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='outputs',
                    help='directory holding current_rankings/detailed/sharp.csv')
    ap.add_argument('--write', action='store_true', help='write newconv.csv into --outputs')
    a = ap.parse_args()

    p = lambda n: os.path.join(a.outputs, f'current_{n}.csv')
    df = load_rankings(p('rankings'))
    nine = load_detailed(p('detailed'))
    sharp = load_sharp(p('sharp'))

    for c in NINE:
        df[c] = df.key.map(lambda k: nine.get(k, {}).get(c, np.nan))
    df['neg05'] = df.key.map(lambda k: sharp.get(k, {}).get('neg05', False))
    df['neg06'] = df.key.map(lambda k: sharp.get(k, {}).get('neg06', False))
    df['old_conv'] = df.key.map(lambda k: sharp.get(k, {}).get('old_conv', np.nan))

    miss = df[NINE[0]].isna().mean()
    if miss > 0.25:
        print(f'⚠️  PropFinder 9-STAT missing for {miss:.0%} of batters — '
              f'model degrades to quant-only for those rows', file=sys.stderr)

    df = score_slate(df)
    # rank on the raw monotone score: isotonic creates probability plateaus,
    # so newconv_p ties; newconv_raw preserves full within-slate ordering.
    df['newconv_rank'] = df.newconv_raw.rank(ascending=False, method='first').astype(int)
    card = build_card(df)

    print(f'\n{"="*74}\n  NEWCONV v1.0  —  {len(df)} batters  |  9-STAT coverage '
          f'{1-miss:.0%}  |  NEG-flagged {int((df.neg05|df.neg06).sum())}\n{"="*74}')
    print('\n  TOP 5 HR CARD  (3 Score + 2 NewConv)\n')
    for i, r in card.iterrows():
        print(f'  {i+1}. [{r.slot:10s}] {r.batter:<26s} {("+"+str(int(r.odds))) if pd.notna(r.odds) else "":>6s}  '
              f'vs {r.pitcher:<20s}')
        print(f'      Score {r.score:.1f} · Vuln {r.vuln:.1f} · PM {r.pm:.3f} · Pwr {r.power:.1f} · '
              f'CONV100 {r.CONV100} (p={r.newconv_p:.3f}, rank {r.newconv_rank}) · '
              f'oldConv {"-" if pd.isna(r.old_conv) else int(r.old_conv)}')
    print(f'\n  NEWCONV TOP 10 (full slate)\n')
    top = df.nsmallest(10, 'newconv_rank')
    for _, r in top.iterrows():
        mark = '★' if r.key in set(card.key) else ' '
        print(f'   {mark} {r.newconv_rank:2d}. {r.batter:<26s} CONV100 {r.CONV100:3d}  '
              f'p={r.newconv_p:.3f}  Score {r.score:.1f}  Vuln {r.vuln:.1f}  '
              f'{("+"+str(int(r.odds))) if pd.notna(r.odds) else ""}')

    if a.write:
        out = os.path.join(a.outputs, 'newconv.csv')
        df[['batter', 'team', 'pitcher', 'odds', 'score', 'vuln', 'pm', 'power',
            'newconv_p', 'CONV100', 'newconv_rank', 'old_conv', 'neg05', 'neg06']] \
            .sort_values('newconv_rank').to_csv(out, index=False)
        print(f'\n  wrote {out}')


if __name__ == '__main__':
    main()
