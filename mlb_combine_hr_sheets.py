import pandas as pd
import re
import pickle
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
file_path   = r"C:\Users\hlee145\Documents\FanDuel Spreadsheets\HR Models\FantasyLabsMLB.xlsm"
output_path = Path(file_path).parent / f"{Path(file_path).stem}.csv"
pkl_path    = Path(file_path).parent / "full_picks.pkl"

# ── Archive cutoff (added 2026-08-27) ─────────────────────────────────────────
# The CSV grew too large for GitHub. Old history was archived separately and a
# fresh CSV/pkl started from here. The .xlsm workbook itself still has every
# HR_YYYY-MM-DD sheet back to April - without this cutoff, a fresh (empty)
# output_path would make every one of those look "new" (see existing_dates
# below) and the whole archive would get re-pulled right back into the new
# file, defeating the point of starting over. This filters the sheet list
# itself, upstream of both the CSV-append and the pkl-append logic, so
# neither one needs its own separate cutoff check.
START_DATE = "2026-04-26"   # inclusive - this date and everything after it

# ── Helpers ────────────────────────────────────────────────────────────────────
today_date = datetime.today().strftime('%Y-%m-%d')

def _sheet_date_str(sheet_name: str) -> str:
    """HR_2026-05-23 → 'Saturday, May 23, 2026'"""
    dt = datetime.strptime(sheet_name.split('_')[1], '%Y-%m-%d')
    return dt.strftime('%A, %B %d, %Y').replace(' 0', ' ')

def _sf(val, default=None):
    try:
        s = str(val).strip().replace('%', '').replace('+', '').replace('N/A', '')
        return float(s) if s else default
    except (ValueError, TypeError):
        return default

def _si(val, default=0):
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default

def _col(row, idx, default=None):
    try:
        return row.iloc[idx] if idx < len(row) else default
    except Exception:
        return default

# ── Find dates already in the CSV ─────────────────────────────────────────────
# FIX (Aug 18 2026): existing_dates used to be the UNION of what's found in the
# CSV *and* what's found in full_picks.pkl. If the two ever desync — a run that
# updated the pkl but failed/was interrupted before the CSV-append step, a CSV
# that got replaced from an older copy, anything — the pkl's memory of a date
# would permanently mark it "already processed" even when the CSV never
# actually received those rows. That's the mechanism behind Aug 17 being
# skipped: the script correctly found the Excel sheet, but the pkl already
# claimed the date, so the CSV-append never ran, and nothing re-checks it on
# later runs either. The CSV file itself is the only thing that should decide
# whether the CSV needs an append — removed the pkl as a source for this
# check. Self-healing: any date silently missing from the CSV now gets
# correctly re-appended the next time this runs, regardless of what the pkl
# thinks. The pkl-merge logic further down already handles re-processing a
# date safely (it removes old entries for any date being re-parsed before
# adding the fresh ones — see `retained_picks` below), so this doesn't risk
# duplicate pkl entries.
existing_dates = set()
if output_path.exists():
    print(f"Reading existing CSV to find processed dates...")
    try:
        # Scan the CSV for separator lines that mark each sheet
        # Separator format: ===== HR_2026-05-23 =====
        # Two independent markers, both scanned from the CSV directly:
        #   1. The combine script's own "===== HR_YYYY-MM-DD =====" separator
        #      (written before every sheet except the very first one ever).
        #   2. Each sheet's own embedded title row — "MLB HOME RUN MODEL — Month DD, YYYY"
        #      — which is part of the sheet's actual data and gets written to
        #      the CSV regardless, including for that first, separator-less sheet.
        # Matching both closes the one gap left by removing the pkl fallback
        # above: the earliest date in the file, which has no separator line.
        sep_pattern   = re.compile(r"=====\s*(HR_\d{4}-\d{2}-\d{2})\s*=====")
        title_pattern = re.compile(r"MLB HOME RUN MODEL.*?—\s*\w+,?\s*(\w+)\s+(\d{1,2}),?\s*(\d{4})")
        with open(output_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            for line in f:
                m = sep_pattern.search(line)
                if m:
                    existing_dates.add(m.group(1).split('_')[1])  # store as YYYY-MM-DD
                    continue
                m2 = title_pattern.search(line)
                if m2:
                    try:
                        dt = datetime.strptime(f"{m2.group(1)} {m2.group(2)} {m2.group(3)}", '%B %d %Y')
                        existing_dates.add(dt.strftime('%Y-%m-%d'))
                    except Exception:
                        pass
        print(f"  Found {len(existing_dates)} dates already in CSV.")
    except Exception as e:
        print(f"  ⚠️  Could not read existing CSV: {e} — will do full rebuild.")
        existing_dates = set()
else:
    print("No existing CSV found — will do full build.")

# ── Sheet discovery — only NEW sheets ─────────────────────────────────────────
print("Reading Excel sheet list...")
pattern = re.compile(r"^HR_\d{4}-\d{2}-\d{2}$")
excel_file = pd.ExcelFile(file_path)

all_hr_sheets = []
for sheet in excel_file.sheet_names:
    if pattern.match(sheet):
        sheet_date = sheet.split('_')[1]
        if sheet_date == today_date:
            print(f"  Excluding today's sheet: {sheet}")
        else:
            all_hr_sheets.append((sheet_date, sheet))

# Sort chronologically
all_hr_sheets.sort(key=lambda x: x[0])

# Apply the archive cutoff (see START_DATE above) before anything downstream
# ever sees these sheets - both the CSV-append logic and the pkl logic key
# off all_hr_sheets/new_sheets, so filtering here covers both in one place.
_pre_cutoff_count = len(all_hr_sheets)
all_hr_sheets = [(d, s) for d, s in all_hr_sheets if d >= START_DATE]
_excluded_count = _pre_cutoff_count - len(all_hr_sheets)
if _excluded_count:
    print(f"  Archive cutoff active (START_DATE={START_DATE}): "
          f"excluding {_excluded_count} sheet(s) older than the cutoff "
          f"(still present in the .xlsm workbook, not pulled into the fresh file).")

# Filter to only sheets not yet in CSV
new_sheets = [(d, s) for d, s in all_hr_sheets if d not in existing_dates]

print(f"Total HR sheets (excl. today): {len(all_hr_sheets)}")
print(f"Already processed: {len(all_hr_sheets) - len(new_sheets)}")
print(f"New sheets to prepend: {len(new_sheets)}")

if not new_sheets:
    print("✓ CSV is already up to date — nothing to prepend.")
else:
    # ── PREPEND new sheets, newest-first (Aug 18 2026) ──────────────────────────
    # Was: append mode, new sheets landed at the end of the file (oldest
    # position). GitHub's published copy of this CSV is newest-first — the
    # local file needs to match. New sheets now go at the TOP, in descending
    # date order (most recent sheet ends up on line 1).
    #
    # One structural detail this requires: whatever sheet currently sits at
    # line 1 was written WITHOUT a preceding "===== HR_... =====" separator,
    # because it was position 1 at the time (see the original append logic's
    # `is_first_sheet_ever` flag). Once new content gets inserted above it,
    # that old top sheet is no longer position 1 and needs a separator added
    # — otherwise its rows silently merge into whatever now precedes them
    # with no boundary marker. Detected via the same title-row pattern used
    # for existing_dates above, scanning just the old file's opening lines.
    csv_has_content = output_path.exists() and output_path.stat().st_size > 0

    old_content = ""
    old_top_needs_separator = False
    if csv_has_content:
        with open(output_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            old_content = f.read()
        # Does the old file already start with a separator? If not, its top
        # sheet needs one synthesized now that it's moving off position 1.
        _old_head = old_content.lstrip('\ufeff').lstrip()
        if not _old_head.startswith('====='):
            _title_pat = re.compile(r"MLB HOME RUN MODEL.*?—\s*\w+,?\s*(\w+)\s+(\d{1,2}),?\s*(\d{4})")
            for _line in old_content.splitlines()[:5]:
                _m = _title_pat.search(_line)
                if _m:
                    try:
                        _dt = datetime.strptime(f"{_m.group(1)} {_m.group(2)} {_m.group(3)}", '%B %d %Y')
                        _old_top_sheet = f"HR_{_dt.strftime('%Y-%m-%d')}"
                        old_top_needs_separator = True
                        print(f"  Old top-of-file sheet ({_old_top_sheet}) has no separator "
                              f"— adding one now that it's moving off position 1.")
                    except Exception:
                        pass
                    break

    # Build the new content block in memory, newest sheet first.
    # Whichever sheet ends up at ABSOLUTE position 1 of the assembled file
    # gets no leading separator — matching the convention already visible in
    # the published CSV (its line 1 is a title row with no "=====" before
    # it). Since new sheets always get written before any old content, the
    # single newest new sheet is always position 1 whenever new_sheets is
    # non-empty — this doesn't depend on whether old content exists.
    import io, csv as _csv_mod
    new_buf = io.StringIO()
    writer = _csv_mod.writer(new_buf)
    for i, (sheet_date, sheet_name) in enumerate(sorted(new_sheets, key=lambda x: x[0], reverse=True)):
        print(f"  Adding to top: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        is_very_first_line_of_file = (i == 0)
        if not is_very_first_line_of_file:
            writer.writerow([])
            writer.writerow([f'===== {sheet_name} ====='])
            writer.writerow([])
        for _, row in df.iterrows():
            writer.writerow(['' if pd.isna(v) else v for v in row])

    # Assemble: new content (newest→...→least-new-of-the-new-batch), then a
    # migrated separator for the old top sheet if needed, then the rest of
    # the old file unchanged.
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_out:
        csv_out.write(new_buf.getvalue())
        if old_top_needs_separator:
            sep_writer = _csv_mod.writer(csv_out)
            sep_writer.writerow([])
            sep_writer.writerow([f'===== {_old_top_sheet} ====='])
            sep_writer.writerow([])
        csv_out.write(old_content)

    print(f"✓ CSV updated (newest sheets now at the top): {output_path}")

# ── Build / update full_picks.pkl ─────────────────────────────────────────────
# Only parse the new sheets for pkl entries, then merge with existing pkl
RANK_MAP    = {'🥇': 1, '🥈': 2, '🥉': 3}
VALID_RANKS = set(RANK_MAP) | {str(i) for i in range(1, 51)}

print("\nUpdating full_picks.pkl...")

# Load existing pkl
existing_model_picks = {}   # (date_str, name_lower) → pick dict (grade-labelled)
existing_all_picks   = []   # all picks already in pkl
if pkl_path.exists():
    try:
        with open(pkl_path, 'rb') as f:
            existing_raw = pickle.load(f)
        existing_all_picks = existing_raw
        for p in existing_raw:
            if p.get('hr_grade') or p.get('has_slm') or p.get('has_highk'):
                key = (p.get('date', ''), p.get('name', '').strip().lower())
                existing_model_picks[key] = p
        print(f"  Loaded {len(existing_all_picks)} existing picks from pkl ({len(existing_model_picks)} grade-labelled)")
    except Exception as e:
        print(f"  ⚠️  Could not load existing pkl: {e} — rebuilding from scratch.")

# Determine which sheets to parse for pkl:
# New sheets always need parsing. Also re-parse ALL sheets if pkl was empty/missing
# (handles first-run case gracefully).
sheets_for_pkl = new_sheets if existing_all_picks else all_hr_sheets

new_picks = []
skipped_rows = 0

for sheet_date, sheet_name in sheets_for_pkl:
    date_str = _sheet_date_str(sheet_name)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    for _, row in df.iterrows():
        if len(row) < 10:
            skipped_rows += 1
            continue

        rank_raw = str(row.iloc[0]).strip()
        if rank_raw not in VALID_RANKS:
            continue

        rank = RANK_MAP.get(rank_raw) or _si(rank_raw, default=None)
        if rank is None:
            continue

        name = str(row.iloc[1]).strip()
        if not name or name in ('nan', 'None', ''):
            continue

        hr_prob = _sf(_col(row, 8))
        if hr_prob is None:
            continue

        status   = str(_col(row, 2, '')).strip()
        hs_raw   = str(_col(row, 17, '')).strip().replace('★', '').replace('~', '')
        edge_raw = str(_col(row, 16, '')).strip().replace('%', '')
        edge     = _sf(edge_raw)
        if edge is not None:
            edge = edge / 100.0

        new_picks.append({
            'date':      date_str,
            'rank':      rank,
            'name':      name,
            'confirmed': '✅' in status or 'Conf' in status,
            'projected': '📋' in status or 'Proj' in status,
            'hr_prob':   hr_prob,
            'score':     _sf(_col(row, 9)),
            'power':     _sf(_col(row, 10)),
            'vuln':      _sf(_col(row, 11)),
            'pm':        _sf(_col(row, 12)),
            'park':      _sf(_col(row, 13)),
            'env':       _sf(_col(row, 14)),
            'pa':        _sf(_col(row, 15)),
            'edge':      edge,
            'hs':        _sf(hs_raw) if hs_raw else None,
            'sig':       _sf(_col(row, 18)),
            'hits':      _si(_col(row, 19, 0)),
            'hr':        _si(_col(row, 20, 0)),
        })

# Merge new picks with existing: existing_all_picks + new (with grade labels preserved)
# Key: remove any existing picks for dates we just re-parsed (avoids dupes)
new_dates_str = {_sheet_date_str(s) for _, s in sheets_for_pkl}
retained_picks = [p for p in existing_all_picks if p.get('date') not in new_dates_str]

# Apply grade-label merge for new picks
merged_new = []
merge_count = 0
for p in new_picks:
    key = (p.get('date', ''), p.get('name', '').strip().lower())
    if key in existing_model_picks:
        merged = dict(existing_model_picks[key])
        merged['hits'] = p.get('hits', merged.get('hits', 0))
        merged['hr']   = p.get('hr',   merged.get('hr', 0))
        for field in ['hr_prob', 'score', 'ranking_score', 'power', 'vuln', 'pm',
                      'park', 'env', 'pa', 'edge', 'hs', 'sig']:
            if p.get(field) is not None:
                merged[field] = p[field]
        merged_new.append(merged)
        merge_count += 1
    else:
        merged_new.append(p)

merged_picks = retained_picks + merged_new
print(f"  Appended {len(merged_new)} new picks ({merge_count} merged with grade labels)")
print(f"  Total picks in pkl: {len(merged_picks)}")

with open(pkl_path, 'wb') as f:
    pickle.dump(merged_picks, f)

# ── Summary ────────────────────────────────────────────────────────────────────
dates_in_pkl = sorted(
    set(p['date'] for p in merged_picks),
    key=lambda s: datetime.strptime(s, '%A, %B %d, %Y'),
    reverse=True
)
hr_rate  = sum(1 for p in merged_picks if p['hr']  > 0) / len(merged_picks) if merged_picks else 0
hit_rate = sum(1 for p in merged_picks if p['hits'] > 0) / len(merged_picks) if merged_picks else 0
labelled = sum(1 for p in merged_picks if p.get('hr_grade') or p.get('has_slm') or p.get('has_highk'))

def _sf2(v, d=0):
    try: return float(v) if v is not None else d
    except: return d

n_sharp_pm  = sum(1 for p in merged_picks if _sf2(p.get('pm')) >= 1.08 and 40 <= _sf2(p.get('score')) < 70)
n_mid_score = sum(1 for p in merged_picks if 50 <= _sf2(p.get('score')) < 60 and _sf2(p.get('pm')) >= 1.04 and 44 <= _sf2(p.get('vuln')) < 53)
n_sig_pm    = sum(1 for p in merged_picks if _sf2(p.get('sig')) >= 5 and _sf2(p.get('pm')) >= 1.04)
n_ic_signal = sum(1 for p in merged_picks if 0 < _sf2(p.get('hs', 99)) < 20 and _sf2(p.get('pm')) >= 1.04 and _sf2(p.get('score')) >= 40)

def _has_note(p, keyword):
    for field in ('hr_grade', 'hit_grade', 'flags', 'notes_summary'):
        val = str(p.get(field, '') or '')
        if keyword in val:
            return True
    return False

n_strong_hr       = sum(1 for p in merged_picks if '🟠 Strong' in str(p.get('hr_grade','')) and 'Gated' not in str(p.get('hr_grade','')) and not p.get('grade_count'))
n_strong_hr_gated = sum(1 for p in merged_picks if '🟠 Strong' in str(p.get('hr_grade','')) and 'Gated' in str(p.get('hr_grade','')))

def _strong_hr_rate(picks, gated=False):
    subset = [p for p in picks if '🟠 Strong' in str(p.get('hr_grade',''))
              and ('Gated' in str(p.get('hr_grade','')) if gated else 'Gated' not in str(p.get('hr_grade','')))
              and not p.get('grade_count') and p.get('hr') is not None]
    if len(subset) < 3: return None, len(subset)
    return sum(1 for p in subset if p.get('hr',0) > 0) / len(subset), len(subset)

shr_rate,  shr_n  = _strong_hr_rate(merged_picks, gated=False)
shrg_rate, shrg_n = _strong_hr_rate(merged_picks, gated=True)

n_l5_prone       = sum(1 for p in merged_picks if _has_note(p, 'HR-prone L5'))
n_l5_suppressing = sum(1 for p in merged_picks if _has_note(p, 'Suppressing L5'))
n_l5_command     = sum(1 for p in merged_picks if _has_note(p, 'Command issues L5'))

def _l5_hr_rate(picks, note_keyword):
    subset = [p for p in picks if _has_note(p, note_keyword) and p.get('hr') is not None]
    if not subset: return None, 0
    return sum(1 for p in subset if _sf2(p.get('hr')) > 0) / len(subset), len(subset)

l5_prone_hr,       l5_prone_n  = _l5_hr_rate(merged_picks, 'HR-prone L5')
l5_suppressing_hr, l5_supp_n   = _l5_hr_rate(merged_picks, 'Suppressing L5')

print(f"✓ PKL saved: {pkl_path}")
print(f"  {len(merged_picks)} picks  |  {len(dates_in_pkl)} slates")
print(f"  Grade-labelled picks: {labelled}/{len(merged_picks)}")
print(f"  Base HR rate: {hr_rate:.1%}  |  Base hit rate: {hit_rate:.1%}")
print(f"  New grade counts (field-detected): SHARP PM={n_sharp_pm}  MID-SCORE={n_mid_score}  SIG+PM={n_sig_pm}  IC+SIGNAL={n_ic_signal}")
print(f"  Strong HR (pts-based): non-gated={n_strong_hr}"
      + (f" ({shr_rate:.1%} actual HR, n={shr_n})" if shr_rate is not None else "")
      + f"  |  gated={n_strong_hr_gated}"
      + (f" ({shrg_rate:.1%} actual HR, n={shrg_n})" if shrg_rate is not None else ""))
print(f"  Pitcher L5 signals: HR-prone={n_l5_prone}"
      + (f" ({l5_prone_hr:.1%} actual HR, n={l5_prone_n})" if l5_prone_n >= 3 else "")
      + f"  |  Suppressing={n_l5_suppressing}"
      + (f" ({l5_suppressing_hr:.1%} actual HR, n={l5_supp_n})" if l5_supp_n >= 3 else "")
      + f"  |  Command issues={n_l5_command}")
print(f"  Most recent slates: {dates_in_pkl[:3]}")
if skipped_rows:
    print(f"  (Skipped {skipped_rows} rows with <10 columns)")
print(f"\n✓ Done! Appended {len(new_sheets)} new sheet(s).")

# ── GitHub Auto-Upload ─────────────────────────────────────────────────────────
import os, subprocess, shutil, urllib.error

_GH_REPO = "nocompharrison/MLB-HR-Model"
_GH_CLONE_DIR = Path(__file__).resolve().parent / "MLB-HR-Model_git"

def _load_gh_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        print(f"  🔑 GitHub token loaded from environment variable")
        return token
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent / "github_token.env")
    except Exception:
        pass
    try:
        candidates.append(Path(os.path.abspath(__file__)).parent / "github_token.env")
    except Exception:
        pass
    candidates.append(Path(os.getcwd()) / "github_token.env")
    for env_file in candidates:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("GITHUB_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    if token:
                        print(f"  🔑 GitHub token loaded from: {env_file}")
                        return token
    print(f"  ⚠️  github_token.env not found in: {[str(c) for c in candidates]}")
    return ""

_GH_TOKEN = _load_gh_token()

def _git_run(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git command failed ({result.returncode}): {' '.join(cmd)}\n"
                           f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result

def _gh_upload(local_path, repo_path, branch="main"):
    """Push a file to the repo via a real local git clone + native `git push`,
    not the GitHub REST API.

    WHY (2026-08-27, second revision). First revision here used the Git Data
    API (blob -> tree -> commit -> ref), which GitHub's own docs recommend
    over the single-request Contents API for larger files - that fixed the
    empty-file failure the Contents API produced, but the Blob API turned
    out to have its OWN ceiling: pushing the ~40MB CSV archive through it
    failed outright with "input was too large to process... consider
    creating the blob in a local clone of the repository and then pushing
    it to GitHub" - which is exactly what this version now does. This
    method has actually been tested end-to-end (fresh clone+push, no-op
    detection, modified-file re-push, byte-for-byte sha256 verification
    against a separate clone) - see github_large_file_push.py, the
    standalone version this was adapted from, for that test history and
    more detail. Both files this script uploads (the CSV and the pkl) are
    exactly the kind of file that can grow large over a season, so this
    needed the version actually proven to handle that, not just the one
    that was an improvement over the last failure.

    Uses a persistent local clone (MLB-HR-Model_git next to this script) so
    only the FIRST run ever does a full clone - every run after that is a
    fast pull + copy + commit + push, which is also more efficient than
    the old approach of re-sending the whole file's content through an API
    call every single day regardless of whether anything changed.
    """
    if not _GH_TOKEN:
        print(f"  ⚠️  GitHub upload skipped — GITHUB_TOKEN not set.")
        return
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"  ⚠️  File not found: {local_path}")
        return
    if shutil.which("git") is None:
        print(f"  ⚠️  GitHub upload skipped — `git` not found on PATH. "
              f"Install Git for Windows (https://git-scm.com/download/win).")
        return

    size_mb = local_path.stat().st_size / (1024 * 1024)
    auth_url = f"https://{_GH_TOKEN}@github.com/{_GH_REPO}.git"
    plain_url = f"https://github.com/{_GH_REPO}.git"

    try:
        # 1. Ensure the local clone exists and matches origin/branch.
        if _GH_CLONE_DIR.exists() and (_GH_CLONE_DIR / ".git").exists():
            _git_run(["git", "remote", "set-url", "origin", auth_url], cwd=str(_GH_CLONE_DIR))
            _git_run(["git", "fetch", "origin", branch], cwd=str(_GH_CLONE_DIR))
            _git_run(["git", "checkout", branch], cwd=str(_GH_CLONE_DIR))
            _git_run(["git", "reset", "--hard", f"origin/{branch}"], cwd=str(_GH_CLONE_DIR))
        elif _GH_CLONE_DIR.exists():
            print(f"  ⚠️  {_GH_CLONE_DIR} exists but isn't a git repo — skipping upload of {repo_path}.")
            return
        else:
            print(f"  📥 First run — cloning {_GH_REPO} to {_GH_CLONE_DIR} (one-time cost)...")
            _GH_CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
            _git_run(["git", "clone", "--branch", branch, auth_url, str(_GH_CLONE_DIR)])
            _git_run(["git", "remote", "set-url", "origin", plain_url], cwd=str(_GH_CLONE_DIR))
            _git_run(["git", "remote", "set-url", "origin", auth_url], cwd=str(_GH_CLONE_DIR))

        # 2. Identity - a fresh clone has none configured, and `git commit`
        # fails outright without one.
        name = subprocess.run(["git", "config", "user.name"], cwd=str(_GH_CLONE_DIR),
                              capture_output=True, text=True).stdout.strip()
        email = subprocess.run(["git", "config", "user.email"], cwd=str(_GH_CLONE_DIR),
                               capture_output=True, text=True).stdout.strip()
        if not name:
            _git_run(["git", "config", "user.name", "MLB-HR-Model Automation"], cwd=str(_GH_CLONE_DIR))
        if not email:
            _git_run(["git", "config", "user.email", "automation@local"], cwd=str(_GH_CLONE_DIR))

        # 3. Copy the file into place and commit only if it actually changed.
        dest = _GH_CLONE_DIR / repo_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        _git_run(["git", "add", repo_path], cwd=str(_GH_CLONE_DIR))
        status = subprocess.run(["git", "status", "--porcelain", "--", repo_path],
                                cwd=str(_GH_CLONE_DIR), capture_output=True, text=True)
        if not status.stdout.strip():
            print(f"  ℹ️  No changes: {repo_path} already matches what's on GitHub.")
            return

        _git_run(["git", "commit", "-m",
                 f"Auto-update {repo_path} — {datetime.today().strftime('%Y-%m-%d %H:%M')}"],
                cwd=str(_GH_CLONE_DIR))
        _git_run(["git", "push", "origin", branch], cwd=str(_GH_CLONE_DIR))
        print(f"  ✅ GitHub Updated: {repo_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ⚠️  GitHub upload failed for {repo_path}: {e}")

print("\n── GitHub Auto-Upload ─────────────────────────────────────────────")
_gh_upload(output_path, "FantasyLabsMLB.csv")
_gh_upload(pkl_path, "full_picks.pkl")
print("──────────────────────────────────────────────────────────────────\n")
