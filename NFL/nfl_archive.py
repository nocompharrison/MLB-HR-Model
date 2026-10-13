"""
nfl_archive.py — week-stamped archive of the NFL model CSVs, decoupled from the mirror.

WHY
---
NFL/cheatsheet.csv, results.csv, optimizer.csv, props.csv and lineups.csv are overwritten
in place every run, so only the current week ever exists on GitHub. That makes it
impossible to fit or validate anything on a multi-week panel — every "finding" ends up
resting on one week. This keeps an immutable week-stamped copy of each.

WHY IT DOESN'T HOOK INTO THE UPLOADER
-------------------------------------
Whatever mirrors the NFL CSVs is not in this repo (the three NFLSolver*.py files are
xlwings solvers that only read/write the workbook; mlb_hr_model.py's _upload_to_github()
handles outputs/current_*.csv only). Rather than depend on finding it, this script reads
the CSVs back from GitHub — where the mirror has already put them — and pushes
week-stamped copies alongside. So it needs no knowledge of, and no edit to, the mirror.

Run it as its own Task Scheduler step, any time after the mirror:

    python nfl_archive.py                  # archive whatever is currently on GitHub
    python nfl_archive.py --dry-run        # show what it would do, change nothing
    python nfl_archive.py --seed cheatsheet=C:\\tmp\\wk05.csv --seed-week 5 --seed-slate Main

Token resolution matches mlb_hr_model.py: GITHUB_TOKEN env var, else a github_token.env
file (script dir, then cwd) containing a GITHUB_TOKEN= line.

NOTES
-----
* Deduplicated by SHA-256 of content, so re-running after an unchanged mirror is a no-op.
  Safe to schedule liberally (your Sunday 11:45 pre-lock run won't bloat anything).
* results.csv is filed by the week ITS OWN ROWS describe, not by lineups.csv's
  "Current Week" pointer — the pointer rolls forward on Tuesday while results.csv still
  holds the week just played, so filing by the pointer would mislabel a whole week.
* results.csv is only complete after MNF, so the index records teams_scored and
  latest.txt always points at the MOST COMPLETE snapshot for a week, never merely the
  newest. Nothing is ever deleted; only the index pick changes.
* NFL/archive/latest.txt is the discovery index, same shape as the scraper's latest.txt.
  Without it the archive is unusable to an agent that cannot list a directory.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import sys
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────

_GH_REPO = "nocompharrison/MLB-HR-Model"
_GH_BRANCH = "main"
_RAW = f"https://raw.githubusercontent.com/{_GH_REPO}/{_GH_BRANCH}/"
_API = f"https://api.github.com/repos/{_GH_REPO}/contents/"

TRACKED = ["cheatsheet", "results", "optimizer", "props", "lineups"]

ARCHIVE_ROOT = "NFL/archive"
INDEX_PATH = f"{ARCHIVE_ROOT}/index.csv"
LATEST_PATH = f"{ARCHIVE_ROOT}/latest.txt"

INDEX_FIELDS = ["captured_utc", "captured_et", "season", "week", "slate", "file",
                "path", "sha256", "rows", "bytes", "teams_scored", "note"]

SEASON_WEEK1_TUESDAY = {2026: datetime(2026, 9, 8), 2025: datetime(2025, 9, 2)}
ET = timezone(timedelta(hours=-4))  # display only


# ── token (same resolution order as mlb_hr_model.py) ───────────────────────────

def _load_gh_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        print("  🔑 GitHub token loaded from environment variable")
        return token
    candidates = []
    for getter in (lambda: Path(__file__).resolve().parent,
                   lambda: Path(os.path.abspath(__file__)).parent,
                   lambda: Path(os.getcwd())):
        try:
            candidates.append(getter() / "github_token.env")
        except Exception:
            pass
    # the NFL scripts live in NFL/, the token file may sit at the repo root
    try:
        candidates.append(Path(__file__).resolve().parent.parent / "github_token.env")
    except Exception:
        pass
    for env_file in candidates:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("GITHUB_TOKEN="):
                    tok = line.split("=", 1)[1].strip()
                    if tok:
                        print(f"  🔑 GitHub token loaded from: {env_file}")
                        return tok
    print(f"  ⚠️  github_token.env not found in any of: {[str(c) for c in candidates]}")
    return ""


# ── GitHub helpers ─────────────────────────────────────────────────────────────

def _fetch_raw(repo_path: str) -> bytes | None:
    """Read a file from the mirror. raw.githubusercontent has no 1MB content cap."""
    try:
        with urllib.request.urlopen(_RAW + repo_path, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  ⚠️  fetch {repo_path}: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  ⚠️  fetch {repo_path}: {e}")
        return None


def _put(token: str, repo_path: str, blob: bytes, message: str) -> bool:
    """Create or replace a file via the contents API (same shape as _github_upload)."""
    if not token:
        print(f"  ⚠️  upload skipped — no token: {repo_path}")
        return False
    url = _API + repo_path
    headers = {"Authorization": f"token {token}",
               "Accept": "application/vnd.github+json",
               "Content-Type": "application/json"}
    sha = None
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=60) as resp:
            sha = json.loads(resp.read()).get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  ⚠️  GET {repo_path}: HTTP {e.code}")
            return False
    payload = {"message": message, "content": base64.b64encode(blob).decode(),
               "branch": _GH_BRANCH}
    if sha:
        payload["sha"] = sha
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers=headers, method="PUT")
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
        print(f"  ✅ {'Updated' if sha else 'Created'}: {repo_path}")
        return True
    except urllib.error.HTTPError as e:
        print(f"  ⚠️  PUT {repo_path}: HTTP {e.code} {e.read()[:200]!r}")
        return False
    except Exception as e:
        print(f"  ⚠️  PUT {repo_path}: {e}")
        return False


# ── metadata from the CSVs themselves ──────────────────────────────────────────

def _rows(blob: bytes) -> int:
    try:
        return max(blob.decode("utf-8", "ignore").count("\n") - 1, 0)
    except Exception:
        return -1


def _reader(blob: bytes):
    return csv.DictReader(io.StringIO(blob.decode("utf-8", "ignore")))


def detect_season(now: datetime) -> int:
    return now.year if now.month >= 3 else now.year - 1


def detect_week(lineups: bytes | None, now: datetime) -> tuple[int, str]:
    """Authoritative = the 'Current Week' cell of lineups.csv (row 2 under row 1)."""
    if lineups:
        try:
            rdr = csv.reader(io.StringIO(lineups.decode("utf-8", "ignore")))
            hdr = next(rdr, [])
            val = next(rdr, [])
            if "Current Week" in hdr:
                i = hdr.index("Current Week")
                if i < len(val) and str(val[i]).strip():
                    return int(float(str(val[i]).strip())), "lineups.csv:Current Week"
        except Exception:
            pass
    anchor = SEASON_WEEK1_TUESDAY.get(detect_season(now))
    if anchor:
        return max(1, min(22, (now - anchor).days // 7 + 1)), "date fallback"
    return 0, "unknown"


def detect_slate(optimizer: bytes | None) -> str:
    if not optimizer:
        return "unknown"
    try:
        rdr = _reader(optimizer)
        if not rdr.fieldnames or "Slate" not in rdr.fieldnames:
            return "unknown"
        seen, vals = set(), []
        for row in rdr:
            v = (row.get("Slate") or "").strip()
            if v and v not in seen:
                seen.add(v)
                vals.append(v)
    except Exception:
        return "unknown"
    if not vals:
        return "unknown"
    order = {"Wednesday": 0, "Wed": 0, "Thursday": 1, "Thu": 1,
             "Sunday": 2, "Sun": 2, "Monday": 3, "Mon": 3}
    vals.sort(key=lambda v: order.get(v, 9))
    return "+".join(vals)


def week_from_results(blob: bytes) -> int | None:
    """results.csv lags the sheet pointer — file it by the week its rows describe."""
    try:
        counts: dict[int, int] = {}
        rdr = _reader(blob)
        if not rdr.fieldnames or "week" not in rdr.fieldnames:
            return None
        for row in rdr:
            raw = (row.get("week") or "").strip()
            if not raw:
                continue
            try:
                wk = int(float(raw))
            except ValueError:
                continue
            counts[wk] = counts.get(wk, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
    except Exception:
        return None


def teams_scored(blob: bytes) -> str:
    """How many teams have a scored row — distinguishes partial from complete."""
    try:
        rdr = _reader(blob)
        if not rdr.fieldnames or "team" not in rdr.fieldnames \
                or "fd_points" not in rdr.fieldnames:
            return ""
        teams = set()
        for row in rdr:
            pts = (row.get("fd_points") or "").strip()
            tm = (row.get("team") or "").strip()
            if tm and pts not in ("", "NA", "NaN"):
                teams.add(tm)
        return str(len(teams))
    except Exception:
        return ""


# ── index / latest.txt ─────────────────────────────────────────────────────────

def _load_index() -> list[dict]:
    blob = _fetch_raw(INDEX_PATH)
    if not blob:
        return []
    try:
        return list(_reader(blob))
    except Exception:
        return []


def _index_blob(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=INDEX_FIELDS, extrasaction="ignore",
                        lineterminator="\n")
    wr.writeheader()
    wr.writerows(rows)
    return buf.getvalue().encode()


def _latest_blob(rows: list[dict]) -> bytes:
    best: dict[tuple, dict] = {}

    def ts(row) -> int:
        try:
            return int(str(row.get("teams_scored") or "").strip())
        except ValueError:
            return -1

    for r in rows:
        key = (str(r.get("season", "")), str(r.get("week", "")),
               str(r.get("file", "")), str(r.get("slate", "")))
        prev = best.get(key)
        if prev is None:
            best[key] = r
            continue
        new_n, old_n = ts(r), ts(prev)
        if new_n >= 0 or old_n >= 0:
            # completeness decides outright; a later but MORE PARTIAL results
            # snapshot must never displace a complete one. Recency breaks ties only.
            if new_n > old_n or (new_n == old_n and
                                 r.get("captured_utc", "") > prev.get("captured_utc", "")):
                best[key] = r
        elif r.get("captured_utc", "") > prev.get("captured_utc", ""):
            best[key] = r

    def sk(r):
        try:
            wk = int(float(str(r.get("week") or 0)))
        except (TypeError, ValueError):
            wk = 0
        return (str(r.get("season", "")), wk, str(r.get("file", "")), str(r.get("slate", "")))

    out = [
        f"# NFL CSV archive | updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"# raw base: {_RAW}",
        f"# {len(rows)} snapshots. Best per (season, week, file, slate).",
        "# results.csv rows show TEAMS scored — pick the highest for a given week.",
        "# SEASON  WK  SLATE             FILE            ROWS  TEAMS  PATH",
    ]
    for r in sorted(best.values(), key=sk):
        out.append(
            f"{r.get('season',''):<8}{str(r.get('week','')):>2}  "
            f"{(r.get('slate','') or '-'):<18}{r.get('file',''):<16}"
            f"{str(r.get('rows','')):>6}  {(r.get('teams_scored','') or '-'):>5}  "
            f"{r.get('path','')}"
        )
    return ("\n".join(out) + "\n").encode()


# ── main ───────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False, seed: tuple[str, bytes] | None = None,
        seed_week: int | None = None, seed_slate: str | None = None) -> int:
    """Returns the number of new snapshots written (0 if nothing changed)."""
    now = datetime.now()
    stamp = datetime.now(timezone.utc)
    token = "" if dry_run else _load_gh_token()
    if not dry_run and not token:
        print("  ⚠️  no token — nothing can be uploaded. Aborting.")
        return 0

    lineups = _fetch_raw("NFL/lineups.csv")
    optimizer = _fetch_raw("NFL/optimizer.csv")
    season = detect_season(now)
    week, wk_src = detect_week(lineups, now)
    slate = detect_slate(optimizer)
    print(f"  season {season}, week {week} ({wk_src}), slate '{slate}'"
          + ("  [DRY RUN]" if dry_run else ""))

    if seed:
        items = [(seed[0], seed[1])]
        week = seed_week if seed_week is not None else week
        slate = seed_slate or slate
    else:
        items = []
        for name in TRACKED:
            blob = (lineups if name == "lineups" else
                    optimizer if name == "optimizer" else
                    _fetch_raw(f"NFL/{name}.csv"))
            if blob:
                items.append((name, blob))
            else:
                print(f"  — missing on GitHub, skipped: NFL/{name}.csv")

    index = _load_index()
    have = {r.get("sha256", "") for r in index}
    new = 0

    for name, blob in items:
        digest = hashlib.sha256(blob).hexdigest()
        if digest in have:
            print(f"  = unchanged, skipped: {name}.csv")
            continue

        file_week = week
        if name == "results" and not seed:
            rw = week_from_results(blob)
            if rw is not None:
                if rw != week:
                    print(f"    note: results.csv holds week {rw}, pointer says {week}"
                          f" -> filing under wk{rw:02d}")
                file_week = rw

        tag = (slate or "unknown").replace("+", "-").replace(" ", "")
        dest = (f"{ARCHIVE_ROOT}/{season}/wk{file_week:02d}/"
                f"{name}_{stamp:%Y%m%dT%H%M%SZ}_{tag}.csv")
        if dry_run:
            print(f"  + would write: {dest}  ({len(blob):,} bytes, {_rows(blob)} rows)")
        elif not _put(token, dest, blob,
                      f"archive: {name}.csv wk{file_week:02d} {stamp:%Y-%m-%d %H:%M}Z"):
            continue
        index.append({
            "captured_utc": f"{stamp:%Y-%m-%d %H:%M:%S}",
            "captured_et": f"{stamp.astimezone(ET):%Y-%m-%d %H:%M:%S}",
            "season": str(season), "week": str(file_week), "slate": slate,
            "file": f"{name}.csv", "path": dest, "sha256": digest,
            "rows": _rows(blob), "bytes": len(blob),
            "teams_scored": teams_scored(blob) if name == "results" else "",
            "note": "seed" if seed else "",
        })
        have.add(digest)
        new += 1

    if not new:
        print("  nothing to do — archive already current")
        return 0
    if dry_run:
        print(f"  [DRY RUN] would add {new} snapshot(s) and refresh index + latest.txt")
        return new
    _put(token, INDEX_PATH, _index_blob(index),
         f"archive: index +{new} ({stamp:%Y-%m-%d %H:%M}Z)")
    _put(token, LATEST_PATH, _latest_blob(index),
         f"archive: latest.txt ({stamp:%Y-%m-%d %H:%M}Z)")
    print(f"  {new} new snapshot(s); index and latest.txt refreshed")
    return new


def main() -> int:
    ap = argparse.ArgumentParser(description="Week-stamp the NFL model CSVs on GitHub.")
    ap.add_argument("--dry-run", action="store_true", help="report only, upload nothing")
    ap.add_argument("--seed", metavar="NAME=PATH",
                    help="back-fill one local file, e.g. cheatsheet=C:\\tmp\\wk05.csv")
    ap.add_argument("--seed-week", type=int)
    ap.add_argument("--seed-slate")
    a = ap.parse_args()

    try:
        if a.seed:
            if "=" not in a.seed:
                print("ERROR: --seed needs NAME=PATH", file=sys.stderr)
                return 2
            name, path = a.seed.split("=", 1)
            p = Path(path)
            if not p.exists():
                print(f"ERROR: seed file not found: {p}", file=sys.stderr)
                return 2
            if a.seed_week is None:
                print("ERROR: --seed requires --seed-week", file=sys.stderr)
                return 2
            run(dry_run=a.dry_run, seed=(name.strip(), p.read_bytes()),
                seed_week=a.seed_week, seed_slate=a.seed_slate)
        else:
            run(dry_run=a.dry_run)
    except Exception:
        # archiving must never be the reason a scheduled chain reports failure
        print("  ⚠️  nfl_archive failed:", file=sys.stderr)
        traceback.print_exc()
        return 0
    return 0


if __name__ == "__main__":
    print("\n── NFL CSV Archive ───────────────────────────────────────────────")
    rc = main()
    print("──────────────────────────────────────────────────────────────────\n")
    raise SystemExit(rc)
