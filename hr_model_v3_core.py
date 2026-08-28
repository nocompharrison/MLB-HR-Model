"""
hr_model_v3_core.py — MLB HR Model v3.0 "Calibrated Edge" core engine
=====================================================================

Replaces the Phase-2 heuristic probability stack. Every function here exists to
fix a specific, measured defect found in the v2 audit (2026-07-27, n=3,446
batter-slates / 72 slates). Each carries its audit citation.

AUDIT BASELINE THIS REPLACES
    v2 model hr_prob      AUC 0.5442   Brier 0.13901
    market implied prob   AUC 0.5930   Brier 0.13771   <- market beat the model
    incremental value of v2 prob over market: LR chi2=0.04, p=0.83 (i.e. none)
    v2 prob orthogonal to market: AUC 0.5112 (coin flip)
    top-decile calibration: 21.1% predicted -> 17.4% actual (ratio 0.83, inverted)

DESIGN RULES (non-negotiable in v3)
    R1. Market price NEVER enters the probability. It is the benchmark, not a feature.
    R2. No probability is reported uncalibrated.
    R3. No signal ships without clearing a permutation null.
    R4. Stake size derives from calibrated probability, never from a tier label.
    R5. Every threshold is either fitted on a held-out fold or declared a prior.
"""

from __future__ import annotations
import json, math, os, datetime as _dt
from collections import defaultdict

import numpy as np

MODEL_VERSION = "3.0.0"
MODEL_CODENAME = "Calibrated Edge"


# ═══════════════════════════════════════════════════════════════════════════
# 1. EMPIRICAL PA DISTRIBUTION            [FIX #3 — negative binomial bug]
# ═══════════════════════════════════════════════════════════════════════════
# v2 BUG: simulate_hr_prob used rng.negative_binomial(pa*4.0, p) where
#   p = 4*pa/(4*pa + pa) = 0.8 CONSTANT, independent of pa. Mean came out
#   right (= pa) which hid the bug, but variance = 1.25*pa, giving SD 2.08.
# MEASURED TRUTH (FantasyLabsMLB.xlsm 'Results' sheet, n=302 batter-games):
#   PA mean 3.596, SD 1.037, var/mean 0.299 -> UNDER-dispersed, not over.
#   The v2 sim was 2.0x too wide, which swamped the 3% lineup-slot fatigue
#   term and is why lineup position showed almost no effect in v2 output.
#
# Distributions below are conditioned on lineup slot, anchored to the measured
# marginal and to published MLB PA-per-slot means (slot 1 ~4.65 down to
# slot 9 ~3.75 for a batter who plays the full game).

_PA_SUPPORT = np.array([1, 2, 3, 4, 5, 6], dtype=float)

_PA_PMF_BY_SLOT = {
    1: np.array([0.010, 0.020, 0.110, 0.420, 0.390, 0.050]),
    2: np.array([0.012, 0.024, 0.125, 0.435, 0.360, 0.044]),
    3: np.array([0.014, 0.028, 0.145, 0.455, 0.320, 0.038]),
    4: np.array([0.016, 0.032, 0.165, 0.470, 0.285, 0.032]),
    5: np.array([0.018, 0.038, 0.190, 0.480, 0.250, 0.024]),
    6: np.array([0.020, 0.044, 0.215, 0.490, 0.215, 0.016]),
    7: np.array([0.024, 0.052, 0.245, 0.495, 0.174, 0.010]),
    8: np.array([0.028, 0.060, 0.275, 0.495, 0.136, 0.006]),
    9: np.array([0.032, 0.070, 0.305, 0.490, 0.100, 0.003]),
}
for _s, _p in _PA_PMF_BY_SLOT.items():
    _PA_PMF_BY_SLOT[_s] = _p / _p.sum()

_PA_PMF_DEFAULT = _PA_PMF_BY_SLOT[5]


def pa_pmf(lineup_spot: int = 5, expected_pa: float | None = None):
    """Return (support, pmf) for plate appearances.

    If `expected_pa` is supplied (e.g. FantasyLabs Est PA, which already
    accounts for the specific game context), the slot pmf is tilted to match
    that mean exactly via exponential reweighting. This keeps the measured
    SHAPE (SD ~1.0) while honouring the per-game mean the data provides —
    v2 got the mean right and the shape catastrophically wrong.
    """
    pmf = _PA_PMF_BY_SLOT.get(int(lineup_spot or 5), _PA_PMF_DEFAULT).copy()
    if expected_pa is None or not np.isfinite(expected_pa) or expected_pa <= 0:
        return _PA_SUPPORT, pmf
    target = float(np.clip(expected_pa, _PA_SUPPORT[0] + 0.05, _PA_SUPPORT[-1] - 0.05))
    lo, hi = -4.0, 4.0
    for _ in range(60):                       # bisection on exponential tilt
        mid = 0.5 * (lo + hi)
        w = pmf * np.exp(mid * _PA_SUPPORT)
        w /= w.sum()
        if float(w @ _PA_SUPPORT) < target:
            lo = mid
        else:
            hi = mid
    w = pmf * np.exp(0.5 * (lo + hi) * _PA_SUPPORT)
    return _PA_SUPPORT, w / w.sum()


# ═══════════════════════════════════════════════════════════════════════════
# 2. CLOSED-FORM GAME PROBABILITY        [FIX #2 — delete the Monte Carlo]
# ═══════════════════════════════════════════════════════════════════════════
# v2 ran 10,000 Bernoulli draws in a Python for-loop per player to estimate a
# quantity with an exact closed form. Measured cost of that choice:
#   rate 0.030 PA 4.3 -> +/-0.00324 SE = 2.7% pure noise, zero info gained
#   rate 0.050 PA 4.6 -> +/-0.00401 SE = 2.0%
# That noise is the same order of magnitude as the edge being hunted, and it
# is ~100x slower. P(>=1 HR) = E_PA[1 - (1-p)^PA]. One line.

def closed_form_hr_prob(rate_per_pa: float,
                        lineup_spot: int = 5,
                        expected_pa: float | None = None,
                        fatigue_mult: float = 1.0) -> float:
    """Exact P(at least one HR). Replaces simulate_hr_prob() entirely.

    `fatigue_mult` keeps the v2 TTO idea (bottom-of-order batters meet a more
    tired starter) but it now actually matters, because the PA distribution is
    no longer wide enough to drown it.
    """
    p = float(rate_per_pa) * float(fatigue_mult)
    if not np.isfinite(p) or p <= 0:
        return 0.0
    p = min(p, 0.35)
    support, pmf = pa_pmf(lineup_spot, expected_pa)
    return float(np.sum(pmf * (1.0 - np.power(1.0 - p, support))))


def closed_form_hit_prob(hit_rate_per_pa: float,
                         lineup_spot: int = 5,
                         expected_pa: float | None = None) -> float:
    """Exact P(at least one hit). Same machinery, hit rate instead of HR rate."""
    p = float(hit_rate_per_pa)
    if not np.isfinite(p) or p <= 0:
        return 0.0
    p = min(p, 0.60)
    support, pmf = pa_pmf(lineup_spot, expected_pa)
    return float(np.sum(pmf * (1.0 - np.power(1.0 - p, support))))


def tto_fatigue_mult(lineup_spot: int = 5, starter_ip: float = 5.5) -> float:
    """Times-through-order fatigue. Bounded, and now non-vestigial."""
    tbf = (int(lineup_spot or 5) - 1) * 3.1 + max(0.0, starter_ip - 5.0) * 4.2
    if tbf <= 18:
        return 1.0
    return float(min(1.12, 1.0 + (tbf - 18) / 6.0 * 0.03))


# ═══════════════════════════════════════════════════════════════════════════
# 3. HIERARCHICAL SHRINKAGE               [FIX #15 — partial pooling priors]
# ═══════════════════════════════════════════════════════════════════════════
# v2 handled small-sample batter rates with hard caps:
#     if power < 78: rate = min(rate, 0.044)
#     if power < 70: rate = min(rate, 0.034)
# A cap is the worst possible shrinkage estimator: it does nothing until it
# does everything, it discards ordering information above the cap, and it is
# blind to sample size. Empirical-Bayes shrinkage handles all three properly.

def shrink_rate(observed_rate: float, n_obs: float,
                prior_rate: float = 0.0335, prior_strength: float = 220.0) -> float:
    """Empirical-Bayes shrinkage of a rate toward a league prior.

    prior_rate     league HR/PA (~3.35%)
    prior_strength pseudo-PA of prior weight. 220 PA ~ the point where an
                   individual HR rate starts to carry more signal than noise.
    A batter with 600 PA keeps ~73% of his own rate; one with 40 PA keeps ~15%.
    """
    if not np.isfinite(observed_rate) or observed_rate < 0:
        return prior_rate
    n = max(0.0, float(n_obs))
    w = n / (n + prior_strength)
    return float(w * observed_rate + (1.0 - w) * prior_rate)


def shrink_matchup(observed: float, n_obs: float, fallback: float) -> float:
    """Shrink a small-sample matchup rate (e.g. L10 BBE vs one pitch type)
    toward the batter's own overall rate. Prior strength 25 batted balls."""
    if not np.isfinite(observed):
        return fallback
    n = max(0.0, float(n_obs))
    w = n / (n + 25.0)
    return float(w * observed + (1.0 - w) * fallback)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ISOTONIC CALIBRATION                    [FIX #8 — held-out calibration]
# ═══════════════════════════════════════════════════════════════════════════
# v2 used a Platt sigmoid with A=0.3931, B=-0.9933 fitted IN-SAMPLE on 873
# picks. Result (measured): top decile 21.1% predicted -> 17.4% actual, and
# non-monotone -- the 16.6% bucket outperformed the 21.1% bucket. The model's
# ordering inverted exactly where bets are placed.
# Isotonic regression is non-parametric, monotone by construction, and here is
# only ever fitted on a fold the scorer did not see.

class IsotonicCalibrator:
    """Monotone probability calibration with out-of-fold fitting.

    P_FLOOR / P_CEILING are physical bounds on a SINGLE-GAME probability, not
    tuning knobs. TAIL_PRIOR_N is the pseudo-count that shrinks thinly-supported
    steps toward the base rate.
    """

    P_CEILING    = 0.320     # nobody is better than ~1-in-3 in one game

    # TAIL_PRIOR_N reduced 40 -> 20 (2026-08-26). Root cause of a live incident:
    # a 272-batter slate produced ~30 batters (several genuinely different raw
    # probabilities, 0.042-0.13) all mapping to the SAME calibrated value
    # (0.1233), because the fitted curve has a flat plateau from x=0.076 to
    # x=0.135 (54 of 200 grid points) where TAIL_PRIOR_N=40 shrinkage pulls
    # everything toward the pool base rate before the isotonic fit is trusted.
    # Tested against the real 2,415-row v3-era training set: N=40 -> 54-point
    # plateau; N=25 -> still 54 points but a different (still flat) value;
    # N=15 -> 36 points; N=8 -> 36 points but the plateau's own value became
    # unstable (0.1233 -> 0.0773), i.e. thin-sample noise, not signal - that
    # is exactly the failure mode TAIL_PRIOR_N exists to prevent. N=20 was
    # chosen as the point that meaningfully narrows the plateau versus 40
    # without dropping into the range where N=8 showed real instability.
    # HONEST LIMIT: this does not eliminate the plateau, only narrows it -
    # the raw probability range below ~0.09 has so few real training
    # examples (support here started at single digits per grid point) that
    # no shrinkage parameter alone can produce a trustworthy gradient there.
    # The actual fix for that is more training data with genuinely low raw
    # probabilities, which only accumulates over time. Re-measure the
    # plateau width directly against future refits rather than assuming
    # this value is still right once n_fit has grown substantially.
    TAIL_PRIOR_N = 20.0

    # ODDS-CONDITIONAL FLOOR (added 2026-07-27, replaces a flat P_FLOOR).
    #
    # THE BUG THIS FIXES. A flat floor of 0.1079 sat ABOVE the true HR rate for
    # every long price, so any bat longer than ~+825 (implied <10.79%) showed
    # POSITIVE edge automatically, regardless of who it was. Kelly then sized
    # only long shots. Measured on the merged CSV:
    #     +680-900 : n=290  actual  9.0%  (0.54x)  implied 11.8%  <- floor 10.8% ABOVE actual
    #     +900+    : n=149  actual 10.7%  (0.64x)  implied  8.3%  <- floor 10.8% ABOVE both
    #     betting >+680 on that manufactured edge: 8.7% HR, 0.52x, flat ROI -7.8%
    # i.e. the floor pointed Kelly straight at the worst cohort in the data.
    #
    # A floor must sit BELOW both the actual rate and the implied probability of
    # its band, or it manufactures edge. Verified against the merged CSV:
    #     band      n     actual  implied  floor
    #     +0-400   1206   21.1%    24.4%   8.0%   OK
    #     +400-600 1466   16.3%    17.4%   6.0%   OK
    #     +600-800  372   10.2%    12.9%   4.5%   OK
    #     +800+     226    8.0%     9.1%   3.5%   OK
    # The floor still exists to stop 0.0% readings (the Willy Adames case); it is
    # now low enough that it can never be the source of an edge signal.
    P_FLOOR_BANDS = ((400, 0.080), (600, 0.060), (800, 0.045), (10**9, 0.035))
    P_FLOOR       = 0.035    # absolute fallback when odds are unknown

    @classmethod
    def floor_for(cls, odds=None):
        """Odds-conditional probability floor. Falls back to the global floor."""
        if odds is None:
            return cls.P_FLOOR
        try:
            o = abs(float(str(odds).replace("+", "").replace(",", "").strip()))
        except (TypeError, ValueError):
            return cls.P_FLOOR
        for hi, fl in cls.P_FLOOR_BANDS:
            if o < hi:
                return fl
        return cls.P_FLOOR

    def __init__(self, name: str = "hr"):
        self.name = name
        self.x_ = None
        self.y_ = None
        self.fitted_on = None
        self.n_fit = 0

    def fit(self, probs, outcomes, groups=None, n_folds: int = 5):
        """Fit on out-of-fold predictions. `groups` should be slate dates so
        that batters from the same slate never straddle the fold boundary --
        they share pitcher, park and weather, so row-wise CV leaks."""
        from sklearn.isotonic import IsotonicRegression
        p = np.asarray(probs, float)
        y = np.asarray(outcomes, float)
        ok = np.isfinite(p) & np.isfinite(y)
        p, y = p[ok], y[ok]
        if len(p) < 50:
            raise ValueError(f"need >=50 rows to calibrate, got {len(p)}")
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        grid = np.linspace(p.min(), p.max(), 200)
        raw = iso.predict(grid)

        # TAIL SHRINKAGE. Each grid point is pulled toward the pool base rate in
        # proportion to how little local support it has. A step backed by 400
        # rows keeps its value; one backed by 12 collapses toward the base rate
        # instead of asserting a 53% single-game HR probability from 18 samples.
        base = float(y.mean())
        half = max((p.max() - p.min()) / 10.0, 1e-6)          # +/- 10% of range
        support = np.array([np.sum(np.abs(p - g) <= half) for g in grid], float)
        w = support / (support + self.TAIL_PRIOR_N)
        self.y_ = w * raw + (1.0 - w) * base

        # PHYSICAL BOUNDS. No batter has a 0% chance of homering, and none has
        # 53%. Bounds are asserted, not hoped for.
        # Fit-time clip uses the LOWEST band floor; the odds-appropriate floor is
        # applied at transform() time, where the price is known.
        self.y_ = np.clip(self.y_, self.P_FLOOR, self.P_CEILING)
        self.y_ = np.maximum.accumulate(self.y_)               # keep monotone
        self.x_ = grid
        self.support_ = support
        self.n_fit = len(p)
        self.fitted_on = _dt.date.today().isoformat()
        return self

    def transform(self, prob: float, odds=None) -> float:
        """Calibrated probability. Pass `odds` so the floor is band-appropriate.

        Without `odds` the most conservative floor (3.5%) applies — that is the
        safe default, since a too-HIGH floor manufactures edge on long shots.

        OUT-OF-RANGE HANDLING (added 2026-08-26). np.interp's default behavior
        clamps any query outside [x_.min(), x_.max()] to the boundary y-value.
        Root-caused via a real slate: training data never saw a raw prob below
        0.0762 (2,415 v3-era rows), but a 272-batter slate produced several
        batters with genuine raw probs as low as 0.042 (weak-power bats like
        a Score-18 batter). Every one of them clamped to the SAME boundary
        value, and because the boundary itself sits in a thinly-supported,
        tail-shrunk region of the curve, a wide band of genuinely different
        low probabilities all collapsed onto one identical output (12.7% on
        6+ different batters, verified in the live diagnostic).
        FIX: extrapolate linearly from the slope of the nearest INTERIOR
        anchor point with real support (>= TAIL_PRIOR_N backing it), rather
        than the boundary point itself, which may be sitting in the shrunk
        region and have near-zero local slope. This preserves monotonicity
        (slope is clamped to >=0) and still respects P_FLOOR/P_CEILING. If no
        adequately-supported anchor exists (e.g. a very small fit), this
        falls back to the original clamp-at-boundary behavior rather than
        guessing at an unsupported slope.
        """
        if self.x_ is None:
            return float(prob)
        p = float(prob)
        flo = self.floor_for(odds)
        if not np.isfinite(p):
            return flo

        if p < self.x_[0] or p > self.x_[-1]:
            extrapolated = self._extrapolate(p)
            if extrapolated is not None:
                return float(np.clip(extrapolated, flo, self.P_CEILING))
            # fall through to the old clamp-at-boundary behavior below

        return float(np.clip(np.interp(p, self.x_, self.y_), flo, self.P_CEILING))

    def _extrapolate(self, p: float):
        """Linear extrapolation anchored to the nearest well-supported interior
        grid point, using the slope from there to the boundary. Returns None
        (signalling "use the old clamp behavior") if support data isn't
        available or no point clears the TAIL_PRIOR_N support threshold."""
        support = getattr(self, "support_", None)
        if support is None or len(support) != len(self.x_):
            return None

        low = p < self.x_[0]
        rng = range(len(self.x_)) if low else range(len(self.x_) - 1, -1, -1)
        anchor = None
        for k in rng:
            if support[k] >= self.TAIL_PRIOR_N:
                anchor = k
                break
        if anchor is None:
            return None

        edge = 0 if low else len(self.x_) - 1
        x_a, y_a = self.x_[anchor], self.y_[anchor]
        x_e, y_e = self.x_[edge], self.y_[edge]
        if x_e == x_a:
            return None
        slope = (y_e - y_a) / (x_e - x_a)
        slope = max(slope, 0.0)   # never let extrapolation run downhill — isotonic stays monotone
        return y_e + slope * (p - x_e)

    __call__ = transform

    def to_dict(self):
        d = {"name": self.name, "x": list(map(float, self.x_)),
             "y": list(map(float, self.y_)), "n_fit": self.n_fit,
             "fitted_on": self.fitted_on, "version": MODEL_VERSION,
             "p_floor": self.P_FLOOR, "p_ceiling": self.P_CEILING,
             "y_min": float(np.min(self.y_)), "y_max": float(np.max(self.y_))}
        # ── ADDED 2026-08-26: without this, support_ never survives the
        # save/load round trip that happens on every scoring run (the
        # calibrator is loaded fresh from JSON for each batter), so the new
        # extrapolation logic in transform() would always silently fall back
        # to the old clamp-at-boundary behavior it was written to replace.
        support = getattr(self, "support_", None)
        if support is not None:
            d["support"] = list(map(float, support))
        return d

    @classmethod
    def from_dict(cls, d):
        c = cls(d.get("name", "hr"))
        c.x_ = np.array(d["x"], float)
        c.y_ = np.array(d["y"], float)
        c.n_fit = d.get("n_fit", 0)
        c.fitted_on = d.get("fitted_on")
        if "support" in d:
            c.support_ = np.array(d["support"], float)
        return c

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except Exception:
            return None


def reliability_report(probs, outcomes, n_bins: int = 10):
    """Reliability diagram data + ECE + Brier skill score vs the base rate.
    v2 never measured any of these."""
    p = np.asarray(probs, float)
    y = np.asarray(outcomes, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {}
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": b, "n": int(m.sum()),
                     "pred": float(p[m].mean()), "actual": float(y[m].mean())})
        ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    return {"bins": rows, "ece": float(ece), "brier": brier,
            "brier_base": brier_base,
            "brier_skill_score": float(1.0 - brier / brier_base) if brier_base else 0.0,
            "n": int(len(p)), "base_rate": float(base)}


# ═══════════════════════════════════════════════════════════════════════════
# 5. MARKET INTERFACE + EDGE          [FIX #1 — break the circularity]
# ═══════════════════════════════════════════════════════════════════════════
# v2's fatal structural defect:
#     market_signal = 1.0 + (implied_prob - 0.38) * 0.5
#     hr_prob = simulate_hr_prob(rate, pa, env, pm, bp_f, mkt, ...)   <- mkt IN
#     edge = hr_prob - market_implied
# Measured consequence: corr(hr_prob, implied) = 0.419, and
#     logit(model) = 0.206 * logit(market) - 1.336
# so "edge" was f(price) - price: a relabelling of price, plus noise. It could
# not detect mispricing because it had no independent opinion to compare.
# In v3 the market appears ONLY below this line, and never upstream of it.

def american_to_prob(odds) -> float | None:
    """American odds -> implied probability, vig included."""
    try:
        o = float(str(odds).replace("+", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    return (-o) / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def american_to_decimal(odds) -> float | None:
    try:
        o = float(str(odds).replace("+", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    return 1.0 + (100.0 / (-o) if o < 0 else o / 100.0)


def devig_two_way(over_odds, under_odds) -> float | None:
    """Remove the bookmaker's margin using both sides of the market.

    v2 compared model probability against the RAW over price, which carries
    4-8% vig on HR props. That guarantees a structurally negative measured
    edge and silently biases every edge threshold. Multiplicative de-vig.
    """
    po, pu = american_to_prob(over_odds), american_to_prob(under_odds)
    if po is None or pu is None or (po + pu) <= 0:
        return po
    return float(po / (po + pu))


def compute_edge(model_prob: float, market_prob: float | None) -> dict:
    """Edge and expected value. model_prob must be market-free (rule R1)."""
    if market_prob is None or not np.isfinite(model_prob):
        return {"edge": None, "edge_pct": None, "ev_per_unit": None, "fair_odds": None}
    edge = float(model_prob - market_prob)
    dec = 1.0 / market_prob if market_prob > 0 else None
    ev = (model_prob * (dec - 1.0) - (1.0 - model_prob)) if dec else None
    fair = (100.0 * (1.0 - model_prob) / model_prob) if model_prob > 0 else None
    return {"edge": edge, "edge_pct": edge * 100.0,
            "ev_per_unit": ev, "fair_odds": fair}


# ═══════════════════════════════════════════════════════════════════════════
# 6. FRACTIONAL KELLY STAKING              [FIX #11 — stake from probability]
# ═══════════════════════════════════════════════════════════════════════════
# v2 sized bets by tier label (MUST PLAY / STRONG PLAY / TRACKING). A tier is
# not a stake. At +250 a 30% probability is a large bet; at -150 it is a pass.
# Kelly is the only sizing rule that is optimal in the long run, and fractional
# Kelly is the standard defence against probability-estimation error -- which,
# given the audit, should be assumed large.

KELLY_MAX_ODDS = 500     # hard ceiling — see RELIABILITY GUARD below


def kelly_stake(model_prob: float, odds, bankroll: float = 1.0,
                fraction: float = 0.25, cap: float = 0.02,
                min_edge: float = 0.02, max_odds: int = KELLY_MAX_ODDS) -> dict:
    """Fractional Kelly. Defaults: quarter-Kelly, hard cap 2% of bankroll.

    The cap matters more than the fraction. If the true probability is off by
    30% (entirely plausible here), full Kelly is ruinous and quarter-Kelly with
    a 2% cap merely underperforms.
    """
    dec = american_to_decimal(odds)
    mp = float(model_prob) if np.isfinite(model_prob) else 0.0
    if dec is None or dec <= 1.0 or mp <= 0.0:
        return {"stake": 0.0, "stake_pct": 0.0, "kelly_full": 0.0,
                "reason": "no price or no probability"}
    # ══ RELIABILITY GUARD (2026-07-27) ═══════════════════════════════
    # The market-free model does not discriminate widely enough to price long
    # shots. Measured on the merged CSV:
    #     raw hr_prob spans 0.0766-0.2935; TRUE rates span 0.080-0.249.
    # The model's output range is NARROWER than reality, and isotonic
    # calibration cannot manufacture discrimination that is not in its input.
    # Tail shrinkage compounds it: the calibrated curve bottoms near 11% while
    # +680-900 bats convert at 9.0% and +900+ at 10.7%.
    #
    # Consequence: at long prices the model reports positive edge purely from
    # compression. Kelly sized ONLY long shots on the 2026-07-27 slate — all 9
    # stakes were +680 to +1000, a cohort that historically runs 8.7% HR / 0.52x
    # with a flat-stake ROI of -7.8%.
    #
    # Note NO odds band beats its own price (the gap IS the vig, 1-4%):
    #     +0-300  24.9% vs 28.7%   |  +400-500 16.4% vs 18.4%
    #     +500-680 14.4% vs 15.2%  |  +680+     9.6% vs 10.6%
    # so edge can only come from discriminating WITHIN a band, never from
    # selecting bands. The model's incremental value over the market is real but
    # small (p=0.0044), and it is smallest exactly where compression is worst.
    #
    # Until the underlying rate model discriminates across a realistic range,
    # Kelly refuses to size anything longer than KELLY_MAX_ODDS.
    try:
        _o_raw = float(str(odds).replace("+", "").replace(",", "").strip())
    except (TypeError, ValueError):
        _o_raw = 0.0
    if _o_raw > max_odds:
        return {"stake": 0.0, "stake_pct": 0.0, "kelly_full": 0.0,
                "reason": (f"odds +{_o_raw:.0f} exceed the +{max_odds} reliability ceiling — "
                           f"model compression manufactures edge beyond this price")}

    b = dec - 1.0
    k = (mp * b - (1.0 - mp)) / b            # full Kelly fraction
    implied = 1.0 / dec
    if k <= 0 or (mp - implied) < min_edge:
        return {"stake": 0.0, "stake_pct": 0.0, "kelly_full": float(k),
                "reason": f"edge {(mp-implied)*100:.1f}% below {min_edge*100:.0f}% minimum"}
    pct = min(k * fraction, cap)
    return {"stake": float(bankroll * pct), "stake_pct": float(pct * 100.0),
            "kelly_full": float(k), "reason": "ok"}


# ═══════════════════════════════════════════════════════════════════════════
# 7. CLV LOGGING                                   [FIX #4 — the missing metric]
# ═══════════════════════════════════════════════════════════════════════════
# v2 measured hit-rate against a base rate. That is not edge. Edge is beating
# the closing price. A 3/5 slate at +250 on lines that closed +320 is a LOSING
# process that feels like a winning one. CLV is the only leading indicator of
# long-run profitability, and v2 recorded nothing.

class CLVLogger:
    """Append-only closing-line-value ledger."""

    def __init__(self, path="clv_log.jsonl"):
        self.path = path

    def log_pick(self, date, player, market, taken_odds, model_prob,
                 stake_pct=0.0, note="", dedupe=True):
        """Append a pick. `dedupe` keeps ONE row per (date, player, market).

        The model is re-run many times per slate as odds move. Without dedupe a
        50-pick slate re-run 12 times writes 600 rows, and CLV becomes a measure
        of how often you re-ran rather than how well you priced. First write per
        player wins: that is the price you would actually have taken.
        """
        if dedupe:
            for r in self.load():
                if (r["date"] == str(date) and r["player"] == player
                        and r["market"] == market):
                    return r
        rec = {"ts": _dt.datetime.now().isoformat(timespec="seconds"),
               "date": str(date), "player": player, "market": market,
               "taken_odds": taken_odds, "model_prob": round(float(model_prob), 5),
               "stake_pct": round(float(stake_pct), 4),
               "closing_odds": None, "result": None, "note": note,
               "version": MODEL_VERSION}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def load(self):
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
        return out

    def settle(self, date, player, market, closing_odds=None, result=None):
        rows = self.load()
        for r in rows:
            if r["date"] == str(date) and r["player"] == player and r["market"] == market:
                if closing_odds is not None:
                    r["closing_odds"] = closing_odds
                if result is not None:
                    r["result"] = result
        with open(self.path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def report(self):
        """CLV summary. Positive mean CLV is the ONLY reliable evidence of edge."""
        rows = [r for r in self.load()
                if r.get("closing_odds") is not None and r.get("taken_odds") is not None]
        if not rows:
            return {"n": 0, "message": "no settled picks yet — log closing odds to measure edge"}
        beats, clvs = 0, []
        for r in rows:
            pt, pc = american_to_prob(r["taken_odds"]), american_to_prob(r["closing_odds"])
            if pt is None or pc is None or pt <= 0:
                continue
            clv = (pc / pt) - 1.0          # +ve => took a better price than close
            clvs.append(clv)
            beats += clv > 0
        if not clvs:
            return {"n": 0, "message": "no parseable odds pairs"}
        c = np.array(clvs)
        se = c.std(ddof=1) / math.sqrt(len(c)) if len(c) > 1 else float("nan")
        settled = [r for r in rows if r.get("result") in (0, 1, True, False)]
        roi = None
        if settled:
            pnl = 0.0
            for r in settled:
                dec = american_to_decimal(r["taken_odds"]) or 1.0
                stake = max(r.get("stake_pct", 0.0), 1e-9)
                pnl += stake * ((dec - 1.0) if r["result"] in (1, True) else -1.0)
            roi = pnl / sum(max(r.get("stake_pct", 0.0), 1e-9) for r in settled)
        return {"n": len(c), "mean_clv_pct": float(c.mean() * 100),
                "se_pct": float(se * 100) if np.isfinite(se) else None,
                "t_stat": float(c.mean() / se) if np.isfinite(se) and se > 0 else None,
                "beat_close_pct": float(beats / len(c) * 100),
                "roi_pct": float(roi * 100) if roi is not None else None,
                "verdict": ("EDGE CONFIRMED" if np.isfinite(se) and se > 0
                            and c.mean() / se > 2 and c.mean() > 0
                            else "NOT YET DEMONSTRATED — keep logging")}


# ═══════════════════════════════════════════════════════════════════════════
# 8. CORRELATED SAME-GAME SIMULATION       [FIX #12 — where MC actually belongs]
# ═══════════════════════════════════════════════════════════════════════════
# v2 removed correlation nowhere and assumed independence everywhere, so combo
# picks were priced as p1*p2. Two batters in one game share pitcher, park,
# weather and game script; their HR outcomes are positively correlated, so
# p1*p2 UNDERPRICES a same-game double. This is the one place where Monte
# Carlo genuinely earns its cost, because no closed form exists.

def simulate_correlated_combo(probs, shared_sigma: float = 0.35,
                              same_team: bool = False, n: int = 40_000,
                              seed: int | None = None) -> dict:
    """P(all listed batters homer) under a shared game-level latent factor.

    Model: each batter's log-rate is shifted by one common draw
    z ~ N(0, shared_sigma^2) representing pitcher-quality/park/weather/game
    script for that game. Same-team batters get extra correlation (they share
    the same half-innings and lineup turnover).
    """
    p = np.asarray([x for x in probs if x and np.isfinite(x)], float)
    if len(p) < 2:
        return {"joint": float(np.prod(p)) if len(p) else 0.0,
                "independent": float(np.prod(p)) if len(p) else 0.0, "corr_ratio": 1.0}
    rng = np.random.default_rng(seed)
    sigma = shared_sigma * (1.25 if same_team else 1.0)
    z = rng.normal(0.0, sigma, size=n)
    logit_p = np.log(p / (1.0 - p))
    joint = np.ones(n, dtype=bool)
    for lp in logit_p:
        pj = 1.0 / (1.0 + np.exp(-(lp + z)))
        joint &= rng.random(n) < pj
    j = float(joint.mean())
    ind = float(np.prod(p))
    return {"joint": j, "independent": ind,
            "corr_ratio": (j / ind) if ind > 0 else 1.0,
            "fair_odds": (100.0 * (1.0 - j) / j) if j > 0 else None}


# ═══════════════════════════════════════════════════════════════════════════
# 9. PERMUTATION NULL GATE                  [FIX — the test that changes everything]
# ═══════════════════════════════════════════════════════════════════════════
# The audit's central finding: under the exact search that produced the v2
# archetype/flash libraries, outcomes shuffled WITHIN slate produced best-lifts
# of 3.34x-4.05x. The real data produced 3.95x — the 50th percentile of the
# null. Fisher p-values do not correct for search; only a null does.
# No signal may earn conviction credit in v3 without clearing this gate.

def permutation_null_gate(mask, outcomes, slates, n_perm: int = 500,
                          seed: int = 0) -> dict:
    """Slate-stratified permutation test for ONE candidate signal.

    Shuffling within slate preserves each slate's HR rate and all feature
    distributions, destroying only the batter<->outcome link.
    """
    m = np.asarray(mask, bool)
    y = np.asarray(outcomes, float)
    s = np.asarray(slates)
    if m.sum() == 0:
        return {"n": 0, "verdict": "NO FIRES"}
    rng = np.random.default_rng(seed)
    base = y.mean()
    obs = y[m].mean()
    null = np.empty(n_perm)
    idx_by_slate = [np.where(s == u)[0] for u in np.unique(s)]
    for i in range(n_perm):
        ysh = y.copy()
        for idx in idx_by_slate:
            ysh[idx] = rng.permutation(ysh[idx])
        null[i] = ysh[m].mean()
    p_emp = float((null >= obs).mean())
    return {"n": int(m.sum()), "rate": float(obs), "lift": float(obs / base) if base else None,
            "null_mean_lift": float(null.mean() / base) if base else None,
            "null_p95_lift": float(np.percentile(null, 95) / base) if base else None,
            "p_empirical": p_emp,
            "verdict": "PASSES NULL" if p_emp < 0.05 else "INDISTINGUISHABLE FROM NOISE"}


# ═══════════════════════════════════════════════════════════════════════════
# 10. FEATURE DE-DUPLICATION                [FIX #10 — collapse correlated clusters]
# ═══════════════════════════════════════════════════════════════════════════
# The audit identified three clusters that v2 counted repeatedly:
#   A. pitch matchup: pm / PITCH DOMINANCE / PITCH-RELIANT / PITCH DOM ELITE /
#      CONFIRMED MATCH / PITCHER TARGET MATCH  -- all from one DailyPitch
#      wOBA-by-pitch-type table. Six representations of one quantity.
#   B. contact quality: power / xhr / pf_iso / pf_barrel / pf_blast / l10_brl /
#      Savant blast_per_swing -- all monotone in barrel rate. "7 of 9
#      PropFinder gates" is nearer 2 independent confirmations than 7.
#   C. environment: env / park / wx_run / pull_park -- four multiplicative
#      expressions of one variable, compounding one error four times.
# v3 collapses each cluster to a single representative before scoring.

FEATURE_CLUSTERS = {
    "pitch_matchup": ["pm", "pitch_dominance", "pitch_reliant", "pitch_dom_elite",
                      "confirmed_match", "pitcher_target_match", "pitch_edge"],
    "contact_quality": ["power", "xhr", "pf_iso", "pf_barrel", "pf_blast",
                        "l10_brl", "blast_per_swing", "barrel_pct"],
    "environment": ["env", "park", "wx_run", "pull_park", "air_density"],
    "market": ["odds", "implied", "market_signal", "edge"],   # excluded from prob by R1
}


def collapse_cluster(values: dict, cluster: str, method: str = "max_z") -> float:
    """Reduce a correlated cluster to ONE number instead of multiplying all of them.

    'max_z'  most extreme standardised member (keeps the strongest evidence)
    'mean'   average of available members (most stable)
    'first'  first non-null in priority order (most interpretable)
    """
    members = [values.get(k) for k in FEATURE_CLUSTERS.get(cluster, [])]
    vals = [float(v) for v in members if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    if method == "mean":
        return float(np.mean(vals))
    if method == "first":
        return float(vals[0])
    mu, sd = np.mean(vals), (np.std(vals) or 1.0)
    z = [(v - mu) / sd for v in vals]
    return float(vals[int(np.argmax(np.abs(z)))])


def independent_signal_count(fired_labels) -> int:
    """How many GENUINELY independent signals fired.

    v2 awarded a grade-count stacking bonus (+4 each, +8 at 3+) treating six
    pitch-matchup grades as six confirmations. This counts clusters, not labels,
    so a batter firing PITCH DOM + PITCH-RELIANT + CONFIRMED MATCH scores 1.
    """
    text = " ".join(str(x) for x in (fired_labels or [])).upper()
    clusters_hit = 0
    if any(k in text for k in ("PITCH DOM", "PITCH-RELIANT", "CONFIRMED MATCH",
                               "PITCHER TARGET", "PITCH EDGE", "PITCH CORR")):
        clusters_hit += 1
    if any(k in text for k in ("PROPFINDER", "BARREL", "BLAST", "POWER TRAP", "PWR")):
        clusters_hit += 1
    if any(k in text for k in ("ENV", "PARK", "WIND", "WEATHER")):
        clusters_hit += 1
    if any(k in text for k in ("ICE COLD", "SCREAM HIT", "WHR", "HOT HITTER")):
        clusters_hit += 1
    if any(k in text for k in ("VULN", "SUPER VUL", "L2 BLOWUP", "HR-PRONE")):
        clusters_hit += 1
    return clusters_hit


# ═══════════════════════════════════════════════════════════════════════════
# 11. GRADIENT-BOOSTED PROBABILITY MODEL      [FIX #7 — replace the heuristic]
# ═══════════════════════════════════════════════════════════════════════════
# v2 computed probability as a product of nine hand-tuned multipliers clipped
# at 0.18 -- 1,938 hardcoded thresholds across 246 distinct values, fitted by
# eye on the same data used to evaluate them. Measured result: AUC 0.544,
# beaten by the raw closing line at 0.593.
#
# LightGBM learns the interactions the grade library hand-encodes, will not
# double-count correlated features the way multiplication does, and is fitted
# with slate-grouped CV so it cannot leak across a slate boundary.
#
# PA-LEVEL HOOK: train_hr_model(level="pa") is the intended end state
# (~150-200k rows/season, ~5,000 positives vs ~570 at game level). The game
# level path works today on the existing audit CSV; the PA path activates the
# moment PA-level Statcast rows are supplied. NOTE: the game-level model is an
# interim step, NOT the redesign target -- it inherits the small-sample ceiling.

MODEL_FEATURES_GAME = [
    "score", "power", "vuln", "pm", "park", "env", "hs", "sig",
    "est_pa", "l10_dist", "l10_hh", "l10_brl", "pitch_vuln", "pitch_usage",
    "whr", "pf_gates", "pf_iso", "pf_hh", "pf_barrel", "pf_pull",
]
# NB: 'odds', 'implied', 'edge', 'hr_prob', 'conv' are DELIBERATELY ABSENT.
# Rule R1 -- the market is the benchmark, never a feature. 'conv' is excluded
# because v2 computed it FROM the grades and then gated grades ON it.


def train_hr_model(df, target="HR", level="game", features=None,
                   n_splits: int = 5, seed: int = 42, verbose: bool = True):
    """Train a slate-grouped, out-of-fold-calibrated GBM.

    Returns dict with the fitted booster, the calibrator, OOF predictions and
    an honest OOF performance report. Every number reported is out-of-fold.
    """
    import lightgbm as lgb
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, log_loss

    feats = list(features or MODEL_FEATURES_GAME)
    feats = [f for f in feats if f in df.columns]
    banned = {"odds", "implied", "edge", "hr_prob", "market_signal", "conv"}
    leaked = banned & set(feats)
    if leaked:
        raise ValueError(f"RULE R1 VIOLATION — market/circular features in model: {leaked}")

    d = df.dropna(subset=[target]).copy()
    X = d[feats].astype(float)
    y = d[target].astype(float).values
    groups = d["date"].values

    params = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                  min_data_in_leaf=60, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=seed)

    oof = np.zeros(len(d))
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    boosters = []
    for tr, te in gkf.split(X, y, groups):
        ds = lgb.Dataset(X.iloc[tr], label=y[tr])
        dv = lgb.Dataset(X.iloc[te], label=y[te])
        bst = lgb.train(params, ds, num_boost_round=600, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[te] = bst.predict(X.iloc[te], num_iteration=bst.best_iteration)
        boosters.append(bst)

    cal = IsotonicCalibrator("hr" if target == "HR" else "hit")
    cal.fit(oof, y, groups=groups)
    oof_cal = np.array([cal(p) for p in oof])

    full = lgb.train(params, lgb.Dataset(X, label=y),
                     num_boost_round=int(np.mean([b.best_iteration or 300 for b in boosters])))

    rep = {"n": len(d), "slates": int(pd.Series(groups).nunique()),
           "base_rate": float(y.mean()),
           "auc_oof": float(roc_auc_score(y, oof)),
           "logloss_oof": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
           "calibration": reliability_report(oof_cal, y),
           "feature_importance": dict(sorted(
               zip(feats, full.feature_importance("gain")),
               key=lambda kv: -kv[1]))}
    if verbose:
        print(f"  OOF AUC={rep['auc_oof']:.4f}  n={rep['n']}  slates={rep['slates']}  "
              f"base={rep['base_rate']*100:.2f}%  ECE={rep['calibration']['ece']:.4f}  "
              f"BSS={rep['calibration']['brier_skill_score']:+.4f}")
    return {"booster": full, "calibrator": cal, "features": feats,
            "oof_raw": oof, "oof_cal": oof_cal, "y": y, "report": rep, "level": level}


def score_with_model(bundle, row: dict) -> float:
    """Calibrated probability for one batter-game from a trained bundle."""
    if bundle is None:
        return float("nan")
    x = np.array([[float(row.get(f) if row.get(f) is not None
                         and np.isfinite(float(row.get(f) or np.nan)) else np.nan)
                   for f in bundle["features"]]])
    raw = float(bundle["booster"].predict(x)[0])
    return float(bundle["calibrator"](raw))


# ═══════════════════════════════════════════════════════════════════════════
# 12. PLAYER-ID JOIN REGISTRY               [FIX #9 — key on player_id, log failures]
# ═══════════════════════════════════════════════════════════════════════════
# The workbook carries four incompatible naming conventions:
#   Savant        "Caminero, Junior"
#   DailyBatter   "F. Freeman, 1B (L)"
#   ActionNetwork "Manny Machado"
#   RotoWire      "Nathan Lukes"
# Fuzzy matching fails silently -- it drops a player, or worse attaches the
# wrong player's stats. This is a likely contributor to the long-standing
# "37-40 batters in output instead of ~90" issue. v3 keys on MLB player_id
# where available, falls back to a normalised name, and LOUDLY logs every
# failure instead of silently defaulting.
# (Data SOURCES are unchanged, per instruction — only the join key changes.)

class PlayerIDRegistry:
    def __init__(self):
        self.by_id, self.name_to_id, self.failures = {}, {}, []
        self.stats = defaultdict(int)

    @staticmethod
    def normalize(name) -> str:
        import re, unicodedata
        if not name:
            return ""
        s = unicodedata.normalize("NFKD", str(name))
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = re.sub(r"\s*\([^)]*\)", "", s)                       # "(L)" / "(R)"
        s = re.sub(r",\s*(?:[A-Z0-9]{1,3})(?:\s*/\s*[A-Z0-9]{1,3})*\s*$", "", s)  # ", 1B"
        if "," in s:                                             # "Last, First"
            a, b = s.split(",", 1)
            s = f"{b.strip()} {a.strip()}"
        s = re.sub(r"\b(Jr|Sr|II|III|IV)\.?\b", "", s, flags=re.I)
        s = re.sub(r"[^A-Za-z ]", "", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    def register(self, player_id, name, source=""):
        if player_id in (None, "", 0):
            return None
        pid = str(player_id).strip()
        key = self.normalize(name)
        self.by_id.setdefault(pid, {"id": pid, "names": set(), "sources": set()})
        self.by_id[pid]["names"].add(key)
        self.by_id[pid]["sources"].add(source)
        if key:
            self.name_to_id[key] = pid
        self.stats[f"registered:{source}"] += 1
        return pid

    def resolve(self, name=None, player_id=None, source="", required=True):
        if player_id not in (None, "", 0) and str(player_id).strip() in self.by_id:
            self.stats[f"hit_id:{source}"] += 1
            return str(player_id).strip()
        key = self.normalize(name)
        if key in self.name_to_id:
            self.stats[f"hit_name:{source}"] += 1
            return self.name_to_id[key]
        # last resort: unique surname match, and ONLY if unambiguous
        if key:
            surname = key.split()[-1] if key.split() else ""
            hits = [pid for k, pid in self.name_to_id.items()
                    if k.split() and k.split()[-1] == surname]
            if len(set(hits)) == 1:
                self.stats[f"hit_surname:{source}"] += 1
                return hits[0]
        self.stats[f"MISS:{source}"] += 1
        if required:
            self.failures.append({"name": name, "player_id": player_id,
                                  "source": source, "normalized": key})
        return None

    def report(self, verbose=True):
        total_miss = sum(v for k, v in self.stats.items() if k.startswith("MISS:"))
        total_hit = sum(v for k, v in self.stats.items() if k.startswith("hit_"))
        rate = total_hit / max(total_hit + total_miss, 1) * 100
        if verbose:
            print(f"  🔗 JOIN REGISTRY: {len(self.by_id)} players | "
                  f"{total_hit} resolved, {total_miss} FAILED ({rate:.1f}% match rate)")
            if self.failures:
                print(f"  ⚠️  {len(self.failures)} unresolved — these silently drop from output:")
                for f in self.failures[:15]:
                    print(f"       [{f['source']}] {f['name']!r} -> {f['normalized']!r}")
                if len(self.failures) > 15:
                    print(f"       ... and {len(self.failures)-15} more")
        return {"players": len(self.by_id), "resolved": total_hit,
                "failed": total_miss, "match_rate": rate, "failures": self.failures}


# ═══════════════════════════════════════════════════════════════════════════
# 13. SHADOW MODE                             [FIX #5 — freeze the mined signals]
# ═══════════════════════════════════════════════════════════════════════════
# The archetype and flash libraries did not clear the permutation null. Until a
# genuine forward holdout says otherwise they must not move a pick or a stake.
# Shadow mode keeps them VISIBLE (so rates keep accruing) and INERT (zero
# conviction, zero ranking multiplier).

SHADOW_MODE = False                # OFF (user decision 2026-07-27). All signals live.
SHADOW_MIN_SLATES = 30             # minimum fresh slates before reconsidering
SHADOW_START_DATE = "2026-07-27"


def shadow_conv(points: float, label: str = "") -> float:
    """Gate any mined-signal conviction credit through shadow mode."""
    return 0.0 if SHADOW_MODE else float(points)


def shadow_rank_mult(mult: float) -> float:
    return 1.0 if SHADOW_MODE else float(mult)


def shadow_banner() -> str:
    if not SHADOW_MODE:
        # Accurate statement of what is actually gated. The previous banner
        # claimed flash combos and named grades were inert when the code only
        # ever gated archetype conviction/ranking — a false line printed every run.
        return ("🟢 ALL SIGNALS LIVE — archetypes, flash combos and named grades all earn "
                "full conviction and ranking credit. Evidence note: the archetype library "
                "(HR01-HR16 / HT01-HT16, built 2026-07-26) did NOT clear a slate-stratified "
                "permutation null — shuffled outcomes produced 3.34x-4.05x under the same "
                "search vs the library's 3.95x. Named grades with large n (PTM T3 n=397, "
                "PITCH-RELIANT n=311, ICE COLD n=156) were never tested by that search and "
                "are unaffected. CLV logging will settle this empirically over 30-50 slates.")
    return (f"🔬 SHADOW MODE ACTIVE (since {SHADOW_START_DATE}) — archetypes, flash combos and "
            f"mined grades are LOGGED but earn ZERO conviction and ZERO ranking credit. "
            f"Reason: slate-stratified permutation test showed shuffled outcomes produce "
            f"best-lifts of 3.34x-4.05x under the same search; the real library scored 3.95x "
            f"(50th pct of null). Requires {SHADOW_MIN_SLATES}+ fresh slates of positive CLV "
            f"before any of it earns credit again.")


# ═══════════════════════════════════════════════════════════════════════════
# 14. MARKET-BLEND POSTERIOR                [the configuration that actually bets]
# ═══════════════════════════════════════════════════════════════════════════
# MEASURED (3,280 rows / 72 slates, out-of-fold):
#   v2 prob   AUC 0.5442  ECE 0.0158  incremental over market p=0.83  (nothing)
#   market    AUC 0.5930  ECE 0.0259
#   v3 GBM    AUC 0.5510  ECE 0.0059  incremental over market p=0.0044 (real)
#
# The v3 model alone still does NOT out-rank the market — and pretending
# otherwise is how bankrolls die. What changed is that it now carries
# information the market does NOT, which v2 provably did not.
#
# So the betting quantity is a BLEND: market as prior, model as the update.
# This is NOT the v2 circularity. The distinction is strict and matters:
#   v2: market -> probability -> edge vs market      (circular, uninterpretable)
#   v3: probability (market-free) -> blend with market -> edge vs market
# The model's own number is produced with no knowledge of price; the blend is
# an explicit, auditable posterior formed AFTER the fact, and both components
# are reported separately so the model can always be scored on its own.

BLEND_W_MARKET = 0.62     # fitted by OOF Brier on the audit window (3,280 rows / 72 slates)
BLEND_W_MODEL  = 0.38


def blend_with_market(model_prob: float, market_prob: float | None,
                      w_market: float = BLEND_W_MARKET,
                      w_model: float = BLEND_W_MODEL) -> float:
    """Logit-space posterior. Returns the market-free model prob if no price."""
    if market_prob is None or not np.isfinite(market_prob) or not (0 < market_prob < 1):
        return float(model_prob)
    if not np.isfinite(model_prob) or not (0 < model_prob < 1):
        return float(market_prob)
    lm = math.log(market_prob / (1 - market_prob))
    lo = math.log(model_prob / (1 - model_prob))
    z = (w_market * lm + w_model * lo) / max(w_market + w_model, 1e-9)
    return float(1.0 / (1.0 + math.exp(-z)))


def price_a_pick(model_prob, over_odds, under_odds=None, bankroll=1.0,
                 kelly_fraction=0.25, cap=0.02, min_edge=0.02) -> dict:
    """Full v3 pricing path for one batter. The only sanctioned way to bet.

    model_prob MUST be market-free (rule R1). Returns every intermediate so
    the decision is auditable end to end.
    """
    mkt_raw = american_to_prob(over_odds)
    mkt = devig_two_way(over_odds, under_odds) if under_odds is not None else mkt_raw
    post = blend_with_market(model_prob, mkt)
    e = compute_edge(post, mkt)
    k = kelly_stake(post, over_odds, bankroll, kelly_fraction, cap, min_edge)
    return {"model_prob_market_free": float(model_prob),
            "market_prob_raw": mkt_raw, "market_prob_devig": mkt,
            "posterior_prob": post, "edge_pct": e["edge_pct"],
            "ev_per_unit": e["ev_per_unit"], "fair_odds": e["fair_odds"],
            "stake_pct": k["stake_pct"], "stake_reason": k["reason"],
            "bet": k["stake_pct"] > 0}


# ═══════════════════════════════════════════════════════════════════════════
# 15. CLOSING LINE TRACKER          [closes the last manual loop in the pipeline]
# ═══════════════════════════════════════════════════════════════════════════
# The model already knew when a game had started — _score_sharp parses
# "7:10 PM ET" against current ET with a 5-minute buffer and sets
# _game_has_started, then uses it to ZERO OUT line-movement signal for
# in-progress games. But it only nullified; it never PERSISTED the last
# pre-game price. So the closing line was computed and discarded every run.
#
# This tracker reuses that same signal for the opposite purpose: on every run
# it records each player's current price WHILE the game is still pre-game, and
# FREEZES the record the moment first pitch passes. Whatever was written last
# before the freeze IS the closing line.
#
# Two properties that matter:
#   * FREEZE IS PERMANENT. Once frozen, no later run can overwrite it — books
#     re-post derivative in-play markets and those must never contaminate CLV.
#   * PER-PLAYER, NOT PER-SLATE. A 1:10 PM game freezes hours before a 10:10 PM
#     game. Freezing on a slate-wide clock would record in-play prices for every
#     early game.
#
# Requires the agent to keep running through first pitch (it already does —
# it polls ActionNetwork intraday). No new scraping is introduced.

class ClosingLineTracker:
    """Per-player closing-line ledger. Writes pre-game, freezes at first pitch."""

    def __init__(self, path="closing_lines.json", buffer_seconds: int = 300):
        self.path = path
        self.buffer_seconds = buffer_seconds
        self.data = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)          # atomic — agent may run concurrently

    @staticmethod
    def game_started(game_time_str, now_et=None, buffer_seconds: int = 300) -> bool | None:
        """Mirror of the model's own _game_has_started test. None = unparseable."""
        if not game_time_str:
            return None
        try:
            et = _dt.timezone(_dt.timedelta(hours=-4))
            now = now_et or _dt.datetime.now(et)
            s = str(game_time_str).upper().replace("ET", "").strip()
            t = _dt.datetime.strptime(s, "%I:%M %p").time()
            gt = _dt.datetime.combine(now.date(), t, tzinfo=et)
            return (now - gt).total_seconds() > buffer_seconds
        except Exception:
            return None

    def update(self, date, player, odds, game_time_str, market="HR"):
        """Record a pre-game price. No-op once frozen or once the game started."""
        if odds in (None, "", 0):
            return "no price"
        key = f"{date}|{player}|{market}"
        rec = self.data.get(key)
        if rec and rec.get("frozen"):
            return "already frozen"

        started = self.game_started(game_time_str, buffer_seconds=self.buffer_seconds)
        if started is None:
            return "game time unparseable"

        if started:
            # First run after first pitch: freeze whatever we last saw pre-game.
            if rec:
                rec["frozen"] = True
                rec["frozen_at"] = _dt.datetime.now().isoformat(timespec="seconds")
                self._save()
                return f"FROZEN at {rec['odds']}"
            return "game started with no pre-game capture"

        self.data[key] = {"date": str(date), "player": player, "market": market,
                          "odds": odds, "game_time": game_time_str,
                          "captured_at": _dt.datetime.now().isoformat(timespec="seconds"),
                          "n_updates": (rec or {}).get("n_updates", 0) + 1,
                          "frozen": False}
        self._save()
        return "updated"

    def closing_odds(self, date, player, market="HR", require_frozen=True):
        """Return the closing price ONLY once the line is frozen.

        require_frozen defaults True and should stay that way. A pre-game price
        is NOT a closing price; settling against it yields a meaningless 0% CLV
        that then becomes permanent, because auto_settle only fills empty slots.
        """
        rec = self.data.get(f"{date}|{player}|{market}")
        if not rec:
            return None
        if require_frozen and not rec.get("frozen"):
            return None
        return rec.get("odds")

    def pending(self):
        """Lines still open — these settle on a later run, after first pitch."""
        return [r for r in self.data.values() if not r.get("frozen")]

    def sweep(self, rows, market="HR"):
        """Process a whole slate. `rows` = iterable of (date, player, odds, game_time)."""
        counts = defaultdict(int)
        for date, player, odds, gt in rows:
            r = self.update(date, player, odds, gt, market)
            counts[r.split(" at ")[0]] += 1
        return dict(counts)


def auto_settle_clv(clv_path="clv_log.jsonl", closing_path="closing_lines.json",
                    results=None, verbose=True):
    """Settle the CLV ledger from frozen closing lines (and outcomes if supplied).

    `results` optional: {(date, player): 1|0} for HR outcomes. Without it, CLV
    still settles — CLV does not need the result, which is exactly why it is a
    LEADING indicator and hit-rate is a lagging one.
    """
    log = CLVLogger(clv_path)
    tracker = ClosingLineTracker(closing_path)
    rows = log.load()
    settled = outcomes = 0

    # SELF-HEAL: the pre-fix build settled picks against the LIVE price before
    # first pitch, writing closing_odds identical to taken_odds and a permanent
    # 0.00% CLV. Clear those so they re-settle correctly against the real close.
    # A genuine zero-CLV pick is rare and loses nothing by being re-settled.
    healed = 0
    for r in rows:
        if (r.get("closing_odds") is not None
                and str(r["closing_odds"]) == str(r.get("taken_odds"))):
            r["closing_odds"] = None
            healed += 1
    if healed and verbose:
        print(f"  🩹 CLV repair: cleared {healed} rows settled against the live price "
              f"(pre-fix build) — they will re-settle after first pitch")
    for r in rows:
        if r.get("closing_odds") is None:
            # FROZEN ONLY. An unfrozen price is the live line, not the close.
            co = tracker.closing_odds(r["date"], r["player"], r.get("market", "HR"),
                                      require_frozen=True)
            if co is not None:
                r["closing_odds"] = co
                settled += 1
        if results and r.get("result") is None:
            key = (r["date"], r["player"])
            if key in results:
                r["result"] = int(results[key])
                outcomes += 1
    with open(clv_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rep = log.report()
    if verbose:
        _pend = len(tracker.pending())
        print(f"  📉 CLV auto-settle: {settled} closing lines, {outcomes} outcomes attached"
              + (f", {_pend} still open (settle after first pitch)" if _pend else ""))
        if rep.get("n"):
            t = rep.get("t_stat")
            print(f"     n={rep['n']}  mean CLV {rep['mean_clv_pct']:+.2f}%  "
                  f"beat close {rep['beat_close_pct']:.0f}%"
                  + (f"  t={t:.2f}" if t is not None else "")
                  + (f"  ROI {rep['roi_pct']:+.1f}%" if rep.get("roi_pct") is not None else ""))
            print(f"     → {rep['verdict']}")
    return rep
