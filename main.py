"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts Grand Theft Auto VI's Twitch viewers in each hour 2..24 of its
release day (2026-11-19 UTC) from the first hour's print and the launch-day
curves of past big launches shipped in the case file.

Model (fit at run time from the case's own comparables, so it adapts to
whatever the grader ships):
  - For every comparable launch with a complete 24-hour day, take the
    log-ratio log(viewers_h / viewers_1) for each hour h = 2..24.
  - Keep only rolling-midnight launches (unlock text mentions "midnight
    local" or "rolling"): GTA VI itself unlocks rolling midnight local, and
    simultaneous-global or early-access launches have very different shapes.
    The case game's own history (Grand Theft Auto VI's pre-launch category,
    tier target-history) is never a launch day and is excluded by the
    completeness + name checks.
  - Fit one pooled slope b of the log-ratio on log(hour-1 viewers): big
    launches peak in hour 1 and stay flat (Elden Ring, Cyberpunk 2077:
    hour-21 ratio ~0.9), small launches grow through the day (Ghost of
    Yotei, Resident Evil Requiem: hour-21 ratio 3.6-11.5). The slope is
    about -0.35 and is stable across leave-one-out folds. Per hour, the
    intercept mu_h and spread sd_h come from the slope-adjusted residuals.
  - q_tau(h) = p1 * exp(mu_h + b * (log(p1) - mean_log_p1) + r_tau(h)),
    r_tau(h) the tau-quantile of hour h's slope-adjusted residuals
    (empirical, not a normal fit: with ~11 launches per hour the residual
    shape is what it is), rounded to 2 decimals half up. Quantiles are
    non-decreasing by construction, positive, and capped at 1e9.

Leave-one-out pinball loss (2x convention, mean over 23h x 5 quantiles) on
the 12 rolling-midnight comparables: 0.2615, vs 0.357 for the house baseline
persistence starter. Deterministic: no clock, no randomness, fixed
iteration order; two runs write identical bytes.

What the grader does: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only, no network. Reads every <id>.in
in /input (one case at a time) and writes <id>.out to /output:
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23
rows, hours 2..24 each once, each q five finite numbers above 0 and at
most 1e9, non-decreasing, at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def day_series(comp: dict) -> dict:
    """A comparable's launch-day hourly viewers, hour 1..24, or {} if incomplete."""
    try:
        ds = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for pt in comp.get("hourly") or []:
        try:
            t = datetime.fromisoformat(pt["from"].replace("Z", "+00:00"))
            v = pt["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and isinstance(v, (int, float)) and v > 0:
            out[h] = v
    return out if len(out) == 24 else {}


def is_rolling(comp: dict) -> bool:
    """True when the unlock text says midnight-local rolling (like GTA VI)."""
    u = (comp.get("unlock") or "").lower()
    return "midnight local" in u or "rolling" in u


def fit(train: list) -> tuple:
    """Pooled slope b, mean log p1, and per-hour (mu, residuals) of log-ratios.

    train is a list of day series (dict hour -> viewers). The slope is fit
    on every (game, hour) pair pooled; mu_h is the slope-adjusted mean
    log-ratio for hour h, and the residuals around it carry the hour's
    spread, used as empirical quantiles instead of a normal assumption:
    with ~11 launches per hour the residual shape is what it is, and the
    empirical quantiles scored better than normal tails in leave-one-out
    (0.2615 vs 0.2656 mean pinball loss).
    """
    xs, ys = [], []
    for s in train:
        lp1 = math.log(s[1])
        for h in range(2, 25):
            xs.append(lp1)
            ys.append(math.log(s[h] / s[1]))
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0
    mu, res = {}, {}
    for h in range(2, 25):
        r = [math.log(s[h] / s[1]) - b * (math.log(s[1]) - mx) for s in train]
        mu[h] = sum(r) / len(r)
        res[h] = sorted(x - mu[h] for x in r)
    return b, mx, mu, res


def quantile(vals: list, tau: float) -> float:
    """Linear-interpolation quantile of a sorted list (numpy's default)."""
    n = len(vals)
    if n == 1:
        return vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    f = pos - lo
    return vals[lo] * (1 - f) + vals[hi] * f


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        # Defensive fallback: never crash the case. Use the last pre-launch
        # point, else 1.
        hist = case.get("history") or []
        p1 = max(1, hist[-1]["viewers"]) if hist else 1
    else:
        p1 = max(1, first["viewers"])

    game_name = (case.get("game") or {}).get("name") or ""
    series = []
    for comp in case.get("comparables") or []:
        if (comp.get("name") or "") == game_name:
            continue  # never calibrate on the target's own history
        s = day_series(comp)
        if s:
            series.append((comp, s))
    rolling = [s for comp, s in series if is_rolling(comp)]
    # Rolling launches are the right shape family; fall back to all
    # comparables only if too few rolling ones shipped.
    train = rolling if len(rolling) >= 5 else [s for _, s in series]
    if not train:
        # No usable comparables at all: fall back to the house baseline.
        zs = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
        return {"hours": [{"hour": h, "q": [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in zs]} for h in range(2, 25)]}

    b, mx, mu, res = fit(train)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        center = mu[h] + b * (math.log(p1) - mx)
        qs = []
        for tau in LEVELS:
            q = p1 * math.exp(center + quantile(res[h], tau))
            q = min(q, 1e9)
            qs.append(round2(max(q, 0.01)))
        # Keep the row non-decreasing after rounding (rounding can tie, never invert).
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
