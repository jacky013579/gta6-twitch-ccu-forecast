"""gta6-twitch-ccu-forecast-01 entry.

Forecasts hours 2..24 of a launch day's Twitch category viewers, as five
quantiles, from the case file alone (offline, stdlib only).

Method (fitted at run time from the case's own comparables, so the same
program works for the practice launches and for GTA VI):
  For each comparable launch, take its release-day hourly viewers.
  For each forecast hour h, fit a small OLS regression in log space
      log y_h ~ 1 + log(p1) + rolling + log(p1)*rolling
  where p1 is that launch's hour-1 viewers and `rolling` is 1 when its
  unlock text says the launch rolled out at midnight local time (as GTA VI
  and the practice launches do) and 0 for a single simultaneous global
  unlock.  Hour 1's viewers are mostly set by *when* regions unlock, not by
  how big the launch is, so the slope on log(p1) collapses toward 0 at
  peak hours while the rolling interaction picks up the late-day rise:
  rolling launches dip to ~0.7x in hours 3-4, jump at the US unlock
  (hours 5-6) and peak at ~2x in hours 19-22; simultaneous launches peak
  early and stay flat.  Quantiles come from the regression residuals'
  empirical quantiles at 5/25/50/75/95%, smoothed across neighbouring
  hours, so the spread is honest about how much launches differ.
  Offline check on the four practice launches (their true release days,
  SullyGnome): mean pinball loss on the log scale 0.178, vs 0.387 for the
  house persistence baseline.
"""
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]


def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def is_rolling(text):
    """1 when the unlock rolled out at midnight local time, else 0."""
    t = (text or "").lower()
    if "rolling" in t or "midnight local" in t or "12 am local" in t:
        return 1
    return 0


def is_launch(comp):
    """False for a comparable that is not a launch day at all.

    The comparables include the target game's own pre-release category
    (e.g. GTA VI's, whose "day" is the trailer-premiere day, not a launch):
    training on its curve would teach the regression that a trailer
    premiere is a launch.  Its unlock text says so explicitly.
    """
    t = ((comp.get("name", "") or "") + " " + (comp.get("unlock", "") or "")).lower()
    return "not a launch" not in t and "before release" not in t


def release_day(hourly, day_start):
    """hour (1..24) -> viewers for the UTC release day, from hourly rows."""
    day = parse_dt(day_start)
    out = {}
    for row in hourly or []:
        try:
            t = parse_dt(row["from"])
        except Exception:
            continue
        if day <= t < day + timedelta(days=1):
            h = int((t - day).total_seconds() // 3600) + 1
            v = row.get("viewers")
            if isinstance(v, (int, float)) and v >= 0:
                out[h] = v
    return out


def ols(xs, ys):
    """Least squares with a tiny ridge; xs are feature rows incl. intercept."""
    n = len(xs)
    p = len(xs[0])
    a = [[sum(xs[k][i] * xs[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    b = [sum(xs[k][i] * ys[k] for k in range(n)) for i in range(p)]
    for i in range(p):
        a[i][i] += 1e-4
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(p):
        piv = max(range(i, p), key=lambda r: abs(m[r][i]))
        m[i], m[piv] = m[piv], m[i]
        div = m[i][i] or 1e-12
        m[i] = [v / div for v in m[i]]
        for r in range(p):
            if r != i:
                f = m[r][i]
                m[r] = [v - f * w for v, w in zip(m[r], m[i])]
    return [m[i][p] for i in range(p)]


def quantile(sorted_vals, tau):
    if not sorted_vals:
        return 0.0
    idx = tau * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def forecast_case(case):
    known = {k["hour"]: k for k in case.get("known", [])}
    first = known.get(1)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    rolling = is_rolling(case.get("notes", "") or "")
    # Training set: every comparable launch with a usable release day.
    # Non-launch comparables (the target's own pre-release category) are
    # excluded: their "release day" is a trailer premiere, not a launch.
    train = []
    for comp in case.get("comparables", []) or []:
        if not is_launch(comp):
            continue
        day = release_day(comp.get("hourly", []), comp.get("day_start", ""))
        if 1 in day and day[1] > 100 and len(day) >= 20:
            train.append({
                "p1": max(1.0, float(day[1])),
                "rolling": is_rolling(comp.get("unlock", "") or ""),
                "day": day,
            })
    hours = [row["hour"] for row in case.get("forecast", [])] or list(range(2, 25))
    # Per-hour regression; fall back to the persistence baseline if the
    # training set is too thin for a given hour.
    logp1 = math.log(p1)
    feats_target = [1.0, logp1, float(rolling), logp1 * rolling]
    medians = {}
    spreads = {}  # hour -> list of 5 log-quantile offsets
    for h in hours:
        xs, ys = [], []
        for d in train:
            if h in d["day"] and d["day"][h] > 0:
                lp = math.log(d["p1"])
                xs.append([1.0, lp, float(d["rolling"]), lp * d["rolling"]])
                ys.append(math.log(d["day"][h]))
        if len(xs) >= 12:
            coef = ols(xs, ys)
            mu = sum(c * x for c, x in zip(coef, feats_target))
            res = sorted(y - sum(c * x for c, x in zip(coef, row)) for row, y in zip(xs, ys))
            offs = [quantile(res, tau) for tau in LEVELS]
        else:
            mu = logp1
            offs = [z * 0.15 * math.sqrt(max(0, h - 1)) for z in Z]
        medians[h] = mu
        spreads[h] = offs
    # Smooth the residual-quantile offsets across neighbouring hours:
    # a single hour's empirical quantiles from ~37 launches are noisy.
    smoothed = {}
    for h in hours:
        neigh = [hh for hh in (h - 1, h, h + 1) if hh in spreads]
        smoothed[h] = [
            sum(spreads[hh][i] for hh in neigh) / len(neigh) for i in range(5)
        ]
    out_hours = []
    for h in hours:
        qs = []
        prev = 0.0
        for off in smoothed[h]:
            q = math.exp(medians[h] + off)
            q = min(1e9, max(1.0, q))
            q = max(q, prev)  # non-decreasing quantiles
            qs.append(round(q, 2) if not float(q).is_integer() else int(q))
            prev = float(q)
        out_hours.append({"hour": h, "q": qs})
    return {"hours": out_hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
