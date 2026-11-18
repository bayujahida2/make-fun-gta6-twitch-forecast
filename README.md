# GTA VI Twitch Forecast - Launch Day

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bayujahida2/make-fun-gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bayujahida2/make-fun-gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

make.fun bounty [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): forecast how many people will be watching Grand Theft Auto VI live on Twitch in each hour 2-24 of its release day (2026-11-19 UTC), from the first hour's print. $250 over 20 places, free entry. New to make.fun? Join with my referral link: https://make.fun/r/0q7mm1bq

## How it works

`main.py` (Python 3.13, standard library only) reads the case file and fits
a model at run time from the case's own comparables: for each past big
launch with a complete 24-hour day, the log-ratio of each hour's viewers to
hour 1's. Only rolling-midnight launches are used (GTA VI unlocks rolling
midnight local), and one pooled slope of the log-ratio on log(hour-1
viewers) is fitted: big launches peak in hour 1 and stay flat, small
launches grow through the day. Per-hour median and spread come from the
slope-adjusted residuals, giving five quantiles (0.05, 0.25, 0.5, 0.75,
0.95) per hour.

Leave-one-out pinball loss on the 12 rolling-midnight comparables: 0.266,
vs 0.357 for the house baseline persistence starter.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out`
(`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the
output directory. Deterministic: two runs write identical bytes.
