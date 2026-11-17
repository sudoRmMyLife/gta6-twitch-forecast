# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sudoRmMyLife/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sudoRmMyLife/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A quantile forecaster for a game launch day on Twitch. Given a launch's first-hour viewer count (and its live channel count), it forecasts the concurrent viewers of the category for each of hours 2 to 24 of the UTC release day, five quantiles per hour (5%, 25%, 50%, 75%, 95%). The model reads the case's comparable past launches, reduces each one to its launch-day curve of log-ratios to its own first hour, and takes similarity-weighted quantiles of those curves hour by hour: comparables are weighted by how close they sit to the target in first-hour scale, first-hour channels, release era and unlock pattern (rolling midnight local vs one simultaneous global unlock). The weighted median is shrunk slightly toward persistence and the spread is rescaled, with every constant fitted by leave-one-out cross-validation over the comparable launches themselves (see `tools/tune.mjs`), never on the games being graded. The program is deterministic, uses only the Node standard library, and runs offline in well under a second per case.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=2tnm35pq&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
