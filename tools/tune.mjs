// Leave-one-out tuner for main.ts.
//
// Every comparable launch in a practice case becomes, in turn, a pseudo
// target: its own first hour is the "known" print, the other comparables are
// the pool, and its real hours 2..24 are the truth. The tuner scores the
// model (and the persistence baseline) with the contest's own pinball loss
// on the log scale, then grid-searches the similarity bandwidths first and
// the calibration constants second. Nothing here is fitted on a graded game:
// the practice launches are never in their own case's comparable pool, and
// the final's pool is the same kind of launches.
//
//   node tools/tune.mjs [path-to-kit-input-dir]

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { forecastCase, DEFAULT_PARAMS } from '../main.ts';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

const inputDir = process.argv[2] ?? join(process.cwd(), '..', 'gta6-twitch-ccu-forecast-01-kit', 'input');
const practice = JSON.parse(readFileSync(join(inputDir, 'p00.in'), 'utf8'));

/** Launch-day hour (1..24) -> {viewers, channels} for one comparable. */
function launchDay(comp) {
  const start = Date.parse(comp.day_start);
  const map = new Map();
  for (const pt of comp.hourly ?? []) {
    const hour = Math.round((Date.parse(pt.from) - start) / 3600000) + 1;
    if (hour < 1 || hour > 24 || map.has(hour)) continue;
    map.set(hour, { viewers: pt.viewers, channels: pt.channels });
  }
  return map;
}

function pinball(y, q, tau) {
  const d = Math.log(y) - Math.log(q);
  return 2 * Math.max(tau * d, (tau - 1) * d);
}

/** Mean pinball loss of a forecast ({hours}) against truth (Map hour->viewers). */
function score(out, truth) {
  let sum = 0;
  let n = 0;
  for (const row of out.hours) {
    const y = truth.get(row.hour);
    if (y === undefined || !(y > 0)) continue;
    for (let j = 0; j < LEVELS.length; j++) {
      sum += pinball(y, row.q[j], LEVELS[j]);
      n += 1;
    }
  }
  return n === 0 ? null : sum / n;
}

function baselineCase(c) {
  const p1 = Math.max(1, c.known[0].viewers);
  return {
    hours: c.forecast.map(({ hour }) => ({
      hour,
      q: Z.map((z) => Math.round(p1 * Math.exp(z * 0.15 * Math.sqrt(hour - 1)) * 100) / 100),
    })),
  };
}

// Build the pseudo-targets once.
const pool = practice.comparables;
const targets = [];
for (const comp of pool) {
  const day = launchDay(comp);
  const h1 = day.get(1);
  if (h1 === undefined || !(h1.viewers > 0)) continue;
  let truthCount = 0;
  const truth = new Map();
  for (let h = 2; h <= 24; h++) {
    const pt = day.get(h);
    if (pt !== undefined && pt.viewers > 0) {
      truth.set(h, pt.viewers);
      truthCount += 1;
    }
  }
  if (truthCount < 18) continue;
  targets.push({
    name: comp.name,
    truth,
    synthetic: {
      day_start: comp.day_start,
      levels: LEVELS,
      known: [{ hour: 1, viewers: h1.viewers, channels: h1.channels ?? null }],
      forecast: Array.from({ length: 23 }, (_, i) => ({ hour: i + 2 })),
      comparables: pool.filter((x) => x !== comp),
      notes: `${comp.name}. Unlock: ${comp.unlock ?? ''}`,
    },
  });
}
console.log(`pseudo-targets: ${targets.length} of ${pool.length} comparables`);

function looLoss(params) {
  let sum = 0;
  let n = 0;
  for (const t of targets) {
    const s = score(forecastCase(t.synthetic, params), t.truth);
    if (s !== null) {
      sum += s;
      n += 1;
    }
  }
  return sum / n;
}

let baseSum = 0;
for (const t of targets) baseSum += score(baselineCase(t.synthetic), t.truth);
console.log(`baseline  LOO loss: ${(baseSum / targets.length).toFixed(4)}`);
console.log(`defaults  LOO loss: ${looLoss(DEFAULT_PARAMS).toFixed(4)}  ${JSON.stringify(DEFAULT_PARAMS)}`);

// Stage 1: similarity weights, calibration at the defaults.
const stage1 = [];
for (const bwScale of [0.8, 1.4, 2.4, 4.5]) {
  for (const bwChan of [0.8, 1.6, 3.2]) {
    for (const bwEra of [2.5, 5, 10]) {
      for (const unlockPen of [0, 0.9, 1.8]) {
        const p = { ...DEFAULT_PARAMS, bwScale, bwChan, bwEra, unlockPen };
        stage1.push({ loss: looLoss(p), params: p });
      }
    }
  }
}
stage1.sort((a, b) => a.loss - b.loss);
console.log('\nstage 1 top 5 (weights):');
for (const r of stage1.slice(0, 5)) console.log(`  ${r.loss.toFixed(4)}  ${JSON.stringify(r.params)}`);

// Stage 2: calibration over the best weight sets.
const stage2 = [];
for (const { params: w } of stage1.slice(0, 6)) {
  for (const medShrink of [0.85, 0.92, 1.0]) {
    for (const spreadScale of [0.85, 1.0, 1.15, 1.3]) {
      for (const baseBlend of [0, 0.15, 0.3]) {
        const p = { ...w, medShrink, spreadScale, baseBlend };
        stage2.push({ loss: looLoss(p), params: p });
      }
    }
  }
}
stage2.sort((a, b) => a.loss - b.loss);
console.log('\nstage 2 top 10 (weights + calibration):');
for (const r of stage2.slice(0, 10)) console.log(`  ${r.loss.toFixed(4)}  ${JSON.stringify(r.params)}`);

// Stage 3: refine around the stage-2 winner, including the median smoothing
// and the late-day spread drift.
const w2 = stage2[0].params;
const stage3 = [];
for (const bwScale of [1.1, 1.4, 1.8]) {
  for (const bwChan of [2.4, 3.2, 4.5]) {
    for (const unlockPen of [1.4, 1.8, 2.3]) {
      for (const smoothMed of [0, 0.2, 0.35]) {
        for (const spreadLate of [0.9, 1.0, 1.1]) {
          const p = { ...w2, bwScale, bwChan, unlockPen, smoothMed, spreadLate };
          stage3.push({ loss: looLoss(p), params: p });
        }
      }
    }
  }
}
stage3.sort((a, b) => a.loss - b.loss);
console.log('\nstage 3 top 5 (refined weights, smoothing, late spread):');
for (const r of stage3.slice(0, 5)) console.log(`  ${r.loss.toFixed(4)}  ${JSON.stringify(r.params)}`);

// Stage 4: calibration once more, finely, around the stage-3 winner.
const w3 = stage3[0].params;
const stage4 = [];
for (const medShrink of [1.2, 1.32, 1.45, 1.6]) {
  for (const spreadScale of [0.85]) {
    for (const baseBlend of [0.07]) {
      const p = { ...w3, medShrink, spreadScale, baseBlend };
      stage4.push({ loss: looLoss(p), params: p });
    }
  }
}
stage4.sort((a, b) => a.loss - b.loss);
console.log('\nstage 4 top 5 (final calibration):');
for (const r of stage4.slice(0, 5)) console.log(`  ${r.loss.toFixed(4)}  ${JSON.stringify(r.params)}`);
console.log(`\nBEST: ${stage4[0].loss.toFixed(4)}  ${JSON.stringify(stage4[0].params)}`);
