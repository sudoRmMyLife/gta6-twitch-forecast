// gta6-twitch-ccu-forecast-01 entry: forecast a game launch day's Twitch
// viewers, hour by hour, from the day's first hour.
//
// Idea: a launch day has a characteristic shape. For every comparable past
// launch in the case, take its launch-day hourly viewers and divide by its own
// first hour, in logs: r_i(h) = ln(v_i(h) / v_i(1)). The target's hour h is
// then forecast as p1 * exp(q_tau({r_i(h)})), where q_tau is a weighted
// quantile over the comparables and the weights reward similarity to the
// target: closeness in first-hour viewers, first-hour live channels, release
// era, and unlock pattern (a rolling midnight-local unlock ramps through the
// UTC day very differently from one simultaneous global unlock). The weighted
// median is shrunk a little toward persistence (ratio 1) and the weighted
// spread is rescaled; both calibrations, and the similarity bandwidths, were
// fitted by leave-one-out cross-validation over the comparables themselves
// (tools/tune.mjs), never on a graded game. Where an hour has no comparable
// data the forecast falls back to the persistence spread.
//
// Node 22, standard library only, no network, deterministic: the same case
// file always produces the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, for the persistence fallback. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** The persistence fallback's spread, in log units per square-root hour. */
const BASE_SIGMA = 0.15;

export interface ForecastParams {
  /** Similarity bandwidth for first-hour viewers, in natural-log units. */
  bwScale: number;
  /** Similarity bandwidth for first-hour channels, in natural-log units. */
  bwChan: number;
  /** Similarity bandwidth for the release-year gap, in years. */
  bwEra: number;
  /** Extra standardised distance when the unlock pattern differs. */
  unlockPen: number;
  /** Shrinkage of the weighted median log-ratio toward 0 (persistence). */
  medShrink: number;
  /** Rescaling of the weighted spread around the median, in log space. */
  spreadScale: number;
  /** Blend of the whole quantile curve toward the persistence fallback. */
  baseBlend: number;
}

export const DEFAULT_PARAMS: ForecastParams = {
  bwScale: 1.6,
  bwChan: 1.4,
  bwEra: 5,
  unlockPen: 0.9,
  medShrink: 0.95,
  spreadScale: 1.1,
  baseBlend: 0.15,
};

interface HourPoint {
  from: string;
  viewers: number;
  channels: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock: string;
  hourly: HourPoint[];
}

export interface CaseFile {
  day_start: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number; channels: number | null }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
  notes: string;
}

type UnlockClass = 'rolling' | 'simultaneous' | 'unknown';

function unlockClass(text: string): UnlockClass {
  const t = (text ?? '').toLowerCase();
  const rolling = t.includes('rolling') || t.includes('midnight local');
  const simultaneous = t.includes('simultaneous') || t.includes('global unlock') || t.includes('single global');
  if (rolling && !simultaneous) return 'rolling';
  if (simultaneous && !rolling) return 'simultaneous';
  return 'unknown';
}

interface CompFeatures {
  /** Launch-day hour (1..24) -> viewers that hour. */
  viewers: Map<number, number>;
  /** Launch-day hour (1..24) -> live channels that hour. */
  channels: Map<number, number>;
  year: number;
  uclass: UnlockClass;
}

const featureCache = new WeakMap<Comparable, CompFeatures>();

function compFeatures(comp: Comparable): CompFeatures {
  const hit = featureCache.get(comp);
  if (hit !== undefined) return hit;
  const start = Date.parse(comp.day_start);
  const viewers = new Map<number, number>();
  const channels = new Map<number, number>();
  for (const pt of comp.hourly ?? []) {
    const at = Date.parse(pt.from);
    if (!Number.isFinite(at) || !Number.isFinite(start)) continue;
    const hour = Math.round((at - start) / 3600000) + 1;
    if (hour < 1 || hour > 24) continue;
    if (typeof pt.viewers === 'number' && pt.viewers > 0 && !viewers.has(hour)) {
      viewers.set(hour, pt.viewers);
    }
    if (typeof pt.channels === 'number' && pt.channels > 0 && !channels.has(hour)) {
      channels.set(hour, pt.channels);
    }
  }
  const feats: CompFeatures = {
    viewers,
    channels,
    year: new Date(start).getUTCFullYear(),
    uclass: unlockClass(comp.unlock ?? ''),
  };
  featureCache.set(comp, feats);
  return feats;
}

/** The weighted quantile of `values` at level tau, weights alongside, linear
 *  interpolation on the empirical distribution. Deterministic. */
function weightedQuantile(values: number[], weights: number[], tau: number): number {
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b] || a - b);
  let total = 0;
  for (const w of weights) total += w;
  const target = tau * total;
  let cum = 0;
  let prevValue = values[order[0]];
  let prevCum = 0;
  for (const idx of order) {
    cum += weights[idx];
    if (cum >= target) {
      if (cum === prevCum) return values[idx];
      const frac = (target - prevCum) / (cum - prevCum);
      return prevValue + frac * (values[idx] - prevValue);
    }
    prevValue = values[idx];
    prevCum = cum;
  }
  return prevValue;
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Forecast one case: 23 rows, hours 2..24, five quantiles each. */
export function forecastCase(c: CaseFile, params: ForecastParams): { hours: Array<{ hour: number; q: number[] }> } {
  const first = (c.known ?? []).find((k) => k.hour === 1);
  const p1 = Math.max(1, first?.viewers ?? 1);
  const c1 = typeof first?.channels === 'number' && first.channels > 0 ? first.channels : null;
  const logP1 = Math.log(p1);
  const logC1 = c1 === null ? null : Math.log(c1);
  const targetYear = new Date(Date.parse(c.day_start)).getUTCFullYear();
  const targetClass = unlockClass(c.notes ?? '');

  // Score every comparable once: its similarity weight and its day-1 anchors.
  const pool: Array<{ feats: CompFeatures; weight: number; logV1: number }> = [];
  for (const comp of c.comparables ?? []) {
    const feats = compFeatures(comp);
    const v1 = feats.viewers.get(1);
    if (v1 === undefined) continue;
    const logV1 = Math.log(v1);
    let d2 = ((logP1 - logV1) / params.bwScale) ** 2;
    const ch1 = feats.channels.get(1);
    if (logC1 !== null && ch1 !== undefined) {
      d2 += ((logC1 - Math.log(ch1)) / params.bwChan) ** 2;
    }
    d2 += ((targetYear - feats.year) / params.bwEra) ** 2;
    if (targetClass !== 'unknown' && feats.uclass !== 'unknown' && targetClass !== feats.uclass) {
      d2 += params.unlockPen ** 2;
    }
    pool.push({ feats, weight: Math.exp(-d2), logV1 });
  }

  const hours = (c.forecast ?? []).map(({ hour }) => {
    const ratios: number[] = [];
    const weights: number[] = [];
    for (const entry of pool) {
      const vh = entry.feats.viewers.get(hour);
      if (vh === undefined) continue;
      ratios.push(Math.log(vh) - entry.logV1);
      weights.push(entry.weight);
    }
    const base = Z.map((z) => z * BASE_SIGMA * Math.sqrt(hour - 1));
    let logs: number[];
    if (ratios.length === 0) {
      logs = base;
    } else {
      const med = weightedQuantile(ratios, weights, 0.5);
      logs = LEVELS.map((tau, j) => {
        const raw = weightedQuantile(ratios, weights, tau);
        const calibrated = params.medShrink * med + params.spreadScale * (raw - med);
        return (1 - params.baseBlend) * calibrated + params.baseBlend * base[j];
      });
    }
    const q: number[] = [];
    let prev = 0;
    for (const lg of logs) {
      let v = round2(p1 * Math.exp(lg));
      if (!(v > 0)) v = prev > 0 ? prev : 0.01;
      if (v > 1e9) v = 1e9;
      if (v < prev) v = prev;
      q.push(v);
      prev = v;
    }
    return { hour, q };
  });
  return { hours };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  mkdirSync(outputDir, { recursive: true });
  const inputs = readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort();
  for (const file of inputs) {
    const id = file.slice(0, -'.in'.length);
    const c = JSON.parse(readFileSync(join(inputDir, file), 'utf8')) as CaseFile;
    if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
      throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
    }
    const out = forecastCase(c, DEFAULT_PARAMS);
    writeFileSync(join(outputDir, `${id}.out`), `${JSON.stringify(out)}\n`);
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
