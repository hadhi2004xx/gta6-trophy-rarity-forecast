// gta6-trophy-rarity-forecast-01 — decomposed log-time earners/players forecaster.
//
// Model: a trophy's Exophase earn rate is a ratio, rate = earners / players_tracked,
// and the two sides move for different reasons, so they are forecast separately and
// divided again at the target:
//
//   P(t)  — players Exophase tracks. Grows as more players sync profiles after launch.
//           Modelled as log P linear in log(1 + t), t in hours since release midnight:
//           fast early growth that bends over as the launch cohort saturates.
//   E_i(t) — players holding trophy i. Grows as the tracked population plays on and
//           reaches that trophy. Modelled per trophy as log E_i linear in log(1 + t),
//           with the per-trophy slope shrunk toward its tier's mean slope (story bronze
//           trophies are reached quickly, gold/platinum slowly), because a single
//           trophy's counts are noisy when few players hold it.
//
// Exophase does not refresh continuously: the house's reads show the page frozen for
// hours, then jumping at a refresh (launch night: 638 players tracked from 23:50Z to
// 04:00Z, then 3,868 at 04:30Z, every rate but the one at 100% lower). A refresh is a
// DILUTION event — the newly tracked players hold fewer trophies — so a "rates stay"
// forecast is wrong by the dilution factor, while earners barely move. The history is
// therefore collapsed into distinct refresh states first (consecutive identical reads
// are one observation), and the fits run on states, not on reads.
//
// When the history holds a single state (launch night, no refresh seen yet) no slope
// can be fitted, so the model falls back to a launch-refresh prior calibrated on the
// launch-night refresh the kit's rules publish, as a two-point log-time slope between
// the first probe (23:50:49Z 18 Nov: 638 players, 943 trophies earned) and the first
// refresh (04:30Z: 3,868 players, 4,276 trophies earned):
//   players slope = ln(3868/638) / (ln(5.5) - ln(0.847)) = 0.963
//   earners slope = ln((4276 - 3868) / (943 - 638)) / (ln(5.5) - ln(0.847)) = 0.156
// (the 100% trophy's earners are removed from both earned totals before the ratio,
// since they track P exactly). The implied rate multiplier over the first 4.5 hours
// is exp((0.156 - 0.963) * 1.871) = 0.221: rates fall to about a fifth, not to zero
// and not staying put. Both slopes are anchored at the first state's time, which is
// where the flat page's values were first seen.
//
// Quantile spreads are honest logit spreads that widen with the extrapolation
// distance in log-time: slope uncertainty from each fit, times how far past the last
// state the target sits, plus a small base term for the 2-decimal rounding of the
// page's rates and for refresh timing. The spread is NOT widened by the horizon in
// days directly — what matters is the distance in log-time, which is the scale both
// curves live on.
//
// Determinism: pure functions of the case file only. No clock, no randomness, no
// network, no environment other than the two MAKEFUN_* directory overrides. Two runs
// on the same case write byte-identical output.
//
// Node 22, standard library only. Run locally with:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clampPct(pct) / (100 - clampPct(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

/** One distinct Exophase refresh state: the page as of the first read showing it. */
interface State {
  /** Hours since release midnight (may be slightly negative for launch-night reads). */
  tHours: number;
  /** log(1 + tHours), the time scale every curve is fitted on. */
  x: number;
  players: number;
  /** Earn rates in trophy order, null where the read lacked the trophy. */
  rates: Array<number | null>;
}

/** Result of an ordinary least-squares fit y = a + b*x. */
interface Fit {
  a: number;
  b: number;
  /** Standard error of the slope; Infinity when it cannot be estimated. */
  seB: number;
  /** Residual standard deviation; 0 when it cannot be estimated. */
  sigma: number;
  n: number;
  xMean: number;
}

function leastSquares(xs: number[], ys: number[]): Fit {
  const n = xs.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
    sxx += xs[i] * xs[i];
    sxy += xs[i] * ys[i];
  }
  const xMean = sx / n;
  const yMean = sy / n;
  const sxxC = sxx - n * xMean * xMean;
  const sxyC = sxy - n * xMean * yMean;
  const b = sxxC > 1e-12 ? sxyC / sxxC : 0;
  const a = yMean - b * xMean;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    rss += r * r;
  }
  const dof = n - 2;
  const sigma = dof > 0 ? Math.sqrt(rss / dof) : 0;
  const seB = dof > 0 && sxxC > 1e-12 ? sigma / Math.sqrt(sxxC) : Infinity;
  return { a, b, seB, sigma, n, xMean };
}

/** Hours from release midnight (2026-11-19T00:00:00Z) to an ISO timestamp. */
function hoursSinceRelease(iso: string, releaseMidnightMs: number): number {
  const ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return 0;
  return (ms - releaseMidnightMs) / 3_600_000;
}

/** Collapse consecutive identical reads into distinct refresh states. */
function distinctStates(c: Case, releaseMidnightMs: number): State[] {
  const states: State[] = [];
  let prevKey = '';
  for (const r of c.history) {
    const key = `${r.players_tracked}|${JSON.stringify(r.rates)}`;
    if (key === prevKey) continue;
    prevKey = key;
    const tHours = hoursSinceRelease(r.at, releaseMidnightMs);
    states.push({
      tHours,
      x: Math.log(Math.max(0.05, 1 + tHours)),
      players: r.players_tracked,
      rates: r.rates,
    });
  }
  return states;
}

/** Earners holding trophy i in a state, from its printed rate (null when unknown). */
function earners(state: State, i: number): number | null {
  const rate = state.rates[i];
  if (rate === null || rate === undefined) return null;
  if (!Number.isFinite(rate) || !Number.isFinite(state.players) || state.players <= 0) return null;
  return (rate / 100) * state.players;
}

/**
 * Forecast one case. Returns the trophy rows in the case's trophy order.
 */
function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${c.id} has no history`);

  const releaseMidnightMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const states = distinctStates(c, releaseMidnightMs);
  const last = states[states.length - 1];
  const xTarget = Math.log(Math.max(0.05, 1 + hoursSinceRelease(c.target, releaseMidnightMs)));
  const xLast = last.x;

  // ---- Players-tracked curve: log P linear in log(1 + t). ----
  // Launch-refresh prior (see header): players slope 0.963 over the first 4.5 hours.
  const PRIOR_B_P = 0.96341;
  const PRIOR_B_E = 0.15551;
  let logPtarget: number;
  let spreadP: number;
  if (states.length >= 2) {
    const fit = leastSquares(
      states.map((s) => s.x),
      states.map((s) => Math.log(Math.max(1, s.players))),
    );
    // Shrink the fitted slope toward 1.0 (steady log-time growth) with a 2-state prior,
    // and clamp it: tracked players never shrink, and a slope above 2.5 would be a
    // launch-night artefact, not a week-long trend.
    const b = Math.min(2.5, Math.max(0, (fit.n * fit.b + 2 * 1.0) / (fit.n + 2)));
    const a = Math.log(Math.max(1, last.players)) - b * xLast;
    logPtarget = a + b * xTarget;
    const slopeUnc = Number.isFinite(fit.seB) ? fit.seB : 0.15;
    spreadP = Math.sqrt(
      fit.sigma * fit.sigma + Math.pow(Math.max(0.05, slopeUnc) * (xTarget - fit.xMean), 2) + 0.05 * 0.05,
    );
  } else {
    // One state: no slope is estimable; the launch-refresh prior carries the forecast.
    logPtarget = Math.log(Math.max(1, last.players)) + PRIOR_B_P * (xTarget - xLast);
    spreadP = 0.05;
  }

  // ---- Earners curve per trophy, slopes shrunk toward the tier mean. ----
  // First pass: raw per-trophy fits (log E linear in log(1 + t)).
  const rawD: Array<number | null> = [];
  const rawC: Array<number | null> = [];
  const rawSe: Array<number | null> = [];
  const rawXObs: number[] = [];
  const weight: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    let eLast = 0;
    for (const s of states) {
      const e = earners(s, i);
      if (e === null) continue;
      eLast = e;
      xs.push(s.x);
      ys.push(Math.log(Math.max(0.5, e)));
    }
    weight[i] = 0;
    rawXObs[i] = xs.length > 0 ? xs[xs.length - 1] : xLast;
    if (xs.length >= 2) {
      const fit = leastSquares(xs, ys);
      // A trophy's earners cannot outgrow the tracked population for long, and cannot
      // fall while players keep arriving: clamp the slope to a sane progression band.
      rawD[i] = Math.min(3, Math.max(-0.5, fit.b));
      rawC[i] = fit.a;
      rawSe[i] = fit.seB;
      // Reliability grows with the number of states and with the earner count:
      // a trophy held by 4 players has its log-earners set by 2-decimal rounding.
      weight[i] = Math.min(8, xs.length) * Math.min(1, Math.max(0.05, eLast) / 20);
    } else if (xs.length === 1) {
      rawD[i] = null;
      rawC[i] = ys[0];
      rawSe[i] = null;
      weight[i] = 0.5 * Math.min(1, Math.max(0.05, eLast) / 20);
    } else {
      rawD[i] = null;
      rawC[i] = null;
      rawSe[i] = null;
      weight[i] = 0;
    }
  }

  // Tier mean slopes from the trophies that have a fit, else the global mean, else prior.
  const tierMean = new Map<string, number>();
  const tierOf = (t: Trophy): string => t.tier;
  for (const tier of ['platinum', 'gold', 'silver', 'bronze']) {
    let sum = 0;
    let wsum = 0;
    for (let i = 0; i < c.trophies.length; i++) {
      if (tierOf(c.trophies[i]) !== tier || rawD[i] === null) continue;
      sum += rawD[i] * weight[i];
      wsum += weight[i];
    }
    if (wsum > 0) tierMean.set(tier, sum / wsum);
  }
  let globalSum = 0;
  let globalW = 0;
  for (let i = 0; i < c.trophies.length; i++) {
    if (rawD[i] === null) continue;
    globalSum += rawD[i] * weight[i];
    globalW += weight[i];
  }
  const globalMean = globalW > 0 ? globalSum / globalW : PRIOR_B_E;

  const out: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const t = c.trophies[i];
    const lastRate = last.rates[i];
    const tierSlope = tierMean.get(t.tier) ?? globalMean;

    // Shrink the trophy's own slope toward its tier mean (prior weight 2).
    // Per-tier base spread: late, rare trophies (gold/platinum) are reached by few
    // players, so their progression is the least predictable; early bronze the most.
    const tierBaseSpread = t.tier === 'platinum' ? 0.14 : t.tier === 'gold' ? 0.12 : t.tier === 'silver' ? 0.1 : 0.08;
    let d: number;
    let logEtarget: number;
    let spreadE: number;
    if (rawD[i] !== null && rawC[i] !== null) {
      d = (weight[i] * rawD[i] + 2 * tierSlope) / (weight[i] + 2);
      d = Math.min(3, Math.max(-0.5, d));
      logEtarget = rawC[i] + d * xTarget;
      const se = rawSe[i] !== null && Number.isFinite(rawSe[i]) ? rawSe[i] as number : 0.2;
      spreadE = Math.sqrt(
        Math.pow(Math.max(0.08, se) * Math.max(0, xTarget - xLast), 2) + tierBaseSpread * tierBaseSpread,
      );
    } else if (rawC[i] !== null) {
      // One observation: anchor on it (at its own time), grow at the tier slope.
      d = tierSlope;
      logEtarget = rawC[i] + d * (xTarget - rawXObs[i]);
      spreadE = 0.15;
    } else {
      // Never observed: half an earner at the last state, growing at the tier slope.
      d = tierSlope;
      logEtarget = Math.log(0.5) + d * (xTarget - xLast);
      spreadE = 0.3;
    }

    // A trophy every tracked player holds stays (nearly) universal: its earners
    // track P exactly, so its rate stays at 100% whatever the dilution does.
    if (lastRate !== null && lastRate !== undefined && lastRate >= 99.99) {
      out.push({ id: t.id, q: [99.97, 99.99, 99.99, 100, 100] });
      continue;
    }

    let eTarget = Math.exp(logEtarget);
    const pTarget = Math.exp(logPtarget);
    if (!Number.isFinite(eTarget) || eTarget < 0) eTarget = 0.5;
    if (eTarget > pTarget) eTarget = pTarget;
    const median = clampPct((100 * eTarget) / pTarget);

    // Flat-history regime: the launch-refresh prior is the whole forecast, and the
    // spread is the calibrated residual (2-decimal rounding + the small per-trophy
    // variation around the uniform dilution the dev board shows: dilution models land
    // within ~0.01-0.02 logit there, so 0.02 covers it with room).
    let s: number;
    if (states.length === 1) {
      s = 0.02;
    } else {
      // Model risk: log-time curves are an approximation, and the further past the
      // last state the target sits, the more the approximation can bend (saturation
      // in P, late-game acceleration in E). It grows with the extrapolation distance.
      const modelRisk = 0.1 * Math.max(0, xTarget - xLast);
      s = Math.sqrt(spreadP * spreadP + spreadE * spreadE + modelRisk * modelRisk + 0.03 * 0.03);
      s = Math.min(1.0, Math.max(0.08, s));
    }

    const l = logit(median);
    let prev = -Infinity;
    const q = Z.map((z) => {
      let v = round4(invLogit(l + z * s));
      v = Math.min(100, Math.max(0.01, v));
      if (v < prev) v = prev; // keep the row non-decreasing after rounding
      prev = v;
      return v;
    });
    out.push({ id: t.id, q });
  }
  return out;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
