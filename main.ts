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
// page's rates and for refresh timing. A refresh is a level shift in the earners
// and players stocks, so each future refresh adds a step to a random walk: the
// fitted regime carries a walk term sized by the detrended per-refresh increment
// scale the history itself shows (MAD of the log-E / log-P increments between
// states), times the square root of the expected number of future refreshes
// (refresh cadence from the history, in clock time). The spread is NOT widened
// by the horizon in days directly — what matters is the distance in log-time,
// which is the scale both curves live on.
//
// v2 recalibration (20 Nov 2026): v1's dev grade (pinball 0.175, coverage 0.191
// at nominal 0.9, typical miss 0.04 pp) showed the spreads were far too narrow —
// the medians were close but 81% of answers fell outside the q05-q95 band. A
// Monte Carlo over refresh moves consistent with those aggregates (a
// two-component mixture fitted to all three: ~20% of trophies land nearly exactly
// on the median, ~80% deviate with SD 0.30 logit; it reproduces coverage 0.183,
// miss 0.040 pp, loss 0.186 at v1's spread) puts the pinball-optimal single
// spread at one refresh near 0.25 logit, with the loss curve flat between 0.19
// and 0.30. The flat-regime spread below is sized to land at ~0.25 at the
// practice distance, and the fitted regime's walk term is validated on a noisy
// synthetic final (v1 fitted: coverage 0.31, loss 0.54). The medians are
// unchanged from v1 — this is a spread fix, not a median chase.
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
/**
 * Per-refresh per-trophy logit scale of a trophy's deviation from uniform
 * dilution, calibrated from the v1 practice aggregates (see header): one unseen
 * refresh moves a trophy's logit rate by ~0.22. The flat-regime spread is built
 * on this term, so the 90% band covers the refresh-move scale the practice showed.
 */
const REFRESH_SD = 0.22;
/** Log-time span of one launch-night refresh (first probe 23:50Z to first refresh 04:30Z). */
const REFRESH_LOG_SPAN = 1.87;
/** Launch-prior rate-slope uncertainty per unit of log-time (bounded by the practice aggregates). */
const PRIOR_SLOPE_SD = 0.06;
/** Cap on the effective number of future refreshes in the fitted walk term. */
const MAX_FUTURE_REFRESHES = 40;
/**
 * Log-scale noise the page's 2-decimal rounding of rates injects into log-earner
 * increments (median rate ~0.8%: 0.005/0.8 ≈ 0.006 per read, ~0.009 per
 * increment). Subtracted in quadrature from the measured walk scale, so pure
 * rounding jitter is not counted as refresh signal.
 */
const ROUNDING_LOG_SD = 0.01;

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

/** Median of a list of numbers. */
function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Robust scale of a list: the median absolute deviation, rescaled to a standard
 * deviation (1.4826 x MAD). Robust, so one launch-night-sized jump does not set
 * the per-refresh scale for the whole history.
 */
function madSd(values: number[]): number {
  if (values.length === 0) return 0;
  const med = median(values);
  return 1.4826 * median(values.map((v) => Math.abs(v - med)));
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

  // Refreshes still to come, in clock time: a refresh process runs on Exophase's
  // clock, not on log-time, so the count is the cadence the history shows times
  // the clock hours from the last state to the target, capped. Each future
  // refresh adds one step to the earners/players random walk (see the spread
  // terms below), so this count sizes the walk, not the trend.
  const tFirstHours = states[0].tHours;
  const tLastHours = last.tHours;
  const tTargetHours = hoursSinceRelease(c.target, releaseMidnightMs);
  const refreshCadencePerHour =
    states.length >= 2 && tLastHours > tFirstHours ? (states.length - 1) / (tLastHours - tFirstHours) : 0;
  const nFutureRefreshes = Math.min(
    MAX_FUTURE_REFRESHES,
    Math.max(0, refreshCadencePerHour * (tTargetHours - tLastHours)),
  );
  const futureWalk = Math.sqrt(nFutureRefreshes);

  // ---- Players-tracked curve: log P linear in log(1 + t). ----
  // Launch-refresh prior (see header): players slope 0.963 over the first 4.5 hours.
  const PRIOR_B_P = 0.96341;
  const PRIOR_B_E = 0.15551;
  let logPtarget: number;
  let spreadP: number;
  /** Per-refresh scale of the players-tracked walk (0 in the flat regime). */
  let rhoP = 0;
  if (states.length >= 2) {
    const logPStates = states.map((s) => Math.log(Math.max(1, s.players)));
    const fit = leastSquares(
      states.map((s) => s.x),
      logPStates,
    );
    // Shrink the fitted slope toward 1.0 (steady log-time growth) with a 2-state prior,
    // and clamp it: tracked players never shrink, and a slope above 2.5 would be a
    // launch-night artefact, not a week-long trend.
    const b = Math.min(2.5, Math.max(0, (fit.n * fit.b + 2 * 1.0) / (fit.n + 2)));
    const a = Math.log(Math.max(1, last.players)) - b * xLast;
    logPtarget = a + b * xTarget;
    // How much one refresh moves log P off its trend: the robust scale of the
    // detrended state-to-state increments. This is the step size of the
    // players-side walk between the last state and the target.
    rhoP = madSd(
      logPStates.slice(1).map((v, j) => v - logPStates[j] - fit.b * (states[j + 1].x - states[j].x)),
    );
    const slopeUnc = Number.isFinite(fit.seB) ? fit.seB : 0.15;
    spreadP = Math.sqrt(
      fit.sigma * fit.sigma +
        Math.pow(rhoP * futureWalk, 2) +
        Math.pow(Math.max(0.05, slopeUnc) * (xTarget - fit.xMean), 2) +
        0.05 * 0.05,
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
  /** Residual SD of the log-earners fit: the per-state noise the trend leaves unexplained. */
  const rawSigma: Array<number | null> = [];
  /** Per-refresh walk scale of log earners (MAD of the detrended state increments). */
  const rawRho: Array<number | null> = [];
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
      rawSigma[i] = fit.sigma;
      // Walk scale from the detrended state-to-state increments (needs >= 4 states
      // for a usable MAD); the increments are the refresh steps, detrended by the
      // trophy's own fitted slope.
      rawRho[i] = null;
      if (xs.length >= 4) {
        const dIncr: number[] = [];
        for (let j = 1; j < xs.length; j++) dIncr.push(ys[j] - ys[j - 1] - fit.b * (xs[j] - xs[j - 1]));
        rawRho[i] = madSd(dIncr);
      }
      // Reliability grows with the number of states and with the earner count:
      // a trophy held by 4 players has its log-earners set by 2-decimal rounding.
      weight[i] = Math.min(8, xs.length) * Math.min(1, Math.max(0.05, eLast) / 20);
    } else if (xs.length === 1) {
      rawD[i] = null;
      rawC[i] = ys[0];
      rawSe[i] = null;
      rawSigma[i] = null;
      rawRho[i] = null;
      weight[i] = 0.5 * Math.min(1, Math.max(0.05, eLast) / 20);
    } else {
      rawD[i] = null;
      rawC[i] = null;
      rawSe[i] = null;
      rawSigma[i] = null;
      rawRho[i] = null;
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

  // Fallback walk scale for trophies without their own increment history: the
  // median per-trophy walk scale over the trophies that have one, else 0.15.
  const rhoList: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) if (rawRho[i] !== null) rhoList.push(rawRho[i] as number);
  const medianRhoE = rhoList.length > 0 ? median(rhoList) : 0.15;
  /** The fallback walk scale with the rounding floor removed (see ROUNDING_LOG_SD). */
  const medianRhoEAdj = Math.sqrt(
    Math.max(0, medianRhoE * medianRhoE - ROUNDING_LOG_SD * ROUNDING_LOG_SD),
  );

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
      const sigmaE = rawSigma[i] !== null ? (rawSigma[i] as number) : 0;
      const rhoERaw = rawRho[i] !== null ? (rawRho[i] as number) : medianRhoE;
      // Rounding jitter is not refresh signal: remove its floor in quadrature.
      const rhoE = Math.sqrt(Math.max(0, rhoERaw * rhoERaw - ROUNDING_LOG_SD * ROUNDING_LOG_SD));
      // Slope uncertainty x distance, plus the fit's residual noise, plus the
      // future-refresh walk (per-refresh scale x sqrt of refreshes to come),
      // plus the per-tier base. The walk term is what v1 missed: without it a
      // multi-refresh horizon is scored as if the next refreshes were noiseless.
      spreadE = Math.sqrt(
        Math.pow(Math.max(0.08, se) * Math.max(0, xTarget - xLast), 2) +
          sigmaE * sigmaE +
          Math.pow(rhoE * futureWalk, 2) +
          tierBaseSpread * tierBaseSpread,
      );
    } else if (rawC[i] !== null) {
      // One observation: anchor on it (at its own time), grow at the tier slope.
      d = tierSlope;
      logEtarget = rawC[i] + d * (xTarget - rawXObs[i]);
      spreadE = Math.sqrt(0.15 * 0.15 + Math.pow(medianRhoEAdj * futureWalk, 2));
    } else {
      // Never observed: half an earner at the last state, growing at the tier slope.
      d = tierSlope;
      logEtarget = Math.log(0.5) + d * (xTarget - xLast);
      spreadE = Math.sqrt(0.3 * 0.3 + Math.pow(medianRhoEAdj * futureWalk, 2));
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
    // spread is the refresh risk the practice showed. v1's 0.02 came from the
    // dev-board claim that dilution models land within ~0.01-0.02 logit; the dev
    // grade proved it wrong (coverage 0.191). The recalibrated spread has three
    // honest parts: one unseen refresh moves a trophy's logit rate by REFRESH_SD
    // (0.22, from the practice aggregates), accumulating over the refreshes the
    // horizon holds (at least one, one per REFRESH_LOG_SPAN of log-time); the
    // launch-prior slope is uncertain by PRIOR_SLOPE_SD per unit of log-time, so
    // its contribution grows with the extrapolation distance; and a 0.03 base for
    // rounding and refresh timing. At the practice distance (d = 1.87) this lands
    // at s = 0.25, the pinball-optimal spread the Monte Carlo found.
    let s: number;
    if (states.length === 1) {
      const dFlat = Math.max(0, xTarget - xLast);
      const nRef = Math.max(1, dFlat / REFRESH_LOG_SPAN);
      s = Math.sqrt(
        REFRESH_SD * REFRESH_SD * nRef +
          PRIOR_SLOPE_SD * PRIOR_SLOPE_SD * dFlat * dFlat +
          0.03 * 0.03,
      );
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
