import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import * as currentRules from '../src/rules.js';

const HELP = `Benchmark unrestricted complete-turn checkmate detection.

Usage: node scripts/benchmark-checkmate.js [options]
  --baseline PATH  Compare against an earlier rules.js module (optional)
  --repeat N       Measured runs per case and implementation (default: 5)
  --time-ms N      Cooperative time cap per run (default: 5000)
  --max-work N     Maximum traversal ticks per run (default: 250000)
  --case NAME      Run one named case instead of the full suite
  --json           Print results and individual samples as JSON
  --help           Show this message

Cases: standard, temporal-mate, dual-evasion, optional-spatial-evasion,
deferred-mate, stalemate. Each implementation receives one untimed warm-up
per case. Measured order alternates when a baseline is supplied. Timing covers
the first legal action or exhaustive no-action proof and terminal classification.
Input creation and independent witness validation are outside the timed region.
Budget exhaustion is reported as unknown, never mate; incomplete runs exit 2.
`;

function integer(value, min, max, label) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < min || number > max) {
    throw new Error(`${label} must be an integer between ${min} and ${max}.`);
  }
  return number;
}

function settings(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index++) {
    const key = args[index];
    if (key === '--help' || key === '--json') parsed[key.slice(2)] = true;
    else if (['--baseline', '--repeat', '--time-ms', '--max-work', '--case'].includes(key)
      && args[index + 1] !== undefined && !args[index + 1].startsWith('--')) {
      parsed[key.slice(2)] = args[++index];
    } else throw new Error(`Unknown option or missing value: ${key}`);
  }
  if (parsed.help) return { help: true };
  return {
    baseline: parsed.baseline ? resolve(parsed.baseline) : null,
    repeat: integer(parsed.repeat ?? 5, 1, 100, 'Repetitions'),
    timeMs: integer(parsed['time-ms'] ?? 5000, 1, 60000, 'Time cap'),
    maxWork: integer(parsed['max-work'] ?? 250000, 1, 10000000, 'Work cap'),
    case: parsed.case ?? null,
    json: parsed.json === true,
  };
}

async function fixtures() {
  const pgn = name => readFile(new URL(`../examples/tactics/${name}.5dpgn`, import.meta.url), 'utf8');
  const [mate, evasion, stale] = await Promise.all(['checkmate', 'dual-evasion', 'stalemate'].map(pgn));
  const required = [[0, 11, 12], [7, 0, 0], [3, 0, 0]];
  const future = [[0, 8, 0], [11, 0, 0], [4, 0, 12]];
  // The optional board must advance first so the subsequent king arrival
  // branches. Excluding optional spatial moves would fabricate a terminal.
  const optional = { board: [[required], null, [structuredClone(future), structuredClone(future), future]],
    action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3] };
  // Black's Nd6 locks the remaining White kings. A complete proof still has
  // to account for optional moves on the other timelines.
  const beforeMate = currentRules.createPosition({ pgn: `[Board "Custom"]
[k7/pn6/K7/8/8/8/6PB/8:0:1:w]
[k7/pn6/K7/8/8/6P1/7B/8:0:1:b]
[k7/p7/K7/2n5/8/6P1/7B/8:0:2:w]
[k7/p7/8/2n5/8/6P1/7B/8:0:2:b]
[k7/p7/8/8/8/6P1/7B/8:0:3:w]
[k7/pn6/K7/n7/8/6P1/7B/8:-1:2:w]
[k7/pn6/K7/1K6/8/8/6PB/8:+1:1:b]` });
  const deferred = currentRules.validateAction(beforeMate, [currentRules.parseMove(beforeMate, '(1T1)Nd6')]);
  return [
    { name: 'standard', expected: 'legal', position: currentRules.createPosition() },
    { name: 'temporal-mate', expected: 'checkmate', position: currentRules.createPosition({ pgn: mate }) },
    { name: 'dual-evasion', expected: 'legal', position: currentRules.createPosition({ pgn: evasion }) },
    { name: 'optional-spatial-evasion', expected: 'legal', position: optional },
    { name: 'deferred-mate', expected: 'checkmate', position: deferred },
    { name: 'stalemate', expected: 'stalemate', position: currentRules.createPosition({ pgn: stale }) },
  ];
}

const median = values => {
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

function run(fixture, implementation, validators, limits) {
  const position = structuredClone(fixture.position), original = structuredClone(position);
  let ticks = 0, status = 'unknown', stoppedReason = null, candidate = null;
  const budgetExceeded = new Error('Benchmark budget exhausted');
  const start = performance.now();
  const tick = () => {
    ticks++;
    if (ticks > limits.maxWork) stoppedReason = 'work-limit';
    else if (performance.now() - start >= limits.timeMs) stoppedReason = 'time-limit';
    if (stoppedReason) throw budgetExceeded;
  };
  const options = { tick };
  if (implementation.rules.createPositionKeyCache) options.keyPosition = implementation.rules.createPositionKeyCache();
  // No policy filter, depth limit, or component cap may turn a missing
  // selected candidate into an exhaustive no-action proof.
  const iterator = implementation.rules.generateActions(position, options);
  try {
    const first = iterator.next();
    if (first.done) status = implementation.rules.inCheck(position) ? 'checkmate' : 'stalemate';
    else { candidate = first.value; status = 'legal'; }
  } catch (error) {
    if (error !== budgetExceeded) throw error;
  } finally { iterator.return?.(); }
  const elapsedMs = performance.now() - start;
  assert.deepEqual(position, original, `${fixture.name}: ${implementation.name} mutated its input`);
  if (status !== 'unknown') assert.equal(status, fixture.expected, `${fixture.name}: wrong result from ${implementation.name}`);
  if (candidate) {
    for (const validator of validators) {
      const replay = validator.rules.validateAction(position, candidate.moves);
      assert.deepEqual(replay, candidate.position, `${fixture.name}: ${validator.name} rejected ${implementation.name}'s witness`);
    }
    assert.deepEqual(position, original, `${fixture.name}: witness validation mutated its input`);
  }
  return { case: fixture.name, implementation: implementation.name, status, ticks, elapsedMs,
    stoppedReason, actionLength: candidate?.moves.length ?? null };
}

async function main() {
  const options = settings(process.argv.slice(2));
  if (options.help) { console.log(HELP); return; }
  const implementations = [{ name: 'current', rules: currentRules }];
  if (options.baseline) implementations.push({ name: 'baseline', rules: await import(pathToFileURL(options.baseline).href) });
  for (const implementation of implementations) {
    for (const name of ['generateActions', 'inCheck', 'validateAction']) {
      if (typeof implementation.rules[name] !== 'function') throw new Error(`${implementation.name} does not export ${name}.`);
    }
  }
  const cases = (await fixtures()).filter(fixture => !options.case || fixture.name === options.case);
  if (!cases.length) throw new Error(`Unknown case: ${options.case}`);
  const samples = [], summaries = [];
  for (const fixture of cases) {
    for (const implementation of implementations) run(fixture, implementation, implementations, options);
    for (let repeat = 0; repeat < options.repeat; repeat++) {
      const order = repeat % 2 ? [...implementations].reverse() : implementations;
      const round = order.map(implementation => ({ ...run(fixture, implementation, implementations, options), repeat: repeat + 1 }));
      const proven = round.filter(sample => sample.status !== 'unknown');
      if (proven.length === 2) assert.equal(proven[0].status, proven[1].status, `${fixture.name}: baseline disagreement`);
      samples.push(...round);
    }
    for (const implementation of implementations) {
      const measured = samples.filter(sample => sample.case === fixture.name && sample.implementation === implementation.name);
      summaries.push({ case: fixture.name, implementation: implementation.name, expected: fixture.expected,
        completed: measured.filter(sample => sample.status !== 'unknown').length, runs: measured.length,
        medianTicks: median(measured.map(sample => sample.ticks)), medianMs: median(measured.map(sample => sample.elapsedMs)) });
    }
  }
  const comparisons = cases.map(fixture => {
    const current = summaries.find(row => row.case === fixture.name && row.implementation === 'current');
    const baseline = summaries.find(row => row.case === fixture.name && row.implementation === 'baseline');
    const comparable = baseline && [current, baseline].every(row => row.completed === row.runs);
    return { case: fixture.name, expected: fixture.expected,
      currentTicks: current.medianTicks, currentMs: current.medianMs,
      baselineTicks: baseline?.medianTicks ?? null, baselineMs: baseline?.medianMs ?? null,
      speedup: comparable ? baseline.medianMs / current.medianMs : null,
      completed: summaries.filter(row => row.case === fixture.name).every(row => row.completed === row.runs) };
  });
  const report = { node: process.version, baseline: options.baseline, repeat: options.repeat,
    timeMs: options.timeMs, maxWork: options.maxWork, warmups: 1, comparisons, summaries, samples };
  if (options.json) console.log(JSON.stringify(report, null, 2));
  else {
    console.table(comparisons.map(row => ({ case: row.case, result: row.completed ? row.expected : 'unknown',
      currentTicks: row.currentTicks, currentMs: +row.currentMs.toFixed(3),
      ...(options.baseline ? { baselineTicks: row.baselineTicks, baselineMs: +row.baselineMs.toFixed(3),
        speedup: row.speedup === null ? null : `${row.speedup.toFixed(2)}x` } : {}) })));
    console.log(`Median of ${options.repeat} measured runs after one warm-up; ${options.timeMs} ms / ${options.maxWork} ticks per run.`);
    console.log('Every completed result matched its fixture; legal witnesses were replayed and inputs remained unchanged.');
  }
  if (samples.some(sample => sample.status === 'unknown')) process.exitCode = 2;
}

main().catch(error => { console.error(error.stack ?? error.message); process.exitCode = 1; });
