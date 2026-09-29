import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createPosition, formatAction, validateAction } from '../src/rules.js';

const HELP = `Benchmark classical search at fixed depth and fixed time.

Usage: npm run benchmark:classical -- [options]
  --baseline PATH   Compare with another search.js and its adjacent source files
  --time-ms N       Fixed-time budget per run (default: BENCH_TIME_MS or 1000)
  --depth-time-ms N Fixed-depth safety time limit (default: 3000)
  --repeat N        Repetitions, with alternating engine order (default: 2)
  --warmup N        Unmeasured warm-up rounds per engine and case (default: 1)
  --depth N         Override the fixed-depth targets
  --mode NAME       both (default), depth, or time
  --case NAME       standard, opening, two-timelines, or temporal (default: all)
  --json            Print machine-readable results
  --help            Show this message

Example:
  npm run benchmark:classical -- --baseline artifacts/classical-search-before/src/search.js

Default depth / quiescence targets: standard 4/2, opening 2/2,
two-timelines 2/1, temporal 2/1. Fixed-time runs search up to depth 64.
Warm-ups use the same targets with a 250 ms limit. Fixed-depth speedups are
reported only for pairs that finish the requested depth and tactical horizon.
Timing excludes legality checks and formatting. All best turns and PVs are
validated; work counters and input immutability are checked on every run.
With node --expose-gc, garbage from earlier runs is collected before timing.
`;

function integer(value, minimum, maximum, label) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < minimum || number > maximum) {
    throw new Error(`${label} must be an integer between ${minimum} and ${maximum}.`);
  }
  return number;
}

function settings(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index++) {
    const key = args[index];
    if (['--help', '--json'].includes(key)) parsed[key.slice(2)] = true;
    else if (['--baseline', '--time-ms', '--depth-time-ms', '--repeat', '--warmup', '--depth', '--mode', '--case'].includes(key)
      && args[index + 1] !== undefined && !args[index + 1].startsWith('--')) parsed[key.slice(2)] = args[++index];
    else throw new Error(`Unknown option or missing value: ${key}`);
  }
  if (parsed.help) return { help: true };
  if (parsed.mode && !['both', 'depth', 'time'].includes(parsed.mode)) throw new Error('--mode must be both, depth, or time.');
  if (parsed.case && !['standard', 'opening', 'two-timelines', 'temporal'].includes(parsed.case)) throw new Error('Unknown benchmark case.');
  return {
    baseline: parsed.baseline ? resolve(parsed.baseline) : null,
    timeMs: integer(parsed['time-ms'] ?? process.env.BENCH_TIME_MS ?? 1000, 1, 3600000, 'Time budget'),
    depthTimeMs: integer(parsed['depth-time-ms'] ?? 3000, 1, 3600000, 'Fixed-depth time limit'),
    repeat: integer(parsed.repeat ?? process.env.BENCH_REPEAT ?? 2, 1, 100, 'Repetitions'),
    warmup: integer(parsed.warmup ?? 1, 0, 100, 'Warm-up rounds'),
    depth: parsed.depth === undefined ? null : integer(parsed.depth, 1, 64, 'Depth'),
    modes: !parsed.mode || parsed.mode === 'both' ? ['depth', 'time'] : [parsed.mode],
    case: parsed.case ?? null,
    json: parsed.json === true,
  };
}

async function loadEngine(name, path) {
  const url = pathToFileURL(path);
  const { analyze } = await import(url.href);
  if (typeof analyze !== 'function') throw new Error(`${path} does not export analyze().`);
  // Include the dependencies that determine classical search behavior so a
  // saved comparison identifies more than just the entry-point revision.
  const sources = {};
  for (const file of ['search.js', 'rules.js', 'evaluate.js', 'heuristics.js', 'royal-safety.js', 'search-cache.js']) {
    sources[file] = createHash('sha256').update(await readFile(file === 'search.js' ? url : new URL(file, url))).digest('hex');
  }
  return { name, path, sources, analyze };
}

function validate(position, original, result) {
  assert.deepEqual(position, original, 'Search modified its input position.');
  if (result.bestAction) validateAction(position, result.bestAction);
  if (result.pv.length) assert.deepEqual(result.pv[0], result.bestAction, 'PV must begin with the best action.');
  let current = position;
  for (const action of result.pv) current = validateAction(current, action);
  assert.deepEqual(position, original, 'PV validation modified the input position.');
  for (const counter of ['nodes', 'searchNodes', 'generationNodes', 'qnodes', 'ttHits', 'cutoffs']) {
    assert(Number.isSafeInteger(result[counter]) && result[counter] >= 0, `Invalid ${counter} counter.`);
  }
  assert.equal(result.nodes, result.searchNodes + result.generationNodes, 'Work counters do not sum to total nodes.');
  assert(result.qnodes <= result.searchNodes, 'Quiescence nodes exceed all search nodes.');
  assert(result.nodes <= result.limits.maxNodes, 'Search exceeded the work limit.');
  assert(result.score === null || Number.isFinite(result.score), 'Search returned an invalid score.');
}

function run(engine, fixture, limits) {
  globalThis.gc?.();
  const position = createPosition(fixture.setup), original = structuredClone(position);
  const start = performance.now();
  const result = engine.analyze(position, limits);
  const wallMs = performance.now() - start;
  validate(position, original, result);
  return {
    engine: engine.name, case: fixture.name, targetDepth: limits.maxDepth,
    requestedQuiescenceDepth: limits.quiescenceDepth,
    depth: result.depth, quiescenceDepth: result.effectiveQuiescenceDepth,
    selectiveDepth: result.selectiveDepth, completed: result.completed,
    targetReached: result.completed && result.stoppedReason === 'depth' && result.depth === limits.maxDepth
      && result.effectiveQuiescenceDepth === limits.quiescenceDepth,
    nodes: result.nodes, searchNodes: result.searchNodes, generationNodes: result.generationNodes,
    tableEntries: result.tableEntries, cacheMemoryBytes: result.cacheMemoryBytes,
    qnodes: result.qnodes, ttHits: result.ttHits, qTtHits: result.qTtHits ?? 0, cutoffs: result.cutoffs,
    searchMs: result.elapsedMs, wallMs, nps: Math.round(result.nodes * 1000 / wallMs),
    score: result.score, scoreType: result.scoreType, status: result.status,
    stoppedReason: result.stoppedReason, searchPolicy: result.searchPolicy,
    turn: result.bestAction ? formatAction(position, result.bestAction) : null,
  };
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function summaries(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.case}:${row.mode}:${row.engine}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return [...groups.values()].map(group => ({
    case: group[0].case, mode: group[0].mode, engine: group[0].engine, runs: group.length,
    allTargetsReached: group[0].mode === 'depth' ? group.every(row => row.targetReached) : null,
    depths: [...new Set(group.map(row => row.depth))],
    quiescenceDepths: [...new Set(group.map(row => row.quiescenceDepth))],
    scores: [...new Set(group.map(row => row.score))],
    medianNodes: median(group.map(row => row.nodes)), totalNodes: group.reduce((sum, row) => sum + row.nodes, 0),
    medianCacheMemoryBytes: median(group.map(row => row.cacheMemoryBytes)),
    medianWallMs: median(group.map(row => row.wallMs)), totalWallMs: group.reduce((sum, row) => sum + row.wallMs, 0),
    minWallMs: Math.min(...group.map(row => row.wallMs)), maxWallMs: Math.max(...group.map(row => row.wallMs)),
    comparableScoreRuns: group.filter(row => row.sameScore !== null).length,
    sameScore: group.every(row => row.sameScore === null) ? null : group.filter(row => row.sameScore !== null).every(row => row.sameScore),
    medianSpeedup: group.every(row => row.speedup !== null) ? median(group.map(row => row.speedup)) : null,
  }));
}

async function main() {
  const options = settings(process.argv.slice(2));
  if (options.help) { console.log(HELP); return; }
  const currentPath = fileURLToPath(new URL('../src/search.js', import.meta.url));
  if (options.baseline === currentPath) throw new Error('Baseline must be a different source file.');
  const engines = [];
  if (options.baseline) engines.push(await loadEngine('baseline', options.baseline));
  engines.push(await loadEngine('current', currentPath));
  const cases = [
    { name: 'standard', setup: { variant: 'standard' }, depth: 4, quiescenceDepth: 2 },
    { name: 'opening', setup: { pgn: await readFile(new URL('../examples/opening.5dpgn', import.meta.url), 'utf8') }, depth: 2, quiescenceDepth: 2 },
    { name: 'two-timelines', setup: { variant: 'two_timelines' }, depth: 2, quiescenceDepth: 1 },
    { name: 'temporal', setup: { pgn: await readFile(new URL('../examples/time-travel.5dpgn', import.meta.url), 'utf8') }, depth: 2, quiescenceDepth: 1 },
  ].filter(fixture => !options.case || options.case === fixture.name);
  const limits = fixture => ({ maxNodes: 2000000, cacheMemoryMb: 128,
    maxDepth: options.depth ?? fixture.depth, quiescenceDepth: fixture.quiescenceDepth });
  // Warm every module and fixture before recording samples. Each analyze call
  // still starts with fresh search tables and ordering history.
  for (let round = 0; round < options.warmup; round++) {
    for (const fixture of cases) for (const engine of engines) run(engine, fixture, { ...limits(fixture), timeMs: 250 });
  }
  const rows = [];
  for (const fixture of cases) {
    for (const mode of options.modes) {
      for (let repeat = 1; repeat <= options.repeat; repeat++) {
        const order = repeat % 2 ? engines : [...engines].reverse();
        for (const engine of order) {
          const row = run(engine, fixture, { ...limits(fixture),
            timeMs: mode === 'depth' ? options.depthTimeMs : options.timeMs,
            maxDepth: mode === 'depth' ? options.depth ?? fixture.depth : 64 });
          Object.assign(row, { mode, repeat });
          if (mode === 'time') row.targetReached = null;
          rows.push(row);
          if (!options.json) console.error(`${fixture.name} ${mode}, run ${repeat}, ${engine.name}: depth ${row.depth}/q${row.quiescenceDepth}, ${row.nodes} nodes, ${Math.round(row.wallMs)} ms`);
        }
      }
    }
  }
  for (const row of rows) {
    const baseline = rows.find(other => other.engine === 'baseline' && other.case === row.case && other.mode === row.mode && other.repeat === row.repeat);
    const comparable = baseline && row.engine !== 'baseline' && row.completed && baseline.completed
      && row.depth === baseline.depth && row.quiescenceDepth === baseline.quiescenceDepth
      && row.searchPolicy === baseline.searchPolicy;
    row.sameScore = comparable ? row.score === baseline.score && row.scoreType === baseline.scoreType : null;
    row.speedup = comparable && row.mode === 'depth' && row.targetReached && baseline.targetReached ? baseline.wallMs / row.wallMs : null;
    row.nodeReduction = row.speedup === null ? null : 1 - row.nodes / baseline.nodes;
  }
  const summary = summaries(rows);
  if (options.json) console.log(JSON.stringify({ options, nodeVersion: process.version, gcBetweenRuns: typeof globalThis.gc === 'function',
    engines: engines.map(({ analyze, ...engine }) => engine), results: rows, summary }, null, 2));
  else {
    console.table(summary.map(row => ({
      case: row.case, mode: row.mode, engine: row.engine, runs: row.runs,
      depth: row.depths.join('/'), qdepth: row.quiescenceDepths.join('/'),
      reached: row.allTargetsReached ?? 'n/a', nodes: Math.round(row.medianNodes),
      ms: Math.round(row.medianWallMs), scores: row.scores.join('/'),
      sameScore: row.sameScore ?? 'n/a', speedup: row.medianSpeedup === null ? 'n/a' : `${row.medianSpeedup.toFixed(2)}x`,
    })));
    console.log('All returned turns and PVs passed legality checks; counters balanced and input positions remained unchanged.');
    console.log('Depth counts complete turns; qdepth is the completed tactical horizon. Timing and paired speedups are medians of warmed runs. Fixed-depth speedups require matching completed horizons. This benchmark does not establish an Elo rating.');
  }
}

main().catch(error => { console.error(`Benchmark failed: ${error.message}`); process.exitCode = 1; });
