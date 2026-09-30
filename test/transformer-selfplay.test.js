import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { parseArguments, trainCandidate, workerAnalyzer } from '../scripts/transformer-selfplay.js';
import { evaluateCandidate } from '../scripts/transformer-selfplay-arena.js';
import { createPosition } from '../src/rules.js';

test('self-play CLI defaults to the continuous 800k preset and accepts overrides', () => {
  const defaults = parseArguments([]);
  assert.equal(defaults.iterations, 0);
  assert.equal(defaults.games, 12);
  assert.equal(defaults.gameConcurrency, 3);
  assert.equal(defaults.maxPlies, 60);
  assert.equal(defaults.arenaConcurrency, 2);
  assert.equal(defaults.maxDepth, 0);
  assert.equal(defaults.maxNodes, 20000);
  assert.equal(defaults.timeMs, 1000);
  assert.equal(defaults.steps, 500);
  assert.equal(defaults.batchSize, 8);
  assert.equal(defaults.maxTokens, 2048);
  assert.equal(defaults.learningRate, .0001);
  assert.equal(defaults.replaySize, 16384);
  assert.equal(defaults.terminalWork, 150000);
  assert.equal(defaults.terminalTimeMs, 3000);
  const budgets = parseArguments(['--time-ms', '600', '--terminal-time-ms', '9000']);
  assert.equal(budgets.timeMs, 600);
  assert.equal(budgets.terminalTimeMs, 9000);
  const { terminalTimeMs, ...legacy } = defaults;
  assert.equal(parseArguments(['--time-ms', '700'], legacy).terminalTimeMs, 700);
  assert.equal(defaults.device, process.env.TRANSFORMER_DEVICE || 'cuda');
  assert.ok(defaults.minPairs <= defaults.arenaPairs);
  const options = parseArguments(['--iterations', '0', '--seed-data', 'none', '--device', 'cpu', '--steps', '3', '--game-concurrency', '4', '--arena-concurrency', '3']);
  assert.equal(options.iterations, 0);
  assert.equal(options.seedData, undefined);
  assert.equal(options.steps, 3);
  assert.equal(options.gameConcurrency, 4);
  assert.equal(options.arenaConcurrency, 3);
  assert.equal(parseArguments(['--iterations', '1']).iterations, 1);
  assert.equal(parseArguments(['--depth', '64']).maxDepth, 64);
  assert.equal(parseArguments(['--depth', '0']).maxDepth, 0);
  assert.equal(parseArguments(['--max-tokens', '16']).maxTokens, 16);
  assert.equal(parseArguments(['--max-tokens', '512']).maxTokens, 512);
  assert.equal(parseArguments(['--max-tokens', '4096']).maxTokens, 4096);
});

test('self-play CLI rejects malformed limits and unsafe promotion thresholds', () => {
  for (const args of [ ['--games', '0'], ['--steps', '1.5'], ['--nodes', 'NaN'], ['--iterations', '-1'],
    ['--game-concurrency', '0'], ['--game-concurrency', '9'], ['--game-concurrency', '1.5'], ['--game-concurrency', 'NaN'],
    ['--arena-concurrency', '0'], ['--arena-concurrency', '9'], ['--arena-concurrency', '1.5'], ['--arena-concurrency', 'NaN'],
    ['--max-tokens', '15'], ['--max-tokens', '4097'], ['--max-tokens', '512.5'], ['--max-tokens', 'NaN'],
    ['--terminal-time-ms', '0'], ['--terminal-time-ms', '60001'], ['--terminal-time-ms', '1.5'],
    ['--exploration', '1.1'], ['--outcome-weight', '-.1'], ['--promotion-score', '.5'], ['--depth', '-1'], ['--depth', '65'],
    ['--arena-pairs', '2', '--min-pairs', '3'], ['--batch-size', '129'], ['--device', 'bogus'], ['--steps'], ['--bogus', '1'] ]) {
    assert.throws(() => parseArguments(args), undefined, args.join(' '));
  }
});

test('stripped npm option names produce actionable guidance instead of guessing positional values', () => {
  assert.throws(() => parseArguments(['0', 'cuda']), /npm\/PowerShell.*node scripts\/transformer-selfplay\.js --iterations 0 --device cuda/);
  const direct = parseArguments(['--iterations', '0', '--device', 'cuda']);
  assert.equal(direct.iterations, 0);
  assert.equal(direct.device, 'cuda');
});

test('retention cannot own active checkpoint or training input paths', () => {
  const run = path.resolve('artifacts/test-selfplay');
  for (const flag of ['--checkpoint', '--seed-data', '--suite', '--arena-suite', '--python']) {
    assert.throws(() => parseArguments(['--run-dir', run, flag, path.join(run, 'iteration-00000001', 'input.pt')]), /managed iteration/);
  }
  assert.throws(() => parseArguments(['--run-dir', run, '--seed-data', path.join(run, 'replay.jsonl')]), /replay.jsonl/);
  for (const file of ['run.json', 'latest.json', 'previous-model.pt', 'replay.jsonl', '.selfplay.lock']) {
    assert.throws(() => parseArguments(['--run-dir', run, '--checkpoint', path.join(run, file)]), /managed output/);
  }
});

test('worker transport honors a per-search cancellation callback and terminates its worker', async () => {
  let stop = false;
  const runtime = { start: async () => ({ model: {} }), evaluate: async () => new Promise(() => {}) };
  const timer = setTimeout(() => { stop = true; }, 50);
  try {
    await assert.rejects(workerAnalyzer(runtime)(createPosition(), {
      maxDepth: 2, maxNodes: 20000, timeMs: 1000, shouldStop: () => stop,
    }), { name: 'AbortError' });
  } finally { clearTimeout(timer); }
});

test('parallel search workers share inference and route independent evaluations correctly', async () => {
  let active = 0, peak = 0, calls = 0;
  const runtime = {
    start: async () => ({ model: {} }),
    async evaluate(positions) {
      calls++; peak = Math.max(peak, ++active);
      await new Promise(resolve => setTimeout(resolve, 5));
      active--;
      return { values: positions.map(() => 75) };
    },
  };
  const analyzer = workerAnalyzer(runtime, undefined, 3);
  try {
    const results = await Promise.all(Array.from({ length: 3 }, () => analyzer(createPosition(), {
      maxDepth: 1, maxNodes: 20000, timeMs: 10000,
    })));
    assert.ok(calls >= 3);
    assert.equal(peak, 1);
    for (const result of results) {
      assert.equal(result.completed, true);
      assert.equal(result.engine, 'transformer');
      assert.equal(result.score, 75);
      assert.ok(result.bestAction.length);
    }
  } finally { await analyzer.close(); }
});

test('parallel arena workers use the correct model for both color assignments', async () => {
  const models = new Set();
  let release;
  const bothModels = new Promise(resolve => { release = resolve; });
  const runtime = (name, score) => ({
    start: async () => ({ model: { name } }),
    async evaluate(positions) {
      models.add(name);
      if (models.size === 2) release();
      await bothModels;
      return { values: positions.map(() => score) };
    },
  });
  const candidate = workerAnalyzer(runtime('candidate', 75), undefined, 2);
  const incumbent = workerAnalyzer(runtime('incumbent', -75), undefined, 2);
  try {
    const arena = await evaluateCandidate({ candidate, incumbent,
      suite: { cases: [{ id: 'tiny', position: createPosition({ pgn: '[Board "Custom"]\n[Size "4x4"]\n[2rk/4/4/KR2:0:1:w]' }) }] },
      pairs: 1, minPairs: 1, gameConcurrency: 2, maxPlies: 1, maxDepth: 1, maxNodes: 20000, timeMs: 10000,
    });
    assert.equal(models.size, 2, 'both model workers must reach inference concurrently');
    assert.deepEqual(arena.games.map(game => game.aColor), [0, 1]);
    for (const [index, game] of arena.games.entries()) {
      assert.equal(game.valid, true);
      assert.equal(game.plies, 1);
      assert.equal(game.moves[0].engine, index === 0 ? 'A' : 'B');
      assert.equal(game.moves[0].search.score, index === 0 ? 75 : -75);
    }
  } finally { release(); await Promise.all([candidate.close(), incumbent.close()]); }
});

test('closing a shared analyzer cancels and drains every search worker', async () => {
  const analyzer = workerAnalyzer({ start: async () => ({ model: {} }), evaluate: async () => new Promise(() => {}) }, undefined, 2);
  const requests = Array.from({ length: 3 }, () => assert.rejects(analyzer(createPosition(), {
    maxDepth: 2, maxNodes: 20000, timeMs: 10000,
  }), { name: 'AbortError' }));
  await new Promise(resolve => setTimeout(resolve, 50));
  await analyzer.close();
  await Promise.all(requests);
  await assert.rejects(analyzer(createPosition(), { timeMs: 10000 }), { name: 'AbortError' });
});

test('closing an analyzer interrupts every shared startup wait', { timeout: 2000 }, async () => {
  const analyzer = workerAnalyzer({ start: () => new Promise(() => {}) }, undefined, 2);
  const requests = Array.from({ length: 3 }, () => assert.rejects(analyzer(createPosition(), {
    maxDepth: 2, maxNodes: 20000, timeMs: 10000,
  }), { name: 'AbortError' }));
  await analyzer.close();
  await Promise.all(requests);
});

test('a per-search stop interrupts startup before a worker exists', { timeout: 2000 }, async () => {
  let stop = false;
  const analyzer = workerAnalyzer({ start: () => new Promise(() => {}) });
  const request = assert.rejects(analyzer(createPosition(), {
    timeMs: 10000, shouldStop: () => stop,
  }), { name: 'AbortError' });
  stop = true;
  try { await request; }
  finally { await analyzer.close(); }
});

async function trainingFixture(t, onSpawn, shouldStop = () => false) {
  const directory = await mkdtemp(path.join(tmpdir(), 'vibe-selfplay-trainer-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const files = Object.fromEntries(['replay', 'incumbent', 'candidate', 'log', 'command']
    .map(name => [name, path.join(directory, name)]));
  const child = new EventEmitter();
  child.stdout = new PassThrough(); child.stderr = new PassThrough();
  child.kill = () => { queueMicrotask(() => child.emit('exit', null, 'SIGTERM')); return true; };
  const training = trainCandidate({ python: 'python', steps: 1, batchSize: 1, maxTokens: 512, learningRate: .001, device: 'cpu' },
    files, 42, shouldStop, () => {}, {
      exitGraceMs: 10, stopGraceMs: 10,
      spawnProcess(executable, args) { queueMicrotask(() => onSpawn(child, executable, args)); return child; },
    });
  return { child, files, training };
}

test('candidate training uses and records the selected context limit', async t => {
  let spawnedCommand;
  const { files, training } = await trainingFixture(t, (child, executable, args) => {
    spawnedCommand = { executable, args };
    child.emit('exit', 0, null);
  });
  await training;
  const command = JSON.parse(await readFile(files.command, 'utf8'));
  assert.deepEqual(command, spawnedCommand);
  assert.equal(command.args[command.args.indexOf('--max-tokens') + 1], '512');
  assert.equal(command.args[command.args.indexOf('--resume') + 1], files.incumbent);
});

test('trainer exit settles without waiting forever for inherited pipes to close', { timeout: 2000 }, async t => {
  const { child, files, training } = await trainingFixture(t, child => {
    child.stdout.write('final training update\n');
    child.emit('exit', 0, null);
    // No close event: a descendant still owns the inherited output pipes.
  });
  await training;
  assert.equal(child.stdout.destroyed, true);
  assert.equal(child.stderr.destroyed, true);
  assert.equal(await readFile(files.log, 'utf8'), 'final training update\n');
});

test('failed trainer exit preserves its status and stderr when pipes remain open', { timeout: 2000 }, async t => {
  const { training } = await trainingFixture(t, child => {
    child.stderr.write('training failed before checkpoint save\n');
    child.emit('exit', 7, null);
  });
  await assert.rejects(training, /Training exited \(7\).*training failed before checkpoint save/);
});

test('trainer cancellation escalates an ignored termination signal and drains output', { timeout: 2000 }, async t => {
  let stop = false;
  const signals = [];
  const { child, training } = await trainingFixture(t, child => {
    child.kill = (signal = 'SIGTERM') => {
      signals.push(signal);
      if (signal === 'SIGKILL') queueMicrotask(() => child.emit('exit', null, signal));
      return true;
    };
    stop = true;
  }, () => stop);
  await assert.rejects(training, { name: 'AbortError' });
  assert.deepEqual(signals, ['SIGTERM', 'SIGKILL']);
  assert.equal(child.stdout.destroyed, true);
});

test('a trainer output error stops the child and reports the stream failure', { timeout: 2000 }, async t => {
  const { training } = await trainingFixture(t, child => child.stdout.emit('error', new Error('output pipe failed')));
  await assert.rejects(training, /output pipe failed/);
});
