#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { Worker } from 'node:worker_threads';
import { createWriteStream } from 'node:fs';
import { once } from 'node:events';
import { finished } from 'node:stream/promises';
import { readFile, mkdir, readdir, lstat, realpath, unlink, rmdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID, createHash } from 'node:crypto';
import { TransformerRuntime, DEFAULT_CHECKPOINT, DEFAULT_PYTHON, PROJECT_ROOT, forwardInference } from '../src/transformer-runtime.js';
import { positionKey } from '../src/rules.js';
import { loadMatchSuite } from './match.js';
import { generateSelfPlayGames } from './transformer-selfplay-games.js';
import { createInferenceQueue } from './transformer-selfplay-inference.js';
import { evaluateCandidate } from './transformer-selfplay-arena.js';
import { atomicWrite, fileHash, acquireRunLock, updateReplay, promoteCheckpoint } from './transformer-selfplay-store.js';
import { recoverInterruptedIterations } from './transformer-selfplay-recovery.js';

const defaults = {
  iterations: 1, games: 8, gameConcurrency: 1, maxPlies: 40, maxNodes: 20000, maxDepth: 2, timeMs: 3000,
  terminalWork: 20000, terminalTimeMs: 3000, exploration: .2, explorationPlies: 12, outcomeWeight: .5,
  steps: 500, batchSize: 16, maxTokens: 4096, learningRate: .0001, replaySize: 8192, seed: 42,
  arenaPairs: 8, arenaConcurrency: 1, minPairs: 4, arenaPlies: 80, promotionScore: .55, keepIterations: 5,
  device: process.env.TRANSFORMER_DEVICE || 'auto',
  checkpoint: process.env.TRANSFORMER_CHECKPOINT || DEFAULT_CHECKPOINT,
  python: process.env.TRANSFORMER_PYTHON || DEFAULT_PYTHON,
  runDir: path.join(PROJECT_ROOT, 'artifacts/transformer/selfplay'),
  seedData: path.join(PROJECT_ROOT, 'artifacts/transformer/training.jsonl'),
  suite: path.join(PROJECT_ROOT, 'examples/matches/training.json'),
  arenaSuite: path.join(PROJECT_ROOT, 'examples/matches/validation.json'),
};

export const getSelfPlayDefaults = () => ({ ...defaults });

export const help = `Usage: node scripts/transformer-selfplay.js [options]
Continuous shortcut: npm run transformer:selfplay:continuous
  --iterations N       Cycles this invocation; 0 = until Ctrl+C (default 1)
  --games N            Self-play games/cycle (8)
  --game-concurrency N Concurrent self-play games, 1..8; shared model (1)
  --plies N            Self-play turn cap (40)
  --nodes N            Per-turn search work (20000)
  --depth N            Neural depth 1..64; 0 grows dynamically (default 2)
  --time-ms N          Per-turn search safety cap (3000)
  --terminal-time-ms N Independent terminal verification time cap (3000)
  --terminal-work N    Full-rules terminal verification budget (20000)
  --exploration X      Random legal-turn probability in early play (0.2)
  --exploration-plies N Early turns eligible for exploration (12)
  --outcome-weight X   Finished-game outcome weight vs search target (0.5)
  --steps N            Additional training updates/cycle (500)
  --batch-size N       Training batch (16)
  --max-tokens N       Training context limit, 16..4096 (4096)
  --learning-rate X    AdamW learning rate (0.0001)
  --replay-size N      Maximum unique replay positions (8192)
  --seed-data FILE     Initial replay JSONL; "none" starts from self-play only
  --arena-pairs N      Distinct starts, each played with colors swapped (8)
  --arena-concurrency N Concurrent arena games, 1..8; shared models (1)
  --min-pairs N        Minimum completed distinct pairs for promotion (4)
  --arena-plies N      Arena game turn cap (80)
  --promotion-score X  Required candidate score, strictly above 0.5 (0.55)
  --suite FILE         Self-play starts (examples/matches/training.json)
  --arena-suite FILE   Evaluation starts (examples/matches/validation.json)
  --checkpoint FILE    Active/UI checkpoint to improve
  --run-dir DIR        Replay, logs, candidate versions (artifacts/transformer/selfplay)
  --keep-iterations N  Retain latest N cycle folders (5)
  --device auto|cuda|cpu --python FILE --seed N
  --help

Existing trained checkpoint required. Only complete pairs score; invalid games or
insufficient completed pairs block promotion. Ctrl+C closes owned processes; rerun to continue
with persisted replay and the active model. Incomplete cycles are not promoted.`;

export function parseArguments(args, initialOptions = defaults) {
  const options = { ...initialOptions };
  const names = { iterations: 'iterations', games: 'games', 'game-concurrency': 'gameConcurrency', plies: 'maxPlies', nodes: 'maxNodes', depth: 'maxDepth',
    'time-ms': 'timeMs', 'terminal-time-ms': 'terminalTimeMs', 'terminal-work': 'terminalWork', exploration: 'exploration', 'exploration-plies': 'explorationPlies',
    'outcome-weight': 'outcomeWeight', steps: 'steps', 'batch-size': 'batchSize', 'max-tokens': 'maxTokens', 'learning-rate': 'learningRate',
    'replay-size': 'replaySize', seed: 'seed', 'arena-pairs': 'arenaPairs', 'arena-concurrency': 'arenaConcurrency', 'min-pairs': 'minPairs',
    'arena-plies': 'arenaPlies', 'promotion-score': 'promotionScore', 'keep-iterations': 'keepIterations' };
  const paths = { checkpoint: 'checkpoint', python: 'python', 'run-dir': 'runDir', 'seed-data': 'seedData', suite: 'suite', 'arena-suite': 'arenaSuite' };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--help') { options.help = true; continue; }
    if (!args[i].startsWith('--')) {
      throw new Error(`Unexpected positional argument ${JSON.stringify(args[i])}. Options require --name VALUE. If npm/PowerShell stripped the option names, run directly: node scripts/transformer-selfplay.js --iterations 0 --device cuda`);
    }
    const flag = args[i].slice(2), value = args[++i];
    if (!value || value.startsWith('--')) throw new Error('Options require --name VALUE. Use --help.');
    if (names[flag]) options[names[flag]] = Number(value);
    else if (paths[flag]) options[paths[flag]] = value;
    else if (flag === 'device') options.device = value;
    else throw new Error(`Unknown option --${flag}.`);
  }
  // Programmatic callers may supply an older saved options object.
  options.terminalTimeMs ??= options.timeMs;
  for (const [name, min, max] of [
    ['iterations', 0, 1000000], ['games', 1, 128], ['gameConcurrency', 1, 8], ['maxPlies', 1, 256], ['maxNodes', 1, 10000000],
    ['maxDepth', 0, 64], ['timeMs', 1, 60000], ['terminalTimeMs', 1, 60000], ['terminalWork', 1, 10000000], ['explorationPlies', 0, 256],
    ['steps', 1, 1000000], ['batchSize', 1, 128], ['maxTokens', 16, 4096], ['replaySize', 1, 100000], ['seed', 0, 0xffffffff],
    ['arenaPairs', 1, 128], ['arenaConcurrency', 1, 8], ['minPairs', 1, 128], ['arenaPlies', 1, 256], ['keepIterations', 1, 100],
  ]) if (!Number.isSafeInteger(options[name]) || options[name] < min || options[name] > max) throw new Error(`Invalid ${name}: expected integer ${min}..${max}.`);
  for (const name of ['exploration', 'outcomeWeight']) if (!Number.isFinite(options[name]) || options[name] < 0 || options[name] > 1) throw new Error(`Invalid ${name}.`);
  if (!(options.learningRate > 0 && options.learningRate <= .1)) throw new Error('Invalid learningRate.');
  if (!(options.promotionScore > .5 && options.promotionScore <= 1)) throw new Error('promotionScore must be >0.5 and <=1.');
  if (options.minPairs > options.arenaPairs) throw new Error('minPairs cannot exceed arenaPairs.');
  if (!['auto', 'cpu', 'cuda'].includes(options.device)) throw new Error('device must be auto, cuda or cpu.');
  for (const name of Object.values(paths)) options[name] = name === 'seedData' && options[name] === 'none' ? undefined : path.resolve(options[name]);
  validateManagedPaths(options);
  return options;
}

function validateManagedPaths(options) {
  // Managed outputs must never alias the active model or read-only inputs.
  const reserved = new Set(['replay.jsonl', 'latest.json', 'run.json', 'previous-model.pt', '.selfplay.lock']);
  for (const name of ['checkpoint', 'seedData', 'suite', 'arenaSuite', 'python']) {
    const relative = options[name] && path.relative(options.runDir, options[name]);
    if (relative && /^iteration-\d+(?:[\\/]|$)/i.test(relative)) throw new Error(`${name} cannot be inside managed iteration folders.`);
    if (relative && reserved.has(relative.toLowerCase())) throw new Error(`${name} conflicts with managed output ${relative}.`);
  }
}

function cancelled() { const error = new Error('Self-play stopped.'); error.name = 'AbortError'; return error; }
function checkStop(shouldStop) { if (shouldStop()) throw cancelled(); }
const emit = (event, fields = {}) => console.log(JSON.stringify({ event, ...fields }));
const json = value => JSON.stringify(value, null, 2) + '\n';

/** CPU search runs in a worker so its safety deadline can interrupt eager rules code. */
export function workerAnalyzer(runtime, shouldStop = () => false, maxConcurrency = 1) {
  if (!Number.isInteger(maxConcurrency) || maxConcurrency < 1 || maxConcurrency > 8) throw new Error('Invalid search worker concurrency.');
  const inference = createInferenceQueue(runtime), active = new Set(), cancellations = new Set();
  const slots = new Set();
  let closed = false;
  const search = async (position, options) => {
    const stopped = () => closed || shouldStop() || Boolean(options.shouldStop?.());
    checkStop(stopped);
    let stopStartup, startupPoll, info;
    const startupCancelled = new Promise((resolve, reject) => {
      stopStartup = () => reject(cancelled());
      cancellations.add(stopStartup);
      startupPoll = setInterval(() => { if (stopped()) stopStartup(); }, 25);
    });
    try { info = await Promise.race([runtime.start(), startupCancelled]); }
    finally { clearInterval(startupPoll); cancellations.delete(stopStartup); }
    checkStop(stopped);
    // A game can hit its transport timeout before its old worker has exited.
    // Hold a physical slot through termination so its replacement cannot
    // transiently exceed the configured number of CPU workers.
    while (slots.size >= maxConcurrency) {
      await Promise.race(slots);
      checkStop(stopped);
    }
    let releaseSlot;
    const slot = new Promise(resolve => { releaseSlot = resolve; });
    slots.add(slot);
    const cancelBuffer = new SharedArrayBuffer(4), cancellation = new Int32Array(cancelBuffer);
    const controller = new AbortController();
    let worker;
    try {
      worker = new Worker(new URL('../src/worker.js', import.meta.url), { workerData: {
        position, cancelBuffer, model: info.model,
        options: { engine: 'transformer', timeMs: options.timeMs, maxNodes: options.maxNodes, maxDepth: options.maxDepth,
          collectPolicyAlternatives: options.collectPolicyAlternatives === true },
      } });
      return await new Promise((resolve, reject) => {
        let finished = false;
        const finish = (error, result) => {
          if (finished) return;
          finished = true; clearInterval(poll); clearTimeout(deadline);
          cancellations.delete(stop);
          controller.abort();
          if (error) reject(error); else resolve(result);
        };
        const stop = () => { Atomics.store(cancellation, 0, 1); finish(cancelled()); };
        cancellations.add(stop);
        const poll = setInterval(() => {
          if (stopped()) stop();
        }, 25);
        const deadline = setTimeout(() => finish(new Error('Search worker exceeded its hard safety deadline.')), options.timeMs + 5000);
        worker.on('message', message => {
          if (finished) return;
          if (message.type === 'evaluate' || message.type === 'policy') void forwardInference(worker, {
            evaluate: (positions, options) => inference.evaluate(positions, { ...options, signal: controller.signal }),
            orderMoves: (position, moves, options) => inference.orderMoves(position, moves, { ...options, signal: controller.signal }),
          }, message);
          else if (message.type === 'result') finish(null, message.result);
          else if (message.type === 'error') finish(new Error(message.error));
        });
        worker.once('error', error => finish(error));
        worker.once('exit', code => { if (!finished) finish(new Error(`Search worker exited without a result (${code}).`)); });
      });
    } finally {
      try { await worker?.terminate(); }
      finally { slots.delete(slot); releaseSlot(); }
    }
  };
  const analyze = (position, options) => {
    const task = search(position, options);
    active.add(task);
    task.then(() => active.delete(task), () => active.delete(task));
    return task;
  };
  analyze.close = async () => {
    closed = true;
    for (const stop of cancellations) stop();
    await Promise.allSettled([...active]);
  };
  return analyze;
}

export async function trainCandidate(options, files, seed, shouldStop, onEvent = emit,
  { spawnProcess = spawn, exitGraceMs = 1000, stopGraceMs = 2000 } = {}) {
  checkStop(shouldStop);
  const args = ['-u', path.join(PROJECT_ROOT, 'neural/train.py'), '--data', files.replay,
    '--resume', files.incumbent, '--output', files.candidate, '--steps', String(options.steps),
    '--batch-size', String(options.batchSize), '--max-tokens', String(options.maxTokens), '--learning-rate', String(options.learningRate),
    '--seed', String(seed), '--device', options.device, '--save-every', String(options.steps),
    '--log-every', String(Math.max(1, Math.ceil(options.steps / 20))),
    '--label', 'Experimental transformer self-play value model; promotion arena recorded separately'];
  await atomicWrite(files.command, json({ executable: options.python, args }));
  const log = createWriteStream(files.log, { flags: 'wx' });
  let poll, heartbeat, exitTimer, killTimer;
  try {
    // Do not launch a trainer unless its exclusive log opened successfully.
    await once(log, 'open');
    checkStop(shouldStop);
    const child = spawnProcess(options.python, args, { cwd: PROJECT_ROOT, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    let tail = '', failure, exited = false, stopping = false;
    const output = chunk => {
      if (!log.destroyed) log.write(chunk);
      tail = (tail + chunk.toString()).slice(-6000);
    };
    child.stdout.on('data', output);
    child.stderr.on('data', output);
    await new Promise((resolve, reject) => {
      const stop = () => {
        if (stopping || exited) return;
        stopping = true;
        child.kill();
        killTimer = setTimeout(() => { if (!exited) child.kill('SIGKILL'); }, stopGraceMs);
      };
      const finish = (code, signal) => {
        if (shouldStop()) reject(cancelled());
        else if (failure) reject(failure);
        else if (code === 0) resolve();
        else reject(new Error(`Training exited (${signal || code}). ${tail.trim()}`));
      };
      const failOutput = error => { failure ??= error; stop(); };
      log.once('error', failOutput);
      child.stdout.once('error', failOutput);
      child.stderr.once('error', failOutput);
      child.once('error', reject);
      child.once('exit', (code, signal) => {
        exited = true;
        clearTimeout(killTimer);
        // Descendants can retain inherited pipes after the trainer has exited,
        // preventing ChildProcess.close forever. Allow buffered output to drain,
        // then release those handles and settle from the actual exit status.
        exitTimer = setTimeout(() => {
          child.stdout.destroy(); child.stderr.destroy();
          finish(code, signal);
        }, exitGraceMs);
      });
      child.once('close', finish);
      poll = setInterval(() => { if (shouldStop()) stop(); }, 100);
      heartbeat = setInterval(() => onEvent('training-progress', { log: files.log }), 30000);
    });
  } finally {
    clearInterval(poll); clearInterval(heartbeat);
    clearTimeout(exitTimer); clearTimeout(killTimer);
    if (!log.destroyed) log.end();
    await finished(log).catch(() => {});
  }
}

async function pruneIterations(runDir, runId, keep) {
  const root = await realpath(runDir);
  const folders = (await readdir(root, { withFileTypes: true }))
    .filter(entry => entry.isDirectory() && /^iteration-\d{8,}$/.test(entry.name))
    .map(entry => entry.name).sort();
  for (const name of folders.slice(0, Math.max(0, folders.length - keep))) {
    const folder = path.join(root, name);
    // Resolve and verify every recursive target; only flat, owned cycle folders are removed.
    if (path.dirname(await realpath(folder)) !== root || (await lstat(folder)).isSymbolicLink()) throw new Error('Unsafe retention path.');
    let manifest;
    try { manifest = JSON.parse(await readFile(path.join(folder, 'iteration.json'), 'utf8')); }
    catch (error) { if (error.code === 'ENOENT' || error instanceof SyntaxError) continue; throw error; }
    if (manifest.runId !== runId) continue;
    const entries = await readdir(folder, { withFileTypes: true });
    if (entries.some(entry => !entry.isFile())) continue;
    for (const entry of entries) await unlink(path.join(folder, entry.name));
    await rmdir(folder);
  }
}

export async function runSelfPlay(options, { shouldStop = () => false, onRuntime = () => {}, onEvent = emit, lockToken } = {}) {
  options = { ...options };
  await mkdir(options.runDir, { recursive: true });
  // Resolve existing symlinks before output validation and checkpoint locking.
  for (const name of ['runDir', 'checkpoint', 'seedData', 'suite', 'arenaSuite', 'python']) if (options[name]) {
    options[name] = await realpath(options[name]).catch(error => { if (error.code !== 'ENOENT') throw error; return options[name]; });
  }
  validateManagedPaths(options);
  const release = await acquireRunLock(options.runDir, { token: lockToken });
  let releaseModel;
  const runtimes = new Set(), analyzers = new Set();
  const openRuntime = checkpoint => {
    const runtime = new TransformerRuntime({ checkpoint, python: options.python, device: options.device });
    runtimes.add(runtime); onRuntime(runtime); return runtime;
  };
  const openAnalyzer = (runtime, concurrency = 1) => {
    const analyzer = workerAnalyzer(runtime, shouldStop, concurrency);
    analyzers.add(analyzer);
    return analyzer;
  };
  const closeRuntimes = async () => {
    for (const runtime of runtimes) runtime.close();
    await Promise.all([...analyzers].map(analyzer => analyzer.close()));
    runtimes.clear(); analyzers.clear();
  };
  try {
    // A common checkpoint lock also excludes runners using different run directories.
    releaseModel = await acquireRunLock(`${options.checkpoint}.selfplay-lock`, { token: lockToken });
    await recoverInterruptedIterations(options);
    const marker = path.join(options.runDir, 'run.json');
    let run;
    try { run = JSON.parse(await readFile(marker, 'utf8')); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (run && (run.version !== 1 || typeof run.runId !== 'string' || !Number.isSafeInteger(run.nextIteration) || run.nextIteration < 1)) throw new Error('Invalid self-play run.json; use a new --run-dir.');
    if (run && run.checkpoint !== options.checkpoint) throw new Error('This run directory belongs to another checkpoint. Choose a new --run-dir.');
    run ??= { version: 1, runId: randomUUID(), checkpoint: options.checkpoint, createdAt: new Date().toISOString(), nextIteration: 1 };
    await atomicWrite(marker, json(run));
    const [trainingSuite, arenaSuite, testSuite] = await Promise.all([loadMatchSuite(options.suite), loadMatchSuite(options.arenaSuite),
      loadMatchSuite(path.join(PROJECT_ROOT, 'examples/matches/transformer-test.json'))]);
    const arenaKeys = new Set(arenaSuite.cases.map(item => positionKey(item.position)));
    if (trainingSuite.cases.some(item => arenaKeys.has(positionKey(item.position)))) throw new Error('Self-play and arena starting positions overlap. Use separate suites.');
    const testKeys = new Set(testSuite.cases.map(item => positionKey(item.position)));
    if ([...trainingSuite.cases, ...arenaSuite.cases].some(item => testKeys.has(positionKey(item.position)))) {
      throw new Error('Training or promotion starts overlap the frozen Transformer test suite. Use separate suites.');
    }
    const excludedKeys = new Set([...arenaKeys, ...testKeys]);
    const replay = path.join(options.runDir, 'replay.jsonl');
    const reports = [];
    for (let count = 0; options.iterations === 0 || count < options.iterations; count++) {
      checkStop(shouldStop);
      const iteration = run.nextIteration++, seed = (options.seed + iteration - 1) >>> 0;
      await atomicWrite(marker, json(run)); // Reserve the number before creating files; interrupted runs never overwrite them.
      const folder = path.join(options.runDir, `iteration-${String(iteration).padStart(8, '0')}`);
      await mkdir(folder);
      const manifest = { runId: run.runId, iteration, seed, options, startedAt: new Date().toISOString() };
      await atomicWrite(path.join(folder, 'iteration.json'), json(manifest));
      await pruneIterations(options.runDir, run.runId, options.keepIterations);
      const files = { replay, incumbent: path.join(folder, 'incumbent.pt'), candidate: path.join(folder, 'candidate.pt'),
        log: path.join(folder, 'train.log'), command: path.join(folder, 'train-command.json') };
      let report = { ...manifest, folder, status: 'running', promoted: false };
      const saveReport = async () => {
        await atomicWrite(path.join(folder, 'report.json'), json(report));
        await atomicWrite(path.join(options.runDir, 'latest.json'), json(report));
      };
      try {
        await saveReport();
        onEvent('iteration-start', { iteration, startedAt: manifest.startedAt });
        checkStop(shouldStop);
        const snapshot = await readFile(options.checkpoint);
        const incumbentHash = createHash('sha256').update(snapshot).digest('hex');
        await atomicWrite(files.incumbent, snapshot);
        const runtime = openRuntime(files.incumbent);
        const info = await runtime.start();
        onEvent('selfplay-start', { iteration, folder, seed, device: info.device, trainedSteps: info.model.trainedSteps,
          gameConcurrency: Math.min(options.gameConcurrency ?? 1, options.games) });
        let completedGames = 0;
        const play = await generateSelfPlayGames({ ...options, positions: trainingSuite.cases, seed,
          analyzePosition: openAnalyzer(runtime, options.gameConcurrency ?? 1), shouldStop,
          metadata: { runId: run.runId, iteration, checkpointSha256: incumbentHash, trainedSteps: info.model.trainedSteps },
          onGame: async (game, samples) => {
            const number = String(game.index + 1).padStart(3, '0');
            await atomicWrite(path.join(folder, `selfplay-${number}.json`), json(game));
            await atomicWrite(path.join(folder, `samples-${number}.jsonl`), samples.map(row => JSON.stringify(row)).join('\n') + (samples.length ? '\n' : ''));
            onEvent('selfplay-game', { iteration, game: game.index + 1, completedGames: ++completedGames,
              result: game.result, reason: game.reason, samples: samples.length });
          } });
        report.selfplay = play.summary; report.incumbentSha256 = incumbentHash;
        await closeRuntimes(); checkStop(shouldStop);
        if (play.games.some(game => !game.valid)) throw new Error('Invalid self-play game; candidate training skipped. See game records.');
        if (!play.samples.length) throw new Error('No completed finite search targets. Increase --nodes/--time-ms or change --suite.');
        report.replay = await updateReplay({ replayPath: replay, newSamples: play.samples, seedData: options.seedData,
          maxSamples: options.replaySize, seed, excludePositionKeys: excludedKeys });
        onEvent('training-start', { iteration, replay: report.replay, steps: options.steps, batchSize: options.batchSize, maxTokens: options.maxTokens });
        await saveReport();
        await trainCandidate(options, files, seed, shouldStop, onEvent);
        checkStop(shouldStop);
        const candidate = openRuntime(files.candidate), incumbent = openRuntime(files.incumbent);
        const candidateInfo = await candidate.start(); await incumbent.start();
        report.candidate = candidateInfo.model; report.candidateSha256 = await fileHash(files.candidate);
        const arenaConcurrency = options.arenaConcurrency ?? 1;
        onEvent('arena-start', { iteration, pairs: options.arenaPairs, trainedSteps: candidateInfo.model.trainedSteps,
          gameConcurrency: Math.min(arenaConcurrency, options.arenaPairs * 2) });
        const arena = await evaluateCandidate({ candidate: openAnalyzer(candidate, arenaConcurrency),
          incumbent: openAnalyzer(incumbent, arenaConcurrency), gameConcurrency: arenaConcurrency,
          suite: arenaSuite, pairs: options.arenaPairs, seed, maxPlies: options.arenaPlies, maxNodes: options.maxNodes,
          maxDepth: options.maxDepth, timeMs: options.timeMs, terminalTimeMs: options.terminalTimeMs, terminalWork: options.terminalWork,
          minPairs: options.minPairs, promotionScore: options.promotionScore, shouldStop,
          onGame: async (game, { index, completed, total }) => {
            await atomicWrite(path.join(folder, `arena-${String(index + 1).padStart(3, '0')}.json`), json(game));
            onEvent('arena-game', { iteration, game: index + 1, completedGames: completed, totalGames: total,
              result: game.result, reason: game.reason });
          } });
        await closeRuntimes(); checkStop(shouldStop);
        await atomicWrite(path.join(folder, 'arena.json'), json(arena));
        report.arena = { summary: arena.summary, decision: arena.decision };
        // Persist the decision before any replacement so even an interrupted promotion is auditable.
        report.status = 'evaluated'; await saveReport();
        checkStop(shouldStop);
        if (arena.decision.promote) {
          report.promotion = await promoteCheckpoint({ candidatePath: files.candidate, activePath: options.checkpoint,
            expectedHash: incumbentHash, backupPath: path.join(folder, 'previous.pt'), shouldStop });
          report.promoted = true;
          await atomicWrite(path.join(options.runDir, 'previous-model.pt'), await readFile(files.incumbent));
        }
        report.status = 'complete'; report.finishedAt = new Date().toISOString();
        await saveReport();
        onEvent('cycle-complete', { iteration, promoted: report.promoted, decision: arena.decision, report: path.join(folder, 'report.json') });
        reports.push({ iteration, promoted: report.promoted, folder });
        // Keep continuous operation's memory bounded as well as replay and disk retention.
        if (reports.length > options.keepIterations) reports.shift();
      } catch (error) {
        report.status = shouldStop() || error.name === 'AbortError' ? 'interrupted' : 'failed';
        report.error = error.message; report.finishedAt = new Date().toISOString(); await saveReport(); throw error;
      } finally { await closeRuntimes(); }
    }
    return reports;
  } finally { await closeRuntimes(); await releaseModel?.(); await release(); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  let stopped = false;
  const runtimes = new Set();
  const stop = () => { stopped = true; for (const runtime of runtimes) runtime.close(); };
  process.on('SIGINT', stop); process.on('SIGTERM', stop);
  try {
    const options = parseArguments(process.argv.slice(2));
    if (options.help) console.log(help);
    else await runSelfPlay(options, { shouldStop: () => stopped, onRuntime: runtime => {
      for (const previous of runtimes) if (previous.closed) runtimes.delete(previous);
      runtimes.add(runtime);
    } });
  } catch (error) {
    if (stopped || error.name === 'AbortError') { emit('stopped', { message: 'Replay and completed reports retained. Rerun the same command to continue.' }); process.exitCode = 130; }
    else { console.error(error.message); process.exitCode = 1; }
  } finally { process.off('SIGINT', stop); process.off('SIGTERM', stop); }
}
