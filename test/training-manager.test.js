import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { TrainingManager, TRAINING_DEFAULTS, validateTrainingOptions } from '../src/training-manager.js';
import { generateSelfPlayGames } from '../scripts/transformer-selfplay-games.js';
import { evaluateCandidate } from '../scripts/transformer-selfplay-arena.js';
import { createPosition, generateActions, positionKey, validateAction } from '../src/rules.js';

function fakeWorker({ stopExits = false } = {}) {
  const worker = new EventEmitter();
  worker.messages = [];
  worker.terminated = false;
  worker.postMessage = message => {
    worker.messages.push(message);
    if (stopExits && message.type === 'stop') queueMicrotask(() => worker.emit('exit', 0));
  };
  worker.terminate = async () => {
    worker.terminated = true;
    worker.emit('exit', 1);
    return 1;
  };
  return worker;
}

async function fixture(t, overrides = {}) {
  const directory = await mkdtemp(path.join(tmpdir(), 'vibe-training-ui-'));
  const runDir = path.join(directory, 'selfplay');
  await mkdir(runDir);
  const workers = [], calls = [];
  const manager = new TrainingManager({
    runDir, checkpoint: path.join(directory, 'model.pt'), python: path.join(directory, 'python'),
    availability: async () => ({ available: true }),
    workerFactory: args => {
      calls.push(args);
      const worker = fakeWorker({ stopExits: true });
      workers.push(worker);
      return worker;
    },
    ...overrides,
  });
  t.after(async () => {
    await manager.close();
    await rm(directory, { recursive: true, force: true });
  });
  return { directory, runDir, manager, workers, calls };
}

function firstLegal(position) {
  const actions = generateActions(position);
  let action;
  try { action = actions.next().value.moves; }
  finally { actions.return(); }
  return { engine: 'transformer', bestAction: action, pv: [action], completed: true,
    status: 'ok', score: 125, depth: 1, nodes: 2, searchNodes: 1, generationNodes: 1,
    stoppedReason: 'depth', searchPolicy: 'transformer-bounded-alpha-beta' };
}

async function persistedIteration(runDir) {
  const id = 'iteration-00000007', folder = path.join(runDir, id);
  await mkdir(folder);
  const position = createPosition();
  const options = { maxPlies: 2, maxNodes: 20000, maxDepth: 1, timeMs: 3000, terminalWork: 20000 };
  const selfplay = await generateSelfPlayGames({ ...options, games: 1, exploration: 0,
    positions: [{ id: 'fixture', position }], analyzePosition: firstLegal,
    metadata: { runId: 'test-run', iteration: 7 } });
  const arena = await evaluateCandidate({ ...options, pairs: 1, minPairs: 1,
    suite: { cases: [{ id: 'fixture', position }] }, candidate: firstLegal, incumbent: firstLegal });
  const records = [selfplay.games[0], ...arena.games];
  const names = ['selfplay-001', 'arena-001', 'arena-002'];
  const report = { iteration: 7, status: 'complete', promoted: false,
    startedAt: '2026-09-23T12:00:00.000Z', finishedAt: '2026-09-23T12:01:00.000Z',
    selfplay: selfplay.summary, arena: { summary: arena.summary, decision: arena.decision } };
  await writeFile(path.join(folder, 'report.json'), JSON.stringify(report));
  await writeFile(path.join(folder, 'iteration.json'), JSON.stringify({ iteration: 7, options: { games: 1 } }));
  await writeFile(path.join(folder, 'train.log'), 'step=1 loss=0.42\nstep=2 loss=0.31\n');
  for (const [index, name] of names.entries()) {
    assert.equal(records[index].valid, true);
    assert.equal(records[index].moves.length, 2);
    await writeFile(path.join(folder, `${name}.json`), JSON.stringify(records[index]));
  }
  return { id, folder, records, names, report };
}

test('training options supply defaults and allow continuous runs with adjusted learning parameters', () => {
  assert.deepEqual(validateTrainingOptions({}), TRAINING_DEFAULTS);
  const options = validateTrainingOptions({ iterations: 0, games: 3, gameConcurrency: 8, device: 'cpu',
    learningRate: 0.001, exploration: 0, outcomeWeight: 1, arenaPairs: 2, arenaConcurrency: 8, minPairs: 1 });
  assert.equal(options.iterations, 0);
  assert.equal(options.games, 3);
  assert.equal(options.gameConcurrency, 8);
  assert.equal(options.arenaConcurrency, 8);
  assert.equal(options.learningRate, 0.001);
  assert.equal(options.device, 'cpu');
  assert.equal(options.steps, TRAINING_DEFAULTS.steps);
  assert.equal(TRAINING_DEFAULTS.iterations, 1, 'Validation does not mutate shared defaults.');
  assert.equal(TRAINING_DEFAULTS.gameConcurrency, 1, 'Training is sequential unless concurrency is requested.');
  assert.equal(TRAINING_DEFAULTS.arenaConcurrency, 1, 'Arena games are sequential unless concurrency is requested.');
  assert.equal(validateTrainingOptions({ gameConcurrency: 4 }).arenaConcurrency, 1, 'Arena concurrency is independent of self-play.');
  assert.equal(validateTrainingOptions({ arenaConcurrency: 4 }).gameConcurrency, 1, 'Arena concurrency does not alter self-play.');
  assert.equal(validateTrainingOptions({ maxDepth: 64 }).maxDepth, 64);
  assert.equal(validateTrainingOptions({ maxDepth: 0 }).maxDepth, 0);
  assert.equal(TRAINING_DEFAULTS.maxDepth, 2, 'Dynamic depth remains opt-in.');
});

test('training options reject malformed numbers, invalid ranges and inconsistent promotion thresholds', () => {
  for (const value of [null, true, false, '', '2', [], {}, NaN, Infinity, -1, 1.5]) {
    assert.throws(() => validateTrainingOptions({ games: value }), /games/i, `Accepted games ${String(value)}`);
    assert.throws(() => validateTrainingOptions({ gameConcurrency: value }), /game.?concurrency/i, `Accepted gameConcurrency ${String(value)}`);
    assert.throws(() => validateTrainingOptions({ arenaConcurrency: value }), /arena.?concurrency/i, `Accepted arenaConcurrency ${String(value)}`);
  }
  for (const options of [
    { games: 0 }, { gameConcurrency: 0 }, { gameConcurrency: 9 }, { arenaConcurrency: 0 }, { arenaConcurrency: 9 },
    { iterations: -1 }, { batchSize: 129 }, { maxDepth: -1 }, { maxDepth: 65 },
    { learningRate: 0 }, { learningRate: 0.11 }, { exploration: -0.01 }, { outcomeWeight: 1.01 },
    { promotionScore: 0.5 }, { promotionScore: 1.01 }, { arenaPairs: 1, minPairs: 2 }, { device: 'shell' },
  ]) assert.throws(() => validateTrainingOptions(options), undefined, JSON.stringify(options));
  for (const value of [null, [], 'options', 12]) assert.throws(() => validateTrainingOptions(value));
});

test('training options reject arbitrary paths, executable settings and unknown or prototype fields', () => {
  for (const name of ['checkpoint', 'python', 'runDir', 'suite', 'arenaSuite', 'seedData', 'env', 'workerFactory', 'constructor', 'unknown']) {
    assert.throws(() => validateTrainingOptions({ [name]: 'untrusted' }), /unknown|unsupported|allowed|unexpected/i, name);
  }
  assert.throws(() => validateTrainingOptions(JSON.parse('{"__proto__":{"games":1}}')), /unknown|unsupported|allowed|unexpected/i);
});

test('only one training start can win while availability is still being checked', async t => {
  let releaseAvailability, checking;
  const entered = new Promise(resolve => { checking = resolve; });
  const pending = new Promise(resolve => { releaseAvailability = resolve; });
  const { manager, workers, calls } = await fixture(t, {
    availability: async () => { checking(); await pending; return { available: true }; },
  });
  const first = manager.start({ games: 2, gameConcurrency: 2, arenaConcurrency: 3, steps: 7 });
  await entered;
  const second = manager.start({ games: 3 });
  releaseAvailability();
  const results = await Promise.allSettled([first, second]);
  assert.equal(results.filter(result => result.status === 'fulfilled').length, 1);
  const rejected = results.find(result => result.status === 'rejected');
  assert.equal(rejected.reason.statusCode, 409);
  assert.equal(workers.length, 1);
  assert.equal(calls[0].options.games, 2);
  assert.equal(calls[0].options.gameConcurrency, 2);
  assert.equal(calls[0].options.arenaConcurrency, 3);
  assert.equal(calls[0].options.steps, 7);
  assert.equal(calls[0].cancelBuffer instanceof SharedArrayBuffer, true);
});

test('training stop signals its worker and remains stopping until the worker has released resources', async t => {
  const worker = fakeWorker();
  let invocation;
  const { manager } = await fixture(t, { workerFactory: args => { invocation = args; return worker; } });
  await manager.start({});
  worker.emit('message', { type: 'event', event: { event: 'selfplay-start', iteration: 4 } });
  const running = await manager.snapshot();
  assert.equal(running.status.state, 'running');
  assert.equal(running.status.iteration, 4);
  assert.equal((await manager.stop()).state, 'stopping');
  assert.equal(Atomics.load(new Int32Array(invocation.cancelBuffer), 0), 1);
  assert.ok(worker.messages.some(message => message.type === 'stop'));
  assert.equal(worker.terminated, false, 'Graceful stop lets the runner persist its report and release locks.');
  await assert.rejects(manager.start({}), error => error.statusCode === 409);
  worker.emit('message', { type: 'complete', state: 'interrupted' });
  worker.emit('exit', 0);
  assert.equal((await manager.snapshot()).status.state, 'interrupted');
});

test('worker failures are visible and server shutdown prevents subsequent starts', async t => {
  const { manager, workers } = await fixture(t);
  await manager.start({});
  workers[0].emit('error', new Error('Fixture training failure'));
  workers[0].emit('exit', 1);
  const failed = await manager.snapshot();
  assert.equal(failed.status.state, 'failed');
  assert.match(failed.status.error, /Fixture training failure/);
  await manager.start({});
  await manager.close();
  assert.ok(workers[1].messages.some(message => message.type === 'stop'));
  await assert.rejects(manager.start({}), /closed|shut/i);
});

test('unavailable training never creates a worker and gives an actionable failure', async t => {
  const { manager, workers } = await fixture(t, {
    availability: async () => ({ available: false, reason: 'Checkpoint is missing.' }),
  });
  assert.equal((await manager.snapshot()).availability.available, false);
  await assert.rejects(manager.start({}), /Checkpoint is missing/i);
  assert.equal(workers.length, 0);
});

test('shutdown during an in-flight start cannot create a late training worker', async t => {
  let releaseAvailability, checking;
  const entered = new Promise(resolve => { checking = resolve; });
  const pending = new Promise(resolve => { releaseAvailability = resolve; });
  const { manager, workers } = await fixture(t, {
    availability: async () => { checking(); await pending; return { available: true }; },
  });
  const starting = manager.start({});
  await entered;
  const closed = manager.close();
  releaseAvailability();
  await closed;
  await assert.rejects(starting, /closed|shut/i);
  assert.equal(workers.length, 0);
});

test('stopping an in-flight start cancels it before a training worker can launch', async t => {
  let releaseAvailability, checking;
  const entered = new Promise(resolve => { checking = resolve; });
  const pending = new Promise(resolve => { releaseAvailability = resolve; });
  const { manager, workers } = await fixture(t, {
    availability: async () => { checking(); await pending; return { available: true }; },
  });
  const starting = manager.start({});
  await entered;
  assert.equal(manager.stop().state, 'stopping');
  releaseAvailability();
  assert.equal((await starting).state, 'interrupted');
  assert.equal(workers.length, 0);
  assert.equal((await manager.snapshot()).availability.available, true);
  await manager.start({});
  assert.equal(workers.length, 1, 'Cancellation does not prevent a later deliberate start.');
});

test('external training locks remain owned by their runner and block local starts', async t => {
  const { manager, runDir, workers } = await fixture(t);
  const filename = path.join(runDir, '.selfplay.lock');
  const lock = JSON.stringify({ pid: process.pid, token: 'external-owner', createdAt: '2026-09-23T12:00:00.000Z' });
  await writeFile(filename, lock);
  const snapshot = await manager.snapshot();
  assert.equal(snapshot.status.state, 'external');
  assert.equal(snapshot.availability.available, false);
  assert.match(snapshot.availability.reason, /outside|terminal/i);
  await assert.rejects(manager.start({}), error => error.statusCode === 409);
  manager.stop();
  await manager.close();
  assert.equal(await readFile(filename, 'utf8'), lock);
  assert.equal(workers.length, 0);
});

async function abandonedIteration(runDir, status = 'running') {
  const folder = path.join(runDir, 'iteration-00000228');
  const report = { runId: 'abandoned-run', iteration: 228, status, promoted: false,
    startedAt: '2026-09-23T12:00:00.000Z', options: { games: 16 } };
  await mkdir(folder);
  await writeFile(path.join(runDir, 'run.json'), JSON.stringify({ version: 1, runId: report.runId, nextIteration: 229 }));
  await writeFile(path.join(folder, 'iteration.json'), JSON.stringify(report));
  await writeFile(path.join(folder, 'report.json'), JSON.stringify(report));
  await writeFile(path.join(runDir, 'latest.json'), JSON.stringify(report));
  return { folder, report };
}

async function exitedPid() {
  const child = spawn(process.execPath, ['-e', ''], { windowsHide: true });
  await once(child, 'exit');
  return child.pid;
}

test('abandoned process locks and running iteration recover together without changing replay or numbering', async t => {
  const { manager, runDir, workers } = await fixture(t);
  const { folder } = await abandonedIteration(runDir);
  const modelDir = `${manager.options.checkpoint}.selfplay-lock`;
  await mkdir(modelDir);
  const owner = { pid: await exitedPid(), token: 'dead-owner', createdAt: '2026-09-23T12:00:00.000Z' };
  for (const directory of [runDir, modelDir]) await writeFile(path.join(directory, '.selfplay.lock'), JSON.stringify(owner));
  await writeFile(path.join(runDir, 'replay.jsonl'), 'saved replay');
  await writeFile(path.join(folder, 'selfplay-001.json'), 'saved game');
  await writeFile(manager.options.checkpoint, 'saved checkpoint');
  const snapshots = await Promise.all([manager.snapshot(), manager.snapshot()]);
  for (const snapshot of snapshots) {
    assert.equal(snapshot.status.state, 'interrupted');
    assert.equal(snapshot.status.iteration, 228);
    assert.equal(snapshot.iterations[0].status, 'interrupted');
    assert.equal(snapshot.availability.available, true);
  }
  const saved = JSON.parse(await readFile(path.join(folder, 'report.json'), 'utf8'));
  assert.match(saved.error, /process exited/i);
  assert.equal(saved.interruption.previousStatus, 'running');
  assert.equal(JSON.parse(await readFile(path.join(runDir, 'latest.json'), 'utf8')).status, 'interrupted');
  assert.equal(JSON.parse(await readFile(path.join(runDir, 'run.json'), 'utf8')).nextIteration, 229);
  assert.equal(await readFile(path.join(runDir, 'replay.jsonl'), 'utf8'), 'saved replay');
  assert.equal(await readFile(path.join(folder, 'selfplay-001.json'), 'utf8'), 'saved game');
  assert.equal(await readFile(manager.options.checkpoint, 'utf8'), 'saved checkpoint');
  assert.equal(workers.length, 0);
  await manager.start({});
  assert.equal(workers.length, 1);
});

test('a live checkpoint owner blocks recovery even when the run directory lock is stale', async t => {
  const { manager, runDir } = await fixture(t);
  const { folder } = await abandonedIteration(runDir);
  const stale = JSON.stringify({ pid: await exitedPid(), token: 'stale' });
  await writeFile(path.join(runDir, '.selfplay.lock'), stale);
  const modelDir = `${manager.options.checkpoint}.selfplay-lock`;
  await mkdir(modelDir);
  await writeFile(path.join(modelDir, '.selfplay.lock'), JSON.stringify({ pid: process.pid, token: 'active-other-run' }));
  const snapshot = await manager.snapshot();
  assert.equal(snapshot.status.state, 'external');
  assert.equal(snapshot.availability.available, false);
  assert.equal(await readFile(path.join(runDir, '.selfplay.lock'), 'utf8'), stale);
  assert.equal(JSON.parse(await readFile(path.join(folder, 'report.json'), 'utf8')).status, 'running');
});

test('worker exit recovers only that invocation locks even while its server PID remains live', async t => {
  const { manager, runDir, calls, workers } = await fixture(t);
  await manager.start({});
  const { folder } = await abandonedIteration(runDir);
  const modelDir = `${manager.options.checkpoint}.selfplay-lock`;
  await mkdir(modelDir);
  for (const directory of [runDir, modelDir]) await writeFile(path.join(directory, '.selfplay.lock'),
    JSON.stringify({ pid: process.pid, token: calls[0].lockToken }));
  workers[0].emit('exit', 1);
  const snapshot = await manager.snapshot();
  assert.equal(snapshot.status.state, 'failed');
  assert.equal(snapshot.availability.available, true);
  assert.equal(JSON.parse(await readFile(path.join(folder, 'report.json'), 'utf8')).status, 'interrupted');
  await manager.start({});
});

test('orphaned evaluated reports retain arena evidence and disclose uncertain promotion', async t => {
  const { manager, runDir } = await fixture(t);
  const { folder, report } = await abandonedIteration(runDir, 'evaluated');
  report.arena = { decision: { promote: true } };
  report.candidateSha256 = 'candidate-hash';
  await writeFile(path.join(folder, 'report.json'), JSON.stringify(report));
  const snapshot = await manager.snapshot();
  assert.equal(snapshot.iterations[0].status, 'interrupted');
  const saved = (await manager.getIteration('iteration-00000228')).report;
  assert.equal(saved.arena.decision.promote, true);
  assert.equal(saved.candidateSha256, 'candidate-hash');
  assert.match(saved.error, /Promotion was in progress/);
});

test('recovery resynchronizes latest when the process exited between the two final report writes', async t => {
  const { manager, runDir } = await fixture(t);
  const { folder, report } = await abandonedIteration(runDir);
  report.status = 'complete';
  report.finishedAt = '2026-09-23T12:01:00.000Z';
  const original = JSON.stringify(report);
  await writeFile(path.join(folder, 'report.json'), original);
  const snapshot = await manager.snapshot();
  assert.equal(snapshot.status.state, 'completed');
  assert.equal(snapshot.availability.available, true);
  assert.deepEqual(JSON.parse(await readFile(path.join(runDir, 'latest.json'), 'utf8')), report);
  assert.equal(await readFile(path.join(folder, 'report.json'), 'utf8'), original);
});

test('a corrupt latest summary does not prevent recovering an intact running report', async t => {
  const { manager, runDir } = await fixture(t);
  await abandonedIteration(runDir);
  await writeFile(path.join(runDir, 'latest.json'), '{broken');
  const snapshot = await manager.snapshot();
  assert.equal(snapshot.status.state, 'interrupted');
  assert.equal(snapshot.availability.available, true);
  assert.equal(JSON.parse(await readFile(path.join(runDir, 'latest.json'), 'utf8')).status, 'interrupted');
});

test('interrupted iteration manifests and unreadable reports remain visible in saved history', async t => {
  const { manager, runDir } = await fixture(t);
  const interrupted = path.join(runDir, 'iteration-00000009');
  const broken = path.join(runDir, 'iteration-00000008');
  await mkdir(interrupted);
  await mkdir(broken);
  await writeFile(path.join(interrupted, 'iteration.json'), JSON.stringify({ iteration: 9, options: { steps: 5 } }));
  await writeFile(path.join(broken, 'report.json'), '{incomplete JSON');
  const snapshot = await manager.snapshot();
  assert.deepEqual(snapshot.iterations.map(item => item.iteration), [9, 8]);
  assert.equal(snapshot.iterations[0].status, 'incomplete');
  assert.equal(snapshot.iterations[1].status, 'unreadable');
  const detail = await manager.getIteration('iteration-00000009');
  assert.equal(detail.report.options.steps, 5);
  assert.deepEqual(detail.games, []);
  assert.equal(detail.log, '');
});

test('persisted self-play and arena games are listed and every replay position is legal', async t => {
  const { manager, runDir, workers } = await fixture(t);
  const { id, names, records, folder } = await persistedIteration(runDir);
  const snapshot = await manager.snapshot();
  assert.equal(snapshot.iterations[0].id, id);
  assert.equal(snapshot.iterations[0].status, 'complete');
  const iteration = await manager.getIteration(id);
  assert.equal(iteration.report.iteration, 7);
  assert.equal(iteration.games.length, 3);
  assert.deepEqual(new Set(iteration.games.map(game => game.kind)), new Set(['selfplay', 'arena']));
  assert.match(iteration.log, /step=2 loss=0.31/);
  for (const [index, gameId] of names.entries()) {
    const filename = path.join(folder, `${gameId}.json`), before = await readFile(filename, 'utf8');
    let position = records[index].initialPosition;
    for (let ply = 0; ply <= records[index].moves.length; ply++) {
      if (ply > 0) position = validateAction(position, records[index].moves[ply - 1].action);
      const replay = await manager.getGame(id, gameId, ply);
      assert.equal(replay.ply, ply);
      assert.equal(replay.totalPlies, records[index].moves.length);
      assert.equal(positionKey(replay.position), positionKey(position));
      assert.ok(Array.isArray(replay.active));
      assert.equal(typeof replay.isEvenTimeline, 'boolean');
    }
    assert.equal(await readFile(filename, 'utf8'), before, 'Review does not rewrite saved games.');
  }
  assert.equal(workers.length, 0, 'Browsing saved runs does not start model or training workers.');
});

test('missing runs and games, traversal and malformed replay indices are rejected', async t => {
  const { manager, runDir } = await fixture(t);
  const { id } = await persistedIteration(runDir);
  await assert.rejects(manager.getIteration('iteration-00000099'), error => error.statusCode === 404);
  await assert.rejects(manager.getGame(id, 'selfplay-999'), error => error.statusCode === 404);
  for (const name of ['../outside', '..\\outside', 'iteration-00000007/..', 'C:\\outside', 'iteration-7']) {
    await assert.rejects(manager.getIteration(name));
  }
  for (const name of ['../report', '..\\report', 'report', 'selfplay-001.json/..']) {
    await assert.rejects(manager.getGame(id, name, 0));
  }
  for (const ply of [-1, 1.5, 3, NaN]) await assert.rejects(manager.getGame(id, 'selfplay-001', ply));
});

test('replay refuses corrupted trajectories instead of displaying an unvalidated position', async t => {
  const { manager, runDir } = await fixture(t);
  const { id, folder, records } = await persistedIteration(runDir);
  const game = structuredClone(records[0]);
  game.moves[0].action = [[[0, 0, 3, 3], [0, 0, 4, 3]]];
  await writeFile(path.join(folder, 'selfplay-001.json'), JSON.stringify(game));
  await assert.rejects(manager.getGame(id, 'selfplay-001', 1));
});
