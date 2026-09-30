import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createApp } from '../src/server.js';
import { TrainingManager } from '../src/training-manager.js';
import { createPosition, formatAction, generateActions, positionKey, validateAction } from '../src/rules.js';

test('invalid training environment leaves classical gameplay available and reports a training setup error', async () => {
  const script = `
    import assert from 'node:assert/strict';
    const { createApp } = await import(${JSON.stringify(new URL('../src/server.js', import.meta.url).href)});
    const app = createApp();
    await new Promise(resolve => app.listen(0, '127.0.0.1', resolve));
    const base = 'http://127.0.0.1:' + app.address().port;
    try {
      assert.equal((await fetch(base + '/api/game')).status, 200);
      const response = await fetch(base + '/api/training');
      assert.equal(response.status, 200);
      const data = await response.json();
      assert.equal(data.availability.available, false);
      assert.match(data.availability.reason, /Invalid training configuration/);
    } finally { await new Promise(resolve => app.close(resolve)); }
  `;
  for (const environment of [
    { TRANSFORMER_DEVICE: 'invalid-device' },
    { TRANSFORMER_DEVICE: 'auto', TRANSFORMER_CHECKPOINT: path.resolve('artifacts/transformer/selfplay/replay.jsonl') },
  ]) {
    await promisify(execFile)(process.execPath, ['--input-type=module', '-e', script], {
      env: { ...process.env, ...environment }, windowsHide: true,
    });
  }
});

async function fixture(t) {
  const directory = await mkdtemp(path.join(tmpdir(), 'vibe-server-training-'));
  const runDir = path.join(directory, 'selfplay');
  await mkdir(runDir);
  const workers = [], invocations = [];
  const manager = new TrainingManager({
    runDir, checkpoint: path.join(directory, 'model.pt'), python: path.join(directory, 'python'),
    leelaConfig: { checkpoint: path.join(directory, 'leela.pt'), runDir: path.join(directory, 'leela-selfplay'),
      seedData: path.join(directory, 'curriculum.jsonl'), python: path.join(directory, 'python'), device: 'cpu' },
    availability: async () => ({ available: true }),
    workerFactory: invocation => {
      const worker = new EventEmitter();
      worker.postMessage = message => {
        if (message.type === 'stop') queueMicrotask(() => {
          worker.emit('message', { type: 'complete', state: 'interrupted' });
          worker.emit('exit', 0);
        });
      };
      worker.terminate = async () => { worker.emit('exit', 1); return 1; };
      invocations.push(invocation);
      workers.push(worker);
      return worker;
    },
  });
  const server = createApp({ trainingManager: manager });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => {
    await new Promise(resolve => server.close(resolve));
    await manager.close();
    await rm(directory, { recursive: true, force: true });
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  const request = async (route, body) => {
    const response = await fetch(base + route, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    return { status: response.status, data: await response.json() };
  };
  return { server, manager, runDir, request, base, workers, invocations };
}

async function writeGame(runDir) {
  const id = 'iteration-00000003', folder = path.join(runDir, id);
  await mkdir(folder);
  const initialPosition = createPosition();
  const actions = generateActions(initialPosition);
  let action;
  try { action = actions.next().value.moves; }
  finally { actions.return(); }
  const finalPosition = validateAction(initialPosition, action);
  const game = {
    gameId: 'fixture:3:1', startId: 'standard', initialPosition, finalPosition,
    initialKey: positionKey(initialPosition), finalKey: positionKey(finalPosition),
    result: 'UNFINISHED', valid: true, reason: 'ply-limit', plies: 1, samples: 1,
    moves: [{ ply: 0, color: 0, action, notation: formatAction(initialPosition, action),
      beforeKey: positionKey(initialPosition), afterKey: positionKey(finalPosition),
      search: { score: 125, completed: true } }],
  };
  await writeFile(path.join(folder, 'report.json'), JSON.stringify({ iteration: 3, status: 'complete', promoted: false }));
  await writeFile(path.join(folder, 'selfplay-001.json'), JSON.stringify(game));
  return { id, game };
}

test('training HTTP flow applies editable parameters, reports progress and accepts graceful stop', async t => {
  const { request, workers, invocations } = await fixture(t);
  const initial = await request('/api/training');
  assert.equal(initial.status, 200);
  assert.equal(initial.data.availability.available, true);
  assert.equal(initial.data.defaults.iterations, 0);
  assert.equal(initial.data.defaults.gameConcurrency, 3);
  assert.equal(initial.data.defaults.arenaConcurrency, 2);
  assert.deepEqual(initial.data.iterations, []);
  const started = await request('/api/training/start', { options: {
    iterations: 2, games: 3, gameConcurrency: 2, arenaConcurrency: 4, steps: 17, batchSize: 4, learningRate: 0.002, device: 'cpu', maxDepth: 0,
  } });
  assert.equal(started.status, 202, started.data.error);
  assert.equal(invocations.length, 1);
  assert.equal(invocations[0].options.steps, 17);
  assert.equal(invocations[0].options.gameConcurrency, 2);
  assert.equal(invocations[0].options.arenaConcurrency, 4);
  assert.equal(invocations[0].options.learningRate, 0.002);
  assert.equal(invocations[0].options.maxDepth, 0, 'Dynamic depth reaches the training worker unchanged.');
  assert.equal((await request('/api/training/start', { options: {} })).status, 409);
  workers[0].emit('message', { type: 'event', event: { event: 'training-start', iteration: 2, steps: 17 } });
  const running = (await request('/api/training')).data;
  assert.equal(running.status.iteration, 2);
  assert.ok(running.status.events.some(event => event.event === 'training-start'));
  const stopped = await request('/api/training/stop', {});
  assert.equal(stopped.status, 200);
  assert.equal(Atomics.load(new Int32Array(invocations[0].cancelBuffer), 0), 1);
  assert.equal((await request('/api/training')).data.status.state, 'interrupted');
});

test('training HTTP validation rejects untrusted options and foreign origins before launching a worker', async t => {
  const { request, base, workers } = await fixture(t);
  for (const options of [
    { python: 'cmd.exe' }, { checkpoint: '../outside.pt' }, { runDir: '../outside' },
    { games: true }, { games: '2' }, { games: 0 }, { learningRate: 0 },
    { gameConcurrency: '2' }, { gameConcurrency: 0 }, { gameConcurrency: 9 }, { gameConcurrency: 1.5 },
    { arenaConcurrency: '2' }, { arenaConcurrency: 0 }, { arenaConcurrency: 9 }, { arenaConcurrency: 1.5 },
    { arenaPairs: 1, minPairs: 2 }, { unknown: 3 },
  ]) {
    const response = await request('/api/training/start', { options });
    assert.equal(response.status, 400, `Accepted ${JSON.stringify(options)}`);
    assert.ok(response.data.error);
  }
  const denied = await fetch(base + '/api/training/start', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://example.com' },
    body: JSON.stringify({ options: {} }),
  });
  assert.equal(denied.status, 403);
  assert.equal(workers.length, 0);
});

test('Leela training HTTP flow selects its checkpoint and namespaces saved game review', async t => {
  const { manager, request, invocations, runDir } = await fixture(t);
  await writeFile(manager.leelaOptions.checkpoint, 'trained LCZero transfer checkpoint');
  await mkdir(manager.leelaOptions.runDir);
  const legacy = await writeGame(runDir);
  const leela = await writeGame(manager.leelaOptions.runDir);
  const snapshot = (await request('/api/training')).data;
  assert.equal(snapshot.leelaAvailability.available, true);
  assert.equal(snapshot.leelaModel.name, 'Leela in a 5D Trenchcoat');
  assert.equal(snapshot.leelaModel.available, true);
  assert.equal(snapshot.leelaDefaults.model, 'leela');
  assert.equal(snapshot.leelaDefaults.batchSize, 4);
  assert.deepEqual(new Set(snapshot.iterations.map(item => item.id)), new Set([legacy.id, `leela__${leela.id}`]));
  const detail = await request(`/api/training/iterations/leela__${leela.id}`);
  assert.equal(detail.status, 200);
  assert.equal(detail.data.games.length, 1);
  const game = await request(`/api/training/iterations/leela__${leela.id}/games/selfplay-001?ply=1`);
  assert.equal(game.status, 200);
  assert.equal(positionKey(game.data.position), leela.game.finalKey);
  const started = await request('/api/training/start', { options: { model: 'leela', steps: 3 } });
  assert.equal(started.status, 202, started.data.error);
  assert.equal(started.data.model, 'leela');
  assert.equal(invocations[0].mode, 'selfplay');
  assert.equal(invocations[0].options.checkpoint, manager.leelaOptions.checkpoint);
  assert.equal(invocations[0].options.runDir, manager.leelaOptions.runDir);
  assert.equal(invocations[0].options.seedData, manager.leelaOptions.seedData);
  assert.equal(invocations[0].options.steps, 3);
  assert.equal(invocations[0].options.batchSize, 4);
  assert.equal((await request('/api/training/start', { options: { model: 'current' } })).status, 409);
  assert.equal((await request('/api/training/stop', {})).status, 200);
  assert.equal((await request('/api/training')).data.status.state, 'interrupted');
});

test('fresh 20M HTTP flow validates settings, starts data generation and shares the stop control', async t => {
  const { request, workers, invocations, base } = await fixture(t);
  const snapshot = (await request('/api/training')).data;
  assert.equal(snapshot.freshAvailability.available, true);
  assert.equal(snapshot.freshDefaults.batchSize, 1);
  assert.equal(snapshot.model20m.available, false);
  for (const options of [{ checkpoint: 'outside.pt' }, { python: 'cmd.exe' }, { resume: 'old.pt' }, { samples: 0 }, { maxTokens: 15 }, { device: 'shell' }]) {
    const rejected = await request('/api/training/fresh/start', { options });
    assert.equal(rejected.status, 400, JSON.stringify(options));
  }
  assert.equal((await fetch(base + '/api/training/fresh/start', { method: 'POST',
    headers: { 'Content-Type': 'application/json', Origin: 'https://example.com' }, body: JSON.stringify({ options: {} }) })).status, 403);
  const started = await request('/api/training/fresh/start', { options: { steps: 5, samples: 32, maxTokens: 512, device: 'cpu' } });
  assert.equal(started.status, 202, started.data.error);
  assert.equal(started.data.mode, 'fresh20m');
  assert.equal(invocations[0].mode, 'fresh20m');
  assert.equal(invocations[0].options.steps, 5);
  assert.equal((await request('/api/training/start', { options: {} })).status, 409);
  workers[0].emit('message', { type: 'event', event: { event: 'data-progress', samples: 16, total: 32 } });
  assert.equal((await request('/api/training')).data.status.phase, 'data');
  assert.equal((await request('/api/training/stop', {})).status, 200);
  assert.equal((await request('/api/training')).data.status.state, 'interrupted');
});

test('saved game browsing returns legal replay positions without touching the analysis game', async t => {
  const { request, runDir, workers } = await fixture(t);
  const { id, game } = await writeGame(runDir);
  const initial = (await request('/api/game')).data;
  const move = initial.moves[0];
  const pending = (await request('/api/move', { revision: initial.revision, move: move.raw })).data;
  assert.equal(pending.pending.length, 1);
  const list = await request('/api/training');
  assert.equal(list.data.iterations[0].id, id);
  const detail = await request(`/api/training/iterations/${id}`);
  assert.equal(detail.status, 200);
  assert.equal(detail.data.games.length, 1);
  for (const ply of [0, 1, 0]) {
    const replay = await request(`/api/training/iterations/${id}/games/selfplay-001?ply=${ply}`);
    assert.equal(replay.status, 200, replay.data.error);
    assert.equal(replay.data.ply, ply);
    assert.equal(positionKey(replay.data.position), positionKey(ply ? game.finalPosition : game.initialPosition));
  }
  const unchanged = (await request('/api/game')).data;
  assert.equal(unchanged.revision, pending.revision);
  assert.deepEqual(unchanged.position, pending.position);
  assert.deepEqual(unchanged.pending, pending.pending);
  assert.equal(unchanged.pgn, pending.pgn);
  assert.equal(workers.length, 0);
});

test('training HTTP routes return errors for missing artifacts and invalid replay paths or plies', async t => {
  const { request, runDir } = await fixture(t);
  const { id } = await writeGame(runDir);
  assert.equal((await request('/api/training/iterations/iteration-00000099')).status, 404);
  assert.equal((await request(`/api/training/iterations/${id}/games/selfplay-999`)).status, 404);
  for (const route of [
    '/api/training/iterations/..%5Coutside',
    `/api/training/iterations/${id}/games/..%5Creport`,
    `/api/training/iterations/${id}/games/report`,
    `/api/training/iterations/${id}/games/selfplay-001?ply=-1`,
    `/api/training/iterations/${id}/games/selfplay-001?ply=1.5`,
    `/api/training/iterations/${id}/games/selfplay-001?ply=two`,
    `/api/training/iterations/${id}/games/selfplay-001?ply=2`,
  ]) {
    const response = await request(route);
    assert.ok([400, 404].includes(response.status), `Accepted ${route}`);
  }
});
