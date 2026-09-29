import test from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { createApp } from '../src/server.js';
import { evaluateDetailed } from '../src/evaluate.js';
import { analyze } from '../src/search.js';

async function fixture(t) {
  const server = createApp();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const request = async (path, body) => {
    const response = await fetch(base + path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    return { status: response.status, data: await response.json() };
  };
  return { base, request };
}

test('heuristic metadata and position breakdown reflect the live revision without editing the game', async t => {
  const { request, base } = await fixture(t);
  const metadata = await request('/api/heuristics');
  assert.equal(metadata.status, 200);
  assert.ok(metadata.data.settings.length > 20);
  assert.equal(metadata.data.defaults.queenValue, 1150);
  assert.equal(metadata.data.defaults.quiescenceDepth, 2);
  for (const setting of metadata.data.settings) {
    assert.equal(metadata.data.defaults[setting.key], setting.default);
    assert.ok(setting.description && setting.group);
    assert.ok(setting.default >= setting.min && setting.default <= setting.max);
  }
  const script = await fetch(base + '/heuristics.js');
  assert.equal(script.status, 200);
  assert.match(script.headers.get('content-type'), /javascript/);
  const game = (await request('/api/game')).data;
  const response = await request('/api/evaluate', { revision: game.revision });
  assert.equal(response.status, 200);
  assert.equal(response.data.revision, game.revision);
  const expected = evaluateDetailed(game.position);
  for (const [key, value] of Object.entries(expected)) assert.equal(response.data.evaluation[key], value, key);
  assert.ok(response.data.evaluation.features.length > 10);
  assert.equal(response.data.evaluation.boards.length, 1);
  assert.deepEqual((await request('/api/game')).data, game);
  const moved = (await request('/api/move', { revision: game.revision, move: game.moves[0].raw })).data;
  assert.equal((await request('/api/evaluate', { revision: game.revision })).status, 409);
  const partial = await request('/api/evaluate', { revision: moved.revision });
  assert.equal(partial.status, 200);
  assert.equal(partial.data.evaluation.total, evaluateDetailed(moved.position).total);
});

test('changing a piece value changes the current-position breakdown and resetting restores it', async t => {
  const { request } = await fixture(t);
  const imported = await request('/api/import', { pgn: '[Size "4x4"]\n[3k/4/4/KQ2:0:1:w]' });
  assert.equal(imported.status, 200, imported.data.error);
  const revision = imported.data.revision;
  const initial = (await request('/api/evaluate', { revision })).data;
  const tuned = (await request('/api/evaluate', { revision, heuristics: { queenValue: 1500 } })).data;
  assert.equal(tuned.evaluation.material - initial.evaluation.material, 350);
  assert.equal(tuned.heuristics.queenValue, 1500);
  const muted = (await request('/api/evaluate', { revision, heuristics: {
    materialWeight: 0, activityWeight: 0, kingSafetyWeight: 0,
    temporalWeight: 0, timelinesWeight: 0, travelWeight: 0,
  } })).data;
  assert.equal(muted.evaluation.total, 0);
  assert.ok(muted.evaluation.features.every(feature => feature.value === 0));
  assert.deepEqual((await request('/api/evaluate', { revision })).data, initial);
  assert.equal((await request('/api/game')).data.revision, revision);
});

test('both tuning endpoints reject malformed or unsupported heuristic parameters', async t => {
  const { request } = await fixture(t);
  const { revision } = (await request('/api/game')).data;
  for (const heuristics of [null, [], 3, { unknownWeight: 1 }, { queenValue: '1500' },
    { queenValue: null }, { materialWeight: -1 }, { quiescenceDepth: 1.5 }, { aspirationWindow: 1000000 }]) {
    for (const path of ['/api/evaluate', '/api/analyze']) {
      const response = await request(path, { revision, heuristics });
      assert.equal(response.status, 400, `${path} accepted ${JSON.stringify(heuristics)}`);
      assert.ok(response.data.error);
    }
  }
});

test('HTTP analysis forwards a custom profile and its quiescence depth through workers', async t => {
  const { request } = await fixture(t);
  const initial = (await request('/api/game')).data;
  const heuristics = { queenValue: 1500, activityWeight: 1.5, quiescenceDepth: 0, aspirationWindow: 0, temporalMovePenalty: 0 };
  const options = { timeMs: 10000, maxDepth: 2, maxNodes: 100000, heuristics };
  const expected = analyze(initial.position, options);
  assert.equal(expected.depth, 2);
  for (const threads of [1, 2]) {
    const started = await request('/api/analyze', { ...options, threads });
    assert.equal(started.status, 202, started.data.error);
    let job;
    const deadline = Date.now() + 15000;
    do {
      await delay(20);
      job = (await request(`/api/analysis/${started.data.jobId}`)).data;
    } while (job.status === 'running' && Date.now() < deadline);
    assert.equal(job.status, 'done', job.error);
    assert.equal(job.result.depth, 2);
    assert.equal(job.result.score, expected.score);
    assert.equal(job.result.limits.quiescenceDepth, 0);
    for (const [key, value] of Object.entries(heuristics)) assert.equal(job.result.heuristics[key], value);
  }
  assert.deepEqual((await request('/api/game')).data.position, initial.position);
});
