import test from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { createApp } from '../src/server.js';

async function fixture(t) {
  const server = createApp();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  const request = async (path, body) => {
    const response = await fetch(base + path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    return { status: response.status, data: await response.json() };
  };
  return { base, request };
}

test('HTTP game flow supports legal moves, stale revision rejection and undo', async t => {
  const { request } = await fixture(t);
  const initial = (await request('/api/game')).data;
  assert.equal(initial.moves.length, 20);
  const move = initial.moves.find(m => m.raw[0][2] === 1 && m.raw[0][3] === 4 && m.raw[1][2] === 3);
  const played = await request('/api/move', { move: move.raw, revision: initial.revision });
  assert.equal(played.status, 200);
  assert.equal(played.data.canSubmit, true);
  const stale = await request('/api/submit', { revision: initial.revision });
  assert.equal(stale.status, 409);
  const submitted = await request('/api/submit', { revision: played.data.revision });
  assert.equal(submitted.data.position.action, 1);
  const undone = await request('/api/undo', {});
  assert.deepEqual(undone.data.position.board, initial.position.board);
});

test('analysis runs separately, produces a legal action, and can be played', async t => {
  const { request } = await fixture(t);
  const initial = (await request('/api/game')).data;
  const created = await request('/api/analyze', { timeMs: 150, maxDepth: 2, quiescenceDepth: 0 });
  assert.equal(created.status, 202);
  assert.equal((await request('/api/game')).status, 200);
  let job;
  const deadline = Date.now() + 8000;
  do {
    await delay(20);
    job = (await request(`/api/analysis/${created.data.jobId}`)).data;
  } while (job.status === 'running' && Date.now() < deadline);
  assert.equal(job.status, 'done', job.error);
  assert.equal(job.result.backend, 'rust');
  assert.ok(Array.isArray(job.result.bestAction));
  assert.equal(job.result.limits.cacheMemoryMb, 128);
  assert.equal(job.result.limits.maxNodes, 2000000);
  const played = await request('/api/play', { jobId: created.data.jobId, revision: initial.revision });
  assert.equal(played.status, 200, played.data.error);
  assert.equal(played.data.position.action, 1);
});

test('local server rejects foreign origins and invalid analysis budgets', async t => {
  const { base, request } = await fixture(t);
  const denied = await fetch(base + '/api/new', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://example.com' }, body: '{}',
  });
  assert.equal(denied.status, 403);
  assert.equal((await request('/api/analyze', { timeMs: -1 })).status, 400);
  assert.equal((await request('/api/new', { variant: 'does-not-exist' })).status, 400);
});

test('analysis forwards node and cache memory budgets to the worker, including disabling the cache', async t => {
  const { request } = await fixture(t);
  for (const cacheMemoryMb of [4096, 0]) {
    const created = await request('/api/analyze', { timeMs: 1000, maxDepth: 1, quiescenceDepth: 0, maxNodes: 1000000000, cacheMemoryMb });
    assert.equal(created.status, 202, created.data.error);
    let job;
    const deadline = Date.now() + 8000;
    do {
      await delay(20);
      job = (await request(`/api/analysis/${created.data.jobId}`)).data;
    } while (job.status === 'running' && Date.now() < deadline);
    assert.equal(job.status, 'done', job.error);
    assert.equal(job.result.limits.maxNodes, 1000000000);
    assert.equal(job.result.limits.cacheMemoryMb, cacheMemoryMb);
    assert.ok(Array.isArray(job.result.bestAction));
    if (cacheMemoryMb === 0) {
      assert.equal(job.result.tableEntries, 0);
      assert.equal(job.result.cacheMemoryBytes, 0);
    } else {
      assert.ok(job.result.tableEntries > 0);
      assert.ok(job.result.cacheMemoryBytes > 0);
      assert.ok(job.result.cacheMemoryBytes <= cacheMemoryMb * 1024 * 1024);
    }
  }
});

test('analysis rejects out-of-range and malformed resource budgets', async t => {
  const { request } = await fixture(t);
  for (const timeMs of [-1, 1, 49, 0.5, 120001, null, true, '', []]) {
    const response = await request('/api/analyze', { timeMs });
    assert.equal(response.status, 400, `Accepted timeMs ${JSON.stringify(timeMs)}`);
    assert.match(response.data.error, /Think time must be/);
  }
  for (const maxNodes of [0, -1, 1.5, 1000000001, null, true, '', []]) {
    const response = await request('/api/analyze', { maxNodes });
    assert.equal(response.status, 400, `Accepted maxNodes ${JSON.stringify(maxNodes)}`);
    assert.match(response.data.error, /Node budget must be an integer/);
  }
  for (const cacheMemoryMb of [-1, 0.5, 4097, null, true, '', []]) {
    const response = await request('/api/analyze', { cacheMemoryMb });
    assert.equal(response.status, 400, `Accepted cacheMemoryMb ${JSON.stringify(cacheMemoryMb)}`);
    assert.match(response.data.error, /Cache memory must be an integer/);
  }
  for (const threads of [0, -1, 1.5, 17, null, true, '', []]) {
    const response = await request('/api/analyze', { threads });
    assert.equal(response.status, 400, `Accepted threads ${JSON.stringify(threads)}`);
    assert.match(response.data.error, /Search threads must be an integer/);
  }
});

test('parallel analysis accepts a thread count and keeps node and cache limits global', async t => {
  const { request } = await fixture(t);
  const created = await request('/api/analyze', {
    timeMs: 1500, maxDepth: 2, quiescenceDepth: 0, threads: 2, maxNodes: 10000, cacheMemoryMb: 16,
  });
  assert.equal(created.status, 202, created.data.error);
  let job;
  const deadline = Date.now() + 8000;
  do {
    await delay(20);
    job = (await request(`/api/analysis/${created.data.jobId}`)).data;
  } while (job.status === 'running' && Date.now() < deadline);
  assert.equal(job.status, 'done', job.error);
  assert.ok(Array.isArray(job.result.bestAction));
  assert.equal(job.result.limits.threads, 2);
  assert.equal(job.result.limits.maxNodes, 10000);
  assert.equal(job.result.limits.cacheMemoryMb, 16);
  assert.ok(job.result.nodes <= 10000);
  assert.ok(job.result.cacheMemoryBytes <= 16 * 1024 * 1024);
});

for (const timeMs of [60000, 0]) test(`stopping a search with think time ${timeMs} preserves the position`, async t => {
  const { request } = await fixture(t);
  const initial = (await request('/api/game')).data;
  const created = await request('/api/analyze', { timeMs, maxDepth: 16 });
  assert.equal(created.status, 202, created.data.error);
  await request(`/api/analysis/${created.data.jobId}/stop`, {});
  let job;
  const deadline = Date.now() + 3000;
  do {
    await delay(20);
    job = (await request(`/api/analysis/${created.data.jobId}`)).data;
  } while (job.status === 'running' && Date.now() < deadline);
  assert.equal(job.status, 'done', job.error);
  assert.equal(job.result.stoppedReason, 'cancelled');
  assert.notEqual(job.result.status, 'checkmate');
  const unchanged = (await request('/api/game')).data;
  assert.equal(unchanged.revision, initial.revision);
  assert.deepEqual(unchanged.position, initial.position);
});

test('zero think time searches without a deadline and still respects node limits', async t => {
  const { request } = await fixture(t);
  for (const threads of [1, 2]) {
    const created = await request('/api/analyze', {
      timeMs: 0, threads, maxDepth: 16, maxNodes: 1000, quiescenceDepth: 0,
    });
    assert.equal(created.status, 202, created.data.error);
    let job;
    const deadline = Date.now() + 8000;
    do {
      await delay(20);
      job = (await request(`/api/analysis/${created.data.jobId}`)).data;
    } while (job.status === 'running' && Date.now() < deadline);
    assert.equal(job.status, 'done', job.error);
    assert.equal(job.result.limits.timeMs, 0);
    assert.equal(job.result.stoppedReason, 'nodes');
    assert.equal(job.result.nodes, 1000);
    assert.ok(Array.isArray(job.result.bestAction));
  }
});

test('malformed imports leave the live position intact and bare FEN imports stay custom', async t => {
  const { request } = await fixture(t);
  const initial = (await request('/api/game')).data;
  const invalid = await request('/api/import', { pgn: '1. e4oops' });
  assert.equal(invalid.status, 400);
  const unchanged = (await request('/api/game')).data;
  assert.equal(unchanged.revision, initial.revision);
  assert.deepEqual(unchanged.position, initial.position);
  const custom = await request('/api/import', { pgn: '[Size "4x4"]\n[3k/4/4/K3:0:1:w]' });
  assert.equal(custom.status, 200, custom.data.error);
  assert.equal(custom.data.position.board[0][0].length, 4);
  assert.match(custom.data.pgn, /\[Board "Custom"\]/);
});
