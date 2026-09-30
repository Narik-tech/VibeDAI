import test from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { createApp } from '../src/server.js';
import { validateAction } from '../src/rules.js';

const NAME = 'Leela in a 5D Trenchcoat';
const TEMPORAL_FEN = '[Board "Custom"]\n[Size "4x4"]\n[3k/4/4/4:0:1:w]\n[4/2k1/2q1/K3:0:2:w]';

function mockRuntime(id, generation) {
  return {
    starts: 0, values: 0, policies: 0, closed: false, available: true,
    describe() { return { id, name: id === 'leela' ? NAME : 'Transformer', available: this.available, status: 'unloaded' }; },
    async start() {
      this.starts++;
      if (!this.available) throw new Error(`${id} checkpoint missing`);
      return { device: 'test', model: { architecture: id === 'leela' ? '5d-lc0-transfer-v1' : '5d-transformer-value-v1',
        checkpoint: `${id}.pt`, device: 'test', trainedSteps: 3, policyAvailable: true,
        policyTrainedSteps: 2, policyVersion: 1, runtimeGeneration: generation } };
    },
    async evaluate(positions, options) {
      assert.equal(options.runtimeGeneration, generation);
      assert(positions.every(position => Array.isArray(position.board)));
      this.values++;
      return { values: positions.map(() => id === 'leela' ? 25 : -40), runtimeGeneration: generation,
        context: positions.map(() => ({ truncated: false, frontierTruncated: false })) };
    },
    async orderMoves(position, moves, options) {
      assert.equal(options.runtimeGeneration, generation);
      assert.equal(options.withMetadata, true);
      assert(Array.isArray(position.board));
      assert(moves.every(move => Array.isArray(move) && move.length >= 2));
      this.policies++;
      return { scores: moves.map((_, index) => -index), runtimeGeneration: generation };
    },
    close() { this.closed = true; },
  };
}

async function fixture(t, overrides = {}) {
  const transformer = overrides.transformer ?? mockRuntime('transformer', 11);
  const leela = overrides.leela ?? mockRuntime('leela', 29);
  const training = { closed: false, close() { this.closed = true; } };
  const server = createApp({ transformerRuntime: transformer, leelaRuntime: leela, trainingManager: training });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  const request = async (route, body) => {
    const response = await fetch(base + route, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    return { status: response.status, data: await response.json() };
  };
  const wait = async id => {
    for (let index = 0; index < 500; index++) {
      const { data } = await request(`/api/analysis/${id}`);
      if (data.status !== 'running') return data;
      await delay(10);
    }
    throw new Error('Leela analysis did not finish.');
  };
  return { request, wait, transformer, leela, server, training };
}

test('engine catalog keeps classical and transformer visible without loading archived Leela', async t => {
  const { request, transformer, leela } = await fixture(t);
  transformer.available = false;
  const { data, status } = await request('/api/engines');
  assert.equal(status, 200);
  assert.deepEqual(data.engines.map(engine => engine.id), ['classical', 'transformer']);
  assert.equal(data.engines.find(engine => engine.id === 'classical').available, true);
  assert.equal(data.engines.find(engine => engine.id === 'transformer').available, false);
  assert.equal(transformer.starts, 0);
  assert.equal(leela.starts, 0);
});

test('Leela routes value and policy independently, validates time travel, and supports Play best', async t => {
  const { request, wait, transformer, leela } = await fixture(t);
  const imported = await request('/api/import', { pgn: TEMPORAL_FEN });
  assert.equal(imported.status, 200, imported.data.error);
  const game = imported.data;
  const created = await request('/api/analyze', { engine: 'leela', maxDepth: 1, timeMs: 2000, quiescenceDepth: 0 });
  assert.equal(created.status, 202, created.data.error);
  const job = await wait(created.data.jobId);
  assert.equal(job.status, 'done', job.error);
  assert.equal(job.result.engine, 'leela');
  assert.equal(job.result.model.architecture, '5d-lc0-transfer-v1');
  assert.equal(job.result.model.runtimeGeneration, 29);
  assert.equal(job.result.completed, true);
  assert(leela.values > 0 && leela.policies > 0);
  assert.equal(job.result.policyCalls, leela.policies);
  assert.equal(transformer.starts + transformer.values + transformer.policies, 0);
  assert(job.result.bestAction.some(([from, to]) => from[0] !== to[0] || from[1] !== to[1]));
  let continuation = game.position;
  for (const action of job.result.pv) continuation = validateAction(continuation, action);
  assert.deepEqual((await request('/api/game')).data.position, game.position);
  const played = await request('/api/play', { jobId: created.data.jobId, revision: game.revision });
  assert.equal(played.status, 200, played.data.error);
  assert.deepEqual(played.data.position, validateAction(game.position, job.result.bestAction));
  const leelaCounts = [leela.starts, leela.values, leela.policies];
  const second = await request('/api/analyze', { engine: 'transformer', maxDepth: 1, timeMs: 2000, quiescenceDepth: 0 });
  assert.equal(second.status, 202, second.data.error);
  const transformerJob = await wait(second.data.jobId);
  assert.equal(transformerJob.status, 'done', transformerJob.error);
  assert.equal(transformerJob.result.engine, 'transformer');
  assert.equal(transformerJob.result.model.architecture, '5d-transformer-value-v1');
  assert(transformer.values > 0 && transformer.policies > 0);
  assert.deepEqual([leela.starts, leela.values, leela.policies], leelaCounts);
});

test('missing Leela checkpoint and inference failure never fall back to Transformer or classical', async t => {
  const { request, wait, transformer, leela } = await fixture(t);
  leela.available = false;
  const missing = await request('/api/analyze', { engine: 'leela' });
  assert.equal(missing.status, 400);
  assert.match(missing.data.error, /leela checkpoint missing/);
  assert.equal(transformer.starts, 0);
  leela.available = true;
  leela.evaluate = async () => { throw new Error('Leela CUDA fixture failure'); };
  await request('/api/import', { pgn: TEMPORAL_FEN });
  const created = await request('/api/analyze', { engine: 'leela', maxDepth: 1, timeMs: 2000 });
  assert.equal(created.status, 202, created.data.error);
  const failed = await wait(created.data.jobId);
  assert.equal(failed.status, 'error');
  assert.match(failed.error, /Leela CUDA fixture failure/);
  assert.equal(failed.result, undefined);
  assert.equal(transformer.starts + transformer.values + transformer.policies, 0);
  const classical = await request('/api/analyze', { engine: 'classical', maxDepth: 1, timeMs: 200 });
  assert.equal(classical.status, 202, classical.data.error);
  assert.equal((await wait(classical.data.jobId)).result.engine, 'classical');
});

test('Leela accepts neural depth bounds and server shutdown closes both owned runtimes', async t => {
  const { request, wait, transformer, leela, server, training } = await fixture(t);
  for (const maxDepth of [-1, 65]) assert.equal((await request('/api/analyze', { engine: 'leela', maxDepth })).status, 400);
  for (const maxDepth of [0, 64]) {
    const created = await request('/api/analyze', { engine: 'leela', maxDepth, maxNodes: 1, timeMs: 1000 });
    assert.equal(created.status, 202, created.data.error);
    const job = await wait(created.data.jobId);
    assert.equal(job.status, 'done', job.error);
    assert.equal(job.result.engine, 'leela');
    assert.equal(job.result.limits.maxDepth, maxDepth);
  }
  await new Promise(resolve => server.close(resolve));
  assert.equal(transformer.closed, true);
  assert.equal(leela.closed, true);
  assert.equal(training.closed, true);
});
