import test from 'node:test';
import assert from 'node:assert/strict';
import childProcess from 'node:child_process';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { analyze } from '../src/rust-engine.js';
import { DEFAULT_HEURISTICS } from '../src/heuristics.js';

function nativeProcess(t, onRequest) {
  const child = new EventEmitter();
  child.stdin = new PassThrough();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.exitCode = null;
  child.kills = 0;
  child.close = (code = 0) => {
    if (child.exitCode !== null) return;
    child.exitCode = code;
    queueMicrotask(() => child.emit('close', code));
  };
  child.kill = () => { child.kills++; child.close(1); };
  child.send = message => child.stdout.write(JSON.stringify(message) + '\n');
  let input = '';
  child.stdin.on('data', chunk => {
    input += chunk.toString();
    let end;
    while ((end = input.indexOf('\n')) >= 0) {
      const request = JSON.parse(input.slice(0, end));
      input = input.slice(end + 1);
      // Real subprocesses cannot reply within a synchronous stdin.write call.
      queueMicrotask(() => onRequest(request, child));
    }
  });
  child.stdin.on('finish', () => child.close());
  t.mock.method(childProcess, 'spawn', () => child);
  return child;
}

test('Rust adapter forwards history and settings, assembles NDJSON and closes its native session', async t => {
  const position = { board: [null, [null, [[1, 2]]]], action: 3 };
  const progress = [], requests = [];
  const child = nativeProcess(t, (request, process) => {
    requests.push(request);
    process.stdout.write('{"type":"progress",');
    process.stdout.write('"result":{"depth":1,"bestAction":[]}}\n');
    process.send({ type: 'result', result: { depth: 2, bestAction: [] } });
  });
  const result = await analyze(position, {
    timeMs: 0, threads: 4, maxNodes: 1200, heuristics: { materialWeight: 2 }, quiescenceDepth: 0, quiescenceWarmup: false,
    onProgress: value => progress.push(value),
  });
  assert.equal(result.backend, 'rust');
  assert.equal(result.depth, 2);
  assert.equal(progress[0].backend, 'rust');
  assert.deepEqual(requests[0].position, position);
  assert.equal(requests[0].type, 'analyze');
  assert.equal(requests[0].options.unlimitedTime, true);
  assert.equal(requests[0].options.threads, 4);
  assert.equal(requests[0].options.maxNodes, 1200);
  assert.equal(requests[0].options.quiescenceWarmup, false);
  assert.deepEqual(requests[0].options.heuristics, { ...DEFAULT_HEURISTICS, materialWeight: 2, quiescenceDepth: 0 });
  assert.equal(child.stdin.writableEnded, true);
  assert.equal(child.kills, 0);
});

test('Rust adapter sends cooperative cancellation and preserves the final native result', async t => {
  const received = [];
  nativeProcess(t, (request, child) => {
    received.push(request.type);
    if (request.type === 'cancel') child.send({ type: 'result', result: { stoppedReason: 'cancelled', bestAction: [] } });
  });
  const result = await analyze({ board: [], action: 0 }, { shouldStop: () => true });
  assert.deepEqual(received, ['analyze', 'cancel']);
  assert.equal(result.stoppedReason, 'cancelled');
});

test('Rust adapter kills a child when progress callbacks fail and removes lifecycle handlers', async t => {
  const listeners = process.listenerCount('exit');
  const child = nativeProcess(t, (_, child) => child.send({ type: 'progress', result: { depth: 1 } }));
  await assert.rejects(analyze({}, { onProgress: () => { throw new Error('consumer failed'); } }), /consumer failed/);
  assert.equal(child.kills, 1);
  assert.equal(process.listenerCount('exit'), listeners);
});

test('Rust adapter rejects corrupted stdout instead of returning a partial result', async t => {
  const child = nativeProcess(t, (_, child) => child.stdout.write('not json\n'));
  await assert.rejects(analyze({}), /invalid JSON/);
  assert.equal(child.kills, 1);
});

test('Rust adapter reports native errors and unexpected process exits', async t => {
  const child = nativeProcess(t, (_, child) => {
    child.stderr.write('invalid board dimensions');
    child.close(2);
  });
  await assert.rejects(analyze({}), /exited without a result \(2\).*invalid board dimensions/);
  assert.equal(child.kills, 0);
});

test('Rust adapter tells users how to build a missing executable', async t => {
  nativeProcess(t, (_, child) => child.emit('error', Object.assign(new Error('missing'), { code: 'ENOENT' })));
  await assert.rejects(analyze({}), /npm run build:classical/);
});

test('Rust adapter rejects invalid settings and circular input before launching a process', async t => {
  const spawn = t.mock.method(childProcess, 'spawn', () => { throw new Error('must not spawn'); });
  await assert.rejects(analyze({}, { heuristics: { materialWeight: 99 } }), /materialWeight must be/);
  const position = {};
  position.board = position;
  await assert.rejects(analyze(position), /circular/i);
  assert.equal(spawn.mock.calls.length, 0);
});
