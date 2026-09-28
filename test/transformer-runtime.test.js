import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter, once } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtemp, writeFile, utimes, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { TransformerRuntime, forwardInference } from '../src/transformer-runtime.js';
import { createApp } from '../src/server.js';

async function fixture(t, options = {}) {
  const directory = await mkdtemp(path.join(tmpdir(), 'vibe-transformer-runtime-'));
  const checkpoint = path.join(directory, 'model.pt');
  await writeFile(checkpoint, 'test checkpoint');
  const children = [];
  const runtime = new TransformerRuntime({
    python:process.execPath, checkpoint, startupMs:1000, requestMs:1000, ...options,
    spawnProcess() {
      const child = new EventEmitter();
      child.stdin = new PassThrough(); child.stdout = new PassThrough(); child.stderr = new PassThrough();
      child.stdin.on('data', data => child.emit('request', JSON.parse(data.toString())));
      child.kill = () => { child.killed = true; queueMicrotask(() => child.emit('exit', 0)); };
      child.respond = message => child.stdout.write(`${JSON.stringify(message)}\n`);
      children.push(child);
      return child;
    },
  });
  t.after(async () => { runtime.close(); await rm(directory, {recursive:true, force:true}); });
  async function ready(policy = false) {
    const starting = runtime.start();
    children.at(-1).respond({ready:true, device:'cpu', model:{trainedSteps:1,
      ...(policy ? {policyAvailable:true, policyTrainedSteps:1, policyVersion:1} : {})}});
    await starting;
    return children.at(-1);
  }
  return {runtime, children, checkpoint, ready};
}

test('closing during shared startup rejects every caller and kills its process', async t => {
  const {runtime, children} = await fixture(t);
  const first = assert.rejects(runtime.start(), /stopped/);
  const second = assert.rejects(runtime.start(), /stopped/);
  assert.equal(children.length, 1);
  runtime.close();
  await Promise.all([first, second]);
  assert.equal(children[0].killed, true);
  assert.equal(runtime.starting, null);
  assert.equal(runtime.state, 'closed');
  await assert.rejects(runtime.start(), /closed/);
});

test('malformed protocol objects and ready messages fail without uncaught callbacks', async t => {
  for (const message of [null, [], {ready:'yes'}, {ready:true, device:'cpu', model:[]}, {id:1, values:[0]}]) {
    const {runtime, children} = await fixture(t);
    const failure = assert.rejects(runtime.start(), /invalid/);
    children[0].respond(message);
    await failure;
    assert.equal(runtime.state, 'error');
    assert.equal(children[0].killed, true);
  }
});

test('closing between ready and evaluate continuation leaves no pending requests', async t => {
  const {runtime, ready} = await fixture(t);
  await ready();
  const evaluation = assert.rejects(runtime.evaluate([{}]), /stopped/);
  runtime.close();
  await evaluation;
  assert.equal(runtime.pending.size, 0);
});

test('invalid evaluation shape stops the process and rejects every pending request', async t => {
  const {runtime, ready} = await fixture(t);
  const child = await ready();
  const next = once(child, 'request');
  const evaluation = assert.rejects(runtime.evaluate([{}]), /invalid evaluations/);
  const [request] = await next;
  child.respond({id:request.id, values:[1], context:{truncated:true}});
  await evaluation;
  assert.equal(runtime.pending.size, 0);
  assert.equal(child.killed, true);
});

test('checkpoint reload rejects old requests and ignores old ready/exit messages', async t => {
  const {runtime, children, checkpoint, ready} = await fixture(t);
  const old = await ready();
  const next = once(old, 'request');
  const oldEvaluation = assert.rejects(runtime.evaluate([{}]), /Checkpoint changed/);
  await next;
  const changed = new Date(Date.now() + 10000);
  await utimes(checkpoint, changed, changed);
  const starting = runtime.start();
  await oldEvaluation;
  assert.equal(children.length, 2);
  assert.equal(old.killed, true);
  old.respond({ready:true, device:'stale', model:{trainedSteps:1}});
  children[1].respond({ready:true, device:'cpu', model:{trainedSteps:2}});
  await starting;
  old.emit('exit', 1);
  assert.equal(runtime.info.device, 'cpu');
  assert.equal(runtime.info.model.trainedSteps, 2);
  const nextRequest = once(children[1], 'request');
  const evaluated = runtime.evaluate([{}]);
  const [request] = await nextRequest;
  children[1].respond({id:request.id, values:[42]});
  assert.deepEqual((await evaluated).values, [42]);
});

test('inference timeout clears pending requests and allows a fresh process', async t => {
  const {runtime, children, ready} = await fixture(t, {requestMs:20});
  await ready();
  await assert.rejects(runtime.evaluate([{}]), /timed out/);
  assert.equal(runtime.pending.size, 0);
  assert.equal(children[0].killed, true);
  await ready();
  assert.equal(children.length, 2);
  assert.equal(runtime.state, 'ready');
});

test('unexpected process exit rejects startup and active inference without waiting for timeouts', async t => {
  const { runtime, children, ready } = await fixture(t);
  const startup = assert.rejects(runtime.start(), /Transformer exited \(SIGKILL\)/);
  children[0].emit('exit', null, 'SIGKILL');
  await startup;
  const child = await ready();
  const next = once(child, 'request');
  const evaluation = assert.rejects(runtime.evaluate([{}]), /Transformer exited \(9\)/);
  await next;
  child.emit('exit', 9, null);
  await evaluation;
  assert.equal(runtime.pending.size, 0);
  assert.equal(runtime.state, 'error');
});

test('output pipe errors reject inference and stop the service without an uncaught error', async t => {
  for (const stream of ['stdout', 'stderr']) {
    const { runtime, ready } = await fixture(t);
    const child = await ready();
    const next = once(child, 'request');
    const evaluation = assert.rejects(runtime.evaluate([{}]), /output pipe failed/);
    await next;
    child[stream].emit('error', new Error('output pipe failed'));
    await evaluation;
    assert.equal(runtime.pending.size, 0);
    assert.equal(child.killed, true);
  }
});

test('server close interrupts an HTTP request waiting for model startup', async t => {
  const {runtime} = await fixture(t);
  let didStart;
  const started = new Promise(resolve => { didStart = resolve; });
  const start = runtime.start.bind(runtime);
  runtime.start = () => { const promise = start(); didStart(); return promise; };
  const server = createApp({transformerRuntime:runtime});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  const request = fetch(`http://127.0.0.1:${server.address().port}/api/analyze`, {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({engine:'transformer'}),
  });
  await started;
  const closed = new Promise(resolve => server.close(resolve));
  const response = await request;
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /stopped/);
  await closed;
  assert.equal(runtime.closed, true);
});

test('inference completion after worker disposal does not create an unhandled rejection', async () => {
  await forwardInference({postMessage() { throw new Error('disposed'); }}, {evaluate:async () => ({values:[0]})}, {type:'evaluate', id:1, positions:[{}]});
});

test('legacy models bypass policy inference while trained heads forward aligned scores', async t => {
  const {runtime, ready} = await fixture(t);
  const child = await ready();
  assert.equal(await runtime.orderMoves({}, [[[0], [1]]]), null);
  assert.equal(runtime.pending.size, 0);
  runtime.stopProcess();
  const trained = await ready(true);
  const next = once(trained, 'request');
  const ordered = runtime.orderMoves({board:[]}, [[[0], [1]], [[1], [2]]]);
  const [request] = await next;
  assert.equal(request.type, 'policy');
  assert.equal(request.moves.length, 2);
  trained.respond({id:request.id, scores:[-2.5, 4]});
  assert.deepEqual(await ordered, [-2.5, 4]);
  assert.equal(child.killed, true);
});

test('invalid policy reply shape fails closed and legacy forwarding reports unavailable', async t => {
  for (const scores of [[], [1, 2], 'no', [null]]) {
    const {runtime, ready} = await fixture(t);
    const child = await ready(true);
    const next = once(child, 'request');
    const failed = assert.rejects(runtime.orderMoves({}, [[[0], [1]]]), /invalid policy scores/);
    const [request] = await next;
    child.respond({id:request.id, scores});
    await failed;
    assert.equal(child.killed, true);
  }
  let response;
  await forwardInference({postMessage(message) { response = message; }}, {}, {type:'policy', id:7, position:{}, moves:[]});
  assert.deepEqual(response, {type:'policyScores', id:7, scores:null});
});

test('policy readiness requires a supported version and completed policy updates', async t => {
  for (const model of [{policyAvailable:true}, {policyAvailable:'yes'},
    {policyAvailable:true, policyVersion:1, policyTrainedSteps:0}]) {
    const {runtime, children} = await fixture(t);
    const failed = assert.rejects(runtime.start(), /invalid policy metadata/);
    children[0].respond({ready:true, device:'cpu', model});
    await failed;
  }
});

test('checkpoint reload rejects an analysis pinned to the old generation before sending values or policy', async t => {
  const {runtime, children, checkpoint, ready} = await fixture(t);
  await ready(true);
  const oldGeneration = runtime.info.model.runtimeGeneration;
  assert.equal(oldGeneration, 1);
  const changed = new Date(Date.now() + 10000);
  await utimes(checkpoint, changed, changed);
  const rejected = assert.rejects(runtime.evaluate([{}], {runtimeGeneration:oldGeneration}), /checkpoint changed during analysis/);
  assert.equal(children.length, 2);
  let dispatched = 0;
  const child = children[1];
  child.on('request', () => { dispatched++; });
  child.respond({ready:true, device:'cpu', model:{trainedSteps:2,
    policyAvailable:true, policyTrainedSteps:1, policyVersion:1}});
  await rejected;
  const freshGeneration = runtime.info.model.runtimeGeneration;
  assert.equal(freshGeneration, oldGeneration + 1);
  await assert.rejects(runtime.orderMoves({}, [[[0], [1]]], {runtimeGeneration:oldGeneration}), /checkpoint changed during analysis/);
  assert.equal(dispatched, 0);
  assert.equal(runtime.pending.size, 0);
  const next = once(child, 'request');
  const evaluation = runtime.evaluate([{}], {runtimeGeneration:freshGeneration});
  const [request] = await next;
  child.respond({id:request.id, values:[13]});
  assert.deepEqual(await evaluation, {id:request.id, values:[13], runtimeGeneration:freshGeneration});
});

test('inference responses retain their request generation when a new service readies before delivery', async t => {
  const {runtime, ready} = await fixture(t);
  const child = await ready(true);
  const oldGeneration = runtime.info.model.runtimeGeneration;
  const valueRequest = once(child, 'request');
  const evaluation = runtime.evaluate([{}], {runtimeGeneration:oldGeneration});
  const [value] = await valueRequest;
  const policyRequest = once(child, 'request');
  const ordering = runtime.orderMoves({}, [[[0], [1]]], {runtimeGeneration:oldGeneration, withMetadata:true});
  const [policy] = await policyRequest;
  // Resolve both service messages, then replace the child before their caller
  // continuations run. Even a bogus Python generation cannot override ownership.
  child.respond({id:value.id, values:[9], runtimeGeneration:999});
  child.respond({id:policy.id, scores:[0.5], runtimeGeneration:999});
  runtime.stopProcess();
  const restarted = ready();
  assert.equal(runtime.info.model.runtimeGeneration, oldGeneration + 1);
  assert.equal((await evaluation).runtimeGeneration, oldGeneration);
  assert.deepEqual(await ordering, {scores:[0.5], runtimeGeneration:oldGeneration});
  await restarted;
  assert.deepEqual(await runtime.orderMoves({}, [[[0], [1]]], {withMetadata:true}),
    {scores:null, runtimeGeneration:oldGeneration + 1});
});

test('forwarding passes generation expectations and rejects mismatched value and policy results', async () => {
  for (const type of ['evaluate', 'policy']) {
    let response, received;
    const runtime = type === 'evaluate'
      ? {evaluate:async (positions, options) => { received = options; return {values:[0], runtimeGeneration:2}; }}
      : {orderMoves:async (position, moves, options) => { received = options; return {scores:[0], runtimeGeneration:2}; }};
    await forwardInference({postMessage(message) { response = message; }}, runtime,
      {type, id:5, positions:[{}], position:{}, moves:[[]], runtimeGeneration:1});
    assert.equal(received.runtimeGeneration, 1);
    if (type === 'policy') assert.equal(received.withMetadata, true);
    assert.match(response.error, /checkpoint changed during analysis/);
    assert.equal(response.values, undefined);
    assert.equal(response.scores, undefined);
  }
  let response;
  await forwardInference({postMessage(message) { response = message; }},
    {orderMoves:async () => ({scores:[0.25], runtimeGeneration:7})},
    {type:'policy', id:6, position:{}, moves:[[]], runtimeGeneration:7});
  assert.deepEqual(response, {type:'policyScores', id:6, scores:[0.25], runtimeGeneration:7});
});
