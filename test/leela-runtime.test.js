import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { EventEmitter, once } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtemp, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { LeelaRuntime } from '../src/leela-runtime.js';
import { LEELA_ID, LEELA_NAME, DEFAULT_LEELA_CHECKPOINT } from '../src/leela-config.js';

const ARCHITECTURE = '5d-lc0-transfer-v1';

async function fixture(t, options = {}) {
  const directory = await mkdtemp(path.join(tmpdir(), 'vibe-leela-runtime-'));
  const checkpoint = path.join(directory, 'leela.pt');
  await writeFile(checkpoint, 'mock checkpoint');
  const children = [], invocations = [];
  const runtime = new LeelaRuntime({ python: process.execPath, checkpoint, startupMs: 1000, requestMs: 1000,
    ...options,
    spawnProcess(executable, args, settings) {
      invocations.push({ executable, args, settings });
      const child = new EventEmitter();
      child.stdin = new PassThrough(); child.stdout = new PassThrough(); child.stderr = new PassThrough();
      child.stdin.on('data', data => child.emit('request', JSON.parse(data.toString())));
      child.kill = () => { child.killed = true; queueMicrotask(() => child.emit('exit', 0)); };
      child.respond = message => child.stdout.write(JSON.stringify(message) + '\n');
      children.push(child);
      return child;
    },
  });
  t.after(async () => { runtime.close(); await rm(directory, { recursive: true, force: true }); });
  const respondReady = (architecture = ARCHITECTURE) => children.at(-1).respond({ ready: true, device: 'cpu', model: {
    architecture, trainedSteps: 8, policyAvailable: true, policyTrainedSteps: 4, policyVersion: 1,
  } });
  return { runtime, checkpoint, children, invocations, respondReady };
}

test('Leela has a dedicated identity and ignores the Transformer checkpoint environment', async () => {
  assert.equal(LEELA_ID, 'leela');
  assert.equal(LEELA_NAME, 'Leela in a 5D Trenchcoat');
  const configUrl = new URL('../src/leela-config.js', import.meta.url).href;
  const runtimeUrl = new URL('../src/leela-runtime.js', import.meta.url).href;
  const source = `
    import { resolveLeelaConfig, DEFAULT_LEELA_CHECKPOINT } from ${JSON.stringify(configUrl)};
    import { LeelaRuntime } from ${JSON.stringify(runtimeUrl)};
    const config = resolveLeelaConfig();
    const runtime = new LeelaRuntime();
    console.log(JSON.stringify({ checkpoint: config.checkpoint, runtimeCheckpoint: runtime.checkpoint,
      expected: DEFAULT_LEELA_CHECKPOINT, description: runtime.describe() }));
    runtime.close();
  `;
  const env = { ...process.env, TRANSFORMER_CHECKPOINT: path.resolve('artifacts/transformer/unrelated.pt') };
  delete env.LEELA_CHECKPOINT;
  const { stdout } = await promisify(execFile)(process.execPath, ['--input-type=module', '-e', source], { env, windowsHide: true });
  const result = JSON.parse(stdout);
  assert.equal(result.checkpoint, DEFAULT_LEELA_CHECKPOINT);
  assert.equal(result.runtimeCheckpoint, DEFAULT_LEELA_CHECKPOINT);
  assert.notEqual(result.checkpoint, env.TRANSFORMER_CHECKPOINT);
  assert.equal(result.description.id, LEELA_ID);
  assert.equal(result.description.name, LEELA_NAME);
});

test('an absent explicit Leela checkpoint fails without spawning or using another model', async t => {
  const { runtime, children } = await fixture(t, { checkpoint: path.join(tmpdir(), `absent-leela-${process.pid}-${Date.now()}.pt`) });
  assert.equal(runtime.describe().available, false);
  await assert.rejects(runtime.start(), /checkpoint|trained|model/i);
  assert.equal(children.length, 0);
  assert.equal(runtime.info, undefined);
});

test('Leela rejects a valid Transformer checkpoint architecture and kills its service', async t => {
  const { runtime, children, respondReady } = await fixture(t);
  const failure = assert.rejects(runtime.start(), /LC0|LCZero|transfer|architecture|Leela/i);
  respondReady('5d-transformer-value-v1');
  await failure;
  assert.equal(children[0].killed, true);
  assert.notEqual(runtime.state, 'ready');
  assert.equal(runtime.info, null);
});

test('Leela accepts transferred weights and keeps value and policy requests on its own child', async t => {
  const { runtime, checkpoint, children, invocations, respondReady } = await fixture(t);
  const starting = runtime.start();
  respondReady();
  const info = await starting;
  assert.equal(info.model.architecture, ARCHITECTURE);
  assert.equal(runtime.describe().id, 'leela');
  assert(invocations[0].args.includes(checkpoint));
  assert.equal(invocations[0].settings.windowsHide, true);
  const generation = info.model.runtimeGeneration;
  const child = children[0];
  const nextValue = once(child, 'request');
  const evaluating = runtime.evaluate([{}], { runtimeGeneration: generation });
  const [valueRequest] = await nextValue;
  child.respond({ id: valueRequest.id, values: [123] });
  assert.deepEqual((await evaluating).values, [123]);
  const nextPolicy = once(child, 'request');
  const ordering = runtime.orderMoves({}, [[[0], [1]], null], { runtimeGeneration: generation });
  const [policyRequest] = await nextPolicy;
  assert.equal(policyRequest.type, 'policy');
  assert.deepEqual(policyRequest.moves, [[[0], [1]], null]);
  child.respond({ id: policyRequest.id, scores: [0.25, -0.5] });
  assert.deepEqual(await ordering, [0.25, -0.5]);
  assert.equal(children.length, 1);
});

test('reloading an incompatible checkpoint cannot reuse an earlier valid Leela service', async t => {
  const { runtime, checkpoint, children, respondReady } = await fixture(t);
  const first = runtime.start();
  respondReady();
  await first;
  const changed = new Date(Date.now() + 10000);
  await utimes(checkpoint, changed, changed);
  const failure = assert.rejects(runtime.evaluate([{}]), /LC0|LCZero|transfer|architecture|Leela/i);
  assert.equal(children.length, 2);
  let requests = 0;
  children[1].on('request', () => { requests++; });
  respondReady('5d-transformer-value-v1');
  await failure;
  assert.equal(children[0].killed, true);
  assert.equal(children[1].killed, true);
  assert.equal(requests, 0);
});
