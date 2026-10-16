import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { setTimeout as delay } from 'node:timers/promises';
import { analyze } from '../src/rust-engine.js';
import { createPosition, validateAction } from '../src/rules.js';
import { nativeTestOptions } from './rust-native-helper.js';

test('native cancellation after progress returns a legal retained recommendation', nativeTestOptions, async () => {
  let stop = false;
  const position = createPosition();
  const result = await analyze(position, {
    timeMs: 0, maxDepth: 64, maxNodes: 1_000_000_000,
    shouldStop: () => stop,
    onProgress: progress => { if (progress.completed) stop = true; },
  });
  assert.equal(result.backend, 'rust');
  assert.equal(result.stoppedReason, 'cancelled');
  assert.ok(Array.isArray(result.bestAction));
  validateAction(position, result.bestAction);
});

test('terminating a Node search worker also ends its native subprocess', { ...nativeTestOptions, timeout: 10_000 }, async t => {
  const worker = new Worker(`
    const { parentPort, workerData } = require('node:worker_threads');
    const childProcess = require('node:child_process');
    const original = childProcess.spawn;
    childProcess.spawn = (...args) => {
      const child = original(...args);
      parentPort.postMessage({ type: 'child', pid: child.pid });
      return child;
    };
    (async () => {
      const { analyze } = await import(workerData.engine);
      const { createPosition } = await import(workerData.rules);
      await analyze(createPosition(), {
        timeMs: 0, maxDepth: 64, maxNodes: 1_000_000_000,
        onProgress: () => parentPort.postMessage({ type: 'progress' }),
      });
    })().catch(error => parentPort.postMessage({ type: 'error', error: error.message }));
  `, {
    eval: true,
    workerData: {
      engine: new URL('../src/rust-engine.js', import.meta.url).href,
      rules: new URL('../src/rules.js', import.meta.url).href,
    },
  });
  t.after(() => worker.terminate());
  let pid;
  await new Promise((resolve, reject) => {
    worker.on('error', reject);
    worker.on('message', message => {
      if (message.type === 'child') pid = message.pid;
      if (message.type === 'error') reject(new Error(message.error));
      if (message.type === 'progress') resolve();
    });
  });
  assert.ok(Number.isInteger(pid), 'the worker started a native child');
  await worker.terminate();
  const alive = () => {
    try { process.kill(pid, 0); return true; }
    catch (error) { if (error.code === 'ESRCH') return false; throw error; }
  };
  const deadline = Date.now() + 3000;
  while (alive() && Date.now() < deadline) await delay(20);
  const orphan = alive();
  if (orphan) process.kill(pid);
  assert.equal(orphan, false, 'closing the worker must close stdin or kill its native child');
});
