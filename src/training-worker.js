import { parentPort, workerData } from 'node:worker_threads';
import { realpath } from 'node:fs/promises';
import path from 'node:path';
import { runSelfPlay } from '../scripts/transformer-selfplay.js';
import { runFreshTraining } from '../scripts/transformer-fresh-training.js';
import { acquireRunLock } from '../scripts/transformer-selfplay-store.js';

const cancellation = new Int32Array(workerData.cancelBuffer);
const runtimes = new Set();
const releases = [];
const shouldStop = () => Atomics.load(cancellation, 0) !== 0;
const stop = () => {
  Atomics.store(cancellation, 0, 1);
  for (const runtime of runtimes) runtime.close();
};
parentPort.on('message', message => { if (message?.type === 'stop') stop(); });
try {
  const fresh = workerData.mode === 'fresh20m';
  if (fresh || (workerData.options.sharedRunDir && path.resolve(workerData.options.sharedRunDir) !== path.resolve(workerData.options.runDir))) {
    releases.push(await acquireRunLock(workerData.options.sharedRunDir, { token: workerData.lockToken }));
  }
  if (fresh) {
    const checkpoint = await realpath(workerData.options.sharedCheckpoint).catch(error => {
      if (error.code !== 'ENOENT') throw error;
      return workerData.options.sharedCheckpoint;
    });
    releases.push(await acquireRunLock(`${checkpoint}.selfplay-lock`, { token: workerData.lockToken }));
  }
  await (fresh ? runFreshTraining : runSelfPlay)(workerData.options, {
    lockToken: workerData.lockToken,
    shouldStop,
    onRuntime(runtime) {
      for (const previous of runtimes) if (previous.closed) runtimes.delete(previous);
      runtimes.add(runtime);
      if (shouldStop()) runtime.close();
    },
    onEvent(event, fields = {}) { parentPort.postMessage({ type: 'event', event: { event, ...fields } }); },
  });
  parentPort.postMessage({ type: 'complete', state: shouldStop() ? 'interrupted' : 'completed' });
} catch (error) {
  parentPort.postMessage({ type: 'complete', state: shouldStop() || error.name === 'AbortError' ? 'interrupted' : 'failed', error: error.message });
} finally {
  // The runner releases its own locks before this message channel closes. Do
  // not forcibly terminate the thread: it owns inference and training children.
  for (const runtime of runtimes) runtime.close();
  for (const release of releases.reverse()) await release();
  parentPort.close();
}
