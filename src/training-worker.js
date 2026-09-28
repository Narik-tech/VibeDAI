import { parentPort, workerData } from 'node:worker_threads';
import { runSelfPlay } from '../scripts/transformer-selfplay.js';

const cancellation = new Int32Array(workerData.cancelBuffer);
const runtimes = new Set();
const shouldStop = () => Atomics.load(cancellation, 0) !== 0;
const stop = () => {
  Atomics.store(cancellation, 0, 1);
  for (const runtime of runtimes) runtime.close();
};
parentPort.on('message', message => { if (message?.type === 'stop') stop(); });
try {
  await runSelfPlay(workerData.options, {
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
  parentPort.close();
}
