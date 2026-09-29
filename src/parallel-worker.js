import { parentPort, workerData } from 'node:worker_threads';
import { createSearchSession } from './search.js';
import { CANCEL, STOP_ITERATION, claimNode, clock, searchCandidate } from './parallel-budget.js';

const shared = new Int32Array(workerData.shared);
const session = createSearchSession(workerData.position, {
  ...workerData.options,
  timeMs: workerData.options.unlimitedTime === true ? workerData.options.timeMs : Math.max(0, workerData.deadline - clock()),
  shouldStop: () => Atomics.load(shared, CANCEL) !== 0 || Atomics.load(shared, STOP_ITERATION) !== 0,
  claimNode: kind => claimNode(shared, workerData.options.maxNodes, kind),
});

parentPort.on('message', ({ position, request }) => {
  try {
    const result = searchCandidate(session, position, request);
    parentPort.postMessage({ result, stats: session.statistics() });
  } catch (error) {
    parentPort.postMessage(session.isInterrupted(error)
      ? { interrupted: session.snapshot().stoppedReason, stats: session.statistics() }
      : { error: error.stack || error.message });
  }
});
