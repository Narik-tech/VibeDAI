import { parentPort, workerData } from 'node:worker_threads';
import { analyze } from './parallel-search.js';
import { formatAction, validateAction } from './rules.js';
import { isNeuralEngine } from './leela-config.js';

let contextTruncated = false, frontierTruncated = false;
const neural = isNeuralEngine(workerData.options.engine);
let requestId = 0;
const pending = new Map();
if (neural) {
  parentPort.on('message', message => {
    if (!['evaluations', 'policyScores'].includes(message.type)) return;
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    if (message.error) request.reject(new Error(message.error));
    else {
      contextTruncated ||= message.context?.some(item => item.truncated) ?? false;
      frontierTruncated ||= message.context?.some(item => item.frontierTruncated) ?? false;
      request.resolve(message.type === 'policyScores' ? message.scores : message.values);
    }
  });
}

function evaluateBatch(positions) {
  const id = ++requestId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    parentPort.postMessage({ type: 'evaluate', id, positions, runtimeGeneration: workerData.model?.runtimeGeneration });
  });
}

function scoreMoves(position, moves) {
  if (!workerData.model?.policyAvailable) return Promise.resolve(null);
  const id = ++requestId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    parentPort.postMessage({ type: 'policy', id, position, moves, runtimeGeneration: workerData.model?.runtimeGeneration });
  });
}

function annotate(result) {
  let position = workerData.position;
  const pvNotation = [];
  for (const action of result.pv ?? []) {
    try {
      pvNotation.push(formatAction(position, action));
      position = validateAction(position, action);
    } catch { break; }
  }
  return {
    ...result,
    engine: workerData.options.engine || 'classical',
    ...(neural ? { model: workerData.model, contextTruncated, frontierTruncated } : {}),
    notation: result.bestAction === null || result.bestAction === undefined
      ? '' : formatAction(workerData.position, result.bestAction),
    pvNotation,
  };
}

try {
  const cancelled = new Int32Array(workerData.cancelBuffer);
  const engine = neural ? (await import('./transformer-search.js')).analyze : analyze;
  const result = await engine(workerData.position, {
    ...workerData.options,
    shouldStop: () => Atomics.load(cancelled, 0) !== 0,
    evaluateBatch,
    scoreMoves,
    onProgress: progress => parentPort.postMessage({ type: 'progress', result: annotate(progress) }),
  });
  parentPort.postMessage({ type: 'result', result: annotate(result) });
} catch (error) {
  parentPort.postMessage({ type: 'error', error: error.message });
} finally {
  parentPort.close();
}
