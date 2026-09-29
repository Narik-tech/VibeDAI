import { Worker } from 'node:worker_threads';
import { setImmediate as yieldTurn } from 'node:timers/promises';
import { analyze as serialAnalyze, createSearchSession, MATE_SCORE } from './search.js';
import { CANCEL, NODES, STOP_ITERATION, SEARCH_NODES, GENERATION_NODES, QNODES, claimNode, clock, searchCandidate } from './parallel-budget.js';

export { MATE_SCORE };
const finite = (value, fallback, min, max) => Number.isFinite(Number(value))
  ? Math.max(min, Math.min(max, Number(value))) : fallback;
const counters = ['searchNodes', 'generationNodes', 'qnodes', 'ttHits', 'qTtHits', 'cutoffs', 'policyLeaves', 'tableEntries', 'cacheMemoryBytes'];

class SearchWorker {
  constructor(data) {
    this.stats = {};
    // --input-type belongs to a string/stdin entry point, never a worker file.
    const execArgv = process.execArgv.filter((arg, index, args) => arg !== '--input-type'
      && !arg.startsWith('--input-type=') && args[index - 1] !== '--input-type');
    this.worker = new Worker(new URL('./parallel-worker.js', import.meta.url), { workerData: data, execArgv });
    const finish = reply => {
      if (reply.stats) this.stats = reply.stats;
      if (this.resolve) { this.reply = reply; this.resolve(reply); this.resolve = null; }
    };
    this.worker.on('message', finish);
    this.worker.on('error', error => { this.failure = error; finish({ error: error.message }); });
    this.worker.on('exit', code => {
      if (!this.closing) {
        this.failure ||= new Error(`Parallel search worker exited (${code}) before completing its work.`);
        finish({ error: this.failure.message });
      }
    });
  }
  run(candidate, request) {
    if (this.failure) throw this.failure;
    this.candidate = candidate;
    this.reply = null;
    this.pending = new Promise(resolve => { this.resolve = resolve; });
    try { this.worker.postMessage({ position: candidate.position, request }); }
    catch (error) {
      // Posting can fail synchronously (for example, a cloning error). The
      // root's cleanup still drains every posted job before closing workers.
      this.reply = { error: error.message };
      this.resolve(this.reply);
      this.resolve = null;
      throw error;
    }
  }
  async close() { this.closing = true; await this.worker.terminate(); }
}

/** Parallel complete-turn root search. `threads` includes the coordinator,
 * which searches the first sibling before distributing the remaining work.
 * Each worker retains its own bounded cache throughout one analysis.
 */
export async function analyze(position, options = {}) {
  const threads = Math.floor(finite(options.threads, 1, 1, 16));
  if (threads === 1) {
    const decorate = result => ({ ...result, threadsUsed: 1, limits: { ...result.limits, threads } });
    return decorate(serialAnalyze(position, {
      ...options, onProgress: options.onProgress ? result => options.onProgress(decorate(result)) : undefined,
    }));
  }
  const started = clock();
  const unlimitedTime = options.unlimitedTime === true;
  const timeMs = finite(options.timeMs, 3000, 0, 3_600_000);
  const maxNodes = Math.floor(finite(options.maxNodes, 2_000_000, 0, 1_000_000_000));
  const cacheMemoryMb = finite(options.cacheMemoryMb, 128, 0, 4096);
  const maxTableEntries = Math.floor(finite(options.maxTableEntries, 100_000, 0, 1_000_000));
  const deadline = unlimitedTime ? Infinity : started + timeMs;
  const shared = new Int32Array(new SharedArrayBuffer(24));
  const workerOptions = { timeMs, unlimitedTime, maxNodes, cacheMemoryMb: cacheMemoryMb / threads, maxTableEntries: Math.floor(maxTableEntries / threads) };
  const pool = [];
  let threadsUsed = 1, callbackError;
  function cancelled() {
    if (options.shouldStop?.()) Atomics.store(shared, CANCEL, 1);
    return Atomics.load(shared, CANCEL) !== 0;
  }
  function aggregate(result) {
    const combined = { ...result };
    for (const field of counters) combined[field] += pool.reduce((sum, worker) => sum + (worker.stats[field] || 0), 0);
    combined.selectiveDepth = Math.max(result.selectiveDepth, ...pool.map(worker => worker.stats.selectiveDepth || 0));
    combined.qnodes = Atomics.load(shared, QNODES);
    combined.searchNodes = Atomics.load(shared, SEARCH_NODES);
    combined.generationNodes = Atomics.load(shared, GENERATION_NODES);
    combined.nodes = combined.searchNodes + combined.generationNodes;
    combined.elapsedMs = Math.max(0, Math.round(clock() - started));
    combined.nps = combined.elapsedMs ? Math.round(combined.nodes * 1000 / combined.elapsedMs) : 0;
    combined.threadsUsed = threadsUsed;
    combined.limits = { ...result.limits, timeMs, maxNodes, cacheMemoryMb, maxTableEntries, threads };
    return combined;
  }
  const session = createSearchSession(position, {
    ...options, ...workerOptions, timeMs: unlimitedTime ? timeMs : Math.max(0, deadline - clock()),
    shouldStop: cancelled, claimNode: kind => claimNode(shared, maxNodes, kind),
    onProgress: options.onProgress ? result => options.onProgress(aggregate(result)) : undefined,
  });
  // Freeze and validate once in the coordinator, then clone the exact effective
  // settings into every worker, including any legacy quiescence-depth override.
  workerOptions.heuristics = session.snapshot().heuristics;
  function check() {
    if (callbackError) throw callbackError;
    if (cancelled()) throw session.interrupt('cancelled');
    if (clock() >= deadline) throw session.interrupt('time');
    if (Atomics.load(shared, NODES) >= maxNodes) throw session.interrupt('nodes');
  }
  function startWorkers() {
    if (pool.length) return;
    for (let index = 1; index < threads; index++) {
      pool.push(new SearchWorker({ position, options: workerOptions, deadline, shared: shared.buffer }));
    }
    threadsUsed = threads;
  }
  async function root(request) {
    check();
    const iterator = session.beginRoot(request);
    let count = 0, best = -1_000_000, pv = [], exhausted = false;
    let alpha = Math.max(request.alpha, -MATE_SCORE), beta = Math.min(request.beta, MATE_SCORE - 1);
    function accept(candidate, child) {
      count++;
      session.acceptRoot(candidate, child, count);
      const value = -child.score;
      if (value > best) { best = value; pv = [candidate.moves, ...child.pv]; }
      alpha = Math.max(alpha, value);
    }
    function next() {
      check();
      const result = iterator.next();
      exhausted = result.done;
      return result.value;
    }
    function collect() {
      for (const worker of pool) if (worker.pending && worker.reply) {
        const reply = worker.reply;
        worker.pending = null;
        if (reply.error) throw new Error(reply.error);
        if (reply.interrupted) {
          check();
          throw session.interrupt(reply.interrupted);
        }
        accept(worker.candidate, reply.result);
        worker.candidate = null;
      }
    }
    try {
      const first = next();
      if (!first) return session.emptyRoot(iterator);
      // Establish a useful bound before spending work on sibling branches.
      accept(first, searchCandidate(session, first.position, { ...request, alpha, beta }, true));
      if (alpha >= beta) return { score: best, pv };
      startWorkers();
      while (true) {
        collect();
        if (alpha >= beta) break;
        for (const worker of pool) if (!worker.pending && !exhausted) {
          const candidate = next();
          if (candidate) worker.run(candidate, { ...request, alpha, beta });
        }
        if (!exhausted) {
          const candidate = next();
          if (candidate) accept(candidate, searchCandidate(session, candidate.position, { ...request, alpha, beta }));
          // Deliver completed jobs, cancellation and progress between local
          // branches; never create a worker for each node or each candidate.
          await yieldTurn();
        } else {
          const pending = pool.filter(worker => worker.pending);
          if (!pending.length) break;
          await Promise.race(pending.map(worker => worker.pending));
        }
      }
      return { score: best, pv };
    } finally {
      iterator.return?.();
      // A root cutoff or interruption invalidates outstanding work, not the
      // whole pool. Drain it before reusing workers for another iteration.
      Atomics.store(shared, STOP_ITERATION, 1);
      await Promise.all(pool.map(worker => worker.pending));
      for (const worker of pool) { worker.pending = null; worker.candidate = null; }
      Atomics.store(shared, STOP_ITERATION, 0);
    }
  }

  let progressTimer;
  const iterations = session.iterations();
  try {
    let step = iterations.next();
    while (!step.done) {
      let result;
      try {
        // The inexpensive depth-one warmup also provides a legal, completed
        // result before worker startup consumes any of the remaining budget.
        if (step.value.remaining === 1) result = session.root(step.value);
        else {
          progressTimer = setInterval(() => {
            try {
              cancelled();
              options.onProgress?.(aggregate(session.snapshot()));
            } catch (error) {
              callbackError = error;
              Atomics.store(shared, CANCEL, 1);
            }
          }, 250);
          try { result = await root(step.value); }
          finally { clearInterval(progressTimer); }
        }
      } catch (error) {
        if (callbackError) throw callbackError;
        step = iterations.throw(error);
        continue;
      }
      if (callbackError) throw callbackError;
      step = iterations.next(result);
    }
    return aggregate(step.value);
  } finally {
    clearInterval(progressTimer);
    Atomics.store(shared, CANCEL, 1);
    await Promise.all(pool.map(worker => worker.close()));
  }
}
