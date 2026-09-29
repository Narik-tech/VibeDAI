import path from 'node:path';
import { open, readdir, lstat, stat, realpath } from 'node:fs/promises';
import { Worker } from 'node:worker_threads';
import { randomUUID } from 'node:crypto';
import { getSelfPlayDefaults, parseArguments } from '../scripts/transformer-selfplay.js';
import { getFreshTrainingDefaults, validateFreshTrainingOptions } from '../scripts/transformer-fresh-training.js';
import { acquireRunLock, inspectRunLock, recoverRunLock, atomicWrite } from '../scripts/transformer-selfplay-store.js';
import { findInterruptedIterations, recoverInterruptedIterations } from '../scripts/transformer-selfplay-recovery.js';
import { DEFAULT_CHECKPOINT, DEFAULT_PYTHON } from './transformer-runtime.js';
import { LEELA_ID, LEELA_NAME, resolveLeelaConfig } from './leela-config.js';
import { raw, positionKey, validateAction } from './rules.js';

const optionFlags = Object.freeze({ iterations: 'iterations', games: 'games', gameConcurrency: 'game-concurrency', maxPlies: 'plies', maxNodes: 'nodes',
  maxDepth: 'depth', timeMs: 'time-ms', terminalWork: 'terminal-work', exploration: 'exploration',
  explorationPlies: 'exploration-plies', outcomeWeight: 'outcome-weight', steps: 'steps', batchSize: 'batch-size',
  learningRate: 'learning-rate', maxTokens: 'max-tokens', replaySize: 'replay-size', seed: 'seed', arenaPairs: 'arena-pairs', arenaConcurrency: 'arena-concurrency',
  minPairs: 'min-pairs', arenaPlies: 'arena-plies', promotionScore: 'promotion-score', keepIterations: 'keep-iterations', device: 'device' });
// Environment mistakes disable training without preventing the classical
// server or saved-game review from loading. Validate them per manager below.
const runnerDefaults = getSelfPlayDefaults();
const safeDefaults = { ...runnerDefaults, checkpoint: DEFAULT_CHECKPOINT, python: DEFAULT_PYTHON,
  device: ['auto', 'cpu', 'cuda'].includes(runnerDefaults.device) ? runnerDefaults.device : 'auto' };
export const TRAINING_DEFAULTS = Object.freeze({ ...Object.fromEntries(Object.keys(optionFlags).map(key => [key, safeDefaults[key]])), model: 'current' });
const ITERATION_ID = /^iteration-\d{8,12}$/;
const FRESH_ID = /^fresh-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const GAME_ID = /^(selfplay|arena)-\d{3,6}$/;
const MAX_GAME_BYTES = 16 * 1024 * 1024;
const MAX_CACHE_BYTES = 32 * 1024 * 1024;
const missing = error => error.code === 'ENOENT';
const failure = (message, statusCode = 400) => Object.assign(new Error(message), { statusCode });
const clone = value => structuredClone(value);

export function validateTrainingOptions(input = {}, initialOptions = safeDefaults) {
  if (!input || typeof input !== 'object' || Array.isArray(input) || ![Object.prototype, null].includes(Object.getPrototypeOf(input))) {
    throw failure('Training options must be a JSON object.');
  }
  const args = [];
  for (const [key, value] of Object.entries(input)) {
    if (key === 'model') {
      if (!['current', '20m', LEELA_ID].includes(value)) throw failure('model must be current, 20m or leela.');
      continue;
    }
    if (!Object.hasOwn(optionFlags, key)) throw failure(`Unknown training option: ${key}.`);
    if (key === 'device' ? typeof value !== 'string' : typeof value !== 'number' || !Number.isFinite(value)) {
      throw failure(`${key} must be ${key === 'device' ? 'auto, cpu or cuda' : 'a finite number'}.`);
    }
    args.push(`--${optionFlags[key]}`, String(value));
  }
  const parsed = parseArguments(args, { ...initialOptions, ...(input.model === LEELA_ID ? { batchSize: 4 } : {}) });
  return { ...Object.fromEntries(Object.keys(optionFlags).map(key => [key, parsed[key]])), model: input.model ?? 'current' };
}

/** File reads remain bounded even when a log grows between stat and read. */
async function readBounded(file, limit, { tail = false } = {}) {
  const handle = await open(file, 'r');
  try {
    const info = await handle.stat();
    if (!info.isFile()) throw failure('Expected a regular training artifact.');
    if (!tail && info.size > limit) throw failure('Training artifact exceeds the review size limit.', 413);
    const size = Math.min(info.size, limit), buffer = Buffer.alloc(size);
    let read = 0;
    while (read < size) {
      const { bytesRead } = await handle.read(buffer, read, size - read, (tail ? Math.max(0, info.size - size) : 0) + read);
      if (!bytesRead) break;
      read += bytesRead;
    }
    return buffer.subarray(0, read).toString('utf8');
  } finally { await handle.close(); }
}

async function readJSON(file, limit = 1024 * 1024) {
  try { return JSON.parse(await readBounded(file, limit)); }
  catch (error) { if (error instanceof SyntaxError) throw failure(`Invalid saved JSON in ${path.basename(file)}.`); throw error; }
}

function assertPosition(position) {
  if (!position || !Number.isSafeInteger(position.action) || position.action < 0 || !Array.isArray(position.board) || position.board.length > 257
    || !Array.isArray(position.promotions) || position.promotions.some(piece => !Number.isInteger(piece) || Math.abs(piece) > 24)) {
    throw failure('The saved game has an invalid starting position.');
  }
  let boards = 0, squares = 0;
  for (const timeline of position.board) {
    if (timeline === null) continue;
    if (!Array.isArray(timeline) || timeline.length > 8192) throw failure('The saved game has an invalid timeline.');
    for (const board of timeline) {
      if (board === null) continue;
      const width = board?.[0]?.length;
      if (!Array.isArray(board) || board.length < 1 || board.length > 16 || !Number.isInteger(width) || width < 1 || width > 16
        || board.some(rank => !Array.isArray(rank) || rank.length !== width || rank.some(piece => !Number.isInteger(piece) || Math.abs(piece) > 24))) {
        throw failure('The saved game has invalid board squares.');
      }
      boards++; squares += board.length * width;
      if (boards > 8192 || squares > 2_000_000) throw failure('The saved position exceeds the review size limit.', 413);
    }
  }
  if (!boards) throw failure('The saved game has no starting boards.');
}

/** Owns UI-launched work; recovers abandoned runs only after verifying their owner exited. */
export class TrainingManager {
  constructor({ runDir = runnerDefaults.runDir, checkpoint = runnerDefaults.checkpoint, python = runnerDefaults.python,
    workerFactory = data => new Worker(new URL('./training-worker.js', import.meta.url), { workerData: data }), availability, leelaConfig } = {}) {
    this.options = { ...runnerDefaults, runDir: path.resolve(runDir), checkpoint: path.resolve(checkpoint), python: path.resolve(python) };
    try { this.options = parseArguments([], this.options); }
    catch (error) {
      this.configurationError = `Invalid training configuration: ${error.message} Check TRANSFORMER_DEVICE, TRANSFORMER_CHECKPOINT and TRANSFORMER_PYTHON before starting.`;
    }
    this.leelaOptions = { ...safeDefaults, ...resolveLeelaConfig(), ...leelaConfig, batchSize: 4,
      model: LEELA_ID, sharedRunDir: this.options.runDir };
    try { this.leelaOptions = parseArguments([], this.leelaOptions); }
    catch (error) {
      this.leelaConfigurationError = `Invalid Leela training configuration: ${error.message} Check LEELA_CHECKPOINT, LEELA_PYTHON and LEELA_DEVICE before starting.`;
    }
    this.workerFactory = workerFactory;
    this.freshRoot = path.join(this.options.runDir, 'fresh20m');
    this.checkAvailability = availability;
    this.state = { state: 'idle', phase: 'idle', iteration: null, startedAt: null, error: null, events: [] };
    this.gameCache = new Map();
    this.summaries = new Map();
    this.cacheBytes = 0;
    this.closed = false;
    this.starting = false;
    this.pendingStop = false;
    this.worker = null;
  }

  async lockDirectories(options = this.options) {
    const checkpoint = await realpath(options.checkpoint).catch(error => { if (!missing(error)) throw error; return options.checkpoint; });
    return [...new Set([...(options.sharedRunDir ? [options.sharedRunDir] : []), options.runDir, `${checkpoint}.selfplay-lock`])];
  }

  async locks(options = this.options) { return (await Promise.all((await this.lockDirectories(options)).map(directory => inspectRunLock(directory)))).filter(Boolean); }

  lockFailure(locks) {
    // A stale run lock must never hide a live owner of the shared checkpoint.
    const lock = locks.find(item => item.live) ?? locks.find(item => !item.recoverable) ?? locks[0];
    if (lock) return { available: false, external: lock.live, startedAt: lock.owner?.createdAt,
      reason: lock.live ? `Training is running outside this page (PID ${lock.owner.pid}). Review is available; stop it in its owning terminal.`
        : `A self-play lock needs manual review (${lock.reason || 'unknown owner'}): ${lock.file}` };
    return null;
  }

  async lockInfo() { return this.lockFailure(await this.locks()); }

  async reconcile(options = this.options) {
    this.recoveries ??= new Map();
    const key = options.runDir;
    if (this.recoveries.has(key)) return this.recoveries.get(key);
    const previous = this.recoveryTail;
    const recovering = (async () => {
      // Namespaces share a global UI lock. Serialize recovery so two snapshot
      // requests cannot race each other while recovering different models.
      await previous?.catch(() => {});
      // A selected model shares the UI run lock. Do not recover that lock
      // while the original checkpoint still has a live owner elsewhere.
      if (options.sharedRunDir && options.runDir !== this.options.runDir) {
        const sharedLocks = await this.locks();
        if (sharedLocks.some(lock => lock.live || !lock.recoverable)) return this.lockFailure(sharedLocks);
      }
      const locks = await this.locks(options);
      if (locks.some(lock => lock.live || !lock.recoverable)) return this.lockFailure(locks);
      if (!locks.length && !(await findInterruptedIterations(options)).length) return null;
      const releases = [];
      try {
        // Holding both locks closes the race with a terminal-launched runner.
        for (const directory of await this.lockDirectories(options)) releases.push(await acquireRunLock(directory));
        const recovered = await recoverInterruptedIterations(options);
        const latest = recovered.at(-1);
        if (latest && this.state.state === 'idle') {
          const state = latest.status === 'complete' ? 'completed' : latest.status;
          Object.assign(this.state, { state, phase: state, iteration: latest.iteration, model: options.model ?? 'current',
            startedAt: latest.startedAt, finishedAt: latest.finishedAt, error: latest.error ?? null });
        }
        return null;
      } catch (error) {
        return { available: false, reason: `Training recovery could not finish: ${error.message}` };
      } finally { for (const release of releases.reverse()) await release(); }
    })();
    this.recoveries.set(key, recovering);
    this.recoveryTail = recovering;
    try { return await recovering; }
    finally {
      if (this.recoveries.get(key) === recovering) this.recoveries.delete(key);
      if (this.recoveryTail === recovering) this.recoveryTail = null;
    }
  }

  async availability({ ignoreOwned = false, mode = 'selfplay', model = 'current', checkpoint = this.options.checkpoint,
    runDir = this.options.runDir, sharedRunDir, python = this.options.python, suite = this.options.suite,
    arenaSuite = this.options.arenaSuite, seedData = this.options.seedData, device = this.options.device } = {}) {
    if (this.settling) await this.finished;
    if (this.closed) return { available: false, reason: 'The local server is shutting down.' };
    if (!ignoreOwned && (this.worker || this.starting)) return { available: false, reason: 'A training run is already active.' };
    const configurationError = model === LEELA_ID ? this.leelaConfigurationError : this.configurationError;
    if (configurationError) return { available: false, reason: configurationError };
    const selectedOptions = { ...this.options, model, runDir, checkpoint, sharedRunDir, python, suite, arenaSuite, seedData, device };
    const lock = await this.reconcile(model === LEELA_ID ? selectedOptions : this.options);
    if (lock) return lock;
    if (model !== LEELA_ID && mode !== 'fresh20m' && runDir !== this.options.runDir) {
      const selectedLock = await this.reconcile(selectedOptions);
      if (selectedLock) return selectedLock;
    }
    if (this.checkAvailability) return this.checkAvailability({ mode, ...selectedOptions });
    for (const [name, file, reason] of [
      ...(mode === 'fresh20m' ? [] : [['checkpoint', checkpoint, model === LEELA_ID
        ? 'Leela checkpoint missing. Train the imported LCZero model first, or set LEELA_CHECKPOINT to its trained checkpoint.'
        : 'No trained checkpoint. Choose Fresh 20M to train a new model.']]),
      ['Python', python, 'Python environment missing. Run npm run transformer:setup.'],
      ['training suite', suite, 'The self-play starting-position suite is missing.'],
      ...(mode === 'fresh20m' ? [] : [['arena suite', arenaSuite, 'The evaluation starting-position suite is missing.']]),
      ...(model === LEELA_ID ? [['Leela curriculum', seedData, 'The Leela curriculum is missing. Run npm run lc0:data before self-play training.']] : []),
    ]) {
      try {
        const info = await stat(file);
        if (!info.isFile()) return { available: false, reason: `The ${name} must be a regular file.` };
        if (model === LEELA_ID && name === 'checkpoint' && !info.size) return { available: false, reason: 'The Leela checkpoint is empty. Restore a trained LCZero transfer checkpoint.' };
      }
      catch (error) { if (missing(error)) return { available: false, reason }; throw error; }
    }
    return { available: true };
  }

  status() { return clone(this.state); }

  async snapshot() {
    const availability = await this.availability();
    const iterations = await this.listIterations();
    const freshAvailability = await this.availability({ mode: 'fresh20m' });
    const freshRuns = await this.listFreshRuns();
    const model20m = this.model20m(freshRuns);
    const training20mAvailability = model20m.available
      ? await this.availability({ checkpoint: model20m.checkpoint, runDir: path.join(path.dirname(model20m.checkpoint), 'selfplay'), sharedRunDir: this.options.runDir })
      : { available: false, reason: 'Complete a Fresh 20M run first.' };
    const leelaModel = await this.leelaModel();
    const leelaAvailability = await this.availability(this.leelaOptions);
    const leelaDefaults = { ...Object.fromEntries(Object.keys(optionFlags).map(key => [key, this.leelaOptions[key]])), model: LEELA_ID };
    let status = this.status();
    const external = availability.external ? availability : leelaAvailability.external ? leelaAvailability : null;
    if (!this.worker && !this.starting && external) {
      const externalModel = availability.external ? 'current' : LEELA_ID;
      const latest = iterations.find(iteration => iteration.model === externalModel);
      status = { ...status, state: 'external', phase: 'external', iteration: latest?.iteration ?? null,
        model: externalModel, startedAt: external.startedAt ?? null, error: external.reason };
    }
    return { defaults: { ...TRAINING_DEFAULTS }, freshDefaults: getFreshTrainingDefaults(), status, availability,
      freshAvailability, training20mAvailability, freshRuns, model20m, leelaModel, leelaAvailability, leelaDefaults, iterations };
  }

  async start(input = {}) {
    return this.startRun('selfplay', validateTrainingOptions(input, input?.model === LEELA_ID ? this.leelaOptions : safeDefaults));
  }

  async startFresh(input = {}) {
    return this.startRun('fresh20m', validateFreshTrainingOptions(input));
  }

  async startRun(mode, editable) {
    if (this.worker || this.starting) throw failure('A training run is already active.', 409);
    // Reserve the manager before awaiting filesystem checks, so simultaneous
    // start requests cannot both launch workers.
    this.starting = true;
    this.pendingStop = false;
    try {
      let options = { ...this.options, ...editable };
      if (mode === 'fresh20m') {
        const runId = `fresh-${randomUUID()}`, runDir = path.join(this.freshRoot, runId);
        options = { ...options, runId, runDir, checkpoint: path.join(runDir, 'model.pt'),
          sharedRunDir: this.options.runDir, sharedCheckpoint: this.options.checkpoint };
      } else if (editable.model === '20m') {
        const selected = this.model20m(await this.listFreshRuns());
        if (!selected.available) throw failure('Complete a Fresh 20M run before starting self-play with the 20M model.', 409);
        options.checkpoint = selected.checkpoint;
        options.seedData = path.join(path.dirname(selected.checkpoint), 'training.jsonl');
        options.runDir = path.join(path.dirname(selected.checkpoint), 'selfplay');
        options.runId = selected.runId;
        options.sharedRunDir = this.options.runDir;
      } else if (editable.model === LEELA_ID) {
        options = { ...this.leelaOptions, ...editable };
      }
      const available = await this.availability({ ...options, ignoreOwned: true, mode });
      if (this.closed) throw failure('The local server is shutting down.', 503);
      if (this.pendingStop) {
        Object.assign(this.state, { state: 'interrupted', phase: 'interrupted', finishedAt: new Date().toISOString() });
        return this.status();
      }
      if (!available.available) throw failure(available.reason, available.external ? 409 : 503);
      this.cancellation = new Int32Array(new SharedArrayBuffer(4));
      this.state = { state: 'running', mode, model: editable.model ?? '20m', phase: 'starting', iteration: null,
        runId: options.runId ?? null, checkpoint: options.checkpoint, startedAt: new Date().toISOString(), error: null, events: [] };
      const lockToken = randomUUID();
      const worker = this.workerFactory({ mode, options, cancelBuffer: this.cancellation.buffer, lockToken });
      this.worker = worker;
      let finished;
      this.finished = new Promise(resolve => { finished = resolve; });
      worker.on('message', message => {
        if (message?.type === 'event' && message.event && typeof message.event.event === 'string') {
          const event = { ...message.event, at: new Date().toISOString() };
          this.state.events.push(event);
          if (this.state.events.length > 100) this.state.events.shift();
          if (Number.isInteger(event.iteration)) this.state.iteration = event.iteration;
          const phases = { 'iteration-start': 'starting', 'selfplay-start': 'selfplay', 'selfplay-game': 'selfplay',
            'data-start': 'data', 'data-progress': 'data', 'training-complete': 'completed',
            'training-start': 'training', 'training-progress': 'training', 'arena-start': 'arena', 'arena-game': 'arena', 'cycle-complete': 'completed' };
          if (phases[event.event]) this.state.phase = phases[event.event];
        }
        if (message?.type === 'complete') {
          this.result = { state: ['completed', 'interrupted', 'failed'].includes(message.state) ? message.state : 'failed', error: message.error ?? null };
        }
      });
      worker.once('error', error => { this.result = { state: 'failed', error: error.message }; });
      worker.once('exit', code => {
        const result = this.result ?? (this.state.state === 'stopping' ? { state: 'interrupted', error: null }
          : { state: 'failed', error: `Training worker exited without a completion report (${code}).` });
        Object.assign(this.state, result, { phase: result.state, finishedAt: new Date().toISOString() });
        this.result = null;
        this.settling = true;
        (async () => {
          try {
            const directories = new Set([...(await this.lockDirectories()), ...(await this.lockDirectories(options)),
              ...(mode === 'fresh20m' ? [options.runDir] : [])]);
            for (const directory of directories) await recoverRunLock(directory, { exitedOwner: { pid: process.pid, token: lockToken } });
            const recovery = await this.reconcile();
            if (recovery && !this.state.error) this.state.error = recovery.reason;
            if (mode !== 'fresh20m' && options.runDir !== this.options.runDir) {
              const selectedRecovery = await this.reconcile(options);
              if (selectedRecovery && !this.state.error) this.state.error = selectedRecovery.reason;
            }
          } catch (error) { this.state.error = [this.state.error, `Training recovery failed: ${error.message}`].filter(Boolean).join(' '); }
          finally { this.worker = null; this.settling = false; finished(); }
        })();
      });
      return this.status();
    } catch (error) {
      if (this.state.state === 'running' && !this.worker) Object.assign(this.state, { state: 'failed', phase: 'failed', error: error.message });
      throw error;
    } finally { this.starting = false; }
  }

  stop() {
    if (this.settling) return this.status();
    if (this.starting && !this.worker) {
      this.pendingStop = true;
      this.state = { state: 'stopping', phase: 'starting', iteration: null, startedAt: null, error: null, events: [] };
    }
    if (this.worker) {
      this.state.state = 'stopping';
      Atomics.store(this.cancellation, 0, 1);
      this.worker.postMessage({ type: 'stop' });
    }
    return this.status();
  }

  async close() { this.closed = true; this.stop(); await this.finished; }

  async leelaModel() {
    let available = false;
    try { const info = await stat(this.leelaOptions.checkpoint); available = info.isFile() && info.size > 0; }
    catch (error) { if (!missing(error)) throw error; }
    return { id: LEELA_ID, name: LEELA_NAME, available, checkpoint: this.leelaOptions.checkpoint };
  }

  model20m(runs) {
    const latest = runs.find(run => run.status === 'complete' && run.checkpointAvailable && run.parameters === 20000257 && run.trainedSteps > 0);
    return latest ? { available: true, checkpoint: latest.checkpoint, parameters: latest.parameters, trainedSteps: latest.trainedSteps, runId: latest.id }
      : { available: false };
  }

  async listFreshRuns() {
    let root, entries;
    try { root = await realpath(this.freshRoot); entries = await readdir(root, { withFileTypes: true }); }
    catch (error) { if (missing(error)) return []; throw error; }
    const runs = [];
    for (const entry of entries.filter(item => FRESH_ID.test(item.name) && item.isDirectory() && !item.isSymbolicLink())) {
      const folder = path.join(root, entry.name);
      try {
        if (path.dirname(await realpath(folder)) !== root) continue;
        let report = await readJSON((await this.artifact(folder, 'report.json')).target);
        if (report.status === 'running' && !(this.worker && this.state.runId === entry.name)) {
          const lock = await inspectRunLock(folder);
          if (!lock || lock.recoverable) {
            if (lock) await recoverRunLock(folder);
            report = { ...report, status: 'interrupted', finishedAt: new Date().toISOString(), error: 'The training process exited before completing this run.' };
            await atomicWrite(path.join(folder, 'report.json'), JSON.stringify(report, null, 2));
          }
        }
        let checkpointAvailable = false;
        try { checkpointAvailable = (await this.artifact(folder, 'model.pt')).info.size > 0; }
        catch (error) { if (!missing(error)) throw error; }
        runs.push({ id: entry.name, status: report.status, startedAt: report.startedAt, finishedAt: report.finishedAt,
          parameters: report.parameters, trainedSteps: report.trainedSteps, loss: report.loss, config: report.config,
          error: report.error, checkpoint: path.join(folder, 'model.pt'), checkpointAvailable });
      } catch (error) {
        if (missing(error)) continue;
        runs.push({ id: entry.name, status: 'unreadable', error: error.message, checkpointAvailable: false });
      }
    }
    return runs.sort((a, b) => String(b.startedAt ?? '').localeCompare(String(a.startedAt ?? '')) || b.id.localeCompare(a.id)).slice(0, 100);
  }

  async folder(id) {
    if (typeof id !== 'string') throw failure('Invalid training iteration ID.');
    const parts = id.split('__');
    const qualified = parts.length === 2 && (parts[0] === LEELA_ID || FRESH_ID.test(parts[0])) && ITERATION_ID.test(parts[1]);
    if (!qualified && !(parts.length === 1 && ITERATION_ID.test(id))) throw failure('Invalid training iteration ID.');
    const root = qualified ? parts[0] === LEELA_ID ? await this.leelaRoot() : await this.selfplayRoot(parts[0])
      : await realpath(this.options.runDir).catch(error => { if (missing(error)) throw failure('Training iteration not found.', 404); throw error; });
    const target = path.join(root, qualified ? parts[1] : id);
    let info;
    try { info = await lstat(target); }
    catch (error) { if (missing(error)) throw failure('Training iteration not found.', 404); throw error; }
    if (!info.isDirectory() || info.isSymbolicLink() || path.dirname(await realpath(target)) !== root) throw failure('Unsafe training iteration path.');
    return target;
  }

  async selfplayRoot(freshId) {
    if (!FRESH_ID.test(freshId)) throw failure('Invalid fresh training ID.');
    const root = await realpath(this.freshRoot).catch(error => { if (missing(error)) throw failure('Training iteration not found.', 404); throw error; });
    let parent = root;
    for (const name of [freshId, 'selfplay']) {
      const target = path.join(parent, name);
      let info;
      try { info = await lstat(target); }
      catch (error) { if (missing(error)) throw failure('Training iteration not found.', 404); throw error; }
      if (!info.isDirectory() || info.isSymbolicLink() || path.dirname(await realpath(target)) !== parent) throw failure('Unsafe training namespace path.');
      parent = target;
    }
    return parent;
  }

  async leelaRoot() {
    let info, root;
    try { info = await lstat(this.leelaOptions.runDir); root = await realpath(this.leelaOptions.runDir); }
    catch (error) { if (missing(error)) throw failure('Training iteration not found.', 404); throw error; }
    if (!info.isDirectory() || info.isSymbolicLink()) throw failure('Unsafe training namespace path.');
    return root;
  }

  async selfplayNamespaces() {
    const namespaces = [{ prefix: '', model: 'current', options: this.options },
      { prefix: `${LEELA_ID}__`, model: LEELA_ID, options: this.leelaOptions }];
    let entries;
    try { entries = await readdir(this.freshRoot, { withFileTypes: true }); }
    catch (error) { if (missing(error)) return namespaces; throw error; }
    for (const entry of entries.filter(item => item.isDirectory() && !item.isSymbolicLink() && FRESH_ID.test(item.name))) {
      let runDir;
      try { runDir = await this.selfplayRoot(entry.name); }
      catch (error) { if (error.statusCode === 404 || error.statusCode === 400 || missing(error)) continue; throw error; }
      namespaces.push({ prefix: `${entry.name}__`, model: '20m', options: { ...this.options,
        model: '20m', runId: entry.name, runDir, checkpoint: path.join(path.dirname(runDir), 'model.pt'), sharedRunDir: this.options.runDir } });
    }
    return namespaces;
  }

  async artifact(folder, filename) {
    const target = path.join(folder, filename), info = await lstat(target);
    if (!info.isFile() || info.isSymbolicLink() || path.dirname(await realpath(target)) !== folder) throw failure('Unsafe training artifact path.');
    return { target, info, stamp: `${info.size}:${info.mtimeMs}:${info.ctimeMs}` };
  }

  async report(folder) {
    try { return await readJSON((await this.artifact(folder, 'report.json')).target); }
    catch (error) {
      if (!missing(error)) throw error;
      const manifest = await readJSON((await this.artifact(folder, 'iteration.json')).target);
      return { ...manifest, status: 'incomplete', promoted: false };
    }
  }

  async listIterations() {
    const iterations = [];
    for (const namespace of await this.selfplayNamespaces()) {
      if (namespace.prefix) {
        // A malformed run marker must not hide independently readable games.
        await this.reconcile(namespace.options).catch(() => null);
      }
      let entries;
      try { entries = await readdir(namespace.options.runDir, { withFileTypes: true }); }
      catch (error) { if (missing(error)) continue; throw error; }
      const ids = entries.filter(entry => entry.isDirectory() && ITERATION_ID.test(entry.name)).map(entry => entry.name).sort().reverse().slice(0, 100);
      for (const localId of ids) {
        const id = namespace.prefix + localId;
        try {
          const report = await this.report(await this.folder(id));
          iterations.push({ id, model: namespace.model, modelRunId: namespace.options.runId ?? null, iteration: report.iteration ?? Number(localId.slice(10)), status: report.status,
            promoted: Boolean(report.promoted), startedAt: report.startedAt ?? null, finishedAt: report.finishedAt ?? null, error: report.error ?? null });
        } catch (error) {
          // Retention can remove a folder during a read. Corrupt retained reports
          // remain visible as errors instead of hiding the rest of the history.
          if (missing(error) || error.statusCode === 404) continue;
          iterations.push({ id, model: namespace.model, modelRunId: namespace.options.runId ?? null, iteration: Number(localId.slice(10)), status: 'unreadable', promoted: false, error: error.message });
        }
      }
    }
    return iterations.sort((a, b) => String(b.startedAt ?? '').localeCompare(String(a.startedAt ?? ''))
      || b.iteration - a.iteration || b.id.localeCompare(a.id)).slice(0, 100);
  }

  async loadGame(folder, id) {
    if (typeof id !== 'string' || !GAME_ID.test(id)) throw failure('Invalid saved game ID.');
    let file;
    try { file = await this.artifact(folder, `${id}.json`); }
    catch (error) { if (missing(error)) throw failure('Saved game not found.', 404); throw error; }
    const cached = this.gameCache.get(file.target);
    if (cached?.stamp === file.stamp) { this.gameCache.delete(file.target); this.gameCache.set(file.target, cached); return cached; }
    if (cached) { this.gameCache.delete(file.target); this.cacheBytes -= cached.bytes; }
    const game = await readJSON(file.target, MAX_GAME_BYTES);
    if (!game || !Array.isArray(game.moves) || game.moves.length > 512) throw failure('Invalid saved game moves.');
    assertPosition(game.initialPosition);
    const entry = { game, stamp: file.stamp, bytes: file.info.size, ply: 0, position: game.initialPosition };
    while (this.gameCache.size && (this.gameCache.size >= 4 || this.cacheBytes + entry.bytes > MAX_CACHE_BYTES)) {
      const first = this.gameCache.keys().next().value;
      this.cacheBytes -= this.gameCache.get(first).bytes; this.gameCache.delete(first);
    }
    this.gameCache.set(file.target, entry); this.cacheBytes += entry.bytes;
    return entry;
  }

  async getIteration(id) {
    const folder = await this.folder(id), report = await this.report(folder), games = [];
    const entries = (await readdir(folder, { withFileTypes: true })).filter(entry => entry.isFile() && /^(selfplay|arena)-\d{3,6}\.json$/.test(entry.name))
      .map(entry => entry.name).sort().slice(0, 384);
    for (const name of entries) {
      const gameId = name.slice(0, -5), file = await this.artifact(folder, name);
      const cached = this.summaries.get(file.target);
      if (cached?.stamp === file.stamp) { games.push(cached.value); continue; }
      const { game } = await this.loadGame(folder, gameId);
      const kind = gameId.startsWith('selfplay-') ? 'selfplay' : 'arena';
      const value = { id: gameId, kind, label: game.startId ?? game.caseId ?? gameId,
        result: game.result ?? 'UNFINISHED', reason: game.reason ?? null, plies: game.moves.length,
        valid: game.valid !== false, samples: game.samples ?? game.sampleCount ?? null };
      this.summaries.set(file.target, { stamp: file.stamp, value });
      if (this.summaries.size > 1024) this.summaries.delete(this.summaries.keys().next().value);
      games.push(value);
    }
    let log = '';
    try { log = await readBounded((await this.artifact(folder, 'train.log')).target, 32 * 1024, { tail: true }); }
    catch (error) { if (!missing(error)) throw error; }
    return { report, games, log };
  }

  async getGame(iterationId, gameId, ply = 0) {
    if (typeof ply === 'string' && /^\d+$/.test(ply)) ply = Number(ply);
    if (!Number.isSafeInteger(ply) || ply < 0) throw failure('Review ply must be a nonnegative integer.');
    const entry = await this.loadGame(await this.folder(iterationId), gameId), { game } = entry;
    if (ply > game.moves.length) throw failure(`Review ply must be between 0 and ${game.moves.length}.`);
    let position = entry.ply <= ply ? entry.position : game.initialPosition;
    const start = entry.ply <= ply ? entry.ply : 0;
    if (start === 0 && game.initialKey && positionKey(position) !== game.initialKey) throw failure('Saved starting position does not match its recorded key.');
    for (let index = start; index < ply; index++) {
      const turn = game.moves[index];
      try {
        if (turn.beforeKey && positionKey(position) !== turn.beforeKey) throw new Error('starting position key mismatch');
        position = validateAction(position, turn.action);
        if (turn.afterKey && positionKey(position) !== turn.afterKey) throw new Error('resulting position key mismatch');
      } catch (error) { throw failure(`Cannot replay saved turn ${index + 1}: ${error.message}`); }
    }
    if (ply === game.moves.length && game.finalKey && positionKey(position) !== game.finalKey) throw failure('Saved final position does not match the replay.');
    entry.position = position; entry.ply = ply;
    return { game, ply, totalPlies: game.moves.length, position,
      active: raw.boardFuncs.active(position.board), present: raw.boardFuncs.present(position.board, position.action),
      isEvenTimeline: raw.boardFuncs.isEvenTimeline(position.board), isTurnZero: raw.boardFuncs.isTurnZero(position.board) };
  }
}
