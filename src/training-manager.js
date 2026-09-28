import path from 'node:path';
import { open, readdir, lstat, stat, realpath } from 'node:fs/promises';
import { Worker } from 'node:worker_threads';
import { randomUUID } from 'node:crypto';
import { getSelfPlayDefaults, parseArguments } from '../scripts/transformer-selfplay.js';
import { acquireRunLock, inspectRunLock, recoverRunLock } from '../scripts/transformer-selfplay-store.js';
import { findInterruptedIterations, recoverInterruptedIterations } from '../scripts/transformer-selfplay-recovery.js';
import { DEFAULT_CHECKPOINT, DEFAULT_PYTHON } from './transformer-runtime.js';
import { raw, positionKey, validateAction } from './rules.js';

const optionFlags = Object.freeze({ iterations: 'iterations', games: 'games', gameConcurrency: 'game-concurrency', maxPlies: 'plies', maxNodes: 'nodes',
  maxDepth: 'depth', timeMs: 'time-ms', terminalWork: 'terminal-work', exploration: 'exploration',
  explorationPlies: 'exploration-plies', outcomeWeight: 'outcome-weight', steps: 'steps', batchSize: 'batch-size',
  learningRate: 'learning-rate', replaySize: 'replay-size', seed: 'seed', arenaPairs: 'arena-pairs', arenaConcurrency: 'arena-concurrency',
  minPairs: 'min-pairs', arenaPlies: 'arena-plies', promotionScore: 'promotion-score', keepIterations: 'keep-iterations', device: 'device' });
// Environment mistakes disable training without preventing the classical
// server or saved-game review from loading. Validate them per manager below.
const runnerDefaults = getSelfPlayDefaults();
const safeDefaults = { ...runnerDefaults, checkpoint: DEFAULT_CHECKPOINT, python: DEFAULT_PYTHON,
  device: ['auto', 'cpu', 'cuda'].includes(runnerDefaults.device) ? runnerDefaults.device : 'auto' };
export const TRAINING_DEFAULTS = Object.freeze(Object.fromEntries(Object.keys(optionFlags).map(key => [key, safeDefaults[key]])));
const ITERATION_ID = /^iteration-\d{8,12}$/;
const GAME_ID = /^(selfplay|arena)-\d{3,6}$/;
const MAX_GAME_BYTES = 16 * 1024 * 1024;
const MAX_CACHE_BYTES = 32 * 1024 * 1024;
const missing = error => error.code === 'ENOENT';
const failure = (message, statusCode = 400) => Object.assign(new Error(message), { statusCode });
const clone = value => structuredClone(value);

export function validateTrainingOptions(input = {}) {
  if (!input || typeof input !== 'object' || Array.isArray(input) || ![Object.prototype, null].includes(Object.getPrototypeOf(input))) {
    throw failure('Training options must be a JSON object.');
  }
  const args = [];
  for (const [key, value] of Object.entries(input)) {
    if (!Object.hasOwn(optionFlags, key)) throw failure(`Unknown training option: ${key}.`);
    if (key === 'device' ? typeof value !== 'string' : typeof value !== 'number' || !Number.isFinite(value)) {
      throw failure(`${key} must be ${key === 'device' ? 'auto, cpu or cuda' : 'a finite number'}.`);
    }
    args.push(`--${optionFlags[key]}`, String(value));
  }
  const parsed = parseArguments(args, safeDefaults);
  return Object.fromEntries(Object.keys(optionFlags).map(key => [key, parsed[key]]));
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
    workerFactory = data => new Worker(new URL('./training-worker.js', import.meta.url), { workerData: data }), availability } = {}) {
    this.options = { ...runnerDefaults, runDir: path.resolve(runDir), checkpoint: path.resolve(checkpoint), python: path.resolve(python) };
    try { this.options = parseArguments([], this.options); }
    catch (error) {
      this.configurationError = `Invalid training configuration: ${error.message} Check TRANSFORMER_DEVICE, TRANSFORMER_CHECKPOINT and TRANSFORMER_PYTHON before starting.`;
    }
    this.workerFactory = workerFactory;
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

  async lockDirectories() {
    const checkpoint = await realpath(this.options.checkpoint).catch(error => { if (!missing(error)) throw error; return this.options.checkpoint; });
    return [this.options.runDir, `${checkpoint}.selfplay-lock`];
  }

  async locks() { return (await Promise.all((await this.lockDirectories()).map(directory => inspectRunLock(directory)))).filter(Boolean); }

  lockFailure(locks) {
    // A stale run lock must never hide a live owner of the shared checkpoint.
    const lock = locks.find(item => item.live) ?? locks.find(item => !item.recoverable) ?? locks[0];
    if (lock) return { available: false, external: lock.live, startedAt: lock.owner?.createdAt,
      reason: lock.live ? `Training is running outside this page (PID ${lock.owner.pid}). Review is available; stop it in its owning terminal.`
        : `A self-play lock needs manual review (${lock.reason || 'unknown owner'}): ${lock.file}` };
    return null;
  }

  async lockInfo() { return this.lockFailure(await this.locks()); }

  async reconcile() {
    if (this.recovering) return this.recovering;
    const recovering = (async () => {
      const locks = await this.locks();
      if (locks.some(lock => lock.live || !lock.recoverable)) return this.lockFailure(locks);
      if (!locks.length && !(await findInterruptedIterations(this.options)).length) return null;
      const releases = [];
      try {
        // Holding both locks closes the race with a terminal-launched runner.
        for (const directory of await this.lockDirectories()) releases.push(await acquireRunLock(directory));
        const recovered = await recoverInterruptedIterations(this.options);
        const latest = recovered.at(-1);
        if (latest && this.state.state === 'idle') {
          const state = latest.status === 'complete' ? 'completed' : latest.status;
          Object.assign(this.state, { state, phase: state, iteration: latest.iteration,
            startedAt: latest.startedAt, finishedAt: latest.finishedAt, error: latest.error ?? null });
        }
        return null;
      } catch (error) {
        return { available: false, reason: `Training recovery could not finish: ${error.message}` };
      } finally { for (const release of releases.reverse()) await release(); }
    })();
    this.recovering = recovering;
    try { return await recovering; }
    finally { if (this.recovering === recovering) this.recovering = null; }
  }

  async availability({ ignoreOwned = false } = {}) {
    if (this.settling) await this.finished;
    if (this.closed) return { available: false, reason: 'The local server is shutting down.' };
    if (!ignoreOwned && (this.worker || this.starting)) return { available: false, reason: 'A training run is already active.' };
    if (this.configurationError) return { available: false, reason: this.configurationError };
    const lock = await this.reconcile();
    if (lock) return lock;
    if (this.checkAvailability) return this.checkAvailability();
    for (const [name, file, reason] of [
      ['checkpoint', this.options.checkpoint, 'No trained checkpoint. Run npm run transformer:data, then npm run transformer:train.'],
      ['Python', this.options.python, 'Python environment missing. Run npm run transformer:setup.'],
      ['training suite', this.options.suite, 'The self-play starting-position suite is missing.'],
      ['arena suite', this.options.arenaSuite, 'The evaluation starting-position suite is missing.'],
    ]) {
      try { if (!(await stat(file)).isFile()) return { available: false, reason: `The ${name} must be a regular file.` }; }
      catch (error) { if (missing(error)) return { available: false, reason }; throw error; }
    }
    return { available: true };
  }

  status() { return clone(this.state); }

  async snapshot() {
    const availability = await this.availability();
    const iterations = await this.listIterations();
    let status = this.status();
    if (!this.worker && !this.starting && availability.external) {
      const latest = iterations[0];
      status = { ...status, state: 'external', phase: 'external', iteration: latest?.iteration ?? null,
        startedAt: availability.startedAt ?? null, error: availability.reason };
    }
    return { defaults: { ...TRAINING_DEFAULTS }, status, availability, iterations };
  }

  async start(input = {}) {
    const editable = validateTrainingOptions(input);
    if (this.worker || this.starting) throw failure('A training run is already active.', 409);
    // Reserve the manager before awaiting filesystem checks, so simultaneous
    // start requests cannot both launch workers.
    this.starting = true;
    this.pendingStop = false;
    try {
      const available = await this.availability({ ignoreOwned: true });
      if (this.closed) throw failure('The local server is shutting down.', 503);
      if (this.pendingStop) {
        Object.assign(this.state, { state: 'interrupted', phase: 'interrupted', finishedAt: new Date().toISOString() });
        return this.status();
      }
      if (!available.available) throw failure(available.reason, available.external ? 409 : 503);
      this.cancellation = new Int32Array(new SharedArrayBuffer(4));
      this.state = { state: 'running', phase: 'starting', iteration: null, startedAt: new Date().toISOString(), error: null, events: [] };
      const lockToken = randomUUID();
      const worker = this.workerFactory({ options: { ...this.options, ...editable }, cancelBuffer: this.cancellation.buffer, lockToken });
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
            for (const directory of await this.lockDirectories()) await recoverRunLock(directory, { exitedOwner: { pid: process.pid, token: lockToken } });
            const recovery = await this.reconcile();
            if (recovery && !this.state.error) this.state.error = recovery.reason;
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

  async folder(id) {
    if (typeof id !== 'string' || !ITERATION_ID.test(id)) throw failure('Invalid training iteration ID.');
    const root = await realpath(this.options.runDir).catch(error => { if (missing(error)) throw failure('Training iteration not found.', 404); throw error; });
    const target = path.join(root, id);
    let info;
    try { info = await lstat(target); }
    catch (error) { if (missing(error)) throw failure('Training iteration not found.', 404); throw error; }
    if (!info.isDirectory() || info.isSymbolicLink() || path.dirname(await realpath(target)) !== root) throw failure('Unsafe training iteration path.');
    return target;
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
    let entries;
    try { entries = await readdir(this.options.runDir, { withFileTypes: true }); }
    catch (error) { if (missing(error)) return []; throw error; }
    const ids = entries.filter(entry => entry.isDirectory() && ITERATION_ID.test(entry.name)).map(entry => entry.name).sort().reverse().slice(0, 100);
    const iterations = [];
    for (const id of ids) {
      try {
        const report = await this.report(await this.folder(id));
        iterations.push({ id, iteration: report.iteration ?? Number(id.slice(10)), status: report.status,
          promoted: Boolean(report.promoted), startedAt: report.startedAt ?? null, finishedAt: report.finishedAt ?? null, error: report.error ?? null });
      } catch (error) {
        // Retention can remove a folder during a read. Corrupt retained reports
        // remain visible as errors instead of hiding the rest of the history.
        if (missing(error) || error.statusCode === 404) continue;
        iterations.push({ id, iteration: Number(id.slice(10)), status: 'unreadable', promoted: false, error: error.message });
      }
    }
    return iterations;
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
