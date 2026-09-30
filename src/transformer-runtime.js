import { spawn } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

export const PROJECT_ROOT = fileURLToPath(new URL('../', import.meta.url));
export const DEFAULT_PYTHON = path.join(PROJECT_ROOT, '.venv-transformer', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
export const DEFAULT_CHECKPOINT = path.join(PROJECT_ROOT, 'artifacts/transformer/model.pt');

/** One owned, persistent Python process per server; workers never own GPU children. */
export class TransformerRuntime {
  constructor({ python = process.env.TRANSFORMER_PYTHON || DEFAULT_PYTHON,
    checkpoint = process.env.TRANSFORMER_CHECKPOINT || DEFAULT_CHECKPOINT,
    device = process.env.TRANSFORMER_DEVICE || 'auto', startupMs = 60000, requestMs = 30000, spawnProcess = spawn } = {}) {
    this.python = python;
    this.checkpoint = path.resolve(checkpoint);
    this.device = device;
    this.startupMs = startupMs;
    this.requestMs = requestMs;
    this.spawnProcess = spawnProcess;
    this.pending = new Map();
    this.nextId = 0;
    this.nextGeneration = 0;
    this.state = 'unloaded';
    this.closed = false;
  }

  describe() {
    const missing = !existsSync(this.checkpoint) ? 'No trained checkpoint. See docs/transformer.md to set up the 800k transformer.'
      : !existsSync(this.python) ? 'Python environment missing. Run npm run transformer:setup.' : null;
    return {
      id: 'transformer', name: 'Transformer', available: !missing && !this.closed,
      status: missing ? 'setup-required' : this.state,
      description: 'Experimental transformer value network with ranked depth search over bounded full-turn candidates.',
      error: missing || this.error || undefined, model: this.info?.model, device: this.info?.device,
    };
  }

  async start() {
    if (this.closed) throw new Error('Transformer runtime is closed.');
    const description = this.describe();
    if (!description.available) throw new Error(description.error);
    const stamp = statSync(this.checkpoint).mtimeMs;
    if (this.info && this.checkpointStamp === stamp) return this.info;
    if (this.starting) return this.starting;
    if (this.child) this.stopProcess(new Error('Checkpoint changed; restarting transformer.'));
    this.state = 'starting';
    this.error = undefined;
    this.checkpointStamp = stamp;
    let child;
    try {
      child = this.spawnProcess(this.python, ['-u', path.join(PROJECT_ROOT, 'neural/service.py'), '--checkpoint', this.checkpoint, '--device', this.device], {
        cwd: PROJECT_ROOT, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });
    } catch (error) {
      this.state = 'error'; this.error = `Cannot start transformer: ${error.message}`;
      throw new Error(this.error);
    }
    this.child = child;
    let stderr = '';
    child.stderr.on('data', chunk => { stderr = (stderr + chunk.toString()).slice(-4000); });
    const lines = createInterface({ input: child.stdout });
    this.lines = lines;
    const starting = new Promise((resolve, reject) => {
      const timer = setTimeout(() => fail(new Error('Transformer startup timed out. Run npm run transformer:doctor.')), this.startupMs);
      this.rejectStartup = error => { clearTimeout(timer); reject(error); };
      const fail = error => {
        clearTimeout(timer);
        if (this.child !== child) return;
        this.state = 'error';
        this.error = error.message;
        this.stopProcess(error);
        reject(error);
      };
      child.once('error', error => fail(new Error(`Cannot start transformer: ${error.message}. Run npm run transformer:setup.`)));
      child.once('exit', (code, signal) => fail(new Error(`Transformer exited (${signal || code}). ${stderr.trim() || 'Run npm run transformer:doctor.'}`)));
      child.stdin.on('error', error => fail(error));
      child.stdout.on('error', error => fail(error));
      child.stderr.on('error', error => fail(error));
      lines.on('error', error => fail(error));
      lines.on('line', line => {
        // The old process can emit buffered output after checkpoint reload/close.
        if (this.child !== child) return;
        let message;
        try { message = JSON.parse(line); }
        catch { fail(new Error('Transformer returned invalid protocol data.')); return; }
        if (!message || typeof message !== 'object' || Array.isArray(message)) {
          fail(new Error('Transformer returned invalid protocol data.')); return;
        }
        if (message.ready !== undefined) {
          if (message.ready === false) { fail(new Error(typeof message.error === 'string' ? message.error : 'Transformer failed to load.')); return; }
          if (this.info || message.ready !== true || typeof message.device !== 'string' || !message.model || typeof message.model !== 'object' || Array.isArray(message.model)) {
            fail(new Error('Transformer returned an invalid ready message.')); return;
          }
          if (message.model.policyAvailable !== undefined && (typeof message.model.policyAvailable !== 'boolean'
            || (message.model.policyAvailable && (message.model.policyVersion !== 1
              || !Number.isSafeInteger(message.model.policyTrainedSteps) || message.model.policyTrainedSteps < 1)))) {
            fail(new Error('Transformer returned invalid policy metadata.')); return;
          }
          clearTimeout(timer);
          this.rejectStartup = null;
          this.info = { ...message, model: { ...message.model, device: message.device,
            runtimeGeneration: ++this.nextGeneration } };
          this.state = 'ready';
          resolve(this.info);
          return;
        }
        if (!this.info || !Number.isSafeInteger(message.id) || message.id < 1) {
          fail(new Error('Transformer returned an invalid request ID.')); return;
        }
        const request = this.pending.get(message.id);
        if (!request) return;
        const policy = request.kind === 'policy';
        const scores = policy ? message.scores : message.values;
        const invalidScores = policy && scores === null ? false
          : !Array.isArray(scores) || scores.length !== request.count || !scores.every(Number.isFinite);
        if ((!message.error && invalidScores)
          || (message.error !== undefined && typeof message.error !== 'string')
          || (message.context !== undefined && (policy || !Array.isArray(message.context) || message.context.length !== request.count || message.context.some(item => !item || typeof item !== 'object' || Array.isArray(item))))) {
          fail(new Error(policy ? 'Transformer returned invalid policy scores.' : 'Transformer returned invalid evaluations.')); return;
        }
        this.pending.delete(message.id);
        clearTimeout(request.timer);
        if (message.error) request.reject(new Error(message.error));
        else request.resolve({ ...message, runtimeGeneration: request.runtimeGeneration });
      });
    }).finally(() => { if (this.starting === starting) this.starting = null; });
    this.starting = starting;
    return starting;
  }

  async evaluate(positions, { runtimeGeneration } = {}) {
    if (!Array.isArray(positions) || positions.length < 1 || positions.length > 128) throw new Error('Transformer batches must contain 1–128 positions.');
    await this.start();
    return this.request({ positions }, positions.length, 'evaluate', runtimeGeneration);
  }

  /** Logits aligned with legal raw moves, larger = preferred by the mover. */
  async orderMoves(position, moves, { runtimeGeneration, withMetadata = false } = {}) {
    if (!Array.isArray(moves) || !moves.length) throw new Error('Policy requires legal component candidates.');
    await this.start();
    this.assertGeneration(runtimeGeneration);
    const response = this.info.model.policyAvailable !== true || moves.length > 16384
      ? { scores: null, runtimeGeneration: this.info.model.runtimeGeneration }
      : await this.request({ type: 'policy', position, moves }, moves.length, 'policy', runtimeGeneration);
    return withMetadata ? { scores: response.scores, runtimeGeneration: response.runtimeGeneration } : response.scores;
  }

  assertGeneration(expected) {
    if (this.closed || !this.child || this.state !== 'ready') throw new Error('Transformer runtime stopped before inference.');
    if (expected !== undefined && (!Number.isSafeInteger(expected) || expected < 1)) throw new Error('Invalid Transformer runtime generation.');
    if (expected !== undefined && expected !== this.info.model.runtimeGeneration) {
      throw new Error('Transformer checkpoint changed during analysis; restart analysis to use the new model.');
    }
  }

  request(body, count, kind, expectedGeneration) {
    this.assertGeneration(expectedGeneration);
    // Capture the service that owns this request. Reading current info when a
    // promise resumes could accidentally stamp an old score with a new model.
    const runtimeGeneration = this.info.model.runtimeGeneration;
    if (this.pending.size >= 8) throw new Error('Transformer is busy; wait for previous analysis to stop.');
    const id = ++this.nextId;
    const payload = `${JSON.stringify({ id, ...body })}\n`;
    if (Buffer.byteLength(payload) > 32 * 1024 * 1024) throw new Error('Transformer batch exceeds the 32 MiB protocol limit.');
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.state = 'error';
        this.error = 'Transformer inference timed out.';
        this.stopProcess(new Error(this.error));
      }, this.requestMs);
      this.pending.set(id, { resolve, reject, timer, count, kind, runtimeGeneration });
      try { this.child.stdin.write(payload); }
      catch (error) { this.stopProcess(error); }
    });
  }

  stopProcess(error = new Error('Transformer stopped.')) {
    const child = this.child;
    this.child = null;
    this.info = null;
    this.lines?.close();
    this.lines = null;
    this.rejectStartup?.(error);
    this.rejectStartup = null;
    child?.stdin.destroy();
    child?.kill();
    for (const request of this.pending.values()) { clearTimeout(request.timer); request.reject(error); }
    this.pending.clear();
    if (this.state !== 'error') this.state = this.closed ? 'closed' : 'unloaded';
  }

  close() { this.closed = true; this.stopProcess(); }
}

export function listEngines(runtime) {
  return { engines: [
    { id: 'classical', name: 'Classical search', available: true, status: 'ready', description: 'CPU alpha-beta search with handcrafted evaluation.' },
    runtime.describe(),
  ] };
}

export async function forwardInference(worker, runtime, message) {
  if (!['evaluate', 'policy'].includes(message?.type)) return;
  const policy = message.type === 'policy';
  let response;
  try {
    const options = { runtimeGeneration: message.runtimeGeneration };
    if (policy) {
      const result = typeof runtime.orderMoves === 'function'
        ? await runtime.orderMoves(message.position, message.moves, { ...options, withMetadata: true }) : null;
      // Older adapters and test doubles retain the scores-only interface.
      response = result && !Array.isArray(result) && typeof result === 'object' ? result : { scores: result };
    } else response = await runtime.evaluate(message.positions, options);
    if (message.runtimeGeneration !== undefined && response.runtimeGeneration !== undefined
      && response.runtimeGeneration !== message.runtimeGeneration) {
      throw new Error('Transformer checkpoint changed during analysis; restart analysis to use the new model.');
    }
  } catch (error) {
    response = { error: error.message };
  }
  // A timed-out or cancelled worker can have exited while inference was queued.
  try { worker.postMessage({ ...response, type: policy ? 'policyScores' : 'evaluations', id: message.id }); }
  catch { /* The worker no longer owns a response channel. */ }
}
