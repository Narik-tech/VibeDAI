import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { lstat, mkdir, realpath, stat } from 'node:fs/promises';
import { once } from 'node:events';
import { finished } from 'node:stream/promises';
import path from 'node:path';
import { DEFAULT_PYTHON, PROJECT_ROOT } from '../src/transformer-runtime.js';
import { acquireRunLock, atomicWrite } from './transformer-selfplay-store.js';

export const FRESH_MODEL_CONFIG = Object.freeze({ width: 512, heads: 8, layers: 6, feedforward: 2048, parameters: 20000257 });
const defaults = {
  device: ['auto', 'cpu', 'cuda'].includes(process.env.TRANSFORMER_DEVICE) ? process.env.TRANSFORMER_DEVICE : 'auto',
  steps: 1000, batchSize: 1, learningRate: .0003, maxTokens: 512, seed: 42,
  samples: 4096, teacherNodes: 2000, teacherTimeMs: 1000,
};
export const getFreshTrainingDefaults = () => ({ ...defaults });
const json = value => `${JSON.stringify(value, null, 2)}\n`;
const emit = (event, fields) => console.log(JSON.stringify({ event, ...fields }));
const absent = error => { if (error.code !== 'ENOENT') throw error; return null; };
function cancelled() { const error = new Error('Fresh training stopped.'); error.name = 'AbortError'; return error; }
function checkStop(shouldStop) { if (shouldStop()) throw cancelled(); }

/** Validate public UI controls; filesystem and executable paths belong to the manager. */
export function validateFreshTrainingOptions(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input) || ![Object.prototype, null].includes(Object.getPrototypeOf(input))) throw new Error('Fresh training options must be a plain object.');
  for (const name of Object.keys(input)) if (!Object.hasOwn(defaults, name)) throw new Error(`Unknown fresh training option: ${name}.`);
  const options = { ...defaults, ...input };
  for (const [name, min, max] of [
    ['steps', 1, 1000000], ['batchSize', 1, 128], ['maxTokens', 16, 4096], ['seed', 0, 0xffffffff],
    ['samples', 1, 100000], ['teacherNodes', 10, 1000000], ['teacherTimeMs', 1, 60000],
  ]) if (!Number.isSafeInteger(options[name]) || options[name] < min || options[name] > max) throw new Error(`Invalid ${name}: expected integer ${min}..${max}.`);
  if (!Number.isFinite(options.learningRate) || options.learningRate <= 0 || options.learningRate > .1) throw new Error('Invalid learningRate.');
  if (!['auto', 'cpu', 'cuda'].includes(options.device)) throw new Error('device must be auto, cpu or cuda.');
  return options;
}

/** The manager allocates an isolated run folder; fresh runs never overwrite checkpoints. */
function validateRunOptions(input) {
  const { runDir, checkpoint, python = process.env.TRANSFORMER_PYTHON || DEFAULT_PYTHON, resume } = input;
  const controls = Object.fromEntries(Object.keys(defaults).filter(name => Object.hasOwn(input, name)).map(name => [name, input[name]]));
  if (resume !== undefined) throw new Error('Fresh training cannot resume an existing checkpoint.');
  const options = { ...validateFreshTrainingOptions(controls), runDir, checkpoint, python };
  for (const name of ['runDir', 'checkpoint', 'python']) {
    if (typeof options[name] !== 'string' || !options[name].trim()) throw new Error(`${name} must be a path.`);
    options[name] = path.resolve(options[name]);
  }
  if (path.dirname(options.checkpoint) !== options.runDir || !/\.pt$/i.test(options.checkpoint)) throw new Error('Fresh checkpoint must be a .pt file directly inside its run directory.');
  return options;
}

/** Run one owned process and wait for exit even when cancellation or logging fails. */
async function runStage(command, log, shouldStop, onLine, {
  spawnProcess = spawn, stopGraceMs = 2000, exitGraceMs = 1000,
} = {}) {
  checkStop(shouldStop);
  const child = spawnProcess(command.executable, command.args, {
    cwd: PROJECT_ROOT, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let poll, killTimer, exitTimer, failure, tail = '', exited = false, stopping = false;
  const buffers = { stdout: '', stderr: '' };
  let failOutput;
  try {
    await new Promise((resolve, reject) => {
      let settled = false;
      const settle = (code, signal) => {
        if (settled) return;
        settled = true;
        if (shouldStop()) reject(cancelled());
        else if (failure) reject(failure);
        else if (code === 0) resolve();
        else reject(new Error(`${command.stage} exited (${signal || code}). ${tail.trim()}`));
      };
      const stop = () => {
        if (stopping || exited) return;
        stopping = true;
        child.kill();
        killTimer = setTimeout(() => { if (!exited) child.kill('SIGKILL'); }, stopGraceMs);
      };
      failOutput = error => { failure ??= error; stop(); };
      const output = (stream, chunk) => {
        if (!log.destroyed) log.write(chunk);
        const text = chunk.toString();
        tail = (tail + text).slice(-6000);
        const lines = (buffers[stream] + text).split(/\r?\n/);
        buffers[stream] = lines.pop().slice(-65536);
        try { for (const line of lines) if (line.trim()) onLine(line, stream); }
        catch (error) { failOutput(error); }
      };
      log.on('error', failOutput);
      for (const stream of ['stdout', 'stderr']) {
        child[stream].on('data', chunk => output(stream, chunk));
        child[stream].once('error', failOutput);
      }
      child.once('error', error => { failure ??= error; if (!child.pid) settle(null); else stop(); });
      child.once('exit', (code, signal) => {
        exited = true; clearTimeout(killTimer);
        // A descendant may retain an inherited pipe after the owned process exits.
        exitTimer = setTimeout(() => { child.stdout.destroy(); child.stderr.destroy(); settle(code, signal); }, exitGraceMs);
      });
      child.once('close', (code, signal) => {
        if (settled) return;
        exited = true;
        try { for (const stream of ['stdout', 'stderr']) if (buffers[stream].trim()) onLine(buffers[stream], stream); }
        catch (error) { failure ??= error; }
        settle(code, signal);
      });
      poll = setInterval(() => { if (shouldStop()) stop(); }, 50);
    });
  } finally {
    clearInterval(poll); clearTimeout(killTimer); clearTimeout(exitTimer);
    log.off('error', failOutput);
  }
}

export async function runFreshTraining(input, {
  shouldStop = () => false, onEvent = emit, lockToken, spawnProcess, stopGraceMs, exitGraceMs,
} = {}) {
  const options = validateRunOptions(input);
  checkStop(shouldStop);
  await mkdir(options.runDir, { recursive: true });
  options.runDir = await realpath(options.runDir);
  options.checkpoint = path.join(options.runDir, path.basename(options.checkpoint));
  const release = await acquireRunLock(options.runDir, { token: lockToken });
  const files = {
    data: path.join(options.runDir, 'training.jsonl'), log: path.join(options.runDir, 'train.log'),
    report: path.join(options.runDir, 'report.json'), checkpoint: options.checkpoint,
  };
  let log, logFailure;
  const report = {
    version: 1, id: path.basename(options.runDir), mode: 'fresh20m', status: 'running', startedAt: new Date().toISOString(),
    folder: options.runDir, checkpoint: path.basename(options.checkpoint), checkpointPath: options.checkpoint, data: files.data, log: files.log,
    model: { ...FRESH_MODEL_CONFIG, policy: true }, options, promoted: false,
  };
  const save = () => atomicWrite(files.report, json(report));
  let ownsReport = false;
  try {
    for (const file of [files.checkpoint, files.data, files.log, files.report,
      path.join(options.runDir, 'data-command.json'), path.join(options.runDir, 'train-command.json')]) {
      if (await lstat(file).catch(absent)) throw new Error('Fresh training requires a new run folder with no previous data, logs or checkpoint.');
    }
    await save(); ownsReport = true;
    log = createWriteStream(files.log, { flags: 'wx' });
    log.on('error', error => { logFailure ??= error; });
    await once(log, 'open');
    const generator = {
      stage: 'Data generation', executable: process.execPath,
      args: [path.join(PROJECT_ROOT, 'scripts/transformer-data.js'), '--output', files.data,
        '--samples', String(options.samples), '--nodes', String(options.teacherNodes),
        '--time-ms', String(options.teacherTimeMs), '--seed', String(options.seed)],
    };
    await atomicWrite(path.join(options.runDir, 'data-command.json'), json(generator));
    onEvent('data-start', { folder: options.runDir, samples: options.samples, teacherNodes: options.teacherNodes });
    await runStage(generator, log, shouldStop, line => {
      const progress = /Teacher labels:\s*(\d+)\/(\d+)/.exec(line);
      if (progress) {
        report.dataProgress = { samples: Number(progress[1]), total: Number(progress[2]), totalSamples: Number(progress[2]) };
        onEvent('data-progress', report.dataProgress);
      }
    }, { spawnProcess, stopGraceMs, exitGraceMs });
    checkStop(shouldStop);
    if (logFailure) throw logFailure;
    if (!(await stat(files.data)).size) throw new Error('Data generation produced an empty training file.');
    const trainer = {
      stage: 'Training', executable: options.python,
      args: ['-u', path.join(PROJECT_ROOT, 'neural/train.py'), '--data', files.data, '--output', files.checkpoint,
        '--width', '512', '--heads', '8', '--layers', '6', '--feedforward', '2048', '--policy', 'on',
        '--device', options.device, '--steps', String(options.steps), '--batch-size', String(options.batchSize),
        '--learning-rate', String(options.learningRate), '--max-tokens', String(options.maxTokens),
        '--seed', String(options.seed), '--save-every', String(Math.min(100, options.steps)),
        '--log-every', String(Math.max(1, Math.min(10, options.steps))),
        '--label', 'Experimental supervised 20M transformer with component policy; strength unverified'],
    };
    report.stage = 'training'; await save();
    await atomicWrite(path.join(options.runDir, 'train-command.json'), json(trainer));
    onEvent('training-start', { steps: options.steps, batchSize: options.batchSize, maxTokens: options.maxTokens, checkpoint: files.checkpoint });
    await runStage(trainer, log, shouldStop, line => {
      let progress;
      try { progress = JSON.parse(line); } catch { return; }
      if (!progress || typeof progress !== 'object' || Array.isArray(progress)) return;
      const { event: trainerEvent, ...fields } = progress;
      report.training = { ...report.training, ...fields, trainerEvent };
      if (trainerEvent === 'start') {
        report.config = fields.config; report.parameters = fields.parameters;
        onEvent('training-start', { ...fields, steps: options.steps, batchSize: options.batchSize, maxTokens: options.maxTokens });
      } else onEvent('training-progress', { ...fields, trainerEvent });
    }, { spawnProcess, stopGraceMs, exitGraceMs });
    checkStop(shouldStop);
    if (logFailure) throw logFailure;
    if (!(await stat(files.checkpoint).catch(absent))?.size) throw new Error('Training completed without a saved checkpoint.');
    if (report.parameters !== FRESH_MODEL_CONFIG.parameters || report.config?.policy_head !== true
      || !['width', 'heads', 'layers', 'feedforward'].every(name => report.config?.[name] === FRESH_MODEL_CONFIG[name])
      || report.training?.trainerEvent !== 'complete' || !Number.isSafeInteger(report.training?.trainedSteps) || report.training.trainedSteps < 1) {
      throw new Error('Trainer did not confirm a completed 20M model with its policy head.');
    }
    log.end(); await finished(log);
    report.trainedSteps = report.training?.trainedSteps;
    report.loss = report.training?.loss;
    report.status = 'complete'; report.finishedAt = new Date().toISOString(); report.checkpointAvailable = true;
    await save();
    onEvent('training-complete', { checkpoint: files.checkpoint, report: files.report, trainedSteps: report.training?.trainedSteps, promoted: false });
    return report;
  } catch (error) {
    if (ownsReport) {
      report.status = shouldStop() || error.name === 'AbortError' ? 'interrupted' : 'failed';
      report.error = error.message; report.finishedAt = new Date().toISOString();
      report.checkpointAvailable = Boolean((await stat(files.checkpoint).catch(absent))?.size);
      await save();
    }
    throw error;
  } finally {
    if (log) { if (!log.destroyed) log.end(); await finished(log).catch(() => {}); }
    await release();
  }
}
