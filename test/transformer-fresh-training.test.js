import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtemp, readFile, rm, writeFile, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { FRESH_MODEL_CONFIG, getFreshTrainingDefaults, runFreshTraining, validateFreshTrainingOptions } from '../scripts/transformer-fresh-training.js';

const field = (args, name) => args[args.indexOf(name) + 1];
const close = (child, code = 0) => {
  child.stdout.end(); child.stderr.end(); child.emit('exit', code, null); child.emit('close', code, null);
};
const trainingStart = { event: 'start', config: { width: 512, heads: 8, layers: 6, feedforward: 2048, max_tokens: 512, policy_head: true }, parameters: 20000257, device: 'cpu' };
async function fixture(t) {
  const runDir = await mkdtemp(path.join(tmpdir(), 'vibe-fresh20m-'));
  t.after(() => rm(runDir, { recursive: true, force: true }));
  return { ...getFreshTrainingDefaults(), runDir, checkpoint: path.join(runDir, 'model.pt'), python: process.execPath, device: 'cpu', steps: 2, samples: 2 };
}
function fakeSpawn(actions, calls) {
  return (executable, args, options) => {
    const child = new EventEmitter(); child.pid = 123;
    child.stdout = new PassThrough(); child.stderr = new PassThrough();
    child.kill = () => { setImmediate(() => close(child, 1)); return true; };
    calls.push({ executable, args, options, child });
    const action = actions.shift();
    setImmediate(() => Promise.resolve().then(() => action(child, args)).catch(error => child.emit('error', error)));
    return child;
  };
}
async function dataSuccess(child, args) {
  await writeFile(field(args, '--output'), '{"position":{},"value":1,"policy":[]}\n');
  child.stderr.write('Teacher labels: 2/'); child.stderr.write('2; approximate values\n');
  close(child);
}
async function trainerSuccess(child, args) {
  child.stdout.write(`${JSON.stringify(trainingStart)}\n`);
  child.stdout.write('{"event":"train","step":1,"loss":0.5}\n');
  await writeFile(field(args, '--output'), 'checkpoint bytes');
  child.stdout.write('{"event":"complete","trainedSteps":2,"loss":0.4}\n');
  close(child);
}

test('fresh training generates policy data then runs the exact 20M architecture without resume', async t => {
  const options = await fixture(t), calls = [], events = [];
  const report = await runFreshTraining(options, {
    spawnProcess: fakeSpawn([dataSuccess, trainerSuccess], calls), onEvent: (event, fields) => events.push({ event, ...fields }),
  });
  assert.equal(calls.length, 2);
  assert.ok(calls[0].args[0].endsWith('transformer-data.js'));
  assert.equal(field(calls[0].args, '--samples'), '2');
  assert.equal(field(calls[0].args, '--nodes'), '2000');
  assert.equal(field(calls[0].args, '--time-ms'), '1000');
  const args = calls[1].args;
  assert.equal(field(args, '--output'), options.checkpoint);
  for (const [name, value] of Object.entries(FRESH_MODEL_CONFIG)) if (name !== 'parameters') assert.equal(field(args, `--${name}`), String(value));
  assert.equal(field(args, '--policy'), 'on');
  assert.equal(field(args, '--max-tokens'), '512');
  assert.equal(field(args, '--batch-size'), '1');
  assert.ok(!args.includes('--resume'));
  assert.ok(calls.every(call => call.options.windowsHide === true));
  assert.equal(report.status, 'complete');
  assert.equal(report.mode, 'fresh20m');
  assert.equal(report.id, path.basename(options.runDir));
  assert.equal(report.parameters, 20000257);
  assert.equal(report.trainedSteps, 2);
  assert.equal(report.checkpoint, 'model.pt');
  assert.equal(report.promoted, false);
  assert.equal(report.config.policy_head, true);
  assert.ok(events.some(event => event.event === 'data-progress' && event.samples === 2 && event.total === 2));
  assert.ok(events.some(event => event.event === 'training-start' && event.parameters === 20000257));
  assert.ok(events.some(event => event.event === 'training-progress' && event.step === 1 && event.loss === .5));
  assert.equal(events.at(-1).event, 'training-complete');
  const saved = JSON.parse(await readFile(path.join(options.runDir, 'report.json'), 'utf8'));
  assert.deepEqual(saved, report);
  assert.match(await readFile(path.join(options.runDir, 'train.log'), 'utf8'), /Teacher labels: 2\/2/);
  const command = JSON.parse(await readFile(path.join(options.runDir, 'train-command.json'), 'utf8'));
  assert.deepEqual(command.args, args);
  await assert.rejects(stat(path.join(options.runDir, '.selfplay.lock')), { code: 'ENOENT' });
});

test('fresh training refuses to overwrite prior data or checkpoints', async t => {
  const options = await fixture(t);
  await writeFile(options.checkpoint, 'existing model');
  await assert.rejects(runFreshTraining(options, { spawnProcess: () => assert.fail('must not start a process') }), /new run folder/);
  assert.equal(await readFile(options.checkpoint, 'utf8'), 'existing model');
});

test('generator failure persists report and logs without starting the trainer', async t => {
  const options = await fixture(t), calls = [];
  await assert.rejects(runFreshTraining(options, {
    onEvent() {}, spawnProcess: fakeSpawn([(child) => { child.stderr.write('teacher failed\n'); close(child, 7); }], calls),
  }), /Data generation exited \(7\).*teacher failed/);
  assert.equal(calls.length, 1);
  const report = JSON.parse(await readFile(path.join(options.runDir, 'report.json'), 'utf8'));
  assert.equal(report.status, 'failed');
  assert.equal(report.checkpointAvailable, false);
});

test('cancel waits for the owned trainer to exit and retains the saved checkpoint', { timeout: 3000 }, async t => {
  const options = await fixture(t), calls = [], signals = [];
  let stopped = false, exited = false;
  const actions = [dataSuccess, async (child, args) => {
    await writeFile(field(args, '--output'), 'saved partial checkpoint');
    child.kill = (signal = 'SIGTERM') => {
      signals.push(signal);
      if (signal === 'SIGKILL') setTimeout(() => { exited = true; close(child, 1); }, 20);
      return true;
    };
    stopped = true;
  }];
  await assert.rejects(runFreshTraining(options, {
    spawnProcess: fakeSpawn(actions, calls), shouldStop: () => stopped, onEvent() {}, stopGraceMs: 10,
  }), { name: 'AbortError' });
  assert.equal(exited, true);
  assert.deepEqual(signals, ['SIGTERM', 'SIGKILL']);
  assert.equal(await readFile(options.checkpoint, 'utf8'), 'saved partial checkpoint');
  const report = JSON.parse(await readFile(path.join(options.runDir, 'report.json'), 'utf8'));
  assert.equal(report.status, 'interrupted');
  assert.equal(report.checkpointAvailable, true);
  await assert.rejects(stat(path.join(options.runDir, '.selfplay.lock')), { code: 'ENOENT' });
});

test('data generation can be cancelled before a trainer is launched', { timeout: 3000 }, async t => {
  const options = await fixture(t), calls = [];
  let stopped = false;
  await assert.rejects(runFreshTraining(options, {
    spawnProcess: fakeSpawn([() => { stopped = true; }], calls), shouldStop: () => stopped, onEvent() {},
  }), { name: 'AbortError' });
  assert.equal(calls.length, 1);
});

test('failed trainer reports stderr even when a descendant holds its pipes open', { timeout: 3000 }, async t => {
  const options = await fixture(t), calls = [];
  await assert.rejects(runFreshTraining(options, {
    spawnProcess: fakeSpawn([dataSuccess, child => {
      child.stderr.write('CUDA out of memory\n'); child.emit('exit', 1, null);
    }], calls), onEvent() {}, exitGraceMs: 10,
  }), /Training exited \(1\).*CUDA out of memory/);
  assert.equal(calls[1].child.stdout.destroyed, true);
});

test('a successful trainer exit without a checkpoint is a failed run', async t => {
  const options = await fixture(t), calls = [];
  await assert.rejects(runFreshTraining(options, {
    spawnProcess: fakeSpawn([dataSuccess, child => close(child)], calls), onEvent() {},
  }), /ENOENT|without a saved checkpoint/);
  const report = JSON.parse(await readFile(path.join(options.runDir, 'report.json'), 'utf8'));
  assert.equal(report.status, 'failed');
});

test('fresh training validates bounded controls and prevents checkpoint escape or resume', () => {
  for (const overrides of [
    { steps: 0 }, { batchSize: 129 }, { maxTokens: 15 }, { maxTokens: 4097 }, { seed: -1 },
    { samples: 0 }, { teacherNodes: 9 }, { teacherTimeMs: 60001 }, { learningRate: NaN },
    { device: 'gpu' }, { checkpoint: path.resolve('artifacts/model.pt') }, { resume: 'old.pt' },
  ]) assert.throws(() => validateFreshTrainingOptions(overrides));
  assert.deepEqual(validateFreshTrainingOptions({}), getFreshTrainingDefaults());
  assert.ok(!Object.hasOwn(getFreshTrainingDefaults(), 'python'));
  for (const input of [null, [], 'options', { python: 'command' }, { runDir: 'dir' }, { unknown: true }]) assert.throws(() => validateFreshTrainingOptions(input));
});

test('trainer must confirm the 20M policy architecture and completed steps', async t => {
  const options = await fixture(t), calls = [];
  await assert.rejects(runFreshTraining(options, {
    spawnProcess: fakeSpawn([dataSuccess, async (child, args) => {
      child.stdout.write(`${JSON.stringify({ ...trainingStart, parameters: 12345 })}\n`);
      await writeFile(field(args, '--output'), 'wrong model');
      child.stdout.write('{"event":"complete","trainedSteps":2}\n');
      close(child);
    }], calls), onEvent() {},
  }), /did not confirm a completed 20M model/);
  const report = JSON.parse(await readFile(path.join(options.runDir, 'report.json'), 'utf8'));
  assert.equal(report.status, 'failed');
});
