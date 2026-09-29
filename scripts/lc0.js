#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { DEFAULT_PYTHON, PROJECT_ROOT } from '../src/transformer-runtime.js';

const help = `Usage: node scripts/lc0.js inspect WEIGHTS.pb.gz
       node scripts/lc0.js data [options]
       node scripts/lc0.js train [options]
       node scripts/lc0.js evaluate [options]
       node scripts/lc0.js smoke [options]
       node scripts/lc0.js test [unittest options]

inspect  Print the LCZero baseline format, tensor inventory, and provenance.
data     Generate a full-rules 5D curriculum with disjoint validation sources.
train    Train or resume an LCZero transfer model on 5D JSONL examples.
evaluate Measure value and component-policy agreement on held-out examples.
smoke    Verify neural inference and complete-turn legality on 5D fixtures.
test     Run neural transfer, legacy, and curriculum regression tests.

Use COMMAND --help for command options.
Python uses TRANSFORMER_PYTHON or .venv-transformer. Create that environment
with npm run transformer:setup if needed.`;

function run(executable, arguments_) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, arguments_, { cwd: PROJECT_ROOT, windowsHide: true, stdio: 'inherit' });
    child.once('error', reject);
    child.once('exit', (code, signal) => code === 0 ? resolve()
      : reject(new Error(signal ? `Command stopped by ${signal}.` : `Command exited with code ${code}.`)));
  });
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  if (!command || command === '--help' || command === '-h') {
    if (args.length) throw new Error('Help accepts no extra arguments.');
    console.log(help);
    return;
  }
  if (!['inspect', 'data', 'train', 'evaluate', 'smoke', 'test'].includes(command)) {
    throw new Error(`Unknown LCZero command: ${command}.\n${help}`);
  }
  if (command === 'data' || command === 'smoke') {
    await run(process.execPath, [path.join(PROJECT_ROOT, 'scripts', `lc0-${command}.js`), ...args]);
    return;
  }
  const python = process.env.TRANSFORMER_PYTHON || DEFAULT_PYTHON;
  if (!existsSync(python)) throw new Error('Run npm run transformer:setup first (use -- --python PATH if needed).');
  if (command === 'test') {
    await run(python, ['-m', 'unittest', 'discover', '-s', 'neural', '-t', '.', ...args]);
    if (!args.includes('--help') && !args.includes('-h')) {
      await run(process.execPath, ['--test', path.join(PROJECT_ROOT, 'test', 'lc0-data.test.js'),
        path.join(PROJECT_ROOT, 'test', 'lc0-smoke.test.js'),
        path.join(PROJECT_ROOT, 'test', 'lc0-training-suite.test.js')]);
    }
    return;
  }
  const script = { inspect: 'lc0_weights.py', train: 'lc0_train.py', evaluate: 'lc0_evaluate.py' }[command];
  await run(python, ['-u', path.join(PROJECT_ROOT, 'neural', script), ...args]);
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
