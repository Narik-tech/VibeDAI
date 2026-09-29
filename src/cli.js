#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { Worker } from 'node:worker_threads';
import { GameSession } from './session.js';
import { TransformerRuntime, forwardInference } from './transformer-runtime.js';
import { LeelaRuntime } from './leela-runtime.js';
import { isNeuralEngine } from './leela-config.js';

const HELP = `Vibe-D AI — full-turn analysis for 5D Chess

Usage: node src/cli.js [options]
  --file PATH       Load a 5DPGN game or 5DFEN position
  --pgn TEXT        Load notation directly (quote it in your shell)
  --variant NAME    Starting variant, default: standard
  --engine NAME     classical (default), transformer, or leela (Leela in a 5D Trenchcoat)
  --time SECONDS    Think time, default: 5 (maximum 3600)
  --depth NUMBER    Maximum complete-turn plies, default: 8; neural engines use 0 for dynamic
  --nodes NUMBER    Search + generation work budget, default: 2000000
  --threads NUMBER  Classical search CPU threads, default: 1 (maximum 16)
  --qdepth NUMBER   Quiescence turn depth, default: 2
  --json           Print machine-readable analysis JSON
  --play           Also print the 5DPGN after playing the recommendation
  --help           Show this message

Examples:
  node src/cli.js --time 30 --depth 10
  node src/cli.js --file examples/time-travel.5dpgn --json
  node src/cli.js --variant two_timelines --time 10 --play

Ctrl+C stops search and returns the last available legal recommendation.
Scores are centipawns from White's perspective; strength is not Elo-rated.
The node budget is shared across search threads.
`;

function parseArgs(args) {
  const parsed = {};
  const values = new Set(['file', 'pgn', 'variant', 'engine', 'time', 'depth', 'nodes', 'threads', 'qdepth']);
  const flags = new Set(['json', 'play', 'help']);
  for (let index = 0; index < args.length; index++) {
    const key = args[index].replace(/^--/, '');
    if (!args[index].startsWith('--')) throw new Error(`Unknown argument: ${args[index]}`);
    if (flags.has(key)) parsed[key] = true;
    else if (values.has(key) && args[index + 1] !== undefined) parsed[key] = args[++index];
    else throw new Error(`Unknown option or missing value: --${key}`);
  }
  if (parsed.file && parsed.pgn) throw new Error('Use either --file or --pgn.');
  return parsed;
}

function integer(value, fallback, min, max, name) {
  const result = value === undefined ? fallback : Number(value);
  if (!Number.isInteger(result) || result < min || result > max) throw new Error(`${name} must be between ${min} and ${max}.`);
  return result;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return console.log(HELP);
  const seconds = args.time === undefined ? 5 : Number(args.time);
  if (!Number.isFinite(seconds) || seconds < 0.01 || seconds > 3600) throw new Error('--time must be between 0.01 and 3600 seconds.');
  const options = {
    engine: args.engine || 'classical',
    timeMs: Math.round(seconds * 1000),
    maxDepth: integer(args.depth, 8, isNeuralEngine(args.engine) ? 0 : 1, 64, '--depth'),
    maxNodes: integer(args.nodes, 2000000, 1, 1000000000, '--nodes'),
    threads: integer(args.threads, 1, 1, 16, '--threads'),
    quiescenceDepth: integer(args.qdepth, 2, 0, 8, '--qdepth'),
  };
  if (!['classical', 'transformer', 'leela'].includes(options.engine)) throw new Error('--engine must be classical, transformer, or leela.');
  const pgn = args.file ? await readFile(args.file, 'utf8') : args.pgn;
  const game = new GameSession({ variant: args.variant, pgn });
  const cancelled = new Int32Array(new SharedArrayBuffer(4));
  const runtime = options.engine === 'leela' ? new LeelaRuntime() : new TransformerRuntime();
  let worker, result;
  const interrupt = () => {
    Atomics.store(cancelled, 0, 1);
    if (!worker) runtime.close();
  };
  process.on('SIGINT', interrupt);
  try {
    const modelInfo = isNeuralEngine(options.engine) ? await runtime.start() : null;
    worker = new Worker(new URL('./worker.js', import.meta.url), {
      workerData: { position: game.position, options, model: modelInfo?.model, cancelBuffer: cancelled.buffer },
    });
    let progress;
    result = await new Promise((resolve, reject) => {
      const deadline = setTimeout(() => {
        void worker.terminate();
        if (progress) resolve({ ...progress, stoppedReason: 'hard-time-limit' });
        else reject(new Error('No legal action was found before the hard time limit. Try a longer budget.'));
      }, options.timeMs + 5000);
      worker.on('message', message => {
        if (message.type === 'evaluate' || message.type === 'policy') { void forwardInference(worker, runtime, message); return; }
        if (message.type === 'progress') {
          progress = message.result;
          if (!args.json) console.error(`depth ${progress.depth} | ${progress.score === null ? 'unscored' : (progress.score / 100).toFixed(2)} | ${progress.nodes} work nodes | ${progress.notation || 'searching'}`);
        }
        if (message.type === 'result') { clearTimeout(deadline); resolve(message.result); }
        if (message.type === 'error') { clearTimeout(deadline); reject(new Error(message.error)); }
      });
      worker.on('error', error => { clearTimeout(deadline); reject(error); });
      worker.on('exit', code => {
        clearTimeout(deadline);
        reject(new Error(`Search worker exited (${code}) before returning a result.`));
      });
    });
  } finally {
    runtime.close();
    process.off('SIGINT', interrupt);
    if (worker) await worker.terminate();
  }
  if (args.play && Array.isArray(result.bestAction)) {
    game.play(result.bestAction);
    result.pgn = game.chess.export();
  }
  if (args.json) console.log(JSON.stringify(result, null, 2));
  else {
    console.log(`\n${result.notation ? `Best turn: ${result.notation}` : `Result: ${result.status}`}`);
    console.log(`White score: ${result.score === null ? 'unavailable' : result.scoreType === 'mate' ? `mate ${result.mateIn}` : (result.score / 100).toFixed(2)} | Depth: ${result.depth} | Nodes: ${result.nodes} | Time: ${Math.round(result.elapsedMs)} ms`);
    if (result.pvNotation?.length) console.log(`Variation: ${result.pvNotation.join(' / ')}`);
    if (result.stoppedReason === 'policy') console.log('Legal turns exist, but none satisfy the optional-board search restriction.');
    else if (!result.completed) console.log('Search is incomplete; increase the budget for a deeper result.');
    if (result.pgn) console.log(`\n${result.pgn}`);
  }
}

main().catch(error => { console.error(`Error: ${error.message}`); process.exitCode = 1; });
