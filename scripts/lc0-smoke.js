#!/usr/bin/env node
// Optional end-to-end check. Owns one runtime and never changes the active model.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { TransformerRuntime } from '../src/transformer-runtime.js';
import { analyze } from '../src/transformer-search.js';
import { applyMove, canSubmit, createPosition, formatAction, pseudoMoves, validateAction } from '../src/rules.js';

const help = `Usage: node scripts/lc0-smoke.js --checkpoint FILE [--device auto|cpu|cuda]
       [--output FILE] [--time-ms 10000] [--nodes 20000]

Loads one trained LCZero transfer checkpoint, checks finite value/policy inference,
and validates searched complete turns and principal variations under the 5D rules.
Fixtures require temporal travel and a compound turn on two 4x4 boards.
This checks integration and legality; it does not measure playing strength.`;

export function parseSmokeArgs(args) {
  const result = { device: 'auto', timeMs: 10000, maxNodes: 20000 };
  for (let index = 0; index < args.length; index++) {
    const option = args[index];
    if (option === '--help') return { help: true };
    if (!['--checkpoint', '--device', '--output', '--time-ms', '--nodes'].includes(option)) throw new Error(`Unknown option: ${option}`);
    const value = args[++index];
    if (!value || value.startsWith('--')) throw new Error(`${option} requires a value.`);
    if (option === '--time-ms' || option === '--nodes') result[option === '--time-ms' ? 'timeMs' : 'maxNodes'] = Number(value);
    else result[option.slice(2)] = value;
  }
  if (!result.checkpoint) throw new Error('--checkpoint is required.');
  if (!['auto', 'cpu', 'cuda'].includes(result.device)) throw new Error('--device must be auto, cpu, or cuda.');
  for (const name of ['timeMs', 'maxNodes']) {
    if (!Number.isInteger(result[name]) || result[name] < 1 || result[name] > 1000000) throw new Error(`Invalid ${name}.`);
  }
  if (result.output && resolve(result.output) === resolve(result.checkpoint)) throw new Error('Report output must differ from the checkpoint.');
  return result;
}

export function smokeFixtures() {
  const square = [[12, 0, 0, 8], [0, 0, 2, 2], [1, 1, 0, 0], [7, 0, 0, 11]];
  return [
    { id: 'temporal-escape-4', requirement: 'time-travel', position: createPosition({
      pgn: '[Board "Custom"]\n[Size "4x4"]\n[3k/4/4/4:0:1:w]\n[4/2k1/2q1/K3:0:2:w]',
    }) },
    { id: 'two-rook4', requirement: 'compound-turn', position: {
      board: [null, [square], [structuredClone(square)]], action: 0, promotions: [8, 7],
    } },
  ];
}

export async function verifyFixture(runtime, info, fixture, limits = {}) {
  const original = structuredClone(fixture.position);
  const generation = { runtimeGeneration: info.model.runtimeGeneration };
  const initial = await runtime.evaluate([fixture.position], generation);
  assert.equal(initial.values.length, 1);
  assert(initial.values.every(Number.isFinite), 'Initial value must be finite.');
  let evaluated = 0, policyCalls = 0, contextTruncated = 0;
  const result = await analyze(fixture.position, {
    timeMs: 10000, maxNodes: 20000, ...limits, maxDepth: 1, candidateLimit: 4,
    innerCandidateLimit: 4, initialCandidates: 2, componentBatchSize: 4, tacticalExtensionDepth: 0,
    evaluateBatch: async positions => {
      const response = await runtime.evaluate(positions, generation);
      evaluated += positions.length;
      contextTruncated += response.context?.filter(context => context.truncated).length ?? 0;
      return response.values;
    },
    scoreMoves: async (position, moves) => {
      const scores = await runtime.orderMoves(position, moves, generation);
      assert(Array.isArray(scores) && scores.length === moves.length && scores.every(Number.isFinite), 'Trained policy must return finite candidate scores.');
      policyCalls++;
      return scores;
    },
  });
  assert(result.completed, `${fixture.id}: search did not complete (${result.stoppedReason}).`);
  assert(result.bestAction?.length && result.pv.length, 'Search must produce a complete legal turn.');
  assert(Number.isFinite(result.score));
  assert(evaluated > 0 && policyCalls > 0, 'Search must use both neural value and policy.');
  assert.deepEqual(result.pv[0], result.bestAction);
  let continuation = fixture.position;
  for (const action of result.pv) continuation = validateAction(continuation, action);
  if (fixture.requirement === 'time-travel') {
    assert(result.bestAction.some(([from, to]) => from[0] !== to[0] || from[1] !== to[1]), 'This position requires time travel.');
  }
  if (fixture.requirement === 'compound-turn') assert(result.bestAction.length >= 2, 'Both boards require a move.');
  let partial = fixture.position;
  for (const move of result.bestAction) partial = applyMove(partial, move);
  assert(canSubmit(partial));
  // The full policy interface can score SUBMIT; search still owns submission.
  const submitCandidates = [...pseudoMoves(partial).slice(0, 63), null];
  const submitScores = await runtime.orderMoves(partial, submitCandidates, generation);
  assert.equal(submitScores?.length, submitCandidates.length);
  assert(submitScores.every(Number.isFinite));
  assert.deepEqual(fixture.position, original, 'Analysis must not mutate its input.');
  return { id: fixture.id, requirement: fixture.requirement, passed: true,
    initialValueCp: initial.values[0], scoreCp: result.score, depth: result.depth,
    completed: result.completed, stoppedReason: result.stoppedReason, elapsedMs: result.elapsedMs,
    evaluations: evaluated, policyCalls, contextTruncated, bestAction: result.bestAction, pv: result.pv,
    notation: formatAction(fixture.position, result.bestAction), legalPv: true,
    submitPolicy: { candidates: submitCandidates.length, score: submitScores.at(-1) } };
}

async function main() {
  const options = parseSmokeArgs(process.argv.slice(2));
  if (options.help) { console.log(help); return; }
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(options.checkpoint)) hash.update(chunk);
  const runtime = new TransformerRuntime({ checkpoint: options.checkpoint, device: options.device });
  try {
    const info = await runtime.start();
    assert.equal(info.model.architecture, '5d-lc0-transfer-v1');
    assert.equal(info.model.policyAvailable, true, 'Checkpoint must have a trained component policy.');
    const fixtures = [];
    for (const fixture of smokeFixtures()) fixtures.push(await verifyFixture(runtime, info, fixture,
      { timeMs: options.timeMs, maxNodes: options.maxNodes }));
    const report = { passed: true, createdAt: new Date().toISOString(), checkpoint: resolve(options.checkpoint),
      checkpointSha256: hash.digest('hex'), device: info.device, architecture: info.model.architecture,
      trainedSteps: info.model.trainedSteps, policyTrainedSteps: info.model.policyTrainedSteps,
      parameters: info.model.parameters, sourceSha256: info.model.baseline?.source_sha256,
      strengthEstablished: false, fixtures };
    if (options.output) {
      await mkdir(dirname(resolve(options.output)), { recursive: true });
      await writeFile(options.output, JSON.stringify(report, null, 2) + '\n');
    }
    console.log(JSON.stringify(report, null, 2));
  } finally { runtime.close(); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
