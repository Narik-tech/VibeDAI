import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { curriculumStage, generateLc0Curriculum, lc0CurriculumStarts, lc0PolicyTargets, parseLc0DataArgs } from '../scripts/lc0-data.js';
import { applyMove, canSubmit, createPosition, generateActions, parseMove, positionKey, submitPosition, validateAction } from '../src/rules.js';

const temporal = ([from, to]) => from[0] !== to[0] || from[1] !== to[1];
const readRows = async path => (await readFile(path, 'utf8')).trim().split('\n').map(JSON.parse);
function firstLegal(position, overrides = {}) {
  const iterator = generateActions(position);
  let action;
  try { action = iterator.next().value?.moves ?? null; }
  finally { iterator.return(); }
  return { bestAction: action, completed: true, score: 237, depth: 1, nodes: 1, status: 'ok', ...overrides };
}
async function outputInTemp(t) {
  const directory = await mkdtemp(join(tmpdir(), 'lc0-curriculum-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return join(directory, 'train.jsonl');
}
function replayLabels(row) {
  let partial = row.position;
  assert.equal(row.lc0Policy.length, row.searchedAction.length + 1);
  for (const [index, label] of row.lc0Policy.entries()) {
    assert.deepEqual(label.position ?? row.position, partial);
    assert.equal(label.componentIndex, index);
    assert.equal(label.moves.includes(null), canSubmit(partial));
    const chosen = label.moves[label.target];
    if (index === row.searchedAction.length) {
      assert.equal(chosen, null);
      assert.deepEqual(submitPosition(partial), validateAction(row.position, row.searchedAction));
    } else {
      assert.deepEqual(chosen, row.searchedAction[index]);
      partial = applyMove(partial, parseMove(partial, chosen));
    }
  }
}

test('compound policy supervises each partial position and legal final SUBMIT', () => {
  const position = createPosition({ variant: 'two_timelines' });
  const before = structuredClone(position);
  const action = firstLegal(position).bestAction;
  assert(action.length >= 2);
  replayLabels({ position, searchedAction: action, lc0Policy: lc0PolicyTargets(position, action) });
  assert.deepEqual(position, before);
  assert.throws(() => lc0PolicyTargets(position, [action[0]]), /cannot be submitted/);
  assert.throws(() => lc0PolicyTargets(position, action, { maxMoves: 1 }), /candidate/);
  assert.throws(() => lc0PolicyTargets(position, Array(256).fill(action[0])), /255 components plus SUBMIT/);
  assert.deepEqual(lc0PolicyTargets(position, null), []);
});

test('SUBMIT competes with optional moves and is absent when submission is illegal', () => {
  const board = [[12, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
  const position = { board: [[board], null, [structuredClone(board), structuredClone(board), structuredClone(board)]], action: 0 };
  const first = parseMove(position, [[0, 0, 0, 0], [0, 0, 0, 1]]);
  const partial = applyMove(position, first);
  assert(canSubmit(partial));
  const second = parseMove(partial, [[2, 2, 0, 0], [2, 2, 0, 1]]);
  const labels = lc0PolicyTargets(position, [first, second]);
  assert(!labels[0].moves.includes(null));
  assert(labels[1].moves.includes(null));
  assert.notEqual(labels[1].moves[labels[1].target], null);
  assert.equal(labels[2].moves[labels[2].target], null);
  replayLabels({ position, searchedAction: [first, second], lc0Policy: labels });
});

test('curriculum keeps related development sources together and covers actual temporal escape', async () => {
  const starts = await lc0CurriculumStarts(), groups = new Map();
  for (const source of starts) {
    assert.equal(curriculumStage(source.position), source.stage);
    if (groups.has(source.group)) assert.equal(groups.get(source.group), source.split);
    groups.set(source.group, source.split);
  }
  assert.equal(starts.find(start => start.id === 'two-mini6-train').group, starts.find(start => start.id === 'mini6-train').group);
  const escape = starts.find(start => start.id === 'temporal-escape-4');
  const iterator = generateActions(escape.position);
  const actions = [...iterator];
  assert(actions.length > 0);
  assert(actions.every(action => action.moves.some(temporal)), 'This tactic must require time travel, not just offer it.');
  assert(starts.filter(start => start.group === 'standard-family').every(start => start.split === 'train'));
});

test('real bounded search writes split-safe three-stage data with travel, compound turns and SUBMIT', async t => {
  const output = await outputInTemp(t);
  const result = await generateLc0Curriculum({ output, samples: 6, nodes: 10000, timeMs: 10000, exploration: 0 });
  const training = await readRows(output), validation = await readRows(result.validationOutput);
  assert.equal(training.length, 3);
  assert.equal(validation.length, 3);
  assert.equal(result.submitTargets, 6);
  assert(result.temporalTargets > 0);
  assert(result.multiComponentTurns >= 2);
  assert.deepEqual(new Set(training.map(row => row.stage)), new Set(['spatial', 'temporal', 'multiverse']));
  const trainGroups = new Set(training.map(row => row.group));
  assert(validation.every(row => !trainGroups.has(row.group)));
  assert.equal(new Set([...training, ...validation].map(row => positionKey(row.position))).size, 6);
  for (const row of [...training, ...validation]) {
    assert.equal(row.teacher.completed, true);
    assert.equal(row.valuePerspective, 'white');
    assert.equal(row.outcomeWhite, null);
    assert.equal(row.targetType, 'classical-5d-search-bootstrap');
    assert.equal(row.lc0PolicyVersion, 1);
    assert(row.policy.length > 0);
    replayLabels(row);
  }
  assert.deepEqual(JSON.parse(await readFile(result.manifestOutput, 'utf8')), result);
});

test('White values remain White-relative on Black turns; turn caps never create draws', async t => {
  const output = await outputInTemp(t);
  const result = await generateLc0Curriculum({ output, samples: 18, nodes: 10000, timeMs: 10000, exploration: 0,
    maxPlies: 2, analyzePosition: position => firstLegal(position) });
  const rows = [...await readRows(output), ...await readRows(result.validationOutput)];
  assert(rows.some(row => row.position.action % 2 === 1));
  assert(rows.every(row => row.value === 237 && row.outcomeWhite === null));
  const restarts = rows.filter(row => row.positionOrigin.kind === 'composed-spatial-restart');
  assert(restarts.length > 0);
  for (const row of restarts) {
    assert.equal(row.stage, 'spatial');
    assert.equal(curriculumStage(row.position), 'spatial');
    assert.notEqual(row.gameId, row.positionOrigin.parentGameId);
    assert.match(row.positionOrigin.sourcePositionSha256, /^[0-9a-f]{64}$/);
    const parent = rows.find(candidate => candidate.gameId === row.positionOrigin.parentGameId);
    if (parent) {
      assert.equal(parent.group, row.group);
      assert.equal(parent.split, row.split);
    }
  }
  const groupSplits = new Map(), gameSplits = new Map();
  for (const row of rows) {
    for (const [key, map] of [[row.group, groupSplits], [row.gameId, gameSplits]]) {
      if (map.has(key)) assert.equal(map.get(key), row.split);
      map.set(key, row.split);
    }
    assert(row.ply < 2);
    replayLabels(row);
  }
});

test('incomplete search guesses do not become training targets or replace published data', async t => {
  const output = await outputInTemp(t);
  await writeFile(output, 'existing data\n');
  let calls = 0;
  await assert.rejects(generateLc0Curriculum({ output, samples: 6, analyzePosition: () => {
    calls++;
    return { completed: false, score: 100, bestAction: null };
  } }), /bounded attempts/);
  assert(calls <= 6 * 50);
  assert.equal(await readFile(output, 'utf8'), 'existing data\n');
  assert.deepEqual(await readdir(join(output, '..')), ['train.jsonl']);
});

test('source split conflicts and invalid CLI limits fail before publication', async t => {
  const output = await outputInTemp(t), starts = await lc0CurriculumStarts();
  const conflict = [...starts, { ...starts[0], split: 'validation' }];
  await assert.rejects(generateLc0Curriculum({ output, samples: 6, starts: conflict }), /crosses splits/);
  await assert.rejects(generateLc0Curriculum({ output, samples: 5 }), /Invalid samples/);
  await assert.rejects(generateLc0Curriculum({ output, nodes: Infinity }), /Invalid nodes/);
  await assert.rejects(generateLc0Curriculum({ output, validationFraction: 0 }), /validationFraction/);
  assert.throws(() => parseLc0DataArgs(['--mystery', '2']), /Usage/);
  assert.deepEqual(parseLc0DataArgs(['--output', 'data.jsonl', '--depth', '2']), { output: 'data.jsonl', maxDepth: 2 });
});
