import { createHash } from 'node:crypto';
import { mkdir, open, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, extname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { analyze } from '../src/search.js';
import { applyMove, canSubmit, createPosition, generateActions, positionKey, pseudoMoves, raw, validateAction } from '../src/rules.js';
import { COMPONENT_POLICY_VERSION, componentPolicyTargets, MAX_POLICY_MOVES } from '../src/transformer-policy.js';

export const LC0_POLICY_VERSION = 1;
export const CURRICULUM_STAGES = ['spatial', 'temporal', 'multiverse'];
const temporal = ([from, to]) => from[0] !== to[0] || from[1] !== to[1];
const digest = text => createHash('sha256').update(text).digest('hex');
const randomFor = seed => {
  let state = seed >>> 0;
  return () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; };
};

/** Spatial means no temporal option at this decision, under the full 5D rules. */
export function curriculumStage(position) {
  if (position.board.filter(Boolean).length > 1) return 'multiverse';
  return pseudoMoves(position).some(temporal) ? 'temporal' : 'spatial';
}

/**
 * Labels describe each partial turn and the decision to submit it. null is the
 * SUBMIT candidate and appears only when full-rules submission is legal.
 * Alternatives are pseudo-legal components, not certified complete turns.
 * Selected components, including every final SUBMIT, form a validated turn.
 */
export function lc0PolicyTargets(position, action, { maxMoves = MAX_POLICY_MOVES } = {}) {
  if (!Number.isInteger(maxMoves) || maxMoves < 1 || maxMoves > MAX_POLICY_MOVES) throw new Error('Invalid policy candidate limit.');
  if (!Array.isArray(action) || !action.length) return [];
  if (action.length > 255) throw new Error('A policy turn is limited to 255 components plus SUBMIT.');
  validateAction(position, action);
  const labels = [];
  let current = position;
  for (let componentIndex = 0; componentIndex <= action.length; componentIndex++) {
    const selected = action[componentIndex] ?? null;
    const moves = pseudoMoves(current);
    if (canSubmit(current)) moves.push(null);
    if (moves.length > maxMoves) throw new Error(`Policy needs ${moves.length} candidates, above the ${maxMoves} limit.`);
    const target = moves.findIndex(move => selected === null ? move === null
      : move !== null && raw.validateFuncs.compareMove(move, selected) === 0);
    if (target < 0) throw new Error('Validated teacher component is absent from the policy mask.');
    labels.push({ ...(componentIndex ? { position: current } : {}), moves, target, componentIndex });
    if (selected !== null) current = applyMove(current, selected);
  }
  return labels;
}

/** Only development starts are read: match validation/tactics suites stay out. */
export async function lc0CurriculumStarts() {
  const suite = JSON.parse(await readFile(new URL('../examples/matches/training.json', import.meta.url), 'utf8'));
  const fixtures = new Map(suite.cases.map(item => [item.id, item.position ?? createPosition({ pgn: item.pgn })]));
  const starts = [];
  const add = (id, group, split, position, description) => starts.push({ id, group, split, position,
    stage: curriculumStage(position), description });
  add('standard', 'standard-family', 'train', createPosition(), 'Standard 5D start');
  for (const id of ['mini5-train', 'rook4-train', 'minor5-train']) {
    const source = fixtures.get(id);
    add(`${id}-initial`, id, 'train', { ...source, board: [[source.board[0][0]]], action: 0 }, 'Initial board of a development miniature');
  }
  add('nikita-time', 'nikita-time', 'validation', fixtures.get('nikita-time'), 'Reserved development spatial source');
  const heldOutMiniature = fixtures.get('mini6-train');
  add('mini6-train-initial', 'mini6-train', 'validation', { ...heldOutMiniature,
    board: [[heldOutMiniature.board[0][0]]], action: 0 }, 'Initial board of the reserved development miniature');
  add('temporal-escape-4', 'temporal-escape-family', 'train', createPosition({
    pgn: '[Board "Custom"]\n[Size "4x4"]\n[3k/4/4/4:0:1:w]\n[4/2k1/2q1/K3:0:2:w]',
  }), 'Composed temporal tactic: the spatially trapped king must travel to a safe historical board');
  for (const id of ['standard-train', 'digolian', 'sac-time', 'boring', 'ng5-theory', 'post-f7-travel']) {
    add(id, 'standard-family', 'train', fixtures.get(id), 'Development Standard opening family');
  }
  for (const id of ['rook4-train', 'minor5-train']) add(id, id, 'train', fixtures.get(id), 'Development temporal miniature');
  add('mini6-train', 'mini6-train', 'validation', fixtures.get('mini6-train'), 'Reserved development temporal source');
  add('two-timelines', 'two-timelines', 'train', createPosition({ variant: 'two_timelines' }), 'Standard two-timeline variant');
  for (const [id, split] of [['rook4-train', 'train'], ['mini6-train', 'validation']]) {
    const source = fixtures.get(id);
    // Two simultaneous initial boards are a composed full-rules setup, not a
    // claim that these boards arose from a played temporal move.
    const board = source.board[0][0];
    add(`two-${id}`, id, split, { ...source, board: [null, [board], [structuredClone(board)]], action: 0 },
      'Composed miniature with two simultaneous required boards');
  }
  // Small datasets should already contain a real compound-turn obligation.
  starts.unshift(starts.splice(starts.findIndex(start => start.id === 'two-rook4-train'), 1)[0]);
  return starts;
}

function optionsWithDefaults(options) {
  const settings = { samples: 192, nodes: 10000, timeMs: 1000, maxDepth: 1, maxPlies: 8, seed: 5,
    validationFraction: 0.2, exploration: 0.35, ...options };
  for (const [name, low, high] of [['samples', 6, 100000], ['nodes', 10, 1000000], ['timeMs', 1, 60000],
    ['maxDepth', 1, 16], ['maxPlies', 1, 128], ['seed', 0, 0xffffffff]]) {
    if (!Number.isInteger(settings[name]) || settings[name] < low || settings[name] > high) throw new Error(`Invalid ${name}: expected an integer from ${low} to ${high}.`);
  }
  if (!Number.isFinite(settings.validationFraction) || settings.validationFraction <= 0 || settings.validationFraction >= 1) throw new Error('validationFraction must be between zero and one.');
  if (!Number.isFinite(settings.exploration) || settings.exploration < 0 || settings.exploration > 1) throw new Error('exploration must be from zero to one.');
  return settings;
}

function explorationAction(position, random, limits) {
  const deadline = performance.now() + limits.timeMs, stopped = new Error('Exploration limit');
  let work = 0;
  const iterator = generateActions(position, { tick() {
    if (work++ >= limits.nodes || performance.now() >= deadline) throw stopped;
  } });
  const choices = [];
  try {
    for (const candidate of iterator) { choices.push(candidate.moves); if (choices.length >= 16) break; }
  } catch (error) { if (error !== stopped) throw error; }
  finally { iterator.return?.(); }
  return choices.length ? choices[Math.floor(random() * choices.length)] : null;
}

/**
 * Complete classical 5D searches supply approximate White-centipawn values.
 * LC0 does not label 5D outcomes. A turn limit is a trajectory cutoff, never a
 * draw. Fixed source-family holdouts stay fixed across seeds and generations.
 */
export async function generateLc0Curriculum(options = {}) {
  const limits = optionsWithDefaults(options);
  const analyzePosition = options.analyzePosition ?? analyze;
  const starts = options.starts ?? await lc0CurriculumStarts();
  const sourceSplits = new Map();
  for (const start of starts) {
    if (!start?.id || !start.group || !['train', 'validation'].includes(start.split) || !CURRICULUM_STAGES.includes(start.stage)) throw new Error('Invalid curriculum source.');
    if (sourceSplits.has(start.group) && sourceSplits.get(start.group) !== start.split) throw new Error(`Source group ${start.group} crosses splits.`);
    if (curriculumStage(start.position) !== start.stage) throw new Error(`Source ${start.id} has the wrong curriculum stage.`);
    sourceSplits.set(start.group, start.split);
  }
  const lanes = [];
  for (const [index, stage] of CURRICULUM_STAGES.entries()) {
    const count = Math.floor(limits.samples / 3) + Number(index < limits.samples % 3);
    const validation = Math.min(count - 1, Math.max(1, Math.round(count * limits.validationFraction)));
    for (const [split, quota] of [['train', count - validation], ['validation', validation]]) {
      const sources = starts.filter(start => start.stage === stage && start.split === split);
      if (!sources.length) throw new Error(`Missing ${split} sources for ${stage}.`);
      lanes.push({ stage, split, quota, sources, samples: 0, attempts: 0, trajectory: 0, current: null,
        random: randomFor((limits.seed + lanes.length * 0x9e3779b9) >>> 0) });
    }
  }
  const output = resolve(options.output ?? 'artifacts/lc0/curriculum.jsonl');
  const base = output.slice(0, output.length - extname(output).length);
  const validationOutput = resolve(options.validationOutput ?? `${base}.validation.jsonl`);
  const manifestOutput = resolve(options.manifestOutput ?? `${base}.manifest.json`);
  const paths = [output, validationOutput, manifestOutput];
  if (new Set(paths.map(path => path.toLowerCase())).size !== paths.length) throw new Error('Train, validation, and manifest paths must differ.');
  const temporary = paths.map(path => `${path}.tmp-${process.pid}`);
  for (const path of paths) await mkdir(dirname(path), { recursive: true });
  const files = [];
  const seen = new Set(), hashes = [createHash('sha256'), createHash('sha256')];
  const counts = { samples: 0, train: 0, validation: 0, componentTargets: 0, submitTargets: 0, temporalTargets: 0,
    multiComponentTurns: 0, incompleteSearches: 0, duplicates: 0, attempts: 0 };
  const usedGroups = new Map();
  try {
    files.push(await open(temporary[0], 'wx'));
    files.push(await open(temporary[1], 'wx'));
    while (lanes.some(lane => lane.samples < lane.quota)) {
      for (const lane of lanes) {
        if (lane.samples >= lane.quota) continue;
        if (++lane.attempts > lane.quota * 50) throw new Error(`Only ${lane.samples}/${lane.quota} ${lane.split} ${lane.stage} labels after bounded attempts; increase --nodes/--time-ms or reduce --samples.`);
        counts.attempts++;
        if (!lane.current || lane.ply >= limits.maxPlies || curriculumStage(lane.current) !== lane.stage) {
          lane.start = lane.sources[lane.trajectory % lane.sources.length];
          lane.current = structuredClone(lane.start.position);
          lane.trajectory++;
          lane.ply = 0;
          lane.gameId = `lc0:${limits.seed}:${lane.stage}:${lane.split}:${lane.trajectory}:${lane.start.id}`;
          lane.origin = { kind: 'curriculum-source', sourceId: lane.start.id };
        }
        const position = lane.current;
        const result = await analyzePosition(position, { timeMs: limits.timeMs, maxNodes: limits.nodes, maxDepth: limits.maxDepth, quiescenceDepth: 0 });
        let next = null;
        if (result.bestAction?.length) next = validateAction(position, result.bestAction);
        if (!result.completed || !Number.isFinite(result.score) || !next) {
          counts.incompleteSearches++;
          lane.current = null;
          continue;
        }
        const key = positionKey(position);
        if (!seen.has(key)) {
          const lc0Policy = lc0PolicyTargets(position, result.bestAction);
          const policy = componentPolicyTargets(position, result.bestAction);
          const group = `lc0-source:${lane.start.group}`;
          const record = { position, value: result.score, valuePerspective: 'white', valueUnit: 'centipawns',
            group, gameId: lane.gameId, split: lane.split, stage: lane.stage, source: lane.start.id, ply: lane.ply, seed: limits.seed,
            positionOrigin: lane.origin,
            searchedAction: result.bestAction, targetType: 'classical-5d-search-bootstrap', outcomeWhite: null,
            lc0PolicyVersion: LC0_POLICY_VERSION, lc0Policy, policyVersion: COMPONENT_POLICY_VERSION, policy,
            teacher: { engine: 'classical', approximate: true, completed: true, depth: result.depth, nodes: result.nodes,
              scoreType: result.scoreType, status: result.status, searchPolicy: result.searchPolicy,
              stoppedReason: result.stoppedReason, elapsedMs: result.elapsedMs,
              limits: { nodes: limits.nodes, timeMs: limits.timeMs, maxDepth: limits.maxDepth } },
          };
          const line = `${JSON.stringify(record)}\n`, fileIndex = lane.split === 'train' ? 0 : 1;
          await files[fileIndex].writeFile(line);
          hashes[fileIndex].update(line);
          seen.add(key);
          lane.samples++; counts.samples++; counts[lane.split]++;
          counts.componentTargets += result.bestAction.length;
          counts.submitTargets++;
          counts.temporalTargets += result.bestAction.filter(temporal).length;
          counts.multiComponentTurns += Number(result.bestAction.length > 1);
          usedGroups.set(group, { split: lane.split, samples: (usedGroups.get(group)?.samples ?? 0) + 1 });
        } else counts.duplicates++;
        if (lane.random() < limits.exploration) {
          const action = explorationAction(position, lane.random, limits);
          if (action) next = validateAction(position, action);
        }
        lane.current = next; lane.ply++;
        if (lane.stage === 'spatial' && next.board.filter(Boolean).length === 1 && curriculumStage(next) !== 'spatial') {
          // The spatial curriculum studies composed ordinary-board setups. End
          // this trajectory once time travel becomes available, then begin a
          // NEW composed game from its frontier, explicitly recording origin.
          // This does not claim the old history survived the new setup.
          const board = next.board.find(Boolean).at(-1), color = next.action % 2;
          lane.current = { ...next, board: [color ? [null, board] : [board]], action: color };
          lane.origin = { kind: 'composed-spatial-restart', parentGameId: lane.gameId,
            sourcePositionSha256: digest(positionKey(next)) };
          lane.trajectory++; lane.ply = 0;
          lane.gameId = `lc0:${limits.seed}:${lane.stage}:${lane.split}:${lane.trajectory}:${lane.start.id}`;
        }
      }
    }
    const manifest = { version: 1, output, validationOutput, manifestOutput, ...counts,
      sha256: { train: hashes[0].digest('hex'), validation: hashes[1].digest('hex') },
      stages: Object.fromEntries(CURRICULUM_STAGES.map(stage => [stage,
        Object.fromEntries(lanes.filter(lane => lane.stage === stage).map(lane => [lane.split, lane.samples]))])),
      groups: Object.fromEntries(usedGroups),
      sourceAssignments: starts.map(({ position, ...source }) => ({ ...source, initialPositionSha256: digest(positionKey(position)) })),
      limits: Object.fromEntries(['samples', 'nodes', 'timeMs', 'maxDepth', 'maxPlies', 'seed', 'validationFraction', 'exploration'].map(key => [key, limits[key]])),
      labelMeaning: 'Completed selective classical 5D search, White centipawns; approximate, not outcomes or LC0 evaluations.',
      spatialMeaning: 'Single timeline with no current temporal component. At the first temporal option, end the episode and begin an explicitly composed game from its frontier; preserve the source split and record the parent game/hash. Rules within each episode are full 5D.',
      policyMeaning: 'Pseudo-legal component alternatives; selected whole turn validated; null is legal SUBMIT.',
      validationMeaning: 'Fixed held-out development source families; not independent playing-strength evidence.',
    };
    await writeFile(temporary[2], `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
    for (const file of files) await file.close();
    for (let index = 0; index < paths.length; index++) await rename(temporary[index], paths[index]);
    return manifest;
  } catch (error) {
    for (const file of files) await file.close().catch(() => {});
    for (const path of temporary) await rm(path, { force: true }).catch(() => {});
    throw error;
  }
}

const usage = 'node scripts/lc0-data.js [--output FILE] [--validation-output FILE] [--samples 192] [--nodes 10000] [--time-ms 1000] [--depth 1] [--max-plies 8] [--seed 5] [--validation-fraction 0.2] [--exploration 0.35]';
export function parseLc0DataArgs(args) {
  const names = { '--output': 'output', '--validation-output': 'validationOutput', '--samples': 'samples', '--nodes': 'nodes',
    '--time-ms': 'timeMs', '--depth': 'maxDepth', '--max-plies': 'maxPlies', '--seed': 'seed',
    '--validation-fraction': 'validationFraction', '--exploration': 'exploration' };
  const options = {};
  for (let index = 0; index < args.length; index += 2) {
    const name = names[args[index]], value = args[index + 1];
    if (!name || value === undefined || value.startsWith('--')) throw new Error(`Usage: ${usage}`);
    options[name] = ['output', 'validationOutput'].includes(name) ? value : Number(value);
  }
  return options;
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.slice(2).some(arg => ['--help', '-h'].includes(arg))) console.log(`${usage}\n\nWrites train/validation JSONL and a manifest. Values are completed classical 5D search labels, never LC0 game outcomes. Each stage reserves at least one validation row; six samples minimum.`);
  else Promise.resolve().then(() => generateLc0Curriculum(parseLc0DataArgs(process.argv.slice(2))))
    .then(result => console.log(JSON.stringify(result, null, 2)))
    .catch(error => { console.error(error.message); process.exitCode = 1; });
}
