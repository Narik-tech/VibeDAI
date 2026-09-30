import { mkdir, open, readFile, readdir, rename, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { analyze } from '../src/search.js';
import { createPosition, positionKey, validateAction } from '../src/rules.js';
import { COMPONENT_POLICY_VERSION, policyTargetsFromSearch } from '../src/transformer-policy.js';
import { loadMatchSuite } from './match.js';
import { sampleExploration, emptyCoverage, actionCoverage, addCoverage } from './transformer-exploration.js';

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; };
}

async function trainingStarts() {
  const starts = [{ id: 'standard', position: createPosition() }, { id: 'two-timelines', position: createPosition({ variant: 'two_timelines' }) }];
  const examples = new URL('../examples/', import.meta.url);
  for (const name of (await readdir(examples)).filter(name => name.endsWith('.5dpgn')).sort()) {
    starts.push({ id: name, position: createPosition({ pgn: await readFile(new URL(name, examples), 'utf8') }) });
  }
  // Use the explicitly designated development starts. Keep the match validation
  // and tactical regression suites out of this bootstrap training dataset.
  const suite = JSON.parse(await readFile(new URL('matches/training.json', examples), 'utf8'));
  for (const fixture of suite.cases) starts.push({ id: fixture.id, position: fixture.position || createPosition({ pgn: fixture.pgn }) });
  return starts;
}

/** Bounded teacher distillation data, not self-play outcomes or strength evidence. */
export async function generateTrainingData({ output = 'artifacts/transformer/training.jsonl', samples = 256, nodes = 1000, seed = 5, timeMs = 1000 } = {}) {
  for (const [name, value, low, high] of [['samples', samples, 1, 100000], ['nodes', nodes, 10, 1000000], ['seed', seed, 0, 0xffffffff], ['timeMs', timeMs, 1, 60000]]) {
    if (!Number.isInteger(value) || value < low || value > high) throw new Error(`Invalid ${name}: expected an integer from ${low} to ${high}.`);
  }
  const starts = await trainingStarts(), random = seededRandom(seed), seen = new Set();
  const heldout = await Promise.all(['validation.json', 'transformer-test.json'].map(name =>
    loadMatchSuite(new URL(`../examples/matches/${name}`, import.meta.url))));
  const excludedKeys = new Set(heldout.flatMap(suite => suite.cases.map(item => positionKey(item.position))));
  const coverage = emptyCoverage(), labelCounts = { completed: 0, incomplete: 0, excluded: 0 };
  const destination = resolve(output), temporary = `${destination}.tmp-${process.pid}`;
  await mkdir(dirname(destination), { recursive: true });
  const file = await open(temporary, 'w');
  let written = 0, attempts = 0, trajectory = 0, current, source, ply = 0;
  try {
    while (written < samples && attempts < samples * 30) {
      if (!current || ply >= 12) {
        const start = starts[trajectory++ % starts.length];
        current = start.position; source = start.id; ply = 0;
      }
      attempts++;
      const result = analyze(current, { timeMs, maxNodes: nodes, maxDepth: 3, quiescenceDepth: 1 });
      const key = positionKey(current);
      if (excludedKeys.has(key)) labelCounts.excluded++;
      if (Number.isFinite(result.score) && !seen.has(key) && !excludedKeys.has(key)) {
        const completed = result.completed === true;
        const record = {
          position: current, value: result.score, weight: completed ? 1 : 0.1,
          labelQuality: completed ? 'completed-search' : 'incomplete-search-fallback',
          teacher: { engine: 'classical', approximate: true, depth: result.depth, nodes: result.nodes,
            completed: result.completed, scoreType: result.scoreType, status: result.status, searchPolicy: result.searchPolicy,
            stoppedReason: result.stoppedReason, elapsedMs: result.elapsedMs, limits: result.limits },
          source, trajectory, ply, seed,
        };
        const policy = completed ? policyTargetsFromSearch(current, result) : [];
        if (policy.length) Object.assign(record, { policyVersion: COMPONENT_POLICY_VERSION, policy });
        await file.writeFile(`${JSON.stringify(record)}\n`);
        seen.add(key); written++;
        labelCounts[completed ? 'completed' : 'incomplete']++;
        if (written % 32 === 0 || written === samples) console.error(`Teacher labels: ${written}/${samples}; approximate White centipawns and component policy targets, not game outcomes.`);
      }
      let action = result.bestAction;
      // Mix teacher play with legal exploration to avoid a single narrow line.
      if (random() < 0.6 || !action) action = sampleExploration(current, random, { maxNodes: nodes, timeMs }).action || action;
      if (!action) current = null;
      else {
        const next = validateAction(current, action);
        addCoverage(coverage, actionCoverage(current, action, next));
        current = next; ply++;
      }
    }
    if (written !== samples) throw new Error(`Only ${written}/${samples} distinct scored positions within the bounded attempt limit. Increase --nodes or reduce --samples.`);
    await file.close();
    await rename(temporary, destination);
    return { output: destination, samples: written, attempts, seed, teacherNodes: nodes, teacherTimeMs: timeMs,
      policyVersion: COMPONENT_POLICY_VERSION, labelCounts, coverage, incompleteLabelWeight: 0.1 };
  } catch (error) {
    await file.close().catch(() => {});
    await rm(temporary, { force: true });
    throw error;
  }
}

function parseArgs(args) {
  const options = {};
  const names = { '--output': 'output', '--samples': 'samples', '--nodes': 'nodes', '--seed': 'seed', '--time-ms': 'timeMs' };
  for (let index = 0; index < args.length; index += 2) {
    const name = names[args[index]], value = args[index + 1];
    if (!name || value === undefined || value.startsWith('--')) throw new Error('Usage: node scripts/transformer-data.js --output FILE --samples 256 --nodes 1000 [--seed 5] [--time-ms 1000]');
    options[name] = name === 'output' ? value : Number(value);
  }
  return options;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  if (args.length === 1 && ['--help', '-h'].includes(args[0])) {
    console.log('Usage: node scripts/transformer-data.js --output FILE --samples 256 --nodes 1000 [--seed 5] [--time-ms 1000]\n\nGenerates bounded classical-teacher White-centipawn labels from development positions and legal continuations. Labels are approximate evaluations, not game outcomes. Defaults: artifacts/transformer/training.jsonl, 256 samples, 1000 work nodes, seed 5, 1000 ms per search.');
  } else {
    Promise.resolve().then(() => generateTrainingData(parseArgs(args))).then(result => console.log(JSON.stringify(result, null, 2)))
      .catch(error => { console.error(error.message); process.exitCode = 1; });
  }
}
