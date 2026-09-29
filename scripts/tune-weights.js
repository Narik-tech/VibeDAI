import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runStrengthSuite } from './strength.js';
import { runMatchSuite, loadMatchSuite } from './match.js';

const help = `Usage: node scripts/tune-weights.js --baseline DIR --output DIR [options]
  --profiles ID,ID  Select profiles (default all 25, including baseline)
  --suite FILE      Match starts (default examples/matches/training.json)
  --nodes N         Per-turn match work (default 2000)
  --plies N         Submitted-turn limit (default 40)
  --tactics-only    Screen tactics without playing matches
The output must be a new directory below the repository. Source files are
copied into isolated variants; production files are never changed. Profiles
describe relative changes to the supplied baseline. Tactical regressions do
not advance to matches. This is a development sweep, not independent validation.`;

// A newly solved position must not hide a lost baseline solution at the same
// budget. Compare individual cases before advancing a candidate to matches.
export function tacticalRegressions(reference, candidate) {
  const solved = new Set(candidate.results.filter(row => row.solved)
    .map(row => JSON.stringify([row.id, row.budget])));
  return reference.results.filter(row => row.solved && !solved.has(JSON.stringify([row.id, row.budget])))
    .map(({ id, budget }) => ({ id, budget }));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
try {
  const args = process.argv.slice(2), options = { suite: 'examples/matches/training.json', nodes: 2000, plies: 40 };
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    if (flag === '--help') { console.log(help); process.exit(0); }
    if (flag === '--tactics-only') { options.tacticsOnly = true; continue; }
    if (!['--baseline', '--output', '--profiles', '--suite', '--nodes', '--plies'].includes(flag)) throw new Error(`Unknown option: ${flag}`);
    const value = args[++i];
    if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value.`);
    options[flag.slice(2)] = ['--nodes', '--plies'].includes(flag) ? Number(value) : value;
  }
  if (!options.baseline || !options.output) throw new Error(help);
  if (!Number.isInteger(options.nodes) || options.nodes < 1 || options.nodes > 1e9 ||
      !Number.isInteger(options.plies) || options.plies < 1 || options.plies > 10000) throw new Error('Invalid nodes or plies.');
  const profiles = JSON.parse(await readFile(new URL('../examples/matches/weight-profiles.json', import.meta.url)));
  const selected = options.profiles ? options.profiles.split(',') : profiles.map(profile => profile.id);
  if (selected.some(id => !profiles.some(profile => profile.id === id))) throw new Error('Unknown profile.');
  const baselineDir = resolve(options.baseline), output = resolve(options.output);
  const evaluator = await readFile(resolve(baselineDir, 'evaluate.js'), 'utf8');
  const baseEngine = (await import(pathToFileURL(resolve(baselineDir, 'search.js')))).analyze;
  if (typeof baseEngine !== 'function') throw new Error('Baseline search.js must export an analyze function.');
  const suite = await loadMatchSuite(options.suite);
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output); // Never overwrite an earlier experiment or engine.
  const reference = await runStrengthSuite({ engine: baseEngine, budgets: [1000, 5000, 20000], timeMs: 60000 });
  await writeFile(resolve(output, 'baseline-tactics.json'), JSON.stringify(reference, null, 2));
  if (reference.summary.some(row => row.invalid || row.timeLimited)) throw new Error('Baseline screen was invalid or time-limited; no comparison was made.');
  const summary = [];
  for (const profile of profiles.filter(profile => selected.includes(profile.id))) {
    const directory = resolve(output, profile.id);
    await mkdir(directory);
    for (const file of ['search.js', 'rules.js', 'search-cache.js']) await copyFile(resolve(baselineDir, file), resolve(directory, file));
    for (const file of ['royal-safety.js', 'heuristics.js']) {
      try { await copyFile(resolve(baselineDir, file), resolve(directory, file)); }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
    }
    let source = evaluator;
    for (const [from, to] of profile.replace || []) {
      if (source.split(from).length !== 2) throw new Error(`Profile ${profile.id} is incompatible with this evaluator: ${from}`);
      source = source.replace(from, to);
    }
    const marker = '  const total = Math.round(';
    if (source.split(marker).length !== 2) throw new Error('Evaluator does not have the expected total-score calculation.');
    const scaling = Object.entries(profile.scales || {}).map(([component, scale]) => `  totals.${component} *= ${scale};`).join('\n');
    source = source.replace(marker, `${scaling}\n${marker}`);
    await writeFile(resolve(directory, 'evaluate.js'), source);
    await writeFile(resolve(directory, 'package.json'), '{"type":"module"}\n');
    const engine = profile.id === 'baseline' ? baseEngine : (await import(pathToFileURL(resolve(directory, 'search.js')))).analyze;
    if (typeof engine !== 'function') throw new Error(`Profile ${profile.id} search.js must export an analyze function.`);
    const tactics = await runStrengthSuite({ engine, budgets: [1000, 5000, 20000], timeMs: 60000 });
    const regressions = tacticalRegressions(reference, tactics);
    const regressed = tactics.summary.some(row => row.invalid || row.timeLimited) || regressions.length > 0;
    await writeFile(resolve(directory, 'tactics.json'), JSON.stringify(tactics, null, 2));
    const result = { id: profile.id, profile, tactics: tactics.summary, regressions, rejected: regressed };
    if (!regressed && !options.tacticsOnly) {
      const matches = await runMatchSuite({ engineA: engine, engineB: baseEngine, suite,
        maxNodes: options.nodes, maxPlies: options.plies, maxDepth: 4, quiescenceDepth: 1, timeMs: 60000, terminalWork: 20000 });
      result.matches = matches.summary;
      await writeFile(resolve(directory, 'matches.json'), JSON.stringify(matches));
    }
    summary.push(result);
    console.log(JSON.stringify(result));
    await writeFile(resolve(output, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
  }
} catch (error) { console.error(error.message); process.exitCode = 1; }
}
