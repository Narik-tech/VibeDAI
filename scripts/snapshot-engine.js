import { execFileSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Keep comparisons self-contained: a search module must use the evaluation,
// rules, and cache implementation from the same revision as its search code.
export async function snapshotEngine(ref, output) {
  if (!ref || !output) throw new Error('A Git revision and output directory are required.');
  const root = resolve(fileURLToPath(new URL('../', import.meta.url)));
  const git = args => execFileSync('git', ['-c', `safe.directory=${root.replaceAll('\\', '/')}`, ...args], { cwd: root });
  const revision = git(['rev-parse', '--verify', '--end-of-options', `${ref}^{commit}`]).toString().trim();
  const target = resolve(output);
  // Resolve all inputs before writing; an invalid revision cannot leave a
  // half-populated comparison engine. Refuse to overwrite existing snapshots.
  const files = ['search.js', 'evaluate.js', 'rules.js', 'search-cache.js'];
  // Older revisions keep royal safety inside rules.js; newer ones import it.
  if (git(['ls-tree', '--name-only', revision, 'src/royal-safety.js']).toString().trim()) files.push('royal-safety.js');
  const sources = files.map(name => git(['show', `${revision}:src/${name}`]));
  await mkdir(dirname(target), { recursive: true });
  await mkdir(target);
  for (let i = 0; i < files.length; i++) await writeFile(resolve(target, files[i]), sources[i], { flag: 'wx' });
  await writeFile(resolve(target, 'package.json'), '{"type":"module"}\n', { flag: 'wx' });
  await writeFile(resolve(target, 'snapshot.json'), `${JSON.stringify({ revision, files }, null, 2)}\n`, { flag: 'wx' });
  return { revision, directory: target };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 4 || process.argv.includes('--help')) {
    console.log('Usage: node scripts/snapshot-engine.js GIT_REVISION OUTPUT_DIRECTORY\nUse a directory below this repository so its installed dependency can resolve.');
    if (!process.argv.includes('--help')) process.exitCode = 1;
  } else {
    try { console.log(JSON.stringify(await snapshotEngine(process.argv[2], process.argv[3]), null, 2)); }
    catch (error) { console.error(error.message); process.exitCode = 1; }
  }
}
