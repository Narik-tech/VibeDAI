import path from 'node:path';
import { lstat, open, readdir, realpath } from 'node:fs/promises';
import { atomicWrite } from './transformer-selfplay-store.js';

const missing = error => error.code === 'ENOENT';
const pending = new Set(['running', 'evaluated', 'incomplete']);
const terminal = new Set(['complete', 'failed', 'interrupted']);
const json = value => `${JSON.stringify(value, null, 2)}\n`;

async function readJSON(file) {
  let handle;
  try {
    const info = await lstat(file);
    if (!info.isFile() || info.isSymbolicLink() || info.size > 1024 * 1024) throw new Error(`Unsafe recovery artifact: ${file}`);
    handle = await open(file, 'r');
    // Bound the read even if a file grows after stat.
    const buffer = Buffer.alloc(Math.min(info.size + 1, 1024 * 1024 + 1));
    let bytesRead = 0;
    while (bytesRead < buffer.length) {
      const part = await handle.read(buffer, bytesRead, buffer.length - bytesRead, bytesRead);
      if (!part.bytesRead) break;
      bytesRead += part.bytesRead;
    }
    if (bytesRead > 1024 * 1024) throw new Error(`Recovery artifact is too large: ${file}`);
    return JSON.parse(buffer.subarray(0, bytesRead).toString('utf8'));
  } catch (error) { if (missing(error)) return null; throw error; }
  finally { await handle?.close(); }
}

async function readLatest(runDir) {
  try { return await readJSON(path.join(runDir, 'latest.json')); }
  catch (error) {
    // latest.json is a derived summary; a valid per-iteration report remains
    // authoritative if that summary was damaged or its final write failed.
    if (error instanceof SyntaxError) return null;
    throw error;
  }
}

/** Discovery is read-only. Recovery must repeat it while holding both run/model locks. */
export async function findInterruptedIterations({ runDir }) {
  const run = await readJSON(path.join(runDir, 'run.json'));
  if (!run) return [];
  if (run.version !== 1 || typeof run.runId !== 'string' || !Number.isSafeInteger(run.nextIteration) || run.nextIteration < 1) {
    throw new Error('Invalid self-play run.json; cannot recover saved iterations.');
  }
  const root = await realpath(runDir);
  const latest = await readLatest(root);
  const entries = (await readdir(root, { withFileTypes: true }))
    .filter(entry => entry.isDirectory() && /^iteration-\d{8,12}$/.test(entry.name)).sort((a, b) => a.name.localeCompare(b.name));
  const reports = [];
  for (const entry of entries) {
    const folder = path.join(root, entry.name);
    if ((await lstat(folder)).isSymbolicLink() || path.dirname(await realpath(folder)) !== root) continue;
    let manifest, report;
    try {
      manifest = await readJSON(path.join(folder, 'iteration.json'));
      report = await readJSON(path.join(folder, 'report.json'));
    } catch (error) {
      // Corrupt history remains available to the review UI as unreadable.
      if (error instanceof SyntaxError) continue;
      throw error;
    }
    if (!manifest || manifest.runId !== run.runId || manifest.iteration !== Number(entry.name.slice(10))) continue;
    report ??= { ...manifest, folder, status: 'incomplete', promoted: false };
    if (report.runId !== run.runId || report.iteration !== manifest.iteration) continue;
    if (pending.has(report.status)) reports.push({ folder, report });
    else if (latest?.runId === report.runId && latest.iteration === report.iteration
      && pending.has(latest.status) && terminal.has(report.status)) reports.push({ folder, report, synchronizeOnly: true });
  }
  return reports;
}

/** Caller holds both self-play locks, so no runner can still be writing these reports. */
export async function recoverInterruptedIterations(options, { reason = 'The training process exited before saving a completion report.' } = {}) {
  const reports = await findInterruptedIterations(options);
  if (!reports.length) return [];
  const latestFile = path.join(options.runDir, 'latest.json');
  const latest = await readLatest(options.runDir);
  const recovered = [];
  for (const { folder, report, synchronizeOnly } of reports) {
    if (synchronizeOnly) {
      await atomicWrite(latestFile, json(report));
      recovered.push(report);
      continue;
    }
    const recoveredAt = new Date().toISOString();
    const promotionPending = report.status === 'evaluated' && report.arena?.decision?.promote && !report.promoted;
    const updated = { ...report, status: 'interrupted', finishedAt: recoveredAt,
      error: reason + (promotionPending ? ' Promotion was in progress; inspect the checkpoint hashes and saved arena decision before assuming which model is active.' : ''),
      interruption: { previousStatus: report.status, recoveredAt } };
    await atomicWrite(path.join(folder, 'report.json'), json(updated));
    if ((latest?.runId === report.runId && latest?.iteration === report.iteration)
      || (!latest && folder === reports.at(-1).folder)) await atomicWrite(latestFile, json(updated));
    recovered.push(updated);
  }
  return recovered;
}
