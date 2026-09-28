import { createReadStream } from 'node:fs';
import { mkdir, open, readFile, realpath, rename, link, unlink, stat, lstat } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { positionKey } from '../src/rules.js';

export const MAX_REPLAY_LINE_BYTES = 4 * 1024 * 1024;
const hash = data => createHash('sha256').update(data).digest('hex');
const missing = error => error.code === 'ENOENT';

async function canonicalPath(file) {
  const absolute = resolve(file);
  try { return await realpath(absolute); }
  catch (error) {
    if (!missing(error)) throw error;
    const parent = dirname(absolute);
    return parent === absolute ? absolute : join(await canonicalPath(parent), basename(absolute));
  }
}

async function distinctPaths(paths) {
  const checked = [];
  for (const [name, file] of Object.entries(paths)) {
    if (file === undefined) continue;
    if (typeof file !== 'string' || !file.trim()) throw new Error(`${name} must be a file path.`);
    const canonical = await canonicalPath(file);
    const key = process.platform === 'win32' ? canonical.toLowerCase() : canonical;
    const info = await stat(file).catch(error => { if (!missing(error)) throw error; return null; });
    for (const previous of checked) {
      if (previous.key === key || (info?.ino && previous.info?.ino === info.ino && previous.info.dev === info.dev)) {
        throw new Error(`${name} and ${previous.name} must be different files.`);
      }
    }
    checked.push({name, key, info});
  }
}

async function writeAtomic(file, data, {exclusive = false, beforeCommit} = {}) {
  if (typeof data !== 'string' && !Buffer.isBuffer(data)) throw new TypeError('Atomic writes require text or a Buffer.');
  const destination = resolve(file);
  await mkdir(dirname(destination), {recursive:true});
  const temporary = join(dirname(destination), `.${basename(destination)}.tmp-${process.pid}-${randomUUID()}`);
  let handle, created = false;
  try {
    handle = await open(temporary, 'wx');
    created = true;
    await handle.writeFile(data);
    await handle.sync();
    await handle.close(); handle = null;
    await beforeCommit?.();
    // link publishes a complete backup without replacing an existing backup.
    // Both publication methods operate within the destination's filesystem.
    if (exclusive) await link(temporary, destination);
    else await rename(temporary, destination);
  } finally {
    await handle?.close().catch(() => {});
    if (created) await unlink(temporary).catch(error => { if (!missing(error)) throw error; });
  }
}

/** Replace a file by renaming a fully written, synced sibling temporary file. */
export async function atomicWrite(file, textOrBuffer) { await writeAtomic(file, textOrBuffer); }

/** Hash incrementally; checkpoints and replay files need not fit in memory. */
export async function fileHash(file) {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(file)) digest.update(chunk);
  return digest.digest('hex');
}

const lockPath = runDir => join(resolve(runDir), '.selfplay.lock');
const sameLockFile = (one, two) => one.isFile() && two.isFile()
  // Windows lstat may report dev=0 while fstat returns the volume serial.
  && (one.dev === two.dev || (process.platform === 'win32' && (one.dev === 0n || two.dev === 0n)))
  && one.ino === two.ino && one.size === two.size
  && one.mtimeNs === two.mtimeNs && one.ctimeNs === two.ctimeNs;

async function readRunLock(runDir, exitedOwner) {
  const file = lockPath(runDir);
  const result = {file, owner:null, live:false, recoverable:false};
  let handle;
  try {
    const info = await lstat(file, {bigint:true});
    if (!info.isFile()) return {...result, reason:info.isSymbolicLink() ? 'symbolic-link' : 'not-a-file'};
    if (info.size > 4096n) return {...result, reason:'malformed'};
    handle = await open(file, 'r');
    if (!sameLockFile(info, await handle.stat({bigint:true}))) return {...result, reason:'lock-changed'};
    // Keep reads bounded even if an outside writer grows the file after stat.
    const bytes = Buffer.alloc(4097);
    let length = 0;
    while (length < bytes.length) {
      const {bytesRead} = await handle.read(bytes, length, bytes.length - length, null);
      if (!bytesRead) break;
      length += bytesRead;
    }
    if (BigInt(length) !== info.size || !sameLockFile(info, await handle.stat({bigint:true}))
      || !sameLockFile(info, await lstat(file, {bigint:true}))) return {...result, reason:'lock-changed'};
    const raw = bytes.subarray(0, length);
    let owner;
    try { owner = JSON.parse(raw.toString('utf8')); }
    catch { return {...result, reason:'malformed'}; }
    result.owner = owner;
    if (!owner || typeof owner !== 'object' || Array.isArray(owner)
      || !Number.isSafeInteger(owner.pid) || owner.pid <= 0 || owner.pid > 0x7fffffff
      || typeof owner.token !== 'string' || !owner.token.trim() || owner.token.length > 256) {
      return {...result, reason:'malformed'};
    }
    // Worker threads share the manager's PID. Its exit handler may attest that
    // this exact invocation exited, while unrelated live owners remain guarded.
    if (owner.pid === process.pid && exitedOwner?.pid === process.pid && owner.token === exitedOwner.token) {
      return {...result, recoverable:true, reason:'execution-exited', raw, info};
    }
    try {
      process.kill(owner.pid, 0);
      return {...result, live:true, reason:'owner-running'};
    } catch (error) {
      // EPERM and unknown probe failures cannot prove that the owner exited.
      if (error.code !== 'ESRCH') return {...result, live:true, reason:'owner-unverifiable'};
    }
    return {...result, recoverable:true, reason:'owner-exited', raw, info};
  } catch (error) {
    if (missing(error)) return null;
    if (error.code === 'EACCES' || error.code === 'EPERM') return {...result, reason:'unreadable'};
    throw error;
  } finally { await handle?.close(); }
}

function lockInspection(snapshot) {
  if (!snapshot) return null;
  const {raw, info, ...inspection} = snapshot;
  return inspection;
}

/** Inspect without modifying a lock. Only ESRCH with valid ownership is recoverable. */
export async function inspectRunLock(runDir) { return lockInspection(await readRunLock(runDir)); }

/**
 * Recover a confirmed exited owner, retaining its exact lock bytes as evidence.
 * The exclusive archive is also a permanent claim for this ownership token:
 * competing recoverers cannot later unlink a new runner's replacement lock.
 * A crash after claiming but before unlinking requires manual review of the
 * claim; never discard claims and retry them automatically.
 */
export async function recoverRunLock(runDir, {exitedOwner} = {}) {
  const snapshot = await readRunLock(runDir, exitedOwner);
  if (!snapshot) return null;
  const inspection = lockInspection(snapshot);
  if (!snapshot.recoverable) return {...inspection, recovered:false};
  const archivePath = `${snapshot.file}.recovered-${hash(snapshot.owner.token)}`;
  let archive;
  try { archive = await open(archivePath, 'wx'); }
  catch (error) {
    if (error.code !== 'EEXIST') throw error;
    return {...inspection, recoverable:false, recovered:false, reason:'recovery-claimed', archivePath};
  }
  try {
    await archive.writeFile(snapshot.raw);
    await archive.sync();
  } finally { await archive.close(); }
  const current = await readRunLock(runDir, exitedOwner);
  if (!current?.recoverable || !sameLockFile(snapshot.info, current.info) || !snapshot.raw.equals(current.raw)) {
    return {...inspection, recoverable:false, recovered:false, reason:'lock-changed', archivePath};
  }
  await unlink(snapshot.file);
  return {...inspection, recovered:true, archivePath, recoveredAt:new Date().toISOString()};
}

/** Hold the lock across the runner, including replay updates and promotion. */
export async function acquireRunLock(runDir, {token = randomUUID()} = {}) {
  if (typeof token !== 'string' || !token.trim() || token.length > 256) throw new Error('Self-play lock token must be a nonempty string of at most 256 characters.');
  await mkdir(runDir, {recursive:true});
  const file = lockPath(runDir);
  let handle;
  // Bounded retries cover a disappearing lock and a successful stale recovery.
  for (let attempt = 0; attempt < 4 && !handle; attempt++) {
    try { handle = await open(file, 'wx'); }
    catch (error) {
      if (error.code !== 'EEXIST') throw error;
      const recovery = await recoverRunLock(runDir);
      if (!recovery || recovery.recovered) continue;
      if (recovery.live) throw new Error(`Self-play is already running under PID ${recovery.owner.pid}; lock: ${file}`);
      throw new Error(`Stale or malformed self-play lock${recovery.owner?.pid ? ` for PID ${recovery.owner.pid}` : ''}: ${file} (${recovery.reason}). Verify no runner is active, then review and remove this lock before restarting.`);
    }
  }
  if (!handle) throw new Error(`Self-play lock kept changing; retry starting the runner: ${file}`);
  try {
    await handle.writeFile(`${JSON.stringify({pid:process.pid, token, createdAt:new Date().toISOString()})}\n`);
    await handle.sync();
  } catch (error) {
    await handle.close(); handle = null;
    await unlink(file).catch(() => {});
    throw error;
  } finally { await handle?.close(); }
  let released = false;
  return async () => {
    if (released) return;
    let owner;
    try { owner = JSON.parse(await readFile(file, 'utf8')); }
    catch (error) { if (missing(error)) { released = true; return; } throw error; }
    if (owner.token !== token || owner.pid !== process.pid) throw new Error(`Self-play lock ownership changed; refusing to remove ${file}.`);
    await unlink(file);
    released = true;
  };
}

function validateSample(record, label) {
  const fail = detail => { throw new Error(`Invalid replay sample (${label}): ${detail}.`); };
  if (!record || typeof record !== 'object' || Array.isArray(record) || !Number.isFinite(record.value)) fail('value must be finite');
  if (Object.hasOwn(record, 'weight') && (!Number.isFinite(record.weight) || record.weight <= 0)) fail('weight must be finite and positive');
  const position = record.position;
  if (!position || typeof position !== 'object' || Array.isArray(position)
    || !Number.isInteger(position.action) || position.action < 0 || position.action > 1_000_000
    || !Array.isArray(position.board) || !position.board.length || position.board.length > 257) fail('invalid position action or timelines');
  let boards = 0;
  for (const timeline of position.board) {
    if (timeline === null) continue;
    if (!Array.isArray(timeline) || timeline.length > 8192) fail('invalid timeline');
    for (const board of timeline) {
      if (board === null) continue;
      const width = board?.[0]?.length;
      if (!Array.isArray(board) || !board.length || board.length > 16 || !Number.isInteger(width) || width < 1 || width > 16) fail('invalid board dimensions');
      for (const rank of board) {
        if (!Array.isArray(rank) || rank.length !== width || rank.some(piece => !Number.isInteger(piece) || Math.abs(piece) > 24)) fail('invalid board squares');
      }
      boards++;
    }
  }
  if (!boards) fail('position has no boards');
  const promotions = position.promotions ?? [];
  if (!Array.isArray(promotions) || promotions.some(piece => !Number.isInteger(piece) || Math.abs(piece) < 1 || Math.abs(piece) > 24)) fail('invalid promotions');
  // Store an independent JSON snapshot even when the producer reuses objects.
  let text;
  try { text = JSON.stringify(record); } catch { fail('sample is not JSON serializable'); }
  if (Buffer.byteLength(text) > MAX_REPLAY_LINE_BYTES) fail('sample exceeds the 4 MiB line limit');
  return {record:JSON.parse(text), key:hash(positionKey(position))};
}

// Balance the rows that actually survive replay selection, not original game
// lengths: deduplication, exclusions and eviction can remove different shares
// of each game. Preserve total self-play weight so seed data keeps its share.
function balanceGameWeights(records) {
  const gameKey = record => record.source === 'transformer-selfplay'
    && typeof record.gameId === 'string' && record.gameId.trim()
    // Existing IDs include the seed, which can be reused when resuming with
    // different CLI options. Iteration provenance keeps those games separate.
    ? JSON.stringify([record.provenance?.runId ?? null, record.provenance?.iteration ?? null, record.gameId]) : null;
  const gameCounts = new Map();
  let samples = 0, ungroupedSamples = 0;
  for (const record of records) {
    if (record.source !== 'transformer-selfplay') continue;
    const key = gameKey(record);
    if (key === null) {
      ungroupedSamples++;
      continue;
    }
    samples++;
    gameCounts.set(key, (gameCounts.get(key) || 0) + 1);
  }
  const weightPerGame = gameCounts.size ? samples / gameCounts.size : null;
  for (const record of records) {
    const key = gameKey(record);
    if (key !== null) {
      record.weight = weightPerGame / gameCounts.get(key);
    }
  }
  return {method:'equal-retained-game-weight', games:gameCounts.size, samples,
    weightPerGame, ungroupedSamples};
}

async function* jsonLines(file) {
  let chunks = [], length = 0, lineNumber = 0;
  function append(chunk) {
    length += chunk.length;
    if (length > MAX_REPLAY_LINE_BYTES) throw new Error(`Replay line exceeds the 4 MiB limit: ${file}:${lineNumber + 1}`);
    chunks.push(chunk);
  }
  function parse() {
    lineNumber++;
    const text = Buffer.concat(chunks, length).toString('utf8').trim();
    chunks = []; length = 0;
    if (!text) return null;
    try { return {record:JSON.parse(text), label:`${file}:${lineNumber}`}; }
    catch { throw new Error(`Invalid replay JSON: ${file}:${lineNumber}`); }
  }
  for await (const chunk of createReadStream(file, {highWaterMark:64 * 1024})) {
    let offset = 0, newline;
    while ((newline = chunk.indexOf(10, offset)) !== -1) {
      append(chunk.subarray(offset, newline));
      const item = parse();
      if (item) yield item;
      offset = newline + 1;
    }
    if (offset < chunk.length) append(chunk.subarray(offset));
  }
  if (length) { const item = parse(); if (item) yield item; }
}

// A bounded hash-priority reservoir samples unique position keys without an
// unbounded set of every key ever seen. Repeated positions keep their last label
// and do not gain extra sampling weight. The heap's root is its largest priority.
class Reservoir {
  constructor(capacity, seed) { this.capacity = capacity; this.seed = seed; this.heap = []; this.byKey = new Map(); }
  add(item) {
    const existing = this.byKey.get(item.key);
    if (existing) { existing.record = item.record; return; }
    const entry = {...item, priority:hash(`${this.seed}:${item.key}`)};
    const heap = this.heap;
    if (heap.length === this.capacity) {
      if (entry.priority >= heap[0].priority) return;
      this.byKey.delete(heap[0].key);
      heap[0] = entry;
      let index = 0;
      while (true) {
        const left = index * 2 + 1, right = left + 1;
        let largest = index;
        if (left < heap.length && heap[left].priority > heap[largest].priority) largest = left;
        if (right < heap.length && heap[right].priority > heap[largest].priority) largest = right;
        if (largest === index) break;
        [heap[index], heap[largest]] = [heap[largest], heap[index]]; index = largest;
      }
    } else {
      heap.push(entry);
      let index = heap.length - 1;
      while (index > 0) {
        const parent = Math.floor((index - 1) / 2);
        if (heap[parent].priority >= heap[index].priority) break;
        [heap[index], heap[parent]] = [heap[parent], heap[index]]; index = parent;
      }
    }
    this.byKey.set(entry.key, entry);
  }
}

/**
 * New samples may be an iterable/async iterable or a JSONL path. Existing replay
 * takes precedence over seedData. Memory retains at most maxSamples historical
 * reservoir records plus maxSamples recent records, regardless of file size.
 */
export async function updateReplay({replayPath, newSamples = [], seedData, maxSamples = 8192, seed = 1, excludePositionKeys = new Set()}) {
  if (typeof replayPath !== 'string' || !replayPath.trim()) throw new Error('replayPath must be a file path.');
  if (!Number.isInteger(maxSamples) || maxSamples < 1 || maxSamples > 1_000_000) throw new Error('maxSamples must be an integer from 1 to 1000000.');
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('seed must be an integer from 0 to 4294967295.');
  if (!(excludePositionKeys instanceof Set) || [...excludePositionKeys].some(key => typeof key !== 'string')) throw new Error('excludePositionKeys must be a Set of positionKey strings.');
  const excludedKeys = new Set([...excludePositionKeys].map(hash));
  let excludedSamples = 0;
  const shouldExclude = item => {
    if (!excludedKeys.has(item.key)) return false;
    excludedSamples++;
    return true;
  };
  await distinctPaths({replayPath, seedData, newSamples:typeof newSamples === 'string' ? newSamples : undefined});
  if (typeof newSamples !== 'string' && !newSamples?.[Symbol.iterator] && !newSamples?.[Symbol.asyncIterator]) throw new Error('newSamples must be an iterable or JSONL file path.');
  const reservoir = new Reservoir(maxSamples, seed), recent = new Map();
  const recentLimit = Math.ceil(maxSamples / 2);
  const replayExists = await stat(replayPath).then(() => true).catch(error => { if (!missing(error)) throw error; return false; });
  const oldPath = replayExists ? replayPath : seedData;
  if (oldPath) {
    try {
      for await (const {record, label} of jsonLines(oldPath)) {
        const item = validateSample(record, label);
        if (!shouldExclude(item)) reservoir.add(item);
      }
    }
    catch (error) {
      if (missing(error) && !replayExists) throw new Error(`Seed data file not found: ${oldPath}. Run npm run transformer:data or select an existing seed file (use --seed-data none to start from new samples only).`);
      throw error;
    }
  }
  async function* incoming() {
    if (typeof newSamples === 'string') yield* jsonLines(newSamples);
    else { let index = 0; for await (const record of newSamples) yield {record, label:`newSamples[${index++}]`}; }
  }
  for await (const {record, label} of incoming()) {
    const item = validateSample(record, label);
    if (shouldExclude(item)) continue;
    const existing = reservoir.byKey.get(item.key);
    if (existing) existing.record = item.record;
    recent.delete(item.key); recent.set(item.key, item);
    if (recent.size > maxSamples) recent.delete(recent.keys().next().value);
  }
  const incomingRecords = [...recent.values()];
  const reserved = incomingRecords.slice(-recentLimit);
  const selectedKeys = new Set(reserved.map(item => item.key));
  const older = reservoir.heap.filter(item => !selectedKeys.has(item.key)).sort((a, b) => a.priority.localeCompare(b.priority)).slice(0, maxSamples - reserved.length);
  for (const item of older) selectedKeys.add(item.key);
  const remaining = maxSamples - reserved.length - older.length;
  const extra = remaining ? incomingRecords.filter(item => !selectedKeys.has(item.key)).slice(-remaining) : [];
  const selected = [...older, ...extra, ...reserved].map(item => item.record);
  if (!selected.length) throw new Error('Replay requires at least one valid sample after exclusions.');
  const weighting = balanceGameWeights(selected);
  const counts = new Map();
  for (const record of selected) {
    const source = typeof record.source === 'string' ? record.source : 'unknown';
    counts.set(source, (counts.get(source) || 0) + 1);
  }
  const text = selected.map(record => {
    const line = JSON.stringify(record);
    // Adding/recalculating weights must not create a file we cannot read back.
    if (Buffer.byteLength(line) > MAX_REPLAY_LINE_BYTES) throw new Error('Weighted replay sample exceeds the 4 MiB line limit.');
    return line;
  }).join('\n') + '\n';
  await atomicWrite(replayPath, text);
  return {samples:selected.length, sourceCounts:Object.fromEntries(counts), sha256:hash(text), excludedSamples, weighting};
}

/**
 * Caller must hold the run lock across evaluation and promotion. Hash checks
 * catch most outside edits, but check+rename is not a cross-process CAS; another
 * writer ignoring this lock can still race the final hash check and rename.
 */
export async function promoteCheckpoint({candidatePath, activePath, expectedHash, backupPath, shouldStop = () => false}) {
  const checkCancellation = () => {
    if (!shouldStop()) return;
    const error = new Error('Checkpoint promotion cancelled.');
    error.name = 'AbortError';
    throw error;
  };
  checkCancellation();
  if (!/^[a-f0-9]{64}$/i.test(expectedHash || '')) throw new Error('expectedHash must be the active checkpoint SHA256.');
  if (!candidatePath || !activePath) throw new Error('candidatePath and activePath are required.');
  if (!backupPath) throw new Error('backupPath is required to preserve the active checkpoint.');
  await distinctPaths({candidatePath, activePath, backupPath});
  const expected = expectedHash.toLowerCase();
  const current = await readFile(activePath);
  if (hash(current) !== expected) throw new Error('Active checkpoint changed; refusing promotion.');
  const candidate = await readFile(candidatePath);
  if (!candidate.length) throw new Error('Candidate checkpoint is empty.');
  checkCancellation();
  try { await writeAtomic(backupPath, current, {exclusive:true, beforeCommit:checkCancellation}); }
  catch (error) {
    if (error.code === 'EEXIST') throw new Error(`Backup already exists; refusing to overwrite ${backupPath}.`);
    throw error;
  }
  await writeAtomic(activePath, candidate, {beforeCommit:async () => {
    if (await fileHash(activePath) !== expected) throw new Error('Active checkpoint changed during promotion; refusing replacement.');
    checkCancellation();
  }});
  return {previousHash:expected, sha256:hash(candidate), backupPath:resolve(backupPath)};
}
