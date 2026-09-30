import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, readdir, mkdir, link, symlink, lstat, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { atomicWrite, fileHash, acquireRunLock, inspectRunLock, recoverRunLock, updateReplay, promoteCheckpoint, MAX_REPLAY_LINE_BYTES } from '../scripts/transformer-selfplay-store.js';
import { positionKey } from '../src/rules.js';

async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), 'vibe-selfplay-store-'));
  t.after(() => rm(directory, {recursive:true, force:true}));
  return name => join(directory, name);
}
const sample = (id, source = 'seed') => ({position:{action:0, board:[[[[id % 25, Math.floor(id / 25) % 25]]]], promotions:[]}, value:id, source});
const jsonl = records => records.map(record => JSON.stringify(record)).join('\n') + '\n';
const recordsAt = async file => (await readFile(file, 'utf8')).trim().split('\n').map(JSON.parse);
async function exitedPid() {
  const child = spawn(process.execPath, ['-e', 'process.exit(0)'], {stdio:'ignore'});
  await once(child, 'exit');
  return child.pid;
}
const gameSample = (id, gameId, gameResult = 'UNFINISHED') => ({
  ...sample(id, 'transformer-selfplay'), gameId, gameResult,
});
const gameWeights = records => {
  const totals = {};
  for (const row of records.filter(row => row.source === 'transformer-selfplay' && row.gameId)) {
    totals[row.gameId] = (totals[row.gameId] || 0) + row.weight;
  }
  return totals;
};

test('atomic writes replace complete text/binary files and incremental hash matches contents', async t => {
  const file = await fixture(t), active = file('nested/active.pt');
  await atomicWrite(active, 'first');
  const bytes = Buffer.from([0, 255, 128, 42]);
  await atomicWrite(active, bytes);
  assert.deepEqual(await readFile(active), bytes);
  assert.equal(await fileHash(active), createHash('sha256').update(bytes).digest('hex'));
  assert.deepEqual(await readdir(file('nested')), ['active.pt']);
  await assert.rejects(atomicWrite(active, {invalid:true}), /text or a Buffer/);
  assert.deepEqual(await readFile(active), bytes);
});

test('failed atomic publication removes its temporary file and preserves destination', async t => {
  const file = await fixture(t);
  await mkdir(file('target'));
  await assert.rejects(atomicWrite(file('target'), 'data'));
  assert.deepEqual(await readdir(file('.')), ['target']);
});

test('run lock rejects a live owner, releases idempotently, and can then be reacquired', async t => {
  const file = await fixture(t);
  assert.equal(await inspectRunLock(file('run')), null);
  const release = await acquireRunLock(file('run'));
  const inspection = await inspectRunLock(file('run'));
  assert.equal(inspection.owner.pid, process.pid);
  assert.equal(inspection.live, true);
  assert.equal(inspection.recoverable, false);
  assert.equal((await recoverRunLock(file('run'))).recovered, false);
  await assert.rejects(acquireRunLock(file('run')), new RegExp(`already running under PID ${process.pid}`));
  await release(); await release();
  const releaseAgain = await acquireRunLock(file('run'));
  await releaseAgain();
});

test('stale/malformed locks explain manual recovery and release does not remove a replacement', async t => {
  const file = await fixture(t), lock = file('.selfplay.lock');
  await writeFile(lock, 'not JSON');
  await assert.rejects(acquireRunLock(file('.')), /Stale or malformed.*Verify no runner is active/);
  assert.equal(await readFile(lock, 'utf8'), 'not JSON');
  await rm(lock);
  const release = await acquireRunLock(file('.'));
  await writeFile(lock, JSON.stringify({pid:process.pid, token:'replacement'}));
  await assert.rejects(release(), /ownership changed/);
  assert.equal(JSON.parse(await readFile(lock, 'utf8')).token, 'replacement');
});

test('a confirmed exited owner is recovered with its exact lock preserved in an archive', async t => {
  const file = await fixture(t), lock = file('.selfplay.lock');
  const owner = {pid:await exitedPid(), token:'exited-owner', createdAt:'2026-09-25T00:00:00.000Z'};
  const raw = `${JSON.stringify(owner)}\n`;
  await writeFile(lock, raw);
  assert.deepEqual(await inspectRunLock(file('.')), {file:lock, owner, live:false, recoverable:true, reason:'owner-exited'});
  const recovery = await recoverRunLock(file('.'));
  assert.equal(recovery.recovered, true);
  assert.equal(await readFile(recovery.archivePath, 'utf8'), raw);
  assert.equal(await inspectRunLock(file('.')), null);
  assert.equal(await recoverRunLock(file('.')), null);
});

test('concurrent starts recover one stale lock and retain exactly one new runner lock', async t => {
  const file = await fixture(t), lock = file('.selfplay.lock');
  const raw = JSON.stringify({pid:await exitedPid(), token:'concurrent-dead-owner'});
  await writeFile(lock, raw);
  const starts = await Promise.allSettled(Array.from({length:16}, () => acquireRunLock(file('.'))));
  const successful = starts.filter(result => result.status === 'fulfilled');
  assert.equal(successful.length, 1);
  assert.equal((await inspectRunLock(file('.'))).live, true);
  const archives = (await readdir(file('.'))).filter(name => name.startsWith('.selfplay.lock.recovered-'));
  assert.equal(archives.length, 1);
  assert.equal(await readFile(file(archives[0]), 'utf8'), raw);
  await successful[0].value();
  assert.equal(await inspectRunLock(file('.')), null);
});

test('an interrupted recovery claim is preserved for manual review', async t => {
  const file = await fixture(t), lock = file('.selfplay.lock'), token = 'interrupted-owner';
  const raw = JSON.stringify({pid:await exitedPid(), token});
  await writeFile(lock, raw);
  const archive = `${lock}.recovered-${createHash('sha256').update(token).digest('hex')}`;
  await writeFile(archive, 'incomplete recovery');
  const recovery = await recoverRunLock(file('.'));
  assert.equal(recovery.recovered, false);
  assert.equal(recovery.reason, 'recovery-claimed');
  await assert.rejects(acquireRunLock(file('.')), /recovery-claimed.*Verify no runner is active/);
  assert.equal(await readFile(lock, 'utf8'), raw);
  assert.equal(await readFile(archive, 'utf8'), 'incomplete recovery');
});

test('EPERM owner probes and malformed ownership never permit automatic recovery', async t => {
  const file = await fixture(t), lock = file('.selfplay.lock');
  const probe = t.mock.method(process, 'kill', () => { throw Object.assign(new Error('Access denied'), {code:'EPERM'}); });
  const owner = {pid:1234, token:'protected-owner'};
  const raw = JSON.stringify(owner);
  await writeFile(lock, raw);
  const inspection = await inspectRunLock(file('.'));
  assert.equal(inspection.live, true);
  assert.equal(inspection.recoverable, false);
  assert.equal((await recoverRunLock(file('.'))).recovered, false);
  assert.equal(await readFile(lock, 'utf8'), raw);
  probe.mock.restore();
  for (const invalid of ['null', '{}', JSON.stringify({pid:1234}), JSON.stringify({pid:1234, token:''}), JSON.stringify({pid:0x80000000, token:'invalid-pid'}), 'x'.repeat(4097)]) {
    await writeFile(lock, invalid);
    assert.equal((await inspectRunLock(file('.'))).reason, 'malformed');
    assert.equal((await recoverRunLock(file('.'))).recovered, false);
    assert.equal(await readFile(lock, 'utf8'), invalid);
  }
  assert.deepEqual(await readdir(file('.')), ['.selfplay.lock']);
});

test('symbolic-link locks are preserved without touching their target', async t => {
  const file = await fixture(t), lock = file('.selfplay.lock'), target = file('target.json');
  const raw = JSON.stringify({pid:await exitedPid(), token:'linked-owner'});
  await writeFile(target, raw);
  try { await symlink(target, lock, 'file'); }
  catch (error) {
    if (error.code === 'EPERM' || error.code === 'EACCES') return t.skip('Creating symlinks is unavailable for this account.');
    throw error;
  }
  assert.equal((await inspectRunLock(file('.'))).reason, 'symbolic-link');
  assert.equal((await recoverRunLock(file('.'))).recovered, false);
  assert.equal((await lstat(lock)).isSymbolicLink(), true);
  assert.equal(await readFile(target, 'utf8'), raw);
});

test('confirmed worker exit recovers only the current process and exact invocation token', async t => {
  const file = await fixture(t);
  const release = await acquireRunLock(file('.'), {token:'exited-worker'});
  assert.equal((await inspectRunLock(file('.'))).recoverable, false);
  assert.equal((await recoverRunLock(file('.'), {exitedOwner:{pid:process.pid + 1, token:'exited-worker'}})).recovered, false);
  assert.equal((await recoverRunLock(file('.'), {exitedOwner:{pid:process.pid, token:'another-worker'}})).recovered, false);
  const recovery = await recoverRunLock(file('.'), {exitedOwner:{pid:process.pid, token:'exited-worker'}});
  assert.equal(recovery.recovered, true);
  assert.equal(recovery.reason, 'execution-exited');
  await release();
  const releaseReplacement = await acquireRunLock(file('.'), {token:'replacement-worker'});
  assert.equal((await recoverRunLock(file('.'), {exitedOwner:{pid:process.pid, token:'exited-worker'}})).recovered, false);
  assert.equal((await inspectRunLock(file('.'))).owner.token, 'replacement-worker');
  await releaseReplacement();
});

test('large incoming batch retains half historical reservoir and half newest unique samples', async t => {
  const file = await fixture(t), seedData = file('seed.jsonl'), replayPath = file('replay.jsonl');
  await writeFile(seedData, jsonl(Array.from({length:100}, (_, i) => sample(i))));
  const newSamples = Array.from({length:400}, (_, i) => sample(i + 100, 'new'));
  const result = await updateReplay({replayPath, seedData, newSamples, maxSamples:10, seed:11});
  const stored = await recordsAt(replayPath);
  assert.equal(result.samples, 10);
  assert.deepEqual(result.sourceCounts, {seed:5, new:5});
  assert.deepEqual(stored.slice(-5).map(record => record.value), [495, 496, 497, 498, 499]);
  assert.equal(new Set(stored.map(record => positionKey(record.position))).size, 10);
  assert.equal(result.sha256, await fileHash(replayPath));
});

test('replay deduplicates latest labels/sources while retaining different sides and history', async t => {
  const file = await fixture(t), seedData = file('seed.jsonl'), replayPath = file('replay.jsonl');
  const white = sample(1), black = {...sample(1), position:{...sample(1).position, action:1}};
  const historyA = {...sample(2), position:{...sample(2).position, board:[[[[1, 0]], [[2, 0]]]]}};
  const historyB = {...sample(2), position:{...sample(2).position, board:[[[[3, 0]], [[2, 0]]]]}};
  await writeFile(seedData, jsonl([white, {...white, value:20, source:'old-update'}, black, historyA, historyB]));
  const updated = {...white, value:123, source:'new-update'};
  const result = await updateReplay({replayPath, seedData, newSamples:[updated, {...updated, value:456}], maxSamples:10});
  const stored = await recordsAt(replayPath);
  assert.equal(result.samples, 4);
  assert.deepEqual(result.sourceCounts, {seed:3, 'new-update':1});
  assert.equal(stored.find(record => positionKey(record.position) === positionKey(white.position)).value, 456);
  assert.equal(new Set(stored.map(record => positionKey(record.position))).size, 4);
});

test('new-only initialization fills capacity, supports async input, and cap one retains newest', async t => {
  const file = await fixture(t);
  async function* incoming() { for (let index = 0; index < 20; index++) yield sample(index, 'selfplay'); }
  const result = await updateReplay({replayPath:file('replay.jsonl'), newSamples:incoming(), maxSamples:7});
  assert.equal(result.samples, 7);
  assert.deepEqual((await recordsAt(file('replay.jsonl'))).map(record => record.value), [13, 14, 15, 16, 17, 18, 19]);
  await updateReplay({replayPath:file('single.jsonl'), newSamples:[sample(0), sample(1)], maxSamples:1});
  assert.equal((await recordsAt(file('single.jsonl')))[0].value, 1);
});

const certifiedSample = (gameId, outcomeWhite, searchScoreWhiteCp = 200) => ({
  ...gameSample(1, gameId, outcomeWhite === 1 ? 'WHITE_WIN' : outcomeWhite === -1 ? 'BLACK_WIN' : 'DRAW'),
  searchScoreWhiteCp, normalizedSearchValue: Math.tanh(searchScoreWhiteCp / 1000),
  outcomeWhite, outcomeWeight: 0.5, targetType: 'outcome-blend',
  targetProvenance: 'certified-full-rules-outcome-and-completed-root-search',
  provenance: { runId: 'test', iteration: 1 },
});

test('fresh bootstrap values retain certified outcomes and repeated ingestion is idempotent', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  const win = certifiedSample('win', 1);
  await updateReplay({ replayPath, newSamples: [win], maxSamples: 10 });
  const bootstrap = { ...gameSample(1, 'unfinished'), value: -400, searchScoreWhiteCp: -400,
    normalizedSearchValue: Math.tanh(-0.4), targetType: 'search-bootstrap', outcomeWhite: null, outcomeWeight: 0,
    targetProvenance: 'completed-root-search-only-unfinished-game' };
  await updateReplay({ replayPath, newSamples: [bootstrap], maxSamples: 10 });
  let [row] = await recordsAt(replayPath);
  assert.equal(row.gameResult, 'UNFINISHED', 'latest trajectory metadata remains honest');
  assert.equal(row.searchScoreWhiteCp, -400);
  assert.equal(row.normalizedSearchValue, Math.tanh(-0.4));
  assert.equal(row.targetType, 'outcome-blend');
  assert.equal(row.outcomeEvidence.observations.length, 1);
  assert.equal(row.value, 1000 * Math.atanh(0.5 * Math.tanh(-0.4) + 0.5));
  await updateReplay({ replayPath, newSamples: [win, win, bootstrap], maxSamples: 10 });
  assert.deepEqual((await recordsAt(replayPath))[0], row);
  await updateReplay({ replayPath, maxSamples: 10 });
  assert.deepEqual((await recordsAt(replayPath))[0], row, 'resuming does not reblend the old target');
});

test('conflicting certified games aggregate separately from newest search and policy evidence', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  const first = certifiedSample('win', 1), second = certifiedSample('loss', -1, -300);
  second.policy = [{ moves: ['a', 'b'], target: 0, targetVersion: 1, targetWeights: [0.2, 0.8] }];
  await updateReplay({ replayPath, newSamples: [first, second, first, second], maxSamples: 10 });
  const [row] = await recordsAt(replayPath);
  assert.equal(row.outcomeEvidence.observations.length, 2);
  assert.equal(row.outcomeWhite, 0);
  assert.equal(row.searchScoreWhiteCp, -300);
  assert.deepEqual(row.policy, second.policy);
  assert.equal(row.value, 1000 * Math.atanh(0.5 * Math.tanh(-0.3)));
  assert.equal(row.weight, 1, 'deduplicated position retains equal-game weighting');
});

test('certified outcome evidence stays bounded and rejects malformed persisted evidence atomically', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  await updateReplay({ replayPath, newSamples: Array.from({ length: 80 }, (_, index) => certifiedSample(`game-${index}`, index % 2 ? 1 : -1)) });
  const [row] = await recordsAt(replayPath), before = await fileHash(replayPath);
  assert.equal(row.outcomeEvidence.observations.length, 64);
  for (const outcomeEvidence of [{ version: 2, observations: [] }, { version: 1, observations: [{ id: 'bad', outcomeWhite: 1, weight: 0.5 }] }]) {
    await assert.rejects(updateReplay({ replayPath, newSamples: [{ ...row, outcomeEvidence }] }), /invalid outcome evidence/);
    assert.equal(await fileHash(replayPath), before);
  }
  for (const targetWeights of [[0, 0], [-1, 2], [1], [NaN, 1]]) {
    await assert.rejects(updateReplay({ replayPath, newSamples: [{ ...sample(2), policy: [{ moves: [0, 1], targetVersion: 1, targetWeights }] }] }), /policy target weights/);
  }
});

test('outcomes retain their recorded mixture weights when run settings differ', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  const first = { ...certifiedSample('light-win', 1), outcomeWeight: 0.2 };
  const second = { ...certifiedSample('heavy-loss', -1, 0), outcomeWeight: 0.8 };
  await updateReplay({ replayPath, newSamples: [first, second] });
  const [row] = await recordsAt(replayPath);
  assert.equal(row.outcomeWeight, 0.5);
  assert(Math.abs(row.outcomeWhite + 0.6) < 1e-12);
  assert(Math.abs(row.normalizedTarget + 0.3) < 1e-12);
  assert.equal(row.normalizedTarget, (1 - row.outcomeWeight) * row.normalizedSearchValue + row.outcomeWeight * row.outcomeWhite);
});

test('arena exclusions remove seeded and incoming samples while preserving the other side and source counts', async t => {
  const file = await fixture(t), seedData = file('seed.jsonl'), replayPath = file('replay.jsonl');
  const excluded = sample(1);
  const opposite = {...sample(1), position:{...sample(1).position, action:1}};
  await writeFile(seedData, jsonl([excluded, opposite, sample(2)]));
  const excludedKey = positionKey(excluded.position);
  const result = await updateReplay({replayPath, seedData, newSamples:[excluded, sample(3, 'selfplay')], maxSamples:10, excludePositionKeys:new Set([excludedKey])});
  const stored = await recordsAt(replayPath);
  assert.equal(result.excludedSamples, 2);
  assert.equal(result.samples, 3);
  assert.deepEqual(result.sourceCounts, {seed:2, selfplay:1});
  assert.equal(stored.some(record => positionKey(record.position) === excludedKey), false);
  assert.equal(stored.some(record => positionKey(record.position) === positionKey(opposite.position)), true);
  const original = await fileHash(replayPath);
  await assert.rejects(updateReplay({replayPath, excludePositionKeys:new Set(stored.map(record => positionKey(record.position)))}), /at least one valid sample after exclusions/);
  assert.equal(await fileHash(replayPath), original);
});

test('short finished and long unfinished games have equal total weight without changing teacher share or targets', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  const newSamples = [sample(0), gameSample(1, 'short', 'WHITE_WIN'),
    ...[2, 3, 4].map(id => gameSample(id, 'long'))];
  const original = structuredClone(newSamples);
  const result = await updateReplay({replayPath, newSamples, maxSamples:10});
  const stored = await recordsAt(replayPath);
  assert.deepEqual(gameWeights(stored), {short:2, long:2});
  assert.equal(stored.reduce((sum, row) => sum + (row.weight ?? 1), 0), 5);
  assert.deepEqual(stored.find(row => row.source === 'seed'), sample(0));
  assert.deepEqual(stored.map(({weight, ...row}) => row), newSamples);
  assert.deepEqual(newSamples, original, 'caller records must remain unchanged');
  assert.deepEqual(result.weighting, {method:'equal-retained-game-weight', games:2,
    samples:4, weightPerGame:2, ungroupedSamples:0});
});

test('weights are rebuilt after deduplication, arena exclusion and replay capacity selection', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  const excluded = gameSample(6, 'long');
  const newSamples = [...[1, 2, 3, 4, 5].map(id => gameSample(id, 'long')),
    excluded, gameSample(5, 'short', 'BLACK_WIN')];
  await updateReplay({replayPath, newSamples, maxSamples:3,
    excludePositionKeys:new Set([positionKey(excluded.position)])});
  const stored = await recordsAt(replayPath);
  assert.deepEqual(stored.map(row => row.value), [3, 4, 5]);
  assert.deepEqual(gameWeights(stored), {long:1.5, short:1.5});
  assert.equal(stored[2].gameResult, 'BLACK_WIN', 'latest duplicate owns the target and game');
  await updateReplay({replayPath, maxSamples:3});
  assert.deepEqual((await recordsAt(replayPath)).sort((a, b) => a.value - b.value), stored,
    'resuming does not compound weights');
  const report = await updateReplay({replayPath, maxSamples:3,
    excludePositionKeys:new Set([positionKey(stored[0].position)])});
  assert.deepEqual(gameWeights(await recordsAt(replayPath)), {long:1, short:1});
  assert.equal(report.weighting.weightPerGame, 1);
});

test('existing unweighted self-play is balanced on resume and ungrouped legacy rows remain usable', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  const legacy = sample(4, 'transformer-selfplay');
  const teacher = {...sample(5), weight:2};
  await writeFile(replayPath, jsonl([gameSample(1, 'old-short', 'DRAW'),
    gameSample(2, 'old-long'), gameSample(3, 'old-long'), legacy, teacher]));
  const report = await updateReplay({replayPath, maxSamples:10});
  const stored = await recordsAt(replayPath), totals = gameWeights(stored);
  assert.equal(totals['old-short'], 1.5);
  assert.equal(totals['old-long'], 1.5);
  assert.deepEqual(stored.find(row => row.value === 4), legacy);
  assert.deepEqual(stored.find(row => row.value === 5), teacher);
  assert.equal(report.weighting.ungroupedSamples, 1);
});

test('invalid sample weights fail atomically even when balancing would replace them', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  await writeFile(replayPath, jsonl([sample(0)]));
  const originalHash = await fileHash(replayPath);
  for (const weight of [0, -1, Infinity, NaN, null, '1', true]) {
    await assert.rejects(updateReplay({replayPath,
      newSamples:[{...gameSample(1, 'bad'), weight}]}), /weight must be finite and positive/);
    assert.equal(await fileHash(replayPath), originalHash);
  }
});

test('reused game IDs from different runs or resumed iterations are weighted separately', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  const newSamples = [
    {...gameSample(1, 'reused'), provenance:{runId:'run-a', iteration:1}},
    {...gameSample(2, 'reused'), provenance:{runId:'run-a', iteration:2}},
    {...gameSample(3, 'reused'), provenance:{runId:'run-a', iteration:2}},
    {...gameSample(4, 'reused'), provenance:{runId:'run-b', iteration:1}},
  ];
  const result = await updateReplay({replayPath, newSamples, maxSamples:10});
  assert.equal(result.weighting.games, 3);
  assert.deepEqual((await recordsAt(replayPath)).map(row => row.weight), [4 / 3, 2 / 3, 2 / 3, 4 / 3]);
});

test('adding weights cannot publish an oversized replay line', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  await writeFile(replayPath, jsonl([sample(0)]));
  const originalHash = await fileHash(replayPath);
  const incoming = {...gameSample(1, 'large'), padding:''};
  incoming.padding = 'x'.repeat(MAX_REPLAY_LINE_BYTES - Buffer.byteLength(JSON.stringify(incoming)));
  await assert.rejects(updateReplay({replayPath, newSamples:[incoming]}), /Weighted replay sample exceeds the 4 MiB/);
  assert.equal(await fileHash(replayPath), originalHash);
});

test('JSONL input streams CRLF, multibyte chunk boundaries, and final line without newline', async t => {
  const file = await fixture(t), incoming = file('incoming.jsonl');
  const first = {...sample(1), note:'é'.repeat(40000)};
  await writeFile(incoming, `\r\n${JSON.stringify(first)}\r\n${JSON.stringify(sample(2))}`);
  await updateReplay({replayPath:file('replay.jsonl'), newSamples:incoming, maxSamples:5});
  const stored = await recordsAt(file('replay.jsonl'));
  assert.equal(stored.length, 2);
  assert.equal(stored[0].note, first.note);
});

test('seeded historical reservoir is reproducible, bounded, and duplicate frequency adds no weight', async t => {
  const file = await fixture(t);
  const originals = Array.from({length:100}, (_, i) => sample(i));
  await writeFile(file('seed.jsonl'), jsonl(originals));
  await writeFile(file('duplicates.jsonl'), jsonl([...originals, ...originals, ...originals]));
  const one = await updateReplay({replayPath:file('one.jsonl'), seedData:file('seed.jsonl'), maxSamples:9, seed:42});
  const two = await updateReplay({replayPath:file('two.jsonl'), seedData:file('duplicates.jsonl'), maxSamples:9, seed:42});
  assert.equal(one.sha256, two.sha256);
  const different = await updateReplay({replayPath:file('three.jsonl'), seedData:file('seed.jsonl'), maxSamples:9, seed:43});
  assert.notEqual(one.sha256, different.sha256);
  await updateReplay({replayPath:file('one.jsonl'), seedData:file('missing-ignored.jsonl'), maxSamples:9, seed:42});
  assert.equal(await fileHash(file('one.jsonl')), one.sha256);
});

test('invalid/empty samples, missing seed, oversize lines, and path collisions preserve existing replay', async t => {
  const file = await fixture(t), replayPath = file('replay.jsonl');
  await atomicWrite(replayPath, jsonl([sample(1)]));
  const original = await fileHash(replayPath);
  for (const invalid of [{...sample(2), value:Infinity}, {...sample(2), position:{}}, {...sample(2), position:{action:0, board:[]}}, {...sample(2), position:{action:0, board:[[[[999]]]]}}]) {
    await assert.rejects(updateReplay({replayPath, newSamples:[invalid]}), /Invalid replay sample/);
    assert.equal(await fileHash(replayPath), original);
  }
  await assert.rejects(updateReplay({replayPath:file('empty.jsonl')}), /at least one valid/);
  await assert.rejects(updateReplay({replayPath:file('new.jsonl'), seedData:file('missing.jsonl')}), /Seed data file not found.*transformer:data/);
  await assert.rejects(updateReplay({replayPath, newSamples:replayPath}), /different files/);
  await writeFile(file('bad.jsonl'), '{broken JSON}\n');
  await assert.rejects(updateReplay({replayPath, newSamples:file('bad.jsonl')}), /Invalid replay JSON/);
  await writeFile(file('oversize.jsonl'), ' '.repeat(MAX_REPLAY_LINE_BYTES + 1));
  await assert.rejects(updateReplay({replayPath, newSamples:file('oversize.jsonl')}), /4 MiB limit/);
  assert.equal(await fileHash(replayPath), original);
});

test('promotion keeps complete prior checkpoint and installs candidate with expected hash', async t => {
  const file = await fixture(t), activePath = file('model.pt'), candidatePath = file('candidate.pt'), backupPath = file('backup/previous.pt');
  const original = Buffer.from([1, 2, 3]), candidate = Buffer.from([9, 8, 7, 0]);
  await writeFile(activePath, original); await writeFile(candidatePath, candidate);
  const expectedHash = await fileHash(activePath);
  const result = await promoteCheckpoint({candidatePath, activePath, expectedHash, backupPath});
  assert.equal(result.previousHash, expectedHash);
  assert.equal(result.sha256, await fileHash(activePath));
  assert.deepEqual(await readFile(activePath), candidate);
  assert.deepEqual(await readFile(backupPath), original);
  assert.deepEqual(await readFile(candidatePath), candidate);
});

test('promotion refuses changed active hash, existing backup, and aliases without modifying files', async t => {
  const file = await fixture(t), activePath = file('model.pt'), candidatePath = file('candidate.pt'), backupPath = file('previous.pt');
  await writeFile(activePath, 'old'); await writeFile(candidatePath, 'new');
  const expectedHash = await fileHash(activePath);
  await writeFile(activePath, 'external edit');
  await assert.rejects(promoteCheckpoint({candidatePath, activePath, expectedHash, backupPath}), /Active checkpoint changed/);
  assert.equal(await readFile(activePath, 'utf8'), 'external edit');
  await writeFile(activePath, 'old'); await writeFile(backupPath, 'preserve me');
  await assert.rejects(promoteCheckpoint({candidatePath, activePath, expectedHash, backupPath}), /Backup already exists/);
  assert.equal(await readFile(backupPath, 'utf8'), 'preserve me');
  await assert.rejects(promoteCheckpoint({candidatePath:activePath, activePath, expectedHash, backupPath}), /different files/);
  await assert.rejects(promoteCheckpoint({candidatePath, activePath, expectedHash, backupPath:activePath}), /different files/);
  await link(activePath, file('alias.pt'));
  await assert.rejects(promoteCheckpoint({candidatePath:file('alias.pt'), activePath, expectedHash, backupPath}), /different files/);
  assert.equal(await fileHash(activePath), expectedHash);
  assert.deepEqual((await readdir(file('.'))).sort(), ['alias.pt', 'candidate.pt', 'model.pt', 'previous.pt']);
});

test('cancelled promotion never changes the active model, including cancellation after backup', async t => {
  const file = await fixture(t), activePath = file('model.pt'), candidatePath = file('candidate.pt'), backupPath = file('previous.pt');
  await writeFile(activePath, 'old'); await writeFile(candidatePath, 'new');
  const expectedHash = await fileHash(activePath);
  await assert.rejects(promoteCheckpoint({candidatePath, activePath, expectedHash, backupPath, shouldStop:() => true}), {name:'AbortError'});
  assert.equal(existsSync(backupPath), false);
  assert.equal(await fileHash(activePath), expectedHash);
  await assert.rejects(promoteCheckpoint({candidatePath, activePath, expectedHash, backupPath, shouldStop:() => existsSync(backupPath)}), {name:'AbortError'});
  assert.equal(await fileHash(activePath), expectedHash);
  assert.equal(await readFile(backupPath, 'utf8'), 'old');
  assert.deepEqual((await readdir(file('.'))).sort(), ['candidate.pt', 'model.pt', 'previous.pt']);
});
