import test from 'node:test';
import assert from 'node:assert/strict';
import { parseSmokeArgs, smokeFixtures, verifyFixture } from '../scripts/lc0-smoke.js';

test('smoke arguments require an explicit checkpoint and preserve its path', () => {
  assert.deepEqual(parseSmokeArgs(['--help']), { help: true });
  assert.equal(parseSmokeArgs(['--checkpoint', 'candidate.pt', '--device', 'cuda']).checkpoint, 'candidate.pt');
  for (const args of [[], ['--unknown'], ['--checkpoint'], ['--checkpoint', 'x', '--nodes', 'NaN'],
    ['--checkpoint', 'x', '--device', 'other'], ['--checkpoint', 'x', '--output', 'x']]) {
    assert.throws(() => parseSmokeArgs(args));
  }
});

test('small smoke fixtures exercise legal travel and compound submissions', async () => {
  const info = { model: { runtimeGeneration: 9 } };
  const runtime = {
    async evaluate(positions, options) {
      assert.equal(options.runtimeGeneration, 9);
      return { values: positions.map(() => 0), context: positions.map(() => ({ truncated: false })) };
    },
    async orderMoves(position, moves, options) {
      assert.equal(options.runtimeGeneration, 9);
      return moves.map(() => 0);
    },
  };
  for (const fixture of smokeFixtures()) {
    for (const timeline of fixture.position.board.filter(Boolean)) {
      for (const board of timeline.filter(Boolean)) assert(board.length <= 4 && board[0].length <= 4);
    }
    const report = await verifyFixture(runtime, info, fixture);
    assert.equal(report.passed, true);
    assert.equal(report.legalPv, true);
    assert(report.evaluations > 0 && report.policyCalls > 0);
    assert.equal(report.submitPolicy.score, 0);
  }
});

test('smoke rejects a missing trained policy before declaring success', async () => {
  const runtime = {
    async evaluate(positions) { return { values: positions.map(() => 0) }; },
    async orderMoves() { return null; },
  };
  await assert.rejects(verifyFixture(runtime, { model: { runtimeGeneration: 1 } }, smokeFixtures()[0]),
    /Trained policy must return finite/);
});
