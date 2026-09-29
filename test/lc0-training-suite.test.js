import test from 'node:test';
import assert from 'node:assert/strict';
import { lc0CurriculumStarts } from '../scripts/lc0-data.js';
import { loadMatchSuite } from '../scripts/match.js';
import { positionKey } from '../src/rules.js';

test('LCZero self-play suite excludes every curriculum validation source family', async () => {
  const [suite, original, starts] = await Promise.all([
    loadMatchSuite(new URL('../examples/matches/lc0-training.json', import.meta.url)),
    loadMatchSuite(new URL('../examples/matches/training.json', import.meta.url)),
    lc0CurriculumStarts(),
  ]);
  assert.equal(suite.version, 1);
  assert.equal(suite.curriculumVersion, 1);
  const reserved = new Set(starts.filter(start => start.split === 'validation').map(start => start.group));
  assert.deepEqual(new Set(suite.heldOutGroups), reserved);
  const expected = original.cases.filter(fixture => !reserved.has(fixture.id));
  assert.deepEqual(suite.cases, expected, 'Preserve every remaining development source.');
  const heldOutKeys = new Set(starts.filter(start => start.split === 'validation').map(start => positionKey(start.position)));
  assert(suite.cases.every(fixture => !heldOutKeys.has(positionKey(fixture.position))));
});
