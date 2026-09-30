import test from 'node:test';
import assert from 'node:assert/strict';
import { createEvaluator, evaluateDetailed, inspectEvaluation } from '../src/evaluate.js';

function sparseHistory(latest) {
  const history = Array.from({ length: latest + 1 }, (_, t) => {
    if (t < (t % 2 ? 7 : 4) || t === 10 || t === 15) return null;
    const squares = Array.from({ length: 8 }, () => Array(8).fill(0));
    squares[0][t % 5 ? 4 : 3] = 12;
    squares[7][4] = 11;
    squares[1][5] = 2;
    squares[6][5] = 1;
    if (t % 4 === 0) squares[2][5] = 2;
    if (t % 3 === 0) squares[5][5] = 1;
    squares[4][5] = 10;
    squares[3][7] = 9;
    return squares;
  });
  return { board: [history], action: latest % 2 };
}

test('sparse historical samples preserve corridor and pawn-entry scores at both parity boundaries', () => {
  // Scores recorded before replacing the Set-based sample schedule. The
  // first even and odd boards begin at different times; the frontier crosses
  // the six-full-turn window for each, with gaps inside the recent history.
  const expected = [
    [7, 0, 0, 7], [8, 0, 0, 100], [9, -13, 70, -27],
    [11, 10, 0, 17], [12, -9, 0, 0], [13, 10, 0, 17],
    [14, 0, 0, 7], [16, 0, 140, 240], [17, 10, 70, 87],
    [18, 0, 140, 57], [19, 10, 0, 17], [20, 5, 0, 104],
    [21, 10, 70, -3], [22, 0, 140, 147], [23, 10, 70, 87],
    [24, 0, 140, 149], [25, 15, 0, 22], [26, 0, 0, 7],
  ];
  const cached = createEvaluator();
  for (const [latest, kingSafety, travel, total] of expected) {
    const position = sparseHistory(latest);
    const result = evaluateDetailed(position);
    assert.deepEqual([result.kingSafety, result.travel, result.total], [kingSafety, travel, total], `turn ${latest}`);
    assert.equal(cached(position), total);
    assert.equal(inspectEvaluation(position).total, total);
  }
});
