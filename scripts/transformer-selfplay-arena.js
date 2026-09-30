import { setImmediate as yieldTurn } from 'node:timers/promises';
import { createPosition, positionKey } from '../src/rules.js';
import { certifyTerminal, runGame, summarizeGames, summarizePairs } from './match.js';

const FINISHED = new Set(['A_WIN', 'B_WIN', 'DRAW']);
const STRENGTH_MIN_PAIRS = 20;
const CONFIDENCE_LEVEL = 0.95;

function integer(name, value, minimum, maximum) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) throw new Error(`Invalid ${name}.`);
}

function thresholds(minPairs, promotionScore) {
  integer('minPairs', minPairs, 1, 10000);
  if (!Number.isFinite(promotionScore) || promotionScore < 0.5 || promotionScore > 1) {
    throw new Error('promotionScore must be between 0.5 and 1.');
  }
}

function certifiedFinish(game) {
  if (game?.valid !== true || !FINISHED.has(game.result)) return false;
  const certificate = game.certificate;
  if (certificate?.verified !== true || certificate.terminal !== true) return false;
  if (game.result === 'DRAW') return certificate.status === 'stalemate' && certificate.winnerColor === null;
  if (certificate.status !== 'checkmate' || ![0, 1].includes(certificate.winnerColor)) return false;
  const winner = certificate.winnerColor === game.aColor ? 'A' : 'B';
  return game.result === `${winner}_WIN`;
}

function gameCompletion(games) {
  const certifiedGames = games.filter(certifiedFinish).length;
  const unfinishedReasons = new Map();
  for (const game of games) {
    if (game?.result !== 'UNFINISHED') continue;
    const reason = game.reason ?? 'unspecified';
    unfinishedReasons.set(reason, (unfinishedReasons.get(reason) ?? 0) + 1);
  }
  return { totalGames: games.length, certifiedGames,
    gameCompletionRate: games.length ? certifiedGames / games.length : null,
    unfinishedReasons: Object.fromEntries(unfinishedReasons) };
}

function pairDetails(pair, games) {
  if (!Array.isArray(pair.gameIndices) || pair.gameIndices.length !== 2 ||
      pair.gameIndices.some(index => !Number.isInteger(index) || index < 0 || index >= games.length)) {
    return { complete: false, eligible: false, reason: 'invalid-pair-games' };
  }
  const played = pair.gameIndices.map(index => games[index]);
  const summary = summarizeGames(played);
  if (played[0].aColor !== 0 || played[1].aColor !== 1 || typeof pair.initialKey !== 'string' ||
      !pair.initialKey || played.some(game => game.initialKey !== pair.initialKey)) {
    return { complete: false, eligible: false, reason: 'unpaired-starts', summary };
  }
  const complete = played.every(certifiedFinish);
  if (!complete) return { complete: false, eligible: false, reason: 'unfinished-or-invalid-pair', summary };
  if (played.some(game => !Array.isArray(game.moves) || game.moves.length < 1 || game.plies !== game.moves.length)) {
    return { complete: true, eligible: false, reason: 'no-meaningful-play', summary };
  }
  const engines = new Set(played.flatMap(game => game.moves.map(move => move.engine)));
  if (!engines.has('A') || !engines.has('B')) {
    return { complete: true, eligible: false, reason: 'both-engines-must-play', summary };
  }
  return { complete: true, eligible: true, reason: null, summary };
}

function assessStrength(score, pairs, { invalidGames, excludedPairs }) {
  const count = score.pairs, estimate = score.aScore;
  // Each observation is the mean score of BOTH color assignments, in [0, 1].
  // A bounded-mean interval avoids treating correlated games as independent,
  // and remains wide for a tiny all-wins sample instead of reporting zero error.
  const radius = count ? Math.sqrt(Math.log(2 / (1 - CONFIDENCE_LEVEL)) / (2 * count)) : null;
  const lower = count ? Math.max(0, estimate - radius) : null;
  const upper = count ? Math.min(1, estimate + radius) : null;
  const planned = new Set(pairs.map((pair, index) => pair.initialKey || `invalid-pair-${index}`)).size;
  const unscored = Math.max(0, planned - count);
  const reasons = [];
  if (invalidGames) reasons.push('invalid-games');
  if (count < STRENGTH_MIN_PAIRS) reasons.push('insufficient-distinct-pairs');
  if (excludedPairs || unscored) reasons.push('unfinished-or-excluded-pairs');
  if (count && lower <= 0.5 && upper >= 0.5) reasons.push('interval-includes-equal-score');
  const status = reasons.length ? 'inconclusive' : lower > 0.5 ? 'candidate-advantage' : 'incumbent-advantage';
  return { status, reasons, minimumPairs: STRENGTH_MIN_PAIRS, distinctPairs: count,
    confidenceInterval: { level: CONFIDENCE_LEVEL, method: 'hoeffding-bounded-pair-means',
      unit: 'distinct-color-swapped-pair', estimate, lower, upper },
    unscoredPairs: unscored,
    // Sensitivity range, not an adjudication: assign every missing pair either
    // zero or one candidate point per game to expose completion selection bias.
    allPlannedPairScoreRange: planned ? { lower: (estimate ?? 0) * count / planned,
      upper: ((estimate ?? 0) * count + unscored) / planned } : null,
    limitation: 'The 95% interval assumes independent representative starting-position pairs. Deterministic selection, related histories and repeated model selection do not establish that assumption. This assessment describes this arena; a frozen final suite is required for independent evaluation. Unfinished pairs receive no result.' };
}

/** A small acceptance gate, not an Elo estimate or a statistical strength proof. */
export function decidePromotion({ pairs = [], games = [], minPairs = 4, promotionScore = 0.55 } = {}) {
  thresholds(minPairs, promotionScore);
  if (!Array.isArray(pairs) || !Array.isArray(games)) throw new Error('pairs and games must be arrays.');
  const seen = new Set(), eligible = [];
  let duplicatePairs = 0, excludedPairs = 0, completePairs = 0;
  for (const pair of pairs) {
    const details = pairDetails(pair, games);
    if (details.complete) completePairs++;
    if (!details.eligible) { excludedPairs++; continue; }
    if (seen.has(pair.initialKey)) { duplicatePairs++; continue; }
    seen.add(pair.initialKey);
    eligible.push({ ...pair, ...details, complete: true });
  }
  const score = summarizePairs(eligible);
  // These describe shared match outcomes, including unfinished and invalid games
  // in the denominator. A color split does not attribute a stop to either engine.
  const completion = { ...gameCompletion(games), totalPairs: pairs.length, completePairs,
    pairCompletionRate: pairs.length ? completePairs / pairs.length : null,
    eligiblePairs: score.pairs, eligiblePairRate: pairs.length ? score.pairs / pairs.length : null,
    byCandidateColor: { white: gameCompletion(games.filter(game => game?.aColor === 0)),
      black: gameCompletion(games.filter(game => game?.aColor === 1)) } };
  const invalidGames = games.filter(game => game?.valid !== true).length;
  let reason;
  if (invalidGames) reason = 'invalid-games';
  else if (score.pairs < minPairs) reason = 'insufficient-complete-distinct-pairs';
  else if (score.aScore <= 0.5) reason = 'no-winning-margin';
  else if (score.aScore < promotionScore) reason = 'below-promotion-score';
  else reason = 'promotion-threshold-met';
  const strengthAssessment = assessStrength(score, pairs, { invalidGames, excludedPairs });
  return { promote: reason === 'promotion-threshold-met', reason, minPairs, promotionScore,
    candidate: 'A', incumbent: 'B', candidateScore: score.aScore,
    candidatePoints: score.aPoints, incumbentPoints: score.bPoints,
    eligiblePairs: score.pairs, eligibleGames: score.pairs * 2, invalidGames, excludedPairs, duplicatePairs,
    completion, strengthAssessment,
    requirement: 'At least minPairs distinct starting positions, two valid certified games with played turns per pair, and candidate score above 50% and at least promotionScore. Any invalid game vetoes promotion.',
    limitation: 'Promotion retains the operational acceptance gate. strengthAssessment is separate and may remain inconclusive even when promotion passes. This repeatedly used deterministic arena is not independent evidence of general strength or an Elo estimate.' };
}

/**
 * Run candidate=A and incumbent=B on identical starts with colors swapped.
 * Both engines must support up to gameConcurrency concurrent search calls.
 * Games and pairs retain their planned order; onGame(game, {index,completed,total})
 * is awaited in completion order, with no overlapping callbacks.
 */
export async function evaluateCandidate({ candidate, incumbent, suite, pairs: requestedPairs = 4, seed = 1,
  maxPlies = 80, maxNodes = 20000, maxDepth = 2, timeMs = 3000, terminalTimeMs = timeMs, terminalWork = 20000,
  minPairs = 4, promotionScore = 0.55, gameConcurrency = 1, shouldStop, onGame } = {}) {
  if (typeof candidate !== 'function' || typeof incumbent !== 'function') throw new Error('candidate and incumbent must be analyze callbacks.');
  if (!suite || !Array.isArray(suite.cases) || !suite.cases.length) throw new Error('suite needs nonempty cases.');
  if (shouldStop !== undefined && typeof shouldStop !== 'function') throw new Error('shouldStop must be a function.');
  if (onGame !== undefined && typeof onGame !== 'function') throw new Error('onGame must be a function.');
  thresholds(minPairs, promotionScore);
  integer('pairs', requestedPairs, 1, 10000);
  integer('gameConcurrency', gameConcurrency, 1, 8);
  integer('seed', seed, 0, 0xffffffff);
  for (const [name, value, minimum, maximum] of [
    ['maxPlies', maxPlies, 0, 10000], ['maxNodes', maxNodes, 0, 1e9], ['maxDepth', maxDepth, 0, 64],
    ['timeMs', timeMs, 1, 3600000], ['terminalTimeMs', terminalTimeMs, 1, 3600000], ['terminalWork', terminalWork, 0, 1e9],
  ]) integer(name, value, minimum, maximum);
  const limits = { engine: 'transformer', maxPlies, maxNodes, maxDepth, timeMs, terminalTimeMs, terminalWork, quiescenceDepth: 0, playOnTimeLimit: true };
  let cancellation = null, failed = false, failure;
  function fail(error) {
    if (!failed) { failed = true; failure = error; }
    if (!cancellation) {
      cancellation = new Error('Transformer self-play arena interrupted.');
      cancellation.name = 'AbortError';
    }
  }
  function checkStopped() {
    if (!cancellation && shouldStop?.()) {
      cancellation = new Error('Transformer self-play arena cancelled.');
      cancellation.name = 'AbortError';
    }
    if (cancellation) throw cancellation;
  }
  const wrap = engine => async (position, gameLimits) => {
    checkStopped();
    let timer;
    const interrupted = new Promise((resolve, reject) => {
      timer = setInterval(() => { try { checkStopped(); } catch (error) { reject(error); } }, 25);
    });
    try {
      const result = await Promise.race([
        Promise.resolve().then(() => {
          checkStopped();
          return engine(position, { ...gameLimits, engine: 'transformer', shouldStop: () => Boolean(cancellation || shouldStop?.()) });
        }),
        interrupted,
      ]);
      if (result?.stoppedReason === 'cancelled') {
        cancellation ??= new Error('Transformer self-play arena cancelled by an engine.');
        cancellation.name = 'AbortError';
      }
      checkStopped();
      return result;
    } catch (error) {
      if (error?.name === 'AbortError') cancellation ??= error;
      throw error;
    } finally { clearInterval(timer); }
  };
  const engineA = wrap(candidate), engineB = wrap(incumbent);
  const games = [], planned = [], pairPlans = [], scheduledCases = [], skippedCases = [], seen = new Set();
  const offset = seed % suite.cases.length;
  // Fix starts and both color assignments before dispatching any games, so
  // completion timing cannot affect selection, game indices, or pair order.
  checkStopped();
  for (let index = 0; index < suite.cases.length && pairPlans.length < requestedPairs; index++) {
    // Each certificate is bounded synchronous work. Yield between cases so
    // cancellation signals and timers are observed during long preparations.
    await yieldTurn();
    checkStopped();
    const fixture = suite.cases[(offset + index) % suite.cases.length];
    if (!fixture || typeof fixture.id !== 'string') throw new Error('Each arena case needs a string id.');
    if (fixture.file && !fixture.position) throw new Error('Load file-based arena fixtures with loadMatchSuite before evaluating.');
    const position = fixture.position ? structuredClone(fixture.position) : createPosition({ pgn: fixture.pgn, variant: fixture.variant });
    const initialKey = positionKey(position);
    if (seen.has(initialKey)) { skippedCases.push({ caseId: fixture.id, reason: 'duplicate-position' }); continue; }
    seen.add(initialKey);
    const certificate = certifyTerminal(position, limits);
    checkStopped();
    if (!certificate.verified || certificate.terminal) {
      skippedCases.push({ caseId: fixture.id, reason: certificate.terminal ? 'terminal-start' : 'uncertified-start', certificate });
      continue;
    }
    scheduledCases.push(fixture.id);
    const gameIndices = [];
    for (const aColor of [0, 1]) {
      gameIndices.push(planned.length);
      planned.push({ caseId: fixture.id, category: fixture.category ?? null, position, aColor });
    }
    pairPlans.push({ caseId: fixture.id, seed, initialKey, gameIndices });
  }
  let nextIndex = 0, completedCount = 0, streamTail = Promise.resolve();
  async function lane() {
    try {
      while (nextIndex < planned.length) {
        checkStopped();
        const index = nextIndex++, { caseId, category, position, aColor } = planned[index];
        const game = { caseId, category, seed, index,
          ...await runGame({ position, engineA, engineB, aColor, ...limits }) };
        checkStopped(); // runGame converts engine exceptions into invalid results.
        games[index] = game;
        const completed = ++completedCount;
        if (onGame) {
          const streaming = streamTail.then(async () => {
            checkStopped();
            await onGame(structuredClone(game), { index, completed, total: planned.length });
            checkStopped();
          });
          // Handle writer failures immediately, interrupt other searches, and
          // let each lane settle before the arena's caller closes its workers.
          streamTail = streaming.catch(fail);
          await streaming;
        }
      }
    } catch (error) { fail(error); }
  }
  await Promise.all(Array.from({ length: Math.min(gameConcurrency, planned.length) }, () => lane()));
  await streamTail;
  if (failed) throw failure;
  checkStopped();
  const paired = pairPlans.map(pair => ({ ...pair, ...pairDetails(pair, games) }));
  const decision = decidePromotion({ pairs: paired, games, minPairs, promotionScore });
  return { engines: { A: 'candidate', B: 'incumbent' }, suiteVersion: suite.version ?? null,
    seed, requestedPairs, gameConcurrency, limits, scheduledCases, skippedCases,
    games, pairs: paired, summary: { ...summarizeGames(games), totalPairs: paired.length,
      completePairs: paired.filter(pair => pair.complete).length,
      eligiblePairs: decision.eligiblePairs, uniqueStartingPositions: new Set(paired.map(pair => pair.initialKey)).size,
      completion: decision.completion, strengthAssessment: decision.strengthAssessment,
      completedPairScore: summarizePairs(paired.map(pair => ({ ...pair, complete: pair.eligible }))) }, decision,
    methodology: 'Deterministic case rotation by seed; distinct full-history starting positions only. Candidate A and incumbent B use equal limits and swapped colors. Validated legal actions may play after a search time limit, including incomplete fallbacks; search interruption remains recorded. Time limits without an action leave games unfinished. Only independently certified checkmate or stalemate finishes games. Unfinished games are not draws and neither evaluation scores nor ply limits adjudicate results. Only complete, valid, played pairs enter promotion scoring; any invalid game blocks promotion. Repeated arena selection can overfit this suite; no statistical strength guarantee.' };
}
