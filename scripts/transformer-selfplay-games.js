import { isDeepStrictEqual } from 'node:util';
import { setImmediate as yieldTurn } from 'node:timers/promises';
import { certifyTerminal } from './match.js';
import { formatAction, positionKey, validateAction } from '../src/rules.js';
import { COMPONENT_POLICY_VERSION, policyTargetsFromSearch } from '../src/transformer-policy.js';
import { sampleExploration, emptyCoverage, actionCoverage, addCoverage } from './transformer-exploration.js';

class Interrupted extends Error {
  constructor(reason) { super(reason); this.reason = reason; }
}
const clone = value => structuredClone(value);
const isScore = value => typeof value === 'number' && Number.isFinite(value);

function settings(options) {
  const limits = { games: 8, gameConcurrency: 1, seed: 5, maxPlies: 40, timeMs: 1000, maxNodes: 20000, maxDepth: 2,
    terminalWork: 20000, exploration: 0.15, explorationPlies: 8, outcomeWeight: 0.5, ...options };
  limits.terminalTimeMs ??= limits.timeMs;
  for (const [name, low, high] of [['games', 1, 10000], ['gameConcurrency', 1, 8], ['seed', 0, 0xffffffff], ['maxPlies', 0, 10000],
    ['timeMs', 1, 60000], ['terminalTimeMs', 1, 60000], ['maxNodes', 0, 1e9], ['maxDepth', 0, 64], ['terminalWork', 0, 1e9], ['explorationPlies', 0, 10000]]) {
    if (!Number.isInteger(limits[name]) || limits[name] < low || limits[name] > high) throw new Error(`Invalid ${name}.`);
  }
  for (const name of ['exploration', 'outcomeWeight']) {
    if (typeof limits[name] !== 'number' || !Number.isFinite(limits[name]) || limits[name] < 0 || limits[name] > 1) throw new Error(`Invalid ${name}.`);
  }
  return Object.fromEntries(['games', 'gameConcurrency', 'seed', 'maxPlies', 'timeMs', 'terminalTimeMs', 'maxNodes', 'maxDepth', 'terminalWork', 'exploration', 'explorationPlies', 'outcomeWeight'].map(name => [name, limits[name]]));
}

/**
 * Generate legal Transformer-vs-itself trajectories and honest value targets.
 * analyzePosition(position, {timeMs,maxNodes,maxDepth,shouldStop}) must return a
 * Transformer search result, with White-centipawn score and a full-turn PV.
 * Up to gameConcurrency games run together; analyzePosition must support that
 * many concurrent calls. Each game has its own deterministic exploration RNG.
 * onGame(game, samples) is awaited after a game's result/validity is final, in
 * completion order with no overlapping callbacks. Returned games and samples
 * remain in original game-index order regardless of completion order.
 * Metadata is copied into provenance, e.g. checkpoint hash and training round.
 */
export async function generateSelfPlayGames(options = {}) {
  const { positions, analyzePosition, shouldStop, onGame, metadata = {} } = options;
  if (!Array.isArray(positions) || !positions.length || positions.some(item => typeof item?.id !== 'string' || !item.id || !item.position)) {
    throw new Error('positions must be a nonempty array of {id,position} starts.');
  }
  if (typeof analyzePosition !== 'function') throw new Error('analyzePosition must be an asynchronous Transformer search callback.');
  if (shouldStop !== undefined && typeof shouldStop !== 'function') throw new Error('shouldStop must be a function.');
  if (onGame !== undefined && typeof onGame !== 'function') throw new Error('onGame must be a function.');
  if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) throw new Error('metadata must be an object.');
  const limits = settings(options), provenance = clone(metadata);
  const completedGames = [];
  let cancelled = false, halted = false, failed = false, failure;
  const stopped = () => {
    if (!halted && shouldStop?.()) { cancelled = true; halted = true; }
    return halted;
  };
  const check = () => { if (stopped()) throw new Interrupted('cancelled'); };

  function terminal(position) {
    check();
    // match.js is the sole authority for finished game results. Its generator
    // ticks bound work and wall time; cancellation is checked before and after
    // this synchronous, bounded operation, never accepted as a terminal proof.
    const certificate = certifyTerminal(position, { terminalWork: limits.terminalWork, terminalTimeMs: limits.terminalTimeMs, timeMs: limits.timeMs });
    check();
    return certificate;
  }
  async function search(position) {
    check();
    let timer, localStop = false;
    // The search owns its advertised time budget. A small transport grace lets
    // it return its latest backed-up result before a stalled callback is cut
    // off. Late results touch only the callback's isolated input copy.
    const deadline = performance.now() + limits.timeMs + 1000;
    const interrupted = new Promise((resolve, reject) => {
      const poll = () => {
        if (stopped() || performance.now() >= deadline) {
          localStop = true; reject(new Interrupted(stopped() ? 'cancelled' : 'search-time-limit')); return;
        }
        timer = setTimeout(poll, Math.min(25, Math.max(1, deadline - performance.now())));
      };
      timer = setTimeout(poll, Math.min(25, Math.max(1, deadline - performance.now())));
    });
    try {
      return await Promise.race([
        Promise.resolve().then(() => analyzePosition(position, {
          timeMs: limits.timeMs, maxNodes: limits.maxNodes, maxDepth: limits.maxDepth,
          collectPolicyAlternatives: true,
          shouldStop: () => localStop || stopped(),
        })), interrupted,
      ]);
    } finally { clearTimeout(timer); }
  }
  const explore = (position, random) => sampleExploration(position, random, { ...limits, check });

  async function playGame(index) {
    // Derive an independent stream from the cycle seed and stable game index,
    // so scheduling and other games' lengths cannot change this trajectory.
    let state = (limits.seed + Math.imul(index, 0x9e3779b9)) >>> 0;
    const random = () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; };
    const start = positions[(limits.seed % positions.length + index) % positions.length];
    let current = clone(start.position);
    const gameId = `${provenance.runId ?? 'selfplay'}:${limits.seed}:${index + 1}`;
    const game = { gameId, startId: start.id, seed: limits.seed, index, result: 'UNFINISHED', valid: true,
      winnerColor: null, outcomeWhite: null, reason: null, limits: clone(limits), provenance: clone(provenance),
      initialPosition: clone(current), initialKey: positionKey(current), moves: [] };
    const pending = [];
    const finish = (reason, extra = {}) => Object.assign(game, { reason, ...extra });
    const finishTerminal = certificate => finish(certificate.status, { certificate,
      result: certificate.winnerColor === null ? 'DRAW' : certificate.winnerColor === 0 ? 'WHITE_WIN' : 'BLACK_WIN',
      winnerColor: certificate.winnerColor, outcomeWhite: certificate.winnerColor === null ? 0 : certificate.winnerColor === 0 ? 1 : -1 });
    try {
      for (let ply = 0; ply <= limits.maxPlies; ply++) {
        // Full-rules validation runs in the coordinator. Yield between plies
        // so other workers' IPC, cancellation and time budgets stay responsive.
        await yieldTurn();
        check();
        if (ply === limits.maxPlies) {
          const certificate = terminal(current);
          if (certificate.verified && certificate.terminal) finishTerminal(certificate);
          else finish(certificate.verified ? 'ply-limit' : certificate.reason, { certificate });
          break;
        }
        const input = clone(current), before = clone(input);
        let result;
        try { result = await search(input); }
        catch (error) {
          if (error instanceof Interrupted) throw error;
          if (stopped() || error.name === 'AbortError') throw new Interrupted('cancelled');
          finish('model-error', { valid: false, error: error.message }); break;
        }
        if (!isDeepStrictEqual(input, before)) { finish('input-mutation', { valid: false }); break; }
        check();
        if (!result || typeof result !== 'object' || result.engine !== 'transformer' || typeof result.completed !== 'boolean') {
          finish('invalid-search-result', { valid: false, error: 'Expected a Transformer search result.' }); break;
        }
        let recorded;
        try { recorded = clone(result); }
        catch (error) { finish('invalid-search-result', { valid: false, error: error.message }); break; }
        game.lastSearch = recorded;
        if (result.score !== null && result.score !== undefined && !isScore(result.score)) {
          finish('invalid-search-score', { valid: false }); break;
        }
        if (result.nodes !== undefined && (!Number.isInteger(result.nodes) || result.nodes < 0 || result.nodes > limits.maxNodes)) {
          finish('invalid-work-accounting', { valid: false }); break;
        }
        if (result.searchNodes !== undefined && result.generationNodes !== undefined &&
          (![result.searchNodes, result.generationNodes].every(value => Number.isInteger(value) && value >= 0) || result.nodes !== result.searchNodes + result.generationNodes)) {
          finish('invalid-work-accounting', { valid: false }); break;
        }
        let searchedNext;
        if (result.bestAction !== null && result.bestAction !== undefined) {
          try { searchedNext = validateAction(current, result.bestAction); }
          catch (error) { finish('illegal-action', { valid: false, error: error.message }); break; }
        }
        try {
          if (!Array.isArray(result.pv)) throw new Error('PV must contain complete legal actions.');
          if (searchedNext ? !result.pv.length || !isDeepStrictEqual(result.pv[0], result.bestAction) : result.pv.length) {
            throw new Error('PV must begin with the searched best action.');
          }
          let pvPosition = current;
          for (const action of result.pv) { check(); pvPosition = validateAction(pvPosition, action); }
        } catch (error) {
          if (error instanceof Interrupted) throw error;
          finish('invalid-pv', { valid: false, error: error.message }); break;
        }
        if (result.stoppedReason === 'cancelled') throw new Interrupted('cancelled');
        const terminalClaim = ['checkmate', 'stalemate'].includes(result.status);
        if (terminalClaim && searchedNext) { finish('false-terminal-claim', { valid: false }); break; }
        if (!searchedNext || terminalClaim) {
          const certificate = terminal(current);
          if (!certificate.verified) { finish(certificate.reason, { certificate }); break; }
          if (terminalClaim && (!certificate.terminal || certificate.status !== result.status)) {
            finish('false-terminal-claim', { valid: false, certificate }); break;
          }
          if (certificate.terminal) finishTerminal(certificate);
          else finish(result.completed ? 'missing-action' : 'incomplete-search', { certificate });
          break;
        }
        let action = result.bestAction, next = searchedNext;
        let explorationInfo = { attempted: false, explored: false, candidates: 0, work: 0 };
        if (ply < limits.explorationPlies && random() < limits.exploration) {
          const exploration = explore(current, random);
          if (exploration.action) {
            action = exploration.action;
            try { next = validateAction(current, action); }
            catch (error) { finish('illegal-exploration-action', { valid: false, error: error.message }); break; }
          }
          explorationInfo = { attempted: true, explored: Boolean(exploration.action), candidates: exploration.candidates,
            work: exploration.work, exhaustive: exploration.exhaustive, stoppedReason: exploration.stoppedReason,
            sampling: exploration.sampling, coverage: exploration.coverage, selectedCoverage: exploration.selectedCoverage };
        }
        check();
        let notation;
        try { notation = formatAction(current, action); }
        catch { notation = null; }
        if (result.completed && isScore(result.score)) pending.push({
          position: clone(current), searchScoreWhiteCp: result.score, source: 'transformer-selfplay', gameId, ply,
          search: clone(recorded), searchedAction: clone(result.bestAction), playedAction: clone(action),
          exploration: clone(explorationInfo), provenance: { ...clone(provenance), seed: limits.seed, startId: start.id },
        });
        game.moves.push({ ply, color: current.action % 2, beforeKey: positionKey(current), afterKey: positionKey(next),
          action: clone(action), searchedAction: clone(result.bestAction), notation, exploration: explorationInfo,
          coverage: actionCoverage(current, action, next), search: recorded });
        current = next;
      }
    } catch (error) {
      if (error instanceof Interrupted) {
        finish(error.reason);
        if (error.reason === 'cancelled') { cancelled = true; halted = true; }
      } else finish('game-error', { valid: false, error: error.message });
    }
    Object.assign(game, { plies: game.moves.length, finalPosition: clone(current), finalKey: positionKey(current) });
    const finished = game.valid && game.result !== 'UNFINISHED';
    const gameSamples = game.valid ? pending.map(row => {
      // Only completed searches from accepted games teach the policy. An
      // exploratory played action must never replace the searched target.
      const policy = policyTargetsFromSearch(row.position, row.search);
      const normalizedSearchValue = Math.tanh(row.searchScoreWhiteCp / 1000);
      const unclampedNormalizedTarget = finished ? (1 - limits.outcomeWeight) * normalizedSearchValue + limits.outcomeWeight * game.outcomeWhite : normalizedSearchValue;
      const normalizedTarget = finished ? Math.max(-0.999, Math.min(0.999, unclampedNormalizedTarget)) : unclampedNormalizedTarget;
      return { ...row, value: finished ? 1000 * Math.atanh(normalizedTarget) : row.searchScoreWhiteCp,
        ...(policy.length ? { policyVersion: COMPONENT_POLICY_VERSION, policy } : {}),
        targetType: finished ? 'outcome-blend' : 'search-bootstrap', normalizedSearchValue,
        normalizedTarget, unclampedNormalizedTarget, outcomeWhite: finished ? game.outcomeWhite : null, outcomeWeight: finished ? limits.outcomeWeight : 0,
        gameResult: game.result, gameReason: game.reason,
        targetProvenance: finished ? 'certified-full-rules-outcome-and-completed-root-search' : 'completed-root-search-only-unfinished-game' };
    }) : [];
    game.samples = gameSamples.length;
    game.coverage = game.moves.reduce((total, move) => addCoverage(total, move.coverage), emptyCoverage());
    game.discardedSamples = game.valid ? 0 : pending.length;
    return { game, samples: gameSamples };
  }
  let nextIndex = 0, streamTail = Promise.resolve();
  async function lane() {
    try {
      while (!stopped() && nextIndex < limits.games) {
        const index = nextIndex++;
        const completed = await playGame(index);
        completedGames[index] = completed;
        if (onGame) {
          const streaming = streamTail.then(async () => {
            if (failed) return;
            try { await onGame(clone(completed.game), clone(completed.samples)); }
            catch (error) { halted = true; failed = true; failure = error; throw error; }
          });
          // Attach the rejection handler immediately; every lane is drained
          // below even when a writer fails while other searches are pending.
          streamTail = streaming.catch(() => {});
          await streaming;
        }
      }
    } catch (error) {
      halted = true;
      if (!failed) { failed = true; failure = error; }
    }
  }
  await Promise.all(Array.from({ length: Math.min(limits.gameConcurrency, limits.games) }, () => lane()));
  if (failed) throw failure;
  const records = completedGames.filter(Boolean).map(completed => completed.game);
  const samples = completedGames.filter(Boolean).flatMap(completed => completed.samples);
  const summary = { requestedGames: limits.games, games: records.length, valid: records.filter(game => game.valid).length,
    invalid: records.filter(game => !game.valid).length, finished: records.filter(game => game.valid && game.result !== 'UNFINISHED').length,
    unfinished: records.filter(game => game.result === 'UNFINISHED').length,
    whiteWins: records.filter(game => game.result === 'WHITE_WIN').length, blackWins: records.filter(game => game.result === 'BLACK_WIN').length,
    draws: records.filter(game => game.result === 'DRAW').length, plies: records.reduce((sum, game) => sum + game.plies, 0),
    samples: samples.length, outcomeSamples: samples.filter(row => row.targetType === 'outcome-blend').length,
    bootstrapSamples: samples.filter(row => row.targetType === 'search-bootstrap').length,
    coverage: records.reduce((total, game) => addCoverage(total, game.coverage), emptyCoverage()),
    discardedSamples: records.reduce((sum, game) => sum + game.discardedSamples, 0), cancelled, seed: limits.seed,
    unfinishedReasons: Object.fromEntries([...new Set(records.filter(game => game.result === 'UNFINISHED').map(game => game.reason))]
      .map(reason => [reason, records.filter(game => game.result === 'UNFINISHED' && game.reason === reason).length])),
  };
  return { games: records, samples, summary };
}
