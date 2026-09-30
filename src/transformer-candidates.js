import { applyMove, canSubmit, generateActionsAsync, isTacticalMove, raw } from './rules.js';

const sign = position => position.action % 2 === 0 ? 1 : -1;
const temporal = ([from, to]) => from[0] !== to[0] || from[1] !== to[1];
const prefixKey = moves => JSON.stringify(moves);

// Without a trained policy, visit captures and temporal ideas first while
// spreading a small inference batch over different source pieces.
function diverseOrder(current, entries) {
  const groups = new Map();
  for (const entry of entries) {
    const priority = Number(isTacticalMove(current, entry.move)) * 2 + Number(temporal(entry.move));
    const key = `${priority}:${entry.move[0].slice(0, 4).join(':')}`;
    if (!groups.has(key)) groups.set(key, { priority, entries: [] });
    groups.get(key).entries.push(entry);
  }
  const sources = [...groups.values()].sort((a, b) => b.priority - a.priority);
  const ordered = [];
  for (let offset = 0; ordered.length < entries.length; offset++) {
    for (const source of sources) if (source.entries[offset]) ordered.push(source.entries[offset]);
  }
  return ordered;
}

// Keep the policy leaders while sampling the rest of its ranking at regular
// intervals. The tail slot includes the final rank, so a small batch does not
// repeatedly exclude the same source pieces. Every entry still appears once.
function spreadBatches(entries, size) {
  if (size < 4 || entries.length <= size) return entries;
  const remaining = entries.slice(), ordered = [];
  const exploration = Math.max(1, Math.floor(size / 4));
  while (remaining.length > size) {
    ordered.push(...remaining.splice(0, size - exploration));
    const indices = Array.from({ length: exploration }, (_, i) =>
      Math.floor((i + 1) * remaining.length / exploration) - 1);
    const sampled = indices.map(index => remaining[index]);
    for (const index of indices.toReversed()) remaining.splice(index, 1);
    ordered.push(...sampled);
  }
  return [...ordered, ...remaining];
}

/**
 * A resumable stream of distinct legal complete turns. Ordinary optional-board
 * moves are omitted, but temporal optional moves receive one of every four
 * admission slots while both families are available. Complete turns are ranked
 * by their submitted position; component values only guide their construction.
 *
 * infer fills the caller's value cache, and valueFor reads White-relative
 * values. scoreMoves optionally supplies trained policy scores for components.
 * selective reports any omitted optional spatial branch, never terminal proof.
 */
export function createCandidateStream(position, {
  ply = 0, tick = () => {}, keyPosition, infer, valueFor,
  probeTerminal = () => null, retainRootCandidate = () => {},
  componentBatchSize = 16, scoreMoves, candidateBranches = 4,
}) {
  const batchSize = Math.max(1, Math.floor(componentBatchSize));
  const partials = new Map(), generated = new Set();
  let selective = false, hasOptionalTemporal = false;
  const options = { tick, keyPosition, skipOptionalSpatial: true,
    onSkipOptionalSpatial: () => { selective = true; } };

  async function* moveBatches(current, moves, prefix, requiredOnly, usedOptional) {
    const present = new Set(raw.boardFuncs.present(current.board, current.action));
    let eligible = moves.map((move, index) => ({ move, index, required: present.has(move[0][0]) }));
    hasOptionalTemporal ||= eligible.some(entry => !entry.required && temporal(entry.move));
    if (requiredOnly) eligible = eligible.filter(entry => entry.required);
    if (!eligible.length) return;
    // A policy encodes the current state once, then scores coordinates. Value
    // inference remains bounded to the components that will actually be tried.
    const policy = scoreMoves ? await scoreMoves(current, eligible.map(entry => entry.move)) : null;
    if (policy != null) {
      if (policy.length !== eligible.length || [...policy].some(value => !Number.isFinite(value))) {
        throw new Error('Component policy must return one finite score per legal move.');
      }
      eligible.forEach((entry, index) => { entry.policy = policy[index]; });
      eligible.sort((a, b) => b.policy - a.policy || a.index - b.index);
    } else eligible = diverseOrder(current, eligible);
    eligible = spreadBatches(eligible, batchSize);
    if (!requiredOnly) {
      // Starting with the optional jump prevents a required-only DFS subtree
      // from using every reserved temporal slot before a jump is considered.
      eligible.sort((a, b) => Number(a.required) - Number(b.required));
    }
    for (let offset = 0; offset < eligible.length; offset += batchSize) {
      const batch = eligible.slice(offset, offset + batchSize), pending = [], rootTurns = [];
      for (const item of batch) {
        tick();
        const partial = applyMove(current, item.move), complete = canSubmit(partial);
        const successor = complete ? { ...partial, action: partial.action + 1 } : partial;
        const key = keyPosition(successor);
        let entry = partials.get(key);
        if (!entry) {
          const terminal = complete ? probeTerminal(successor, ply + 1, 64) : null;
          entry = { position: successor, terminal, complete, value: terminal
            ? terminal.score * sign(successor) : valueFor(successor) };
          partials.set(key, entry);
          if (!terminal && !Number.isFinite(entry.value)) pending.push(successor);
        }
        item.entry = entry;
        const turn = [...prefix, item.move];
        usedOptional.set(prefixKey(turn), usedOptional.get(prefixKey(prefix)) || !item.required);
        if (ply === 0 && entry.complete) {
          rootTurns.push({ turn, entry });
          retainRootCandidate(turn, entry.position, entry.terminal);
        }
      }
      if (pending.length) await infer(pending);
      for (const item of batch) {
        item.entry.value ??= valueFor(item.entry.position);
        if (!Number.isFinite(item.entry.value)) throw new Error('Transformer did not return a finite component value.');
      }
      for (const { turn, entry } of rootTurns) retainRootCandidate(turn, entry.position, entry.terminal);
      batch.sort((a, b) => (!requiredOnly && !usedOptional.get(prefixKey(prefix))
        ? Number(a.required) - Number(b.required) : 0)
        || (b.entry.value - a.entry.value) * sign(current)
        || (b.policy ?? 0) - (a.policy ?? 0) || a.index - b.index);
      yield batch.map(item => item.move);
    }
  }

  async function scoreCandidate(candidate) {
    const key = keyPosition(candidate.position);
    let entry = partials.get(key);
    if (!entry) {
      const terminal = probeTerminal(candidate.position, ply + 1, 64);
      entry = { position: candidate.position, complete: true, terminal,
        value: terminal ? terminal.score * sign(candidate.position) : valueFor(candidate.position) };
      partials.set(key, entry);
      if (!terminal && !Number.isFinite(entry.value)) {
        await infer([candidate.position]);
        entry.value = valueFor(candidate.position);
      }
    }
    if (!Number.isFinite(entry.value)) throw new Error('Transformer did not return a finite candidate value.');
    return { ...candidate, position: entry.position, candidateScore: entry.value };
  }

  async function* family(requiredOnly) {
    // Multiple required boards create large subtrees below each first move.
    // Partition the root's ordered components into a few independent lanes;
    // one complete turn per lane keeps an early subtree from filling the cap.
    // The lanes share lazily scored root batches, so widening does not multiply
    // root inference. All deeper prefixes still use the common rules traversal.
    const count = raw.boardFuncs.present(position.board, position.action).length > 1
      ? Math.max(1, Math.min(8, Math.floor(candidateBranches))) : 1;
    const shared = [], rootOptional = new Map([[prefixKey([]), false]]);
    let rootBatches, rootDone = false;
    async function batchAt(index, current, moves) {
      rootBatches ??= moveBatches(current, moves, [], requiredOnly, rootOptional);
      while (shared.length <= index && !rootDone) {
        const next = await rootBatches.next();
        if (next.done) rootDone = true;
        else shared.push(next.value);
      }
      return shared[index];
    }
    const lanes = Array.from({ length: count }, (_, lane) => {
      const usedOptional = new Map([[prefixKey([]), false]]);
      const legal = generateActionsAsync(position, { ...options,
        orderMoves: async function* (current, moves, prefix) {
          if (prefix.length) { yield* moveBatches(current, moves, prefix, requiredOnly, usedOptional); return; }
          let offset = 0;
          for (let index = 0; ; index++) {
            const batch = await batchAt(index, current, moves);
            if (!batch) return;
            const selected = batch.filter((_move, i) => (offset + i) % count === lane);
            offset += batch.length;
            for (const move of selected) usedOptional.set(prefixKey([move]), rootOptional.get(prefixKey([move])));
            if (selected.length) yield selected;
          }
        },
      });
      return { legal, usedOptional, done: false };
    });
    try {
      while (lanes.some(lane => !lane.done)) for (const lane of lanes) {
        if (lane.done) continue;
        const next = await lane.legal.next();
        if (next.done) { lane.done = true; continue; }
        const candidate = next.value;
        if (!requiredOnly && !lane.usedOptional.get(prefixKey(candidate.moves))) continue;
        const key = keyPosition(candidate.position);
        if (!generated.has(key)) {
          generated.add(key);
          yield await scoreCandidate(candidate);
        }
      }
    } finally {
      for (const lane of lanes) await lane.legal.return?.();
      await rootBatches?.return?.();
    }
  }

  async function* stream() {
    const required = family(true), optional = family(false);
    let requiredDone = false, optionalDone = false, admitted = 0;
    try {
      while (!requiredDone || !optionalDone) {
        const chooseOptional = !optionalDone && (requiredDone || (hasOptionalTemporal && admitted % 4 === 3));
        const next = await (chooseOptional ? optional : required).next();
        if (next.done) {
          if (chooseOptional) optionalDone = true;
          else {
            requiredDone = true;
            // Required traversal inspected every reachable prefix. If it saw
            // no optional jump, a second traversal cannot introduce one.
            if (!hasOptionalTemporal) optionalDone = true;
          }
          continue;
        }
        admitted++;
        yield next.value;
      }
    } finally {
      await required.return?.();
      await optional.return?.();
    }
  }
  const iterator = stream();
  Object.defineProperty(iterator, 'selective', { get: () => selective });
  return iterator;
}
