# Search performance and GPU assessment

The search policy allows ordinary, same-board moves only on boards currently
required to advance the present. On inactive timelines and boards ahead of the
present, search considers only moves to another board (another timeline or
another time). This also excludes ordinary captures, promotions, en passant,
and castling from optional boards. The policy is recalculated after each move
of a partial turn, because branching can change the present and active set.
Preferred turns, fallback recommendations, and tactical continuations use the
same policy. Manual play and the rules validator retain all legal moves.

This is a selective search policy. In particular, advancing an optional board
can make a later arrival branch instead of merge. The rules tests preserve
that distinction even though search deliberately omits such spatial moves.
If the restricted action tree is empty, the engine checks for one unrestricted
legal turn solely to distinguish a policy limit from actual mate or stalemate.
It does not evaluate or recommend that excluded turn. A root policy limit
returns `status: incomplete`, `stoppedReason: policy`, and no recommendation;
an internal policy limit uses static evaluation. Results identify the policy
with `searchPolicy: present-spatial` and count internal limits in `policyLeaves`.
Multi-turn mating lines remain subject to the selected search policy; the
tactical suite checks its expected mates against unrestricted legal replies.

## CPU improvements

- Exact position keys reuse encodings of immutable historical boards within
  each search. The complete history, side, and promotion set remain in the
  key; this introduces no hash-collision risk. Public `positionKey` calls still
  observe edits to caller-owned boards, and each search starts a fresh cache.
  Complete serialized history bodies are also weakly cached by their outer
  board container, so submitted turns and repeated search windows can reuse
  them without keeping discarded positions alive. Side and promotion keys
  remain distinct.
- Temporal attacks use small integer lookup tables instead of allocating and
  serializing vectors for every piece/royal pair. Directional pawn and brawn
  royal threats are included. Geometry is checked against the pinned rules
  implementation, including missing boards, blockers, and even timelines.
- Tactical search reuses a searched capture as evidence that a legal turn
  exists, avoiding duplicate generation. Quiet-only positions and terminal
  positions still receive legality checks.
- Positive legal-turn proofs, check status, and static evaluations are reused
  across tactical horizons. Searched scores remain isolated by horizon. A
  legal move outside the search policy never supplies a permitted-turn proof.
  Exhausted tactical trees and empty full-turn trees retain exact results,
  including normalized mate distances, within the existing cache budget.
- The present is found by scanning backward from each active frontier and is
  computed once per partial action state. Move application copies only the
  timeline containers that change; untouched history stays shared. Preferred
  and killer move sets are also reused while constructing a complete turn.
- Mate-distance bounds avoid searching scores that cannot improve an already
  found mate. Iterative search uses a 60-centipawn aspiration window starting
  at depth two, with a full-window retry whenever the estimate falls outside.
- Aspiration retries reuse the previous root bound. Move ordering computes
  scalar priorities without retaining a feature object for every generated
  move. String keys are built only when a preferred, killer, or history lookup
  needs them. Learned priorities are refreshed on every ordering pass.
- Move generation indexes moves by source timeline, so consuming a board skips
  its remaining moves together. Exact position keys reuse unchanged timeline
  encodings. Ordinary spatial moves copy only the ranks they change; temporal
  moves, castling, and en passant still use the upstream move implementation.
- Evaluation uses shared movement tables, pawn counts and rank bounds per
  file, and precomputed corridor directions. Historical royal scans are reused,
  and ordinary king zones avoid coordinate strings and maps. Overlapping royal
  zones in variants retain their deduplication. These changes preserve scores,
  including custom heuristic contributions.
- Classical search shares immutable piece and royal scans across action
  generation and forced-pass check detection for the entire search session.
  Evaluation also reuses board-local king zones and pawn-defender counts.
  Timeline coordinates, historical weights and corridor blockers are still
  evaluated in each position's history. These weak caches do not keep discarded
  boards alive; public rule checks and evaluation still observe caller edits.

These changes target deeper completed searches within a fixed budget. They
do not establish an Elo gain. `node scripts/strength.js --nodes
1000,5000,20000,50000 --repeat 2 --json` reports deterministic work-budget
results; `--engine PATH` permits comparison with an earlier search module.

Compared with commit `34c712b`, the local checks produced these work counts
(search visits plus generation ticks). Both engines used quiescence depth two
and a 100,000-node ceiling. Each case was repeated three times; elapsed times
below are medians and vary by hardware and load.

| Position and requested depth | Previous nodes | Current nodes | Completed depth, previous → current | Median ms, previous → current |
| --- | ---: | ---: | --- | --- |
| Standard opening, depth 4 | 22,333 | 17,306 | 4 → 4 | 479 → 311 |
| Locked king, depth 3 | 8,713 | 3,654 | 3 → 3 | 85 → 33 |
| Temporal queen mate, depth 2 | 283 | 222 | 1 → 1, immediate mate | 10 → 4 |
| Defended pawn opening, depth 2 | 38,849 | 100,000, limit reached | 2 → 1 | 952 → 4,058 |

The last case is a regression from the requested optional-board policy:
omitting spatial continuations can remove cheap cutoffs and leave expensive
temporal branches. The change is not a universal speed or strength gain.
The curated 12-case tactical suite improves from 9 to 10 solved at 5,000 work
nodes, and retains 12/12 at 20,000 and 50,000. Two repeats at each budget
produced identical search results and no invalid principal variations or
false terminal certificates. The suite's defended-pawn case requires depth
one with recapture analysis; the deeper depth-two experiment above is a
separate diagnostic.

### Repeating classical search comparisons

```sh
npm run benchmark:classical
npm run benchmark:classical -- --baseline path/to/baseline/search.js --repeat 3 --json
```

The baseline must keep its matching rules, evaluation, heuristics, royal-safety,
and cache modules alongside `search.js`. `scripts/snapshot-engine.js` can save
these from a Git revision. The comparison warms both engines, alternates their
order, and measures fixed-depth and fixed-time runs on four positions. Fixed
depth timings are comparable only when both searches complete the requested
full-turn depth and tactical horizon. The report validates every returned
principal variation, input immutability, and node accounting, and records
source hashes with individual timings. Use `--mode depth` or `--mode time` to
run either measurement alone.

On 2026-09-29, Node 22.15.1, three warmed comparisons against the working tree
saved before the latest changes produced the following medians. Both engines
completed the same requested depth and tactical horizon with identical scores;
each run used a fresh search cache and their execution order alternated.

| Position | Depth / tactical horizon | Previous work nodes | Current work nodes | Previous ms | Current ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard | 4 / 2 | 18,073 | 17,947 | 331 | 317 |
| Opening | 2 / 2 | 4,598 | 4,361 | 72 | 62 |
| Two timelines | 2 / 1 | 26,184 | 23,579 | 534 | 444 |
| Temporal | 2 / 1 | 9,516 | 9,393 | 208 | 197 |

Across all twelve measured fixed-depth searches, total elapsed time fell by
11.9% and work nodes by 5.3%. Under a deterministic 25,000-node limit, the
two-timeline case completed depth two instead of depth one in all three
comparisons. At a one-second time limit, both engines still completed the same
depths on these four fixtures. These local results show reduced work and
latency; the standard-position timing ranges overlap, and the measured depth
gain is at the fixed work budget. All 530 tests
passed, and the 12-case tactical suite passed twice at 20,000 nodes per case.

The local reports `artifacts/classical-search-final.json` and
`artifacts/classical-search-work-final.json` retain timings, settings, source
hashes, and validation results. Their baseline is the initial working-tree
snapshot in `artifacts/classical-search-before/src`, including the local
evaluation changes already present when the comparison began.

### Allocation and root-cache comparison

Against commit `9de95b5`, five warmed, alternating comparisons on Node 22.15.1
produced these medians. Every run completed the requested full-turn depth and
tactical horizon with the same score as the baseline.

| Position | Depth / tactical horizon | Previous work nodes | Current work nodes | Previous ms | Current ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard | 4 / 2 | 17,947 | 17,676 | 323 | 279 |
| Opening | 2 / 2 | 4,361 | 4,361 | 58 | 51 |
| Two timelines | 2 / 1 | 23,579 | 23,579 | 478 | 391 |
| Temporal | 2 / 1 | 9,393 | 9,393 | 201 | 156 |

Total fixed-depth elapsed time fell by 16.4% over the twenty measured searches.
At 500 ms, completed depths were unchanged on all four fixtures, while median
work throughput increased. Under a fixed 17,700-work-node budget, the standard
position completed depth four instead of depth three in all three repeats,
with tactical horizon two. Most of the gain is lower cost per node; root bound
reuse also removes 271 work nodes from the standard depth-four search.

All 535 tests passed. The twelve-case tactical suite passed twice at 20,000
work nodes per case. Evaluation scores, component contributions, and inspected
features also matched the baseline on 300 seeded varied positions. Timings
depend on hardware and load; these fixtures do not establish playing strength.

The local reports are `artifacts/classical-depth-speed-final.json`,
`artifacts/classical-depth-speed-work.json`, and
`artifacts/classical-depth-speed-strength.json`. The timing and work reports
include source hashes, settings, and per-run results. Repeat the timing check
with:

```sh
node scripts/snapshot-engine.js 9de95b5 artifacts/classical-depth-speed-baseline
node scripts/benchmark-classical.js --baseline artifacts/classical-depth-speed-baseline/search.js --repeat 5 --warmup 2 --time-ms 500 --json
```

### Search session reuse and lazy ordering

Classical search now shares immutable royal-safety scans across complete-turn
generation and forced-pass checks. Its evaluator caches board-local king zones
and pawn-defender counts, then applies the current history, coordinates and
weights on each visit. Each search owns fresh weak caches, so discarded boards
can be collected and later searches observe caller edits. Public evaluation
and rule checks remain uncached.

Move ordering selects the highest-priority component in one pass and defers
sorting the rest until another move is requested. Stable ties and the original
floating-point arithmetic are preserved. Priorities are captured before yielding
so a deeper sibling's history updates cannot change a suspended ordering pass.

On 2026-09-29 with Node 22.15.1, five warmed, alternating comparisons against
`f9a0ba9` produced these median work counts at a one-second time limit:

| Position | Previous nodes | Current nodes | Throughput gain | Completed depth, both |
| --- | ---: | ---: | ---: | ---: |
| Standard | 53,263 | 60,568 | 13.7% | 4 |
| Opening | 27,526 | 30,422 | 10.5% | 2 |
| Two timelines | 53,217 | 60,635 | 13.9% | 2 |
| Temporal | 37,095 | 44,778 | 20.7% | 2 |

All fixed-depth runs retained the baseline scores and work counts. An additional
three-repeat standard depth-five comparison completed the same 161,075 work
nodes in median 3,353 → 2,943 ms, a 12.2% reduction. Timings were variable:
the shallower combined benchmark's total fixed-depth time fell only 1.7%, and
its standard depth-four median increased from 304 to 438 ms. At three seconds,
both engines still completed depth four; current median work was 3.5% lower.
These measurements support lower cost in several cases, without a consistent
extra completed ply or a universal timing improvement.

All 543 tests passed, including tied and fractional move priorities, sibling
history updates, immutable cache reuse, caller edits, and parallel search. The
twelve-case tactical suite passed twice at 20,000 work nodes with no invalid
lines. Baseline evaluation scores and complete feature inspections matched on
228 position/profile combinations.

Reports are saved locally in `artifacts/classical-session-cache-final.json`,
`artifacts/classical-session-cache-deeper.json`, and
`artifacts/classical-session-cache-strength.json`. Timing reports include
source hashes, individual runs, PV legality and input-immutability validation.
Reproduce with:

```sh
node scripts/snapshot-engine.js f9a0ba9 artifacts/session-cache-baseline
node scripts/benchmark-classical.js --baseline artifacts/session-cache-baseline/search.js --repeat 5 --warmup 2 --time-ms 1000 --depth-time-ms 20000 --json
node scripts/benchmark-classical.js --baseline artifacts/session-cache-baseline/search.js --case standard --depth 5 --repeat 3 --warmup 2 --time-ms 3000 --depth-time-ms 20000 --json
```

### Reusing move geometry across search passes

Classical search now reuses pseudo-legal move geometry when another search
window visits the same immutable history, side, and promotion list. Each
traversal still checks complete-turn legality, applies the optional-board
policy, and refreshes learned move-ordering bonuses. Weak references let
discarded histories leave the cache. A new search always creates fresh caches.

Forced-pass check detection shares read-only board snapshots and copies only
the timeline containers it extends. Quiet spatial history uses exact numeric
keys for supported board sizes, avoiding repeated strings. Cache writes avoid
temporary eviction arrays when space is available. Evaluation groups temporal
targets by color and half-turn parity, avoids projected attacker copies, and
stops ray tests once they cannot improve a contribution. Evaluation weights,
search horizons, and pruning policy are unchanged.

On 2026-09-29, Node 22.15.1, five warmed runs per engine compared these changes
with commit `178e474f2b4c55ce9c501b8b9be00b1e6f6a866b`. Execution order
alternated. All fixed-depth runs completed with identical scores and work
counts; the harness validated every PV and input immutability.

| Position | Depth / tactical horizon | Work nodes, both engines | Previous median ms | Current median ms |
| --- | --- | ---: | ---: | ---: |
| Standard | 4 / 2 | 17,676 | 408 | 284 |
| Opening | 2 / 2 | 4,361 | 64 | 63 |
| Two timelines | 2 / 1 | 23,579 | 471 | 369 |
| Temporal | 2 / 1 | 9,393 | 207 | 165 |
| Standard, deeper comparison | 5 / 2 | 161,075 | 3,328 | 2,998 |

Total elapsed time across the first four fixtures fell 21.0%. The separate
depth-five comparison fell 10.6% in total time. Its individual times ranged
from 3,123–3,658 ms before and 2,902–3,111 ms after. The opening fixture's
timing ranges overlap, so its small median difference is inconclusive.
At two seconds, median work increased by 3.6–22.5% across the four fixtures,
but both engines still completed the same depths. A separate five-run standard
comparison at the default three-second budget also completed depth four for
both engines, with overlapping throughput. Higher throughput helps
reach deeper iterations sooner; it does not guarantee another full depth at
every time limit.

Validation passed all 549 tests and all 12 tactical cases at both 20,000 and
50,000 work nodes, repeated twice. A separate comparison preserved the full
evaluation breakdown on 800 position/profile combinations. New regressions
cover repeated geometry traversal, frozen and sparse forced-pass histories,
16-by-16 ordering, temporal evaluation maxima, and bounded cache eviction.

Reproduce the timing comparison with a baseline snapshot below the repository:

```sh
node scripts/snapshot-engine.js 178e474f2b4c55ce9c501b8b9be00b1e6f6a866b artifacts/classical-before-reuse
node scripts/benchmark-classical.js --baseline artifacts/classical-before-reuse/search.js --time-ms 2000 --depth-time-ms 10000 --repeat 5 --warmup 2 --json
node scripts/benchmark-classical.js --baseline artifacts/classical-before-reuse/search.js --case standard --mode depth --depth 5 --depth-time-ms 15000 --repeat 5 --warmup 2 --json
node scripts/benchmark-classical.js --baseline artifacts/classical-before-reuse/search.js --case standard --mode time --time-ms 3000 --repeat 5 --warmup 2 --json
```

### Deduplicating appended history within a turn

Partial turns generated from one position share all of its existing history.
Duplicate detection now encodes only newly appended boards and their timeline
indices, reusing the search's board encodings. This avoids allocating and
hashing the common history for every component move. The keys are exact and
local to that traversal; transposition-table keys still include the complete
history, side, and promotions. Move order, legal actions, and work counts stay
the same.

On 2026-09-29 with Node 22.15.1, five warmed, alternating runs against the
working tree saved immediately before this change produced these medians:

| Position | Depth / tactical horizon | Work nodes, both engines | Previous ms | Current ms |
| --- | --- | ---: | ---: | ---: |
| Standard | 4 / 2 | 17,676 | 321 | 302 |
| Opening | 2 / 2 | 4,361 | 67 | 40 |
| Two timelines | 2 / 1 | 23,579 | 461 | 381 |
| Temporal | 2 / 1 | 9,393 | 179 | 138 |
| Standard, deeper comparison | 5 / 2 | 161,075 | 2,867 | 2,759 |

All completed scores matched. Total elapsed time across the first four
fixtures fell 15.7%. Standard depth five improved 3.8% at the median, with
overlapping timing ranges (2,549–3,064 ms before, 2,486–2,932 ms after).
Timing varies by position and system load.

At one second, median work throughput increased 10.4–19.8% across these four
positions; both engines completed the same depths. At three seconds, both
completed standard depth five in every run. A 400 ms comparison on two
timelines completed depth two in five of five optimized runs versus four of
five baseline runs. Faster traversal can finish another depth near a time
boundary, but the gain depends on the position and budget.

All 556 tests passed. The twelve tactical cases passed at both 20,000 and
50,000 work nodes, repeated twice, with no invalid principal variations.
New regressions compare full-history and appended-history traversal, including
commuting moves, temporal branch ordering, sparse history, preferred empty
turns, duplicate promotions, and unchanged work accounting.

The local baseline is `artifacts/classical-search-start/src`, including the
search and evaluation changes already present at the start of this task.
Reports in `artifacts/classical-search-optimized-depth.json` and
`artifacts/classical-search-optimized-depth5.json` record source hashes,
individual timings, PV legality, input immutability, and work accounting.
The corresponding `-time.json`, `-time3s.json`, `-time400ms.json`, and
`-strength.json` reports retain the timed and tactical comparisons.

```sh
node scripts/benchmark-classical.js --baseline artifacts/classical-search-start/src/search.js --mode depth --repeat 5 --warmup 2 --depth-time-ms 15000 --json
node scripts/benchmark-classical.js --baseline artifacts/classical-search-start/src/search.js --case standard --mode depth --depth 5 --repeat 5 --warmup 2 --depth-time-ms 15000 --json
node scripts/benchmark-classical.js --baseline artifacts/classical-search-start/src/search.js --mode time --time-ms 1000 --repeat 5 --warmup 2 --json
node scripts/strength.js --nodes 20000,50000 --repeat 2 --strict --json
```

## Checkmate detection

Royal safety now tests piece-to-royal geometry directly instead of generating
every opponent move. Each action traversal caches piece and royal locations
on immutable snapshots. Unsafe partial turns are rejected before serializing
their full history. If a spatial move is also unsafe when played alone from
the turn's original position, the generator remembers that outcome and skips
it in later combinations. This conservative form of constraint reuse was
inspired by [cwmtt's checkmate solver](https://github.com/penteract/cwmtt/blob/master/Game/Chess/TimeTravel/FastCheckmate.lhs).

Temporal arrivals and component-dependent attacks still receive full legality
checks. Optional moves remain available, and an interrupted proof remains
unknown. Both synchronous and asynchronous action generation share these
optimizations, so Classical and Transformer terminal probes benefit.

Run the dedicated benchmark, optionally against a saved earlier engine:

```sh
npm run benchmark:checkmate
node scripts/snapshot-engine.js 5bf686f artifacts/checkmate-baseline-5bf686f
node scripts/benchmark-checkmate.js --baseline artifacts/checkmate-baseline-5bf686f/rules.js --repeat 11
```

The benchmark validates legal witnesses with both implementations, checks
input immutability, and includes actual mate, stalemate, multiple-board
evasions, and an evasion that requires advancing an optional board first.
`--json` includes individual timings; `--max-work` and `--time-ms` bound proofs.
A run that reaches a cap reports `unknown` and exits with code 2.

On Node 22.15.1, a local comparison on 2026-09-29 against `5bf686f` measured
the following medians over 11 runs after warm-up, alternating implementations:

| Position | Previous work ticks | Current work ticks | Previous ms | Current ms |
| --- | ---: | ---: | ---: | ---: |
| Temporal mate | 61 | 61 | 2.812 | 0.525 |
| Deferred mate after Nd6 | 5,643 | 4,835 | 39.735 | 9.375 |
| Optional spatial evasion | 31 | 27 | 0.329 | 0.101 |

The deferred mate used 14% fewer work ticks and ran about 4.2 times faster.
These are small local fixtures; timing varies with position, hardware, JIT
warm-up, and load. Direct attack checks are compared with upstream move
enumeration across all piece types, both colors, sparse histories, blockers,
promotion sets, and mutable public positions. Exhaustive action comparisons
also cover temporary checks and attacks that arise only from combinations.

### Reducing ordering allocations and impossible attack checks

Move ordering now keeps a numeric priority array for each visit instead of a
weak-cache entry and feature object per generated move. The first move and
stable ordering of the remainder are unchanged, including fractional heuristic
weights and priorities captured before a sibling updates the history table.

Royal-safety checks return before scanning historical targets when no opponent
has a latest board to move from. Evaluation also skips ordinary pawn temporal
attacks when the required adjacent timeline is absent. Brawns retain their
additional attack directions. These changes preserve the search tree, scores,
and work accounting.

On 2026-09-29, five warmed, alternating comparisons against `7954421` on
Node 22.15.1 produced these median wall times. Each engine started with fresh
search caches. All fixed-depth runs completed with matching scores and work
counts, and every returned PV passed legality and input-immutability checks.

| Position | Depth / tactical horizon | Work nodes, both engines | Previous ms | Current ms |
| --- | --- | ---: | ---: | ---: |
| Standard | 4 / 2 | 17,676 | 275 | 237 |
| Opening | 2 / 2 | 4,361 | 46 | 49 |
| Two timelines | 2 / 1 | 23,579 | 429 | 386 |
| Temporal | 2 / 1 | 9,393 | 177 | 159 |
| Standard, longer search | 5 / 2 | 161,075 | 3,840 | 2,623 |

Standard depth five took 32% less median time. With an equal 3,300 ms budget,
the new engine completed depth five in all five runs; the baseline completed
depth five once and depth four four times. Both used tactical horizon two.
These are local measurements: the short opening fixture was slightly slower,
and hardware, garbage collection, and load affect the depth reached at a time
limit. The changes do not establish a playing-strength rating.

All 551 tests passed, including exact ordering traces, tactical search,
parallel search, sparse and even timelines, and upstream royal-safety checks.
The reports in `artifacts/classical-optimized-depth.json`,
`artifacts/classical-optimized-depth5.json`, and
`artifacts/classical-optimized-time.json` retain per-run results and source
hashes. Reproduce with:

```sh
node scripts/snapshot-engine.js 795442102120b86d60ede66f534c015a4fce4f4d artifacts/classical-allocation-baseline
node scripts/benchmark-classical.js --baseline artifacts/classical-allocation-baseline/search.js --mode depth --depth-time-ms 15000 --repeat 5 --json
node scripts/benchmark-classical.js --baseline artifacts/classical-allocation-baseline/search.js --case standard --mode depth --depth 5 --depth-time-ms 15000 --repeat 5 --json
node scripts/benchmark-classical.js --baseline artifacts/classical-allocation-baseline/search.js --case standard --mode time --time-ms 3300 --repeat 5 --json
node --test --test-concurrency=2
```

### Avoiding allocations for impossible moves

Classical search now walks the pinned library's nonpawn movement directions
using numeric coordinates. It allocates move arrays only for reachable,
unblocked destinations, preserving the library's direction and capture order.
Pawns, brawns and unmoved kings retain the library's special-move handling.
Ordinary pawns skip temporal geometry when their required adjacent timeline
is absent; brawns still check captures into their own past.

Five warmed, alternating comparisons on 2026-09-29 against `48d5875`, using
Node 22.15.1 and fresh search caches, produced these median wall times:

| Position | Depth / tactical horizon | Work nodes, both engines | Previous ms | Current ms |
| --- | --- | ---: | ---: | ---: |
| Standard | 4 / 2 | 17,676 | 204 | 204 |
| Opening | 2 / 2 | 4,361 | 34 | 31 |
| Two timelines | 2 / 1 | 23,579 | 291 | 265 |
| Temporal | 2 / 1 | 9,393 | 121 | 103 |
| Standard, longer search | 5 / 2 | 161,075 | 2,498 | 2,298 |

The longer standard search took 8% less time. All fixed-depth runs completed
with identical scores and work counts, legal principal variations, and
unchanged input positions. Short-run timing is noisy; the shallow standard
fixture showed essentially no change. These measurements do not establish
a playing-strength rating.

With a 2,400 ms budget, the optimized engine reached depth five in four of
five runs, compared with two of five baseline runs. Both used tactical
horizon two. The remaining runs completed depth four; time-limited depth
depends on garbage collection and machine load.

Differential tests compare complete ordered move arrays against the library
for every piece type and both colors, including fairy pieces, negative
unmoved flags, sparse and even timelines, blockers, royal exclusions,
castling, en passant, promotions and seeded mixed multiverses.

Reports in `artifacts/classical-speed-final-depth.json` and
`artifacts/classical-speed-final-depth5.json` include per-run timings and
source hashes; `artifacts/classical-speed-final-time.json` records the timed
comparison. Reproduce with:

```sh
node scripts/snapshot-engine.js 48d5875318c1d56389c8feb3d51732ed5fd4900c artifacts/classical-speed-baseline-20260929
node scripts/benchmark-classical.js --baseline artifacts/classical-speed-baseline-20260929/search.js --mode depth --depth-time-ms 15000 --repeat 5 --warmup 2 --json
node scripts/benchmark-classical.js --baseline artifacts/classical-speed-baseline-20260929/search.js --case standard --mode depth --depth 5 --depth-time-ms 15000 --repeat 5 --warmup 2 --json
node scripts/benchmark-classical.js --baseline artifacts/classical-speed-baseline-20260929/search.js --case standard --mode time --time-ms 2400 --repeat 5 --warmup 2 --json
node --test --test-concurrency=2
```

### Direct legal-turn existence probes

Classical search now uses a `firstOnly` action probe when it only needs to
establish that a legal turn exists: fallback selection, static quiescence
boundaries, and terminal verification. From a single timeline with a valid
mover-color frontier, each generated move consumes the only playable source.
Any new branch belongs to the opponent, so checking royal safety proves that
the move completes the turn. This avoids constructing the general partial-turn
traversal and serializing histories solely to deduplicate a result that will
never be requested again.

Preferred turns, tactical-only requests, sparse frontiers, and multiboard
positions retain the general traversal. Checked quiescence horizons still
search every required evasion. Both synchronous and asynchronous probes close
their suspended ordering iterators after the first result.

A local comparison on 2026-09-29 used Node 22.15.1, revision
`f07d8c14de088c5b6f5d52a85dc7643a541a4915` as the baseline, five alternating
pairs, and three warm-up rounds. Garbage from previous searches was collected
before each timed run using `--expose-gc`. Median fixed-depth wall times were:

| Position | Depth / quiescence | Baseline | Updated |
| --- | --- | ---: | ---: |
| Standard | 4 / 2 | 223 ms | 206 ms |
| Opening | 2 / 2 | 47 ms | 40 ms |
| Two timelines | 2 / 1 | 347 ms | 352 ms |
| Temporal | 2 / 1 | 124 ms | 117 ms |
| Standard | 5 / 2 | 2,490 ms | 2,175 ms |

The depth-five median fell by 12.7%. Completed scores and work counts matched
in every pair; all returned PVs were legal and inputs remained unchanged.
Timing varied across trials. An equal-time check at 2.3 seconds reached depth
five in three of five baseline runs and two of five updated runs, so this
measurement does **not** establish a consistent extra completed depth at that
budget. The two-timeline fixed-depth case was also slightly slower.

The regression suite passed all 581 tests, including first-result equivalence,
preferred replay, unsafe-move options, sparse histories, asynchronous cleanup,
checked horizons, policy boundaries, and exact search-order traces.

The raw reports are `artifacts/classical-direct-witness-depth.json`,
`artifacts/classical-direct-witness-depth5.json`, and
`artifacts/classical-direct-witness-time.json`. Reproduce them with:

```sh
node scripts/snapshot-engine.js f07d8c14de088c5b6f5d52a85dc7643a541a4915 artifacts/classical-opt-baseline-20260929
node --expose-gc scripts/benchmark-classical.js --baseline artifacts/classical-opt-baseline-20260929/search.js --mode depth --depth-time-ms 15000 --repeat 5 --warmup 3 --json
node --expose-gc scripts/benchmark-classical.js --baseline artifacts/classical-opt-baseline-20260929/search.js --case standard --mode depth --depth 5 --depth-time-ms 15000 --repeat 5 --warmup 3 --json
node --expose-gc scripts/benchmark-classical.js --baseline artifacts/classical-opt-baseline-20260929/search.js --case standard --mode time --time-ms 2300 --repeat 5 --warmup 3 --json
node --test --test-concurrency=2
```

## Parallel CPU search

Classical analysis supports a configurable root-search pool through the app's
**Search threads** control, CLI `--threads N`, and the asynchronous `analyze`
export in `src/parallel-search.js`. The synchronous `src/search.js` API remains
single-threaded. The default is one thread; the pool is opt-in.

Depth-one passes run on the calling search thread. At depth two and beyond,
the preferred root turn is searched first to establish an alpha-beta bound.
The other complete-turn branches are distributed lazily between that thread
and up to `threads - 1` workers, with at most one outstanding job per worker.
Workers retain private transposition tables and ordering history across jobs
and iterations within an analysis. Scout searches use their dispatch-time
bound; an improving probe is re-searched before it can become the best turn.

The time budget includes pool startup. All threads reserve work from one
atomic node budget, including action generation. Cache bytes and table-entry
limits are divided across the pool, not multiplied by its size. Total process
RAM remains higher because every worker has a JavaScript runtime and copied
position history. Cancellation and root cutoffs stop outstanding work; an
interrupted iteration keeps the last completed score, PV, and tactical horizon.
All workers close before the parallel API returns. `threadsUsed` reports
whether the pool activated; shallow or terminal searches can use only one.

Parallel work changes scheduling, cache reuse, and tied-move ordering. Fixed
work-budget results need not match a single-thread run. Completed searches
are checked for matching scores at the same depth and quiescence horizon;
the regression suite also covers mate distance, multiboard turns, policy
limits, live accounting, cancellation, and input immutability.

Use the benchmark to measure your machine and positions:

```sh
node scripts/benchmark-parallel.js --time-ms 3000 --repeat 3
node scripts/benchmark-parallel.js --mode depth --case standard --depth 5 --time-ms 10000 --repeat 3
```

It compares 1, 2, and 4 threads at equal requested depths and equal think
times, validates every returned PV, and includes startup and cleanup in wall
time. Fixed-depth speedups are reported only when both searches reach the
requested depth. A speedup in nodes per second alone does not establish
deeper search, because parallel branches can duplicate work or miss cutoffs.

A local comparison on 2026-09-24 (Node 22.15.1, Intel i5-12600KF) used the standard opening, depth five,
quiescence depth two, a 10-second ceiling, and three runs per configuration
with rotating run order. All nine runs completed depth five with score +65
centipawns and legal PVs. Median wall times, including worker startup and
cleanup, were:

| Search threads | Median wall time | Speedup over one thread |
| --- | ---: | ---: |
| 1 | 8,164 ms | 1.00x |
| 2 | 7,534 ms | 1.08x |
| 4 | 8,222 ms | 0.99x |

These are modest gains, not linear scaling. Parallel searches did more work
because branches used older bounds and separate caches. Shorter
locked-king and two-timeline checks also included slowdowns, so one thread
remains the default and two is a useful first comparison on longer searches.
An equal-time check with three 3-second runs per configuration completed
depth four at all thread counts, with the same score; higher throughput did
not reach an extra full depth within that budget.

## GPU feasibility

A local check found an NVIDIA GeForce RTX 3060 with 12 GiB of VRAM. A CPU
profile of the five benchmark positions, with 1.5 seconds per search, found:

| Work | Sampled CPU self time |
| --- | ---: |
| Rule generation and upstream move geometry | 59.2% |
| Position evaluation | 15.0% |
| Search orchestration | 13.9% |
| Garbage collection | 10.1% |

Full-history serialization alone accounted for 16.5% of the samples and
temporal-attack evaluation for 9.6%. These are observations from the previous
implementation on this machine, not portable performance guarantees.

Even a hypothetically free GPU evaluator could improve that workload by at
most about 1.18 times before accounting for transfers and synchronization.
The current alpha-beta search consumes one child score before deciding which
branch to visit next. Its small, hand-written evaluations provide little
parallel arithmetic per dispatch. NVIDIA's
[CUDA best-practices guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#what-runs-on-a-cuda-enabled-device)
recommends substantial parallel work and minimizing host/device transfers.

GPU support is feasible through a compute runtime such as
[Dawn's Node WebGPU bindings](https://github.com/dawn-gpu/node-webgpu), but is
not enabled in this implementation. A promising future use is training and
batched inference for a learned policy/value model, which requires training
data, a validated model, and search designed to batch evaluations. The current
engine has none of those prerequisites. It continues to use CPU and RAM;
raising the search-cache limit does not allocate GPU VRAM.
