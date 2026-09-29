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
- Aspiration retries now reuse the previous root bound. Cached move features
  have a consistent object layout and are reused during stable sorting;
  string keys are built only when a preferred, killer, or history lookup needs
  them. Learned priorities are refreshed on every ordering pass.
- Move generation indexes moves by source timeline, so consuming a board skips
  its remaining moves together. Exact position keys reuse unchanged timeline
  encodings. Ordinary spatial moves copy only the ranks they change; temporal
  moves, castling, and en passant still use the upstream move implementation.
- Evaluation uses shared movement tables, pawn counts and rank bounds per
  file, and precomputed corridor directions. Historical royal scans are reused,
  and ordinary king zones avoid coordinate strings and maps. Overlapping royal
  zones in variants retain their deduplication. These changes preserve scores,
  including custom heuristic contributions.

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
