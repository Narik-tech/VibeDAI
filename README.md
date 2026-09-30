# Vibe-D AI

A local analysis and play workbench, command-line engine, and JavaScript library for **5D Chess with Multiverse Time Travel**.

The classical engine searches **complete submitted turns**, including turns that require moves on several timelines. Ordinary moves are considered only on boards currently required to advance the present; optional boards contribute only cross-board moves. It uses iterative deepening, principal variation alpha-beta search, full-turn move ordering, transposition caching in both normal and tactical search, capture/promotion quiescence, and a multiverse evaluation. The shared rules layer supports historical travel, branching, inactive timelines, castling, en passant, promotions, and the variant pieces supported by the pinned rules dependency.

The UI offers **Classical search**, an experimental **Transformer**, and **Leela in a 5D Trenchcoat**, which uses the LCZero transfer model. The neural engines have separate checkpoints and training choices. These engines have no established Elo. Search budgets, depth, principal variation, and incomplete results are visible so behavior can be measured and improved.

The **Position heuristics** panel below the boards explains the displayed position's static evaluation, with signed centipawn contributions for individual features and timeline frontiers. Sliders and number inputs adjust component weights, positional terms, piece values, and search ordering. Changes refresh the breakdown and apply to subsequent classical analyses and automatic replies, including parallel searches. Settings persist in this browser; **Reset defaults** restores the shipped profile. Changing a setting invalidates the previous recommendation. The static total evaluates the displayed position (including a partial turn); the Engine score comes from searching continuations, so the two can differ. Neural engines use their own models and do not use these controls. Legal move rules and the present-spatial search policy remain fixed.

## Run

Install [Node.js 22 or newer](https://nodejs.org/) and run from this directory:

```sh
npm ci
npm start
```

Open **http://127.0.0.1:5173**. Everything runs on your computer. No API key, cloud service, or account is required. Classical search runs without a GPU or Python installation. Set `PORT` to use a different local port.

Choose the engine in the Engine panel. The selection and resource settings are saved in your browser and apply to analysis and automatic engine replies. Switching engines cancels the current analysis and clears its recommendation. **Max nodes** (1–1,000,000,000), think time, and depth bound search; reaching the node limit is shown below the results. Set **Think time** to **Infinite (0)** to remove the time limit for either engine; node and depth limits still apply, and **Stop analysis** stops the search early. **Search cache (RAM)** (Off or 16 MiB–4 GiB) applies to Classical search, where a full cache evicts old entries while search continues.

**Search threads** selects 1, 2, 4, 8, or 16 CPU threads for Classical search. One thread remains the default. Multiple threads share the time, node, and cache budgets and search different complete-turn branches after depth-one warmup. Worker startup, copied history, and duplicated work can outweigh the gain on small searches; compare 1, 2, and 4 on your positions with `npm run benchmark:parallel`. See [parallel search behavior and measurements](docs/performance.md#parallel-cpu-search).

Classical search does not allocate GPU VRAM. Profiling found rule generation and history handling to dominate its runtime; its small, sequential evaluations do not provide enough batched work to justify GPU transfers. See [the classical engine performance and GPU assessment](docs/performance.md). The cache control caps estimated RAM retained for saved search results, including full position-history keys and move arrays; total process memory is higher. The selected amount is a budget, not an up-front allocation. Results display estimated cache usage against that budget. Choose Off to disable the cache.

To prepare the transformer, run these commands from the project folder, then select **Transformer · experimental** and choose **Refresh** in the UI:

```sh
npm run transformer:setup
npm run transformer:data
npm run transformer:train
npm run transformer:doctor
```

The transformer uses bounded historical context and candidate turns to keep training and inference manageable on a local GPU, with a CPU option. Its dynamic search ranks candidates and True Evaluations at each depth, balances evaluation work by the leading True prefix, and expands eligible ranks as that shared prefix grows. Max depth supports up to 64 complete turns. **Dynamic (0)** starts at depth 1 and raises the ceiling by one when every searched depth's top 20 entries are True Evaluations (or all entries when fewer than 20 exist). Use `--engine transformer --depth 0` on the command line. Full history still determines move legality. Training on a small generated dataset creates a usable experimental checkpoint; it does not establish playing strength. See [transformer architecture, training, GPU settings, and limits](docs/transformer.md) for configuration and validation commands. The UI shows setup guidance until a local runtime and checkpoint are available.

To improve an existing transformer through self-play, run `node scripts/transformer-selfplay.js --iterations 0 --device cuda` and stop with Ctrl+C. The loop generates legal games, trains on a bounded replay buffer, and promotes a candidate only after passing paired matches against the incumbent. Use `--iterations 1` for one cycle. The shortcut `npm run transformer:selfplay:continuous` runs continuously and selects CUDA when available. Use direct `node` invocation when setting options: PowerShell's `npm.ps1` forwarding can strip flag names. See [self-play settings, promotion rules, logs and recovery](docs/transformer-selfplay.md).

You can also open **Training** in the app header, or visit **http://127.0.0.1:5173/training**. Adjust the self-play, search, training, and promotion settings, then start or stop a run from the browser. The page shows progress, retained cycle reports, and training logs. Select a self-play or arena game to review its multiverse turn by turn, including recorded search scores and results. Review is read-only and leaves the analysis board untouched. Training requires the local Python environment and an existing checkpoint; the setup commands above prepare both.

Select a piece on a playable board, then a highlighted destination on any board. A time-travel move can create a new timeline. Continue until the turn can be submitted, then choose **Submit turn**. **Analyze** recommends an entire remaining turn; **Play best** applies it and submits. The opponent selector enables automatic engine replies. Undo removes one pending move, or a whole submitted turn when no moves are pending.

Import/export uses **5DPGN** and **5DFEN**, not ordinary chess FEN. An import preserves the history needed for time-travel moves. Export records submitted turns; submit or undo pending moves before saving a complete game. Undo history starts at the imported position.

## Train from an LCZero baseline

The [LCZero-to-5D training guide](docs/lc0-transfer.md) describes the new transfer
pipeline for the supplied T3 512×15×16h model. It imports the spatial transformer,
trains timeline-aware value and legal-candidate policy adapters, and connects the
result to the existing Transformer engine and self-play promotion loop. Use
`node scripts/lc0.js --help` for inspection, curriculum generation, training,
evaluation, and runtime verification. The short validation run confirms the
pipeline works; it does not establish playing strength.

Choose **Leela in a 5D Trenchcoat** under **Analysis engine** to analyze with the
trained LCZero model. Set **Engine plays** to White or Black to use it as an
opponent. In **Training**, choose the same name to resume its self-play training.
Both use `artifacts/lc0/best.pt` by default; `LEELA_CHECKPOINT` selects another
trained transfer checkpoint. Leela's training history is stored separately in
`artifacts/lc0/selfplay`.

## Command line

```sh
node src/cli.js --time 30 --depth 10
node src/cli.js --time 30 --depth 10 --threads 4
node src/cli.js --file examples/opening.5dpgn --time 10 --json
node src/cli.js --variant two_timelines --time 10 --play
node src/cli.js --help
```

`--time` is seconds. `--nodes` bounds search and action-generation work together. `--depth` counts **complete player turns**, not individual piece moves. `--qdepth` controls tactical continuation at the horizon. Ctrl+C requests cancellation and returns the best available legal turn. A worker protects the command line and web server from long searches; a hard deadline backs up cooperative cancellation.

Scores are centipawns from **White's** perspective. A `null` score means no evaluation is available yet. For Classical search, `completed: true` means at least one full depth iteration finished under the search policy, not that the game has been solved. A later interrupted iteration does not replace the last completed iteration. For Transformer search, it means a True root evaluation is available or the root was proved terminal; interruption retains the latest backed-up values. Transformer `depth` reports the deepest True Evaluation, while `pvDepth` reports the selected line's length. `status: incomplete` never means checkmate. `mateIn` is measured in complete-turn plies and is signed by the winning color; classical multi-turn mating lines are subject to the optional-board search restriction. Classical search first completes depth-one passes with capture extensions increasing from zero to the configured limit, then deepens the complete-turn search. `effectiveQuiescenceDepth` records what the completed pass used.

## Engine API

```js
import { createPosition, validateAction } from './src/rules.js';
import { analyze } from './src/search.js';

const position = createPosition({ variant: 'standard' });
const result = analyze(position, {
  timeMs: 5000,
  maxDepth: 8,
  maxNodes: 2_000_000,
  quiescenceDepth: 2,
  cacheMemoryMb: 128, // MiB of estimated search-cache RAM; 0 disables the cache
  onProgress: info => console.log(info.depth, info.score),
});
if (result.bestAction !== null) {
  const next = validateAction(position, result.bestAction);
}
```

`analyze` is synchronous; call it in a worker if your application has an event loop to keep responsive. `src/worker.js` shows cancellation with shared memory. An action is an ordered array of raw piece moves. Coordinates are `[timelineIndex, halfTurnIndex, rankIndex, fileIndex]`. Read [the rules and compatibility notes](docs/rules.md) before using raw coordinates.

For parallel Classical search, import `analyze` from `src/parallel-search.js` and `await analyze(position, { threads: 4, timeMs: 5000, maxDepth: 8 })`. The existing synchronous `src/search.js` API remains single-threaded. The parallel API accepts 1–16 threads, reports the requested count in `limits.threads` and the count actually activated in `threadsUsed`, and closes its workers before resolving. Search still performs synchronous work on its calling thread; the app and CLI run it in their existing background worker. Parallel scheduling can change tied moves and results at interrupted work budgets; one thread retains deterministic work-budget behavior.

## Verification

```sh
npm test
npm run benchmark
npm run benchmark:classical
npm run benchmark:checkmate
npm run benchmark:parallel
npm run strength -- --nodes 50000 --repeat 2 --strict
npm run match -- --engine-a src/search.js --engine-b path/to/baseline/search.js --nodes 10000 --plies 40 --output artifacts/match.json
npm run selfplay
```

Tests cover temporal geometry, history immutability, present shifting, complete-turn legality, optional inactive boards, tactical search, interruption, game export/import, and HTTP integration. The benchmark reports local throughput and verifies returned actions. Self-play is a diagnostic and stops at a turn limit; it does not assign an Elo or count an unfinished game as a draw.

`npm run benchmark:classical -- --baseline path/to/baseline/search.js --repeat 3` compares warmed classical searches at fixed depth and fixed time, alternating engine order. It checks full PV legality, input immutability, and work accounting. `--json` records source hashes and individual measurements; `--mode depth` or `--mode time` runs just one comparison.

The tactical suite in `examples/tactics/` measures captures for both colors, coordinated multi-board captures and evasions, temporal mates, knight underpromotion, defended captures, and terminal positions. `npm run strength -- --nodes 1000,5000,20000,50000 --repeat 2 --json` reports results at fixed work budgets. Each run validates the full principal variation and checks that the input history is unchanged; repeats check deterministic search results. `--strict` exits with failure if a selected case is unsolved or invalid. Use `--case ID` to inspect one case and `--engine PATH` to compare an earlier compatible search module. These are curated regressions, not an independent rating.

Paired matches use the independent miniature and opening positions in `examples/matches/suite.json`, giving each engine both colors under equal search/generation work limits. `--suite FILE`, `--case ID,ID`, `--depth N`, and `--qdepth N` control the comparison; alternate modules must export `analyze`. `--seed 1,2 --opening-plies 2` adds reproducible legal opening variations. Seeds without opening plies repeat the same starts and do not add independent evidence. Reports preserve full move traces, validate actions and principal variations, check input immutability, and expose deterministic self-match discrepancies when both engine paths are identical.

Only exhaustive full-rule verification awards a checkmate win or stalemate draw. Verification can become expensive on large multiverses; `--terminal-work N` and `--time-ms N` bound it, and reaching either cap leaves the game `UNFINISHED`. Turn caps, missing moves, time stops and search-policy boundaries also stay unfinished; a legal fallback from an incomplete node-limited search can continue. There is no repetition or evaluation-based draw adjudication. Reports include scores among finished games and separately among complete color-swapped pairs: prefer the paired figure, because differing unfinished rates can bias the former. The suite is a development diagnostic, not an Elo rating or broad strength guarantee.

[The heuristic experiment report](docs/heuristic-tuning.md) records 54 alternatives, paired matches, rejected changes, and reproduction commands. `scripts/snapshot-engine.js` freezes a Git revision for comparison, and `scripts/tune-weights.js` screens isolated evaluation profiles before matches. No candidate established a reliable improvement in this experiment, so the production heuristics were retained.

Search retains the exact order of a preferred full turn, including optional moves after a legal submission. Quiet-move history transfers across half-turns, so useful ordering survives as the search deepens. Tactical cache entries are isolated by horizon and share the configured table-size limit; `qTtHits` reports their reuse. The checked-horizon boundary also verifies whether an evasion itself ends the game before assigning a static score. Immutable board encodings are reused within each search while keeping exact, complete-history position keys. Temporal evaluation uses allocation-free integer geometry and includes pawn and brawn royal threats.

Classical search also reuses piece scans, king zones, and pawn-defender counts from unchanged historical boards throughout each search. Move ordering selects the best component first and sorts the remaining moves only if search needs them. These optimizations preserve the searched tree and scores while reducing the time needed to reach each depth; see [local comparison results](docs/performance.md#search-session-reuse-and-lazy-ordering).

Tactical passes also reuse proven legal-turn existence, check status, and static evaluations across horizons while keeping searched scores separate. Empty full-turn trees are cached with mate-distance normalization. Move generation scans backward from timeline frontiers, calculates the present once per partial state, and copies only timeline containers changed by a move.

Repeated search passes also reuse move geometry from immutable histories. Move ordering avoids per-move feature objects, and royal safety and pawn evaluation skip impossible attack checks. In a local five-run comparison, standard depth five fell from a median 3.84 to 2.62 seconds with unchanged scores and work counts. At 3.3 seconds, all five optimized runs reached depth five, compared with one baseline run. See [the measurements and reproduction commands](docs/performance.md#reducing-ordering-allocations-and-impossible-attack-checks).

Partial-turn duplicate detection now encodes only newly appended boards, while search-table keys retain the complete history. A further five-run comparison reduced total fixed-depth time by 15.7% across four fixtures with identical scores and work counts. See [the measurements and reproduction commands](docs/performance.md#deduplicating-appended-history-within-a-turn).

Classical search also uses direct legal-turn existence probes on single timelines, avoiding partial-turn traversal and history serialization at static leaves. A local comparison reduced the standard depth-five median from 2.49 to 2.18 seconds, with identical scores and work counts. Equal-time trials did not show a consistent extra completed depth; see [the measurements and limits](docs/performance.md#direct-legal-turn-existence-probes).

Classical search now uses compact exact history keys, direct comparisons for short preferred/killer move lists, and prefiltered single-timeline movement directions. A local two-second comparison reached depth five in four of five updated runs versus one of five baseline runs, while estimated depth-five cache memory fell by 44.8%. Fixed-depth timings varied, including a slight temporal-fixture slowdown; see [the full comparison](docs/performance.md#compact-history-keys-and-cheaper-move-ordering).

The locked-king puzzle in `examples/locked-king.5dpgn` is a performance regression: depth three with two capture-extension plies must complete within 20,000 search/generation work nodes. The original search stalled at depth one because it explored already-lost partial turns and lengthy sequences of checks at the tactical horizon. Search now rejects irreversible royal attacks early, reuses unchanged move geometry within a turn, and directly generates tactical actions during quiescence. The app reports live work counts and the depth currently being searched separately from completed depth.

For longer diagnostics, set `BENCH_TIME_MS`, `SELFPLAY_TIME_MS`, or `SELFPLAY_PLIES` in your shell. Increase think time before increasing depth: requested depth is only a ceiling, and the full-turn branching factor can grow rapidly.

## Design and limits

- **Immutable history:** search shares unchanged past boards, but position keys include all history. Identical current boards with different pasts are different positions.
- **Legal complete turns:** move generation is lazy and deduplicates equivalent partial states. A king may be exposed during a partial turn; every king must be safe on submission. Search allows cross-board moves on optional boards, including after the present has shifted, but excludes ordinary same-board moves there. The required boards are recomputed after each component move. Manual play retains every legal move.
- **No false mate from a timer:** mate/stalemate is classified only after exhaustive legal-action enumeration. The upstream eager action enumerator and timeout-based mate getters are bypassed.
- **Evaluation:** weighted frontier material, development, mobility, pawn structure, king exposure, temporal pressure, and weaknesses across timelines. Temporal pressure follows the pinned movement vectors, including directional pawn/brawn captures and royal and fairy pieces, respects historical blockers and missing boards, and connects only matching half-turn colors. Historical material is not repeatedly counted. Direct royal pressure samples six past snapshots; king-zone protection also retains the first snapshot of each half-turn color. Weights are hand tuned and not statistically calibrated.
- **Search:** no beam cap or chess null-move assumption in normal-depth search. Ordinary moves on optional boards are deliberately excluded, including captures, promotions, and castling. Quiescence uses the same policy and searches captures/promotions up to its configured limit. If checked at that limit, it evaluates permitted legal evasions for one further turn, including terminal detection after the evasion. An exhausted restricted tree is checked against full rules before classifying mate/stalemate; a policy-limited root returns no recommendation and `stoppedReason: policy`. This selective search and finite horizon can miss tactics. Time budgets can expire before depth one on a large multiverse.
- **Transformer:** an experimental engine incrementally constructs complete turns, ranks their submitted positions, and balances selective deepening with progressive widening. Optional exploration uses temporal moves only. Bounded checking extensions, exact-history value reuse, and an optional trained component policy improve search allocation. Context and candidate limits can still miss tactics; playing strength is unmeasured. See [the transformer guide](docs/transformer.md).
- **Compatibility:** the pinned community rules implementation is not an official Thunkspace engine. Regression coverage is substantial but cannot certify every Steam variant. There is no live Steam integration, opening book, tablebase, repetition adjudication, or distributed search.

For further strength measurement, expand the tactical suite and run paired engine matches before tuning evaluation weights. More search time is useful, but no finite setting guarantees optimal play.

King safety includes potential time-travel corridors toward kings and adjacent friendly pawns. Pawns with few nonroyal defenders receive more protection priority, which makes the early f2/f7 pawns especially important. The heuristic follows same-timeline time/rank and time/rank/file rays through recorded history, then projects the current arrangement beyond the frontier for up to six steps. It considers compatible opposing sliders, fades with material phase, and uses the worst sampled exposure instead of summing historical copies. This rewards Nf3 closing the f2–f3 route, d4 closing the f2–e3–d4 route, and c3 closing the e1–d2–c3 route after d4 vacates d2. The geometry applies to both colors and custom king placements; it is a development heuristic, not an opening book or a proof of future safety. Tactical search and direct royal pressure still handle concrete attacks and other travel directions.

Timeline evaluation values unused capacity to create active branches. If White has used W timeline slots and Black B, their remaining active-branch capacities are `max(0, B + 1 - W)` and `max(0, W + 1 - B)`. Slots follow each side's outermost timeline index, including inactive branches, to match the rules for sparse and even layouts. The first unreciprocated branch costs 180 centipawns of reserve advantage; further overextension adds a penalty. Travel to an existing frontier spends no branch reserve. These bounded evaluation costs require compensation from the resulting position while allowing forced defenses, material wins, and mating travel.

The `travel` evaluation component rewards an unobstructed route from a frontier piece to a historical pawn beside an enemy royal when the pawn has no nonroyal spatial defender and the attacker can still create an active branch. The strongest such opportunity receives up to 140 centipawns per color, with reduced weight for inactive timelines and low-value attackers. A setup that depends on the opponent's reply leaving the arrangement intact receives half credit. Historical blockers, missing boards, and half-turn parity still apply; copied targets and multiple attackers do not multiply the bonus. This is potential attacking value, with complete-turn legality and the strength of an actual capture determined by search.

## Sources and license

Movement and notation are based on [5D Chess JS](https://gitlab.com/5d-chess/5d-chess-js), pinned to npm version **1.2.1**, by Shaun Wu and its contributors. Additional primary implementation and notation references are recorded in [docs/rules.md](docs/rules.md). The included temporal-attack example follows the opening illustrated by [ftxi's 5D Chess engine](https://ftxi.github.io/5dchess_engine/).

This project is **AGPL-3.0-or-later**, consistent with the rules dependency. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). It is an independent project and is not affiliated with Thunkspace.
