# Transformer engine

The UI's **Transformer · 800k** engine uses a separately trained neural
value evaluator and bounded search over complete legal turns. The existing
classical engine remains available. Move generation and legality always use the
complete 5D position, including its history; the neural model receives a bounded
representation of that position.

This is an experimental trainable engine. Bootstrap checkpoints imitate
approximate scores from the classical engine; a small dataset does not establish
playing strength. There is no random-weight or classical-evaluation fallback
when a transformer checkpoint is missing or incompatible.

## Setup and use

From the project directory with Python 3.10+ installed:

```powershell
npm run transformer:setup
npm run transformer:data
node scripts/transformer.js train --width 128 --heads 4 --layers 4 --feedforward 384
npm run transformer:doctor
npm start
```

Skip data generation and training if `artifacts/transformer/model.pt` already
exists. Then choose **Transformer · 800k** in the analysis controls. Setup creates the
project-local `.venv-transformer` environment and installs the tested PyTorch
2.14.0 CUDA 12.6 wheel from the official PyTorch package index. The NVIDIA driver
must support that CUDA runtime; a separate CUDA toolkit is unnecessary for the
wheel. `npm run transformer:setup -- --cpu` installs the CPU wheel instead.

If Python is not on PATH, pass its executable to setup:

```powershell
npm run transformer:setup -- --python "C:\path\to\python.exe"
```

Use `TRANSFORMER_PYTHON` to select a different Python environment,
`TRANSFORMER_CHECKPOINT` to select a different checkpoint, and
`TRANSFORMER_DEVICE=auto|cuda|cpu` for analysis device selection. `auto` uses CUDA
when available and otherwise runs the same model on CPU. An explicit `cuda`
request fails if CUDA is unavailable. Training and doctor also accept `--device`.

The default checkpoint is `artifacts/transformer/model.pt`. Artifacts and the
virtual environment are local generated files, excluded from version control.
The service loads a state dictionary with `torch.load(weights_only=True)`, checks
architecture/encoding versions and finite weights, and rejects checkpoints with
zero completed training steps.

The command-line interface selects the same engine:

```powershell
npm run analyze -- --engine transformer --time 3 --depth 3
```

Select **Dynamic (0)** in the analysis depth control, or pass `--depth 0`, to
let the search increase its depth ceiling as the leading evaluations become
ready. Fixed depths from 1 through 64 remain available.

## Search architecture

The transformer uses asynchronous selective search over complete legal turns.
Move generation retains the full multiverse history. The model orders and scores
candidates; it never authorizes a move or certifies mate.

### Incremental generation and widening

Each position initially admits up to **8 complete turns**, then resumes its
candidate generator in growing batches, normally **16 → 32 → 64**. The default
root and reply caps are both 64 (configurable up to 256). Widening competes with
deepening, so an expansion no longer has to construct its entire capped list
before a promising continuation receives search. A frontier with no remaining
work widens immediately; otherwise width grows approximately with the square
root of the work spent in that branch.

At each component prefix, successor values are evaluated in **batches of 16** by
default, bounded by the service maximum of 128. Later batches are requested only
when the generator reaches them. A trained component policy, when available,
orders the supplied components from one encoding of that prefix. Without one,
a deterministic order spreads the batch across source pieces and favors tactical
and temporal components. Values then order the components within each batch.
This is selective ordering: a small cap can omit a strong move in a later batch.
For batches of at least four components, approximately one quarter of slots
sample later ranks, including the tail, without increasing the batch size.
On turns with multiple required boards, up to four first-component traversals
alternate their complete-turn admissions. They share root policy/value batches
and deduplicate complete histories. `candidateBranches: 1` disables this
interleaving; required/optional admission quotas remain the same.

Required-board-only turns receive three of every four admission slots while both
families are available. The fourth slot explores a turn using an **optional
temporal move**. Optional moves whose source and destination have the same
timeline and half-turn are excluded from candidate construction and legal
fallback recommendations. Required/optional status is recomputed after each
component because time travel can change the present. Optional temporal moves
can precede required moves when that order changes temporal branching.

Move orders are merged only when their complete histories agree. Boards are
not assumed independent. All streams, including suspended component batches,
close on completion, interruption, cancellation, or model failure.

### Values, coverage, and scheduling

A **Candidate Evaluation** is the White-relative value of the final submitted
position. Intermediate component values guide turn construction, but are not
averaged into the completed candidate's score. This avoids assigning different
candidate values to commuting sequences with different intermediate positions.

A **True Evaluation** is a scheduler-selected submitted-position value with a
full legal terminal check. The term does not mean exact minimax truth. Once
selected continuations exist, the parent backs up the best evaluated continuation
for its mover. Unsearched alternatives do not become proof of a forced loss.
The static submitted-state score and backed search estimate are kept separately.

The scheduler first evaluates up to **three leading root contenders**, then the
two strongest admitted opponent replies to each expandable contender. It
subsequently balances the leading continuation, progressive widening, competitive
sideline catch-up, and exploration. Every eighth scheduled operation reserves an
exploration opportunity for a less-visited branch. Catch-up has a separate
periodic allocation and considers score separation, evaluation changes, forcing
positions, and continuation length; inferior-looking branches do not receive
unlimited work merely because their lines are short.

A demoted line already explored more deeply than a stronger alternative does not
claim catch-up priority. This suppression follows its descendants. Displayed
rankings retain parent-first order: replies to a stronger parent appear before
replies to a weaker one. The displayed consecutive True prefix remains a coverage
statistic, but no longer gates every depth's scheduling.

**Dynamic depth (maxDepth: 0)** begins at one complete turn. Its ceiling increases
one turn at a time when the current leading contenders have selected values,
initial reply coverage where applicable, and live continuations reaching the
ceiling. Resolved terminal lines need no further depth. There is no global
20-entry quota. The ceiling never decreases and is capped at 64.

### Tactical extensions and proofs

Checks and temporal royal threats detected by the full-rules inCheck function
can extend a line up to **two complete turns** beyond its ordinary ceiling by
default. Such expansion admits at most **four reply candidates** per position.
Captures and timeline creation alone do not trigger an extension. Extension
plies and width, as well as ordinary time/work limits, bound this extra search.
When a dynamic ceiling rises, formerly extended nodes can resume generation up
to the normal reply cap.

Preliminary terminal probes during construction are limited to 64 generation
work nodes. An unfinished probe leaves status unknown and permits a neural
ordering value. Selected evaluations resume their legal traversal in slices of
256 work nodes by default (`terminalProbeWork`). Deferred proofs alternate with
other search work and do not become True evaluations until resolved. At most
128 unresolved traversals are retained. Up to 32 selected proofs are pinned;
eviction closes the oldest unselected construction traversal.
Single-timeline checks use the direct legal-existence path. Terminal probes remain **unrestricted**,
including optional same-board moves: excluding such moves from search cannot
manufacture mate or stalemate.

A certified winning witness can prove a win. Proving a forced loss requires all
legal replies to have been exhausted and proved losing. Candidate caps, omitted
optional spatial branches, and interrupted generation cannot establish that
exhaustion. If legal turns exist but all are outside the optional-move policy,
the result reports stoppedReason: policy with no playable recommendation.

### Reuse and reporting

A per-analysis LRU cache reuses neural values across separately constructed
identical histories. Its default bounds are **4,096 entries / 32 MiB** of estimated
key/value storage, plus weak object-local value references. The key includes
full history, promotion options, and the absolute action number seen by the
model. It stores no depth-dependent mate score and is not shared between models
or analyses. The retained search tree and suspended generators are separately
bounded by candidate, depth, and work limits. The former auxiliary candidate
cache is unnecessary because generation state belongs to each tree node.
Server, CLI, and self-play requests pin the inference process generation. If a
checkpoint reloads during analysis, that analysis fails explicitly instead of
mixing new predictions with values cached from the previous model.

Results use searchPolicy: transformer-adaptive-depth and optionalMovePolicy:
temporal-only. They retain the existing rankings/depthStats interface and add
wideningSteps, policyCalls, valueCacheHits, and stage timings. Each displayed
entry includes staticScore, searchedReplies, generatedReplies, repliesExhaustive,
forcing, and visits. These distinguish line depth from defensive coverage.

Depth is the deepest selected True evaluation, pvDepth is the returned principal
variation length, and selectiveDepth includes generation and terminal probing.
Tactical extensions can exceed currentMaxDepth; effectiveQuiescenceDepth reports
the observed extension. Dynamic results preserve limits.maxDepth: 0 and expose
dynamicDepthThreshold: 3 as the initial contender count, not a per-depth quota.

Progress uses the existing 100 ms cadence and bounded top-ten depth snapshots.
completed: true means a selected root evaluation exists or the root was proved
terminal; it does not mean every alternative at the reported depth was examined.
Interruption retains the latest backed result. Before any root True evaluation,
accepted neural batches can improve the legal scored fallback, but partial turns
and expired batches never become playable recommendations.

Advanced analyze callers can set initialCandidates, componentBatchSize, candidateBranches, terminalProbeWork,
tacticalExtensionDepth (or quiescenceDepth), extensionCandidateLimit,
maxValueCacheEntries, and valueCacheMemoryMb. The usual timeMs, maxNodes,
candidateLimit, innerCandidateLimit, cancellation, and fixed/dynamic depth limits
still apply. Disabling tacticalExtensionDepth gives a strict complete-turn horizon.

### Benchmarking

Run node scripts/benchmark-transformer.js --time-ms 1000 --nodes 20000 --depth 3
for an equal-budget diagnostic over the tactical suite plus standard and
two-timeline positions. Optional --checkpoint, --device, --search-module, and
--output arguments support controlled comparisons. Use the same checkpoint and
hardware for both versions. Compare tactical targets, legal PVs, time to first
observed target selection, and generation/inference cost, alongside line depth.
These development fixtures do not establish Elo or independent playing strength.

## Architecture and GPU budget

| Component | Default |
| --- | --- |
| Transformer blocks | 6, pre-normalization |
| Hidden width / attention heads | 512 / 8 |
| Feed-forward width | 2,048, GELU |
| Parameters | 20,000,257 with the optional policy head (~20M); 19,142,657 value-only |
| Context | At most 4,096 tokens including CLS |
| Training | AdamW, batch 16, CUDA float16 AMP, gradient norm clipped to 1 |
| Prediction | CLS value head, scalar white-relative score; optional component policy logits |

Tokens represent occupied squares and board markers. Piece identity/color,
signed unmoved flags, board side to move, and frontier status have categorical
embeddings. Coordinates encode timeline, absolute half-turn, rank, file, age
relative to that timeline's frontier, and board dimensions, using sinusoidal
features plus signed logarithmic coordinates. Global features include action
side, action number, history/timeline sizes, even-timeline mode, available
promotion pieces, and context coverage. Empty board markers preserve board
existence even without pieces. In even-timeline variants, the two zero-labelled
timelines receive adjacent distinct coordinates, matching rule-engine movement.

Positions within the budget retain every token. On overflow, selection drops
whole boards furthest from the nearest playable board first. Playable boards
are timeline frontiers whose side to move matches the action, including future
and inactive timelines. Distance is the sum of signed timeline separation and
half-turn separation divided by two. If no frontier matches the mover, all
frontiers serve as distance anchors. Ties prefer newer boards, then lower signed
timeline coordinates, so selection is deterministic.

Selected boards keep their marker and every occupied square. Selection stops
when the next closest board cannot fit; unused capacity does not admit a farther,
smaller board. Very small custom budgets may retain only CLS if no closest board
fits. Every prediction reports token counts, `truncated`, and
`frontierTruncated` (whether any latest board was omitted). Discarded context can
hide relevant tactics from evaluation; legal search still uses full history.
Training and inference share this selection policy.

With the previous 694,017-parameter model on this workstation's
**NVIDIA GeForce RTX 3060, 12 GiB**, PyTorch
2.14.0+cu126 completed three training updates at batch 16 × 512
tokens in **0.379 seconds**, with **206.4 MiB peak allocated** and **262 MiB
peak reserved** CUDA memory. These allocator figures exclude driver/context and
other applications. These historical measurements do not describe the new 20M
model. Benchmark the current default architecture with:

```powershell
node scripts/transformer.js doctor --benchmark --device cuda --batch-size 1
```

The benchmark creates no checkpoint. It exercises forward/backward passes,
mixed precision, and optimizer state at a fixed 512-token workload. It does not
measure the 4,096-token maximum. The 20M model needs about 76 MiB for float32
weights alone; gradients, AdamW state, activations, and attention add to that.
Start at batch 1 and measure memory with representative positions before
increasing the batch or context length, especially when training the policy head.

The initial local bootstrap of the previous small model trained for 1,000 updates on 256 teacher positions
in **41.213 seconds**, reaching **206.4 MiB peak allocated / 264 MiB reserved**.
An end-to-end HTTP smoke check on this workstation completed standard-position
analysis in about 650 ms and a two-timeline position in about 85 ms, and verified
legal principal variations, Play best, and cancellation. These timings are
small smoke cases with bounded candidate search; they are not a general speed
or strength estimate. The bootstrap has no held-out validation claim.

## Training

### Training the 20M model

The 20M model is archived and hidden from the UI. Existing local fresh runs and
their self-play history are in `archive/models-20260930/fresh20m`; the archive's
`manifest.json` records original paths and checkpoint hashes. The **Training** page
offers **Self-play · 800k model** for the active checkpoint. The CLI still
supports the larger architecture and explicit archived checkpoint paths.

New training runs use the 20M architecture by default (19.14M value-only;
20.00M when policy labels enable the optional policy head). Existing checkpoints
retain their saved dimensions when loaded or resumed, including in self-play.
Changing the defaults does not enlarge trained weights; start a fresh run to
train the larger architecture. For a separate candidate with a smaller initial
memory budget:

```powershell
node scripts/transformer.js train --data artifacts/transformer/training.jsonl --output artifacts/transformer/model-20m.pt --batch-size 1 --max-tokens 512 --device cuda
```

The standard context budget remains 4,096; this example explicitly uses 512
during training to reduce memory. Evaluate the candidate on held-out data before
selecting it with `TRANSFORMER_CHECKPOINT`. Self-play can then resume that larger
checkpoint. Its CLI defaults to a 4,096-token training budget; use `--max-tokens`
to adjust it. Resuming the old checkpoint
continues training the old architecture.
To create a model with the previous dimensions, pass
`--width 128 --heads 4 --layers 4 --feedforward 384`.

### Component policy head

New teacher datasets and accepted self-play samples contain optional component
policy targets from the searched complete-turn action. Each target includes the
prefix position and a variable candidate list, filtered to required or temporal
components. Targets follow the searched action, including when self-play actually
chooses an exploratory move. Invalid/discarded games and incomplete searches do
not produce policy labels.

The optional policy head shares the transformer encoder and scores move
descriptors against the encoded prefix. Timeline/time coordinates, piece flags,
promotions, castling, and en-passant endpoints remain distinguishable. Padded
candidate lists are masked for cross-entropy training. Self-play also records
soft targets from evaluated complete-turn alternatives sharing the exact prefix.
Scores are mover-relative with a 250 cp softmax temperature; only observed
choices participate in this conditional loss. Unsearched choices receive no
negative supervision. Old single-target datasets remain supported. One prefix is sampled
uniformly per value row during training to bound shuffle-buffer memory; existing
sample/game weighting is retained.

Training defaults to --policy auto: it trains the head when policy labels are
available. --policy on requires those labels; --policy off trains values only.
--policy-weight controls the policy loss weight (default 1.0). Resuming an older
value-only checkpoint adds the head while preserving the value weights and
compatible optimizer state. Existing datasets/checkpoints remain supported.
A policy is advertised as available only after a supervised update with multiple
candidate choices; absent or untrained heads never supply random move rankings.

To train a separate candidate checkpoint from newly generated teacher labels:

```powershell
node scripts/transformer-data.js --output artifacts/transformer/policy-training.jsonl --samples 4096 --nodes 2000
node scripts/transformer.js train --data artifacts/transformer/policy-training.jsonl --resume artifacts/transformer/model.pt --output artifacts/transformer/policy-model.pt --policy on --steps 1000
```

Select that checkpoint with TRANSFORMER_CHECKPOINT after evaluating it. Continuous
self-play can also learn the policy automatically from its newly labeled replay
rows and uses its existing arena promotion checks. Adding this feature does not
replace the active checkpoint; a legacy checkpoint continues using value-guided
ordering until a trained policy checkpoint is selected.

### Value training and self-play

For iterative training from the model's own games, see [continuous self-play](transformer-selfplay.md).
Run `node scripts/transformer-selfplay.js --iterations 0 --device cuda` after creating
a checkpoint; candidates are evaluated before they replace the UI's active model.
The shortcut `npm run transformer:selfplay:continuous` selects CUDA when available
and requires no forwarded arguments. Use direct `node` invocation for options:
PowerShell's `npm.ps1` wrapper can strip forwarded flag names.

[The September 23 expanded training report](transformer-training-20260923.md)
records the locally installed checkpoint, validation comparison, and backups.

The data generator writes bounded classical-teacher examples from development
positions and legal continuations. It excludes designated match validation and
tactical regression suites. Labels remain approximate classical search scores,
not game outcomes. A larger example run is:

```powershell
npm run transformer:data -- --samples 4096 --nodes 2000 --seed 7
npm run transformer:train -- --steps 5000 --batch-size 16 --device cuda
```

Training defaults to 1,000 optimizer updates, learning rate 0.0003, weight decay
0.01, seed 42, dropout 0.1, and a shuffle buffer of 128 encoded positions. The
JSONL file is streamed and repeated as needed; it is never loaded in full.
Each line has this shape, with the raw rules-library position representation:

```json
{"position":{"board":[[[[12,0],[0,11]]]],"action":0,"promotions":[9,10]},"value":125}
```

`value` is always finite **white-relative centipawns**, regardless of the side to
move. The network minimizes MSE against `tanh(value / 1000)`; inference converts
back using `1000 * atanh(clamp(output, -0.999, 0.999))`. Neural scores are bounded
to about ±3,800 cp and do not claim mate. Legal terminal detection belongs to
search. Additional row fields such as teacher provenance are allowed.

Supply separate held-out JSONL data to measure generalization:

```powershell
npm run transformer:train -- --data artifacts/transformer/train.jsonl --validation-data artifacts/transformer/validation.jsonl --steps 1000
npm run transformer:train -- --resume artifacts/transformer/model.pt --steps 1000
```

With `--resume`, steps are additional completed updates and the checkpoint's
architecture and optimizer state are reused. Shuffle order and AMP scaler state
restart from the provided seed, so resumption is not bit-for-bit equivalent to
an uninterrupted run. CUDA kernels may also be nondeterministic.

To keep the best validation checkpoint while also saving the latest training
state, pass `--best-output` with a different path from `--output`. A resumed
model is evaluated before training and remains the best checkpoint if no later
checkpoint improves validation MSE. Use enough `--validation-batches` to cover
the entire validation file. For example:

```powershell
npm run transformer:train -- --resume artifacts/transformer/model.pt --data artifacts/transformer/train.jsonl --validation-data artifacts/transformer/validation.jsonl --steps 5000 --validation-batches 64 --output artifacts/transformer/latest.pt --best-output artifacts/transformer/best.pt
npm run transformer:evaluate -- --data artifacts/transformer/validation.jsonl --checkpoints artifacts/transformer/model.pt artifacts/transformer/best.pt --device cuda --output artifacts/transformer/comparison.json
```

The evaluator streams every validation row and reports normalized value MSE,
centipawn MAE with both targets and predictions clipped to the model's output
range, and context truncation counts. For a trained policy it also reports
top-1 accuracy, top-8/top-16 target retention, full-list cross-entropy, and the
conditional supervised loss on non-forced prefixes. Best-checkpoint selection
continues to use value MSE; use policy metrics and the paired arena to assess
the resulting search. This measures agreement with the teacher;
it does not by itself measure playing strength. Keep exact duplicate positions
out of both training and validation, and use separate games or starting histories
where possible.

Checkpoints save every 100 updates and on normal completion through an atomic
file replacement. They include architecture, encoding version, parameters,
optimizer, trained-step count, examples seen, data path and SHA256, seed, loss,
optional validation MSE, and a human-readable experimental label. Validation
measures up to 16 batches by default (`--validation-batches`); it does not test
playing strength. Logs report elapsed time and truncation; completion reports
peak CUDA allocator memory. An interrupted run can resume from the last saved
checkpoint.

For the 20M model, start with `--batch-size 1`; the default batch remains 16.
Reduce `--max-tokens` as needed if long positions exceed available memory.
`--width`, `--heads`, `--layers`, and `--feedforward` configure new models and
cannot change an existing model via resume. Width accepts 32–512, heads accepts
1, 2, 4, or 8 and must divide width, layers accepts 1–8, and feed-forward
width accepts hidden width–2,048. `--max-tokens` accepts 16–4,096 and
defaults to 4,096 for both new and resumed training. Training saves the effective
budget with the checkpoint. Inference and evaluation load existing weights with
the current 4,096-token budget, including older checkpoints trained with smaller
contexts; loading does not rewrite those files. Context length does not change
parameter shapes, so no weight conversion or retraining is required. Run training
during a separate period from latency-sensitive analysis for predictable GPU usage.

## Validation and worker protocol

```powershell
npm run transformer:test
npm run transformer:smoke
npm test
```

Python tests cover history and movement coordinates, signed piece flags,
deterministic context selection, frontier overflow reporting, loss reduction,
padding invariance, checkpoint round trips, training resumption, and worker
protocol errors. These verify implementation behavior; they do not establish
chess strength.

The persistent worker accepts one JSON object per line on stdin; stdout contains
only JSONL protocol messages. It is launched as:

```text
python -u neural/service.py --checkpoint artifacts/transformer/model.pt --device auto
```

Startup emits `{ "ready": true, "device": "cuda", "model": { ... } }` or
`{ "ready": false, "error": "..." }` followed by a nonzero exit. Each request
`{ "id": 1, "positions": [position] }` returns `{ "id": 1, "values": [125.0],
"context": [{ "tokens": 34, "totalTokens": 34, "truncated": false,
"frontierTruncated": false }], "device": "cuda" }`. Requests accept 1–128
positions and are internally evaluated in batches of 1 for models with at least
10M parameters, or 16 for smaller models. `--batch-size` overrides this default;
the ready response includes the effective `batchSize`. Invalid requests
return `{ "id": 1, "error": "..." }`. Input lines are limited to 32 MiB.

Implementation references: [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html),
[mixed precision](https://docs.pytorch.org/docs/2.14/amp.html), and
[safe state-dictionary loading](https://docs.pytorch.org/docs/2.14/notes/serialization.html).
