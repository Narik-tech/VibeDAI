# LCZero transfer to 5D chess

This implementation imports the supplied **T3 512×15×16h** LCZero spatial
network and trains new multiverse value and candidate-move heads. It produces
self-contained checkpoints usable by this project's Transformer engine and
self-play runner. The original model and existing engine checkpoint stay intact.

The approach is **shared spatial features + timeline attention + legal-prefix
learning**. It transfers chess representations; LCZero's orthodox evaluation is
not treated as a 5D game result. This is an experimental training pipeline. The
initial short run verifies training and legal engine operation, not strength.

## Run

Use the existing project Python environment (`npm run transformer:setup` on a
fresh installation). The importer needs PyTorch and the Python standard library;
NumPy, protobuf, TensorFlow, and a native LCZero executable are unnecessary.

```powershell
# Inventory the actual baseline, including format and SHA256.
node scripts/lc0.js inspect "D:\Downloads\t3-512x15x16h-distill-swa-2767500.pb.gz"

# Generate train/validation JSONL and a provenance manifest.
node scripts/lc0.js data --samples 192 --nodes 10000 --time-ms 1000

# Reproduce the short validation run performed during implementation.
node scripts/lc0.js train --weights "D:\Downloads\t3-512x15x16h-distill-swa-2767500.pb.gz" --data artifacts/lc0/curriculum.jsonl --validation-data artifacts/lc0/curriculum.validation.jsonl --output artifacts/lc0/model.pt --best-output artifacts/lc0/best.pt --device cuda --steps 64 --batch-size 4 --max-boards 16 --save-every 16 --log-every 16

# Value/policy teacher agreement, and actual legal engine operation.
node scripts/lc0.js evaluate --checkpoint artifacts/lc0/best.pt --data artifacts/lc0/curriculum.validation.jsonl --device cuda --output artifacts/lc0/evaluation.json
node scripts/lc0.js smoke --checkpoint artifacts/lc0/best.pt --device cuda
node scripts/lc0.js test
```

Run `node scripts/lc0.js --help` for commands; each command accepts `--help`.
Use direct `node` invocation for flags in PowerShell. npm aliases are also
available: `lc0:inspect`, `lc0:data`, `lc0:train`, `lc0:evaluate`, `lc0:smoke`,
and `lc0:test`.

Leela is archived and hidden from the analysis and training menus. Existing local
artifacts are stored in `archive/models-20260930/lc0`; the archive's
`manifest.json` records original paths and checkpoint hashes. The commands above
describe creating a new transfer model. To analyze with the archived checkpoint
from the command line:

```powershell
$env:LEELA_CHECKPOINT = 'archive/models-20260930/lc0/best.pt'
node src/cli.js --engine leela --time 3 --depth 3
```

The CLI requires a trained LCZero transfer checkpoint and uses its own inference
process. `LEELA_PYTHON` and `LEELA_DEVICE` select the Python environment and device;
otherwise they follow `TRANSFORMER_PYTHON`/`TRANSFORMER_DEVICE`.

The archived self-play history retains its original checkpoint paths. To resume
that run, restore the LCZero directory to the original location recorded in the
manifest. For a separate experiment, copy the checkpoint and use a new run directory.

## What is transferred

| Component | Treatment |
| --- | --- |
| Dense positional preprocessing and input embedding | Imported |
| Embedding gates, FFN, and normalization | Imported |
| All 15 attention/FFN blocks and normalization | Imported |
| Per-layer smolgen and shared attention-bias projection | Imported |
| Orthodox policy, WDL/value, and moves-left heads | Replaced |
| Fairy-piece/unmoved/square features and board pooling | New, trainable |
| Attention across timelines/history, White value, candidate policy | New, trainable |

The supplied file contains 85,708,452 parameters. The loader transfers
**83,261,184 parameters in 390 tensors**, and explicitly reports the omitted
2,447,268 orthodox-head parameters. The default 5D model adds 1,986,049
parameters, for **85,247,233 total**. Initial training freezes the spatial body.

Source SHA256:
`78541c4abc0bc81e8145e1f9a4c35a934ebbefff70f11f33215751d55a886352`.

The supplied protobuf uses an older network-format declaration with newer
embedding/head fields. The importer applies LCZero's corresponding legacy
format upgrade, then checks the actual tensor layout. Unsupported architectures,
missing body tensors, malformed gzip/protobuf, and nonfinite weights fail.
There is no partial import with randomly filled missing body layers.

Tensor shapes and equations follow the official [network schema](https://github.com/LeelaChessZero/lc0/blob/master/proto/net.proto),
[loader](https://github.com/LeelaChessZero/lc0/blob/master/src/neural/loader.cc),
and [BLAS backend](https://github.com/LeelaChessZero/lc0/blob/master/src/neural/backends/blas/network_blas.cc).
The implementation includes dense positional features, Mish/Swish, residual
scaling, and smolgen. **Numerical parity against a native LCZero executable has
not been benchmarked.** Tests verify tensor coverage, layout, equations,
reconstruction, gradients, and finite output from the actual file.

## Representing the multiverse

1. Select up to `max_boards` snapshots. Frontier boards take priority, followed
   by nearby history; coordinates retain actual timeline and half-turn identity.
2. Run the shared LCZero tower on each selected board. Use current pieces and
   up to seven contiguous prior snapshots from the same timeline, with ranks
   oriented toward that board's side to move. History stops at a missing slot.
3. Add learned square identity, all 24 variant-piece codes, and unmoved flags.
   A square mask excludes padding on miniature boards. Learned pooling makes
   one summary per board; square features remain available to the policy.
4. A two-layer transformer relates the board summaries using timeline, time,
   distance from frontier, side, board dimensions, action count, and global
   multiverse features. A global token produces the White-relative value.
5. Candidate logits combine that global context with source/destination square
   features, their routed board features, all move endpoints, promotions, and
   temporal displacement. Castling and en-passant endpoint descriptors are
   retained. `null` represents a rules-supplied SUBMIT candidate.

This is a versioned **5D input projection**, not LCZero's ordinary chess encoder.
Orthodox castling, repetition, en-passant, and rule-50 auxiliary planes are zero;
the 5D rules layer still handles the actual rights and full history. Classical112
side-to-move and constant planes are supplied. Boards smaller than 8×8 are
padded; boards larger than 8×8 fail explicitly. Fairy pieces have separate
learned features rather than being misidentified as orthodox pieces.

The default context is 16 boards. `--max-boards` changes that budget and
`--board-batch` bounds how many boards pass through the spatial tower together
(default 8). Both can be changed on resume. `max_tokens` also bounds the router
sequence; it counts board summaries plus the global token for this architecture.
Reported truncation includes omitted frontier boards. It never changes legality.

The current engine uses policy logits to order supplied component moves and
uses its existing legal-submission logic. The transfer policy interface can
score SUBMIT, and the curriculum trains it, but search does not yet use a learned
SUBMIT score to decide when to end a turn. Candidate limits and the existing
optional-temporal-move search restriction still apply.

## Training procedure

### 1. Balanced curriculum with frozen spatial weights

The data generator balances three stages:

- **Spatial:** a single timeline with no currently available temporal component.
  It includes explicit composed spatial restarts. Each restart begins a new
  game and records its origin; it is not presented as uninterrupted played history.
- **Temporal:** historical targets and travel, including a composed position
  whose spatially trapped king must escape through time.
- **Multiverse:** branching and simultaneous board obligations, including turns
  requiring two component moves before submission.

Only completed classical 5D searches supply bootstrap values. The value is White
centipawns, mapped to `tanh(cp / 1000)`. These labels are approximate search
estimates. Incomplete searches are discarded, and turn limits never become draws.

Each selected full turn is replay-validated. Every prefix carries candidate
components and a target; SUBMIT appears only when submission is legal. Alternative
components are pseudo-legal at that prefix and may fail to lead to a complete
legal turn. The engine remains responsible for full-turn legality.

Training and validation use fixed source families. Related miniature and composed
sources remain together; games never cross the split. The generator removes
duplicate full-history positions. Training additionally rejects overlap involving
any parent or supervised policy-prefix position.

The loss combines dataset-weighted value MSE and masked policy cross-entropy.
One uniformly sampled prefix per value row bounds memory and keeps a long
compound turn from gaining more policy weight merely because it has more moves.
Mixed precision, gradient clipping, and bounded board batches control training
memory. Policy availability is advertised only after informative policy updates.

### 2. Optional cautious spatial fine-tuning

Once adapters learn useful temporal behavior, unfreeze a small number of the
last spatial blocks at a separate low learning rate:

```powershell
node scripts/lc0.js train --resume artifacts/lc0/best.pt --data artifacts/lc0/curriculum.jsonl --validation-data artifacts/lc0/curriculum.validation.jsonl --output artifacts/lc0/finetuned.pt --device cuda --steps 1000 --batch-size 4 --unfreeze-last 2 --backbone-lr 0.000001 --learning-rate 0.0001 --anchor-weight 0.01
```

An L2 penalty anchors these parameters to their values when first unfrozen.
Anchors survive refreezing and later resume. Changing the unfrozen block count
starts a new optimizer; continuing within a stage restores optimizer moments.
Architecture dimensions cannot be changed during resume. Width, router depth,
feedforward width, and dropout are initialization settings.

### 3. Self-play with promotion matches

The existing runner now resumes this architecture automatically. Start from
the trained checkpoint and use the LC0 training suite, which excludes the
curriculum's reserved source families:

```powershell
node scripts/transformer-selfplay.js --checkpoint artifacts/lc0/best.pt --run-dir artifacts/lc0/selfplay --seed-data artifacts/lc0/curriculum.jsonl --suite examples/matches/lc0-training.json --iterations 1 --device cuda --batch-size 4 --steps 500
```

The active checkpoint can be replaced only after the runner's paired promotion
gate passes. Keep curriculum validation for diagnostics; use the separate arena
suite for promotion. Self-play samples use complete-turn search targets and
verified completed-game outcomes. Unfinished games remain unfinished. Existing
replay component labels are supported; they do not yet add SUBMIT supervision.
See [self-play details](transformer-selfplay.md) for replay, results, and recovery.

## Evidence from the initial short run

On the local RTX 3060 (12 GB), the generated dataset contained 153 training and
39 held-out rows, with 15 selected temporal moves and 46 compound turns. Each
stage contributed 51 training and 13 validation rows.

| Measurement | Result |
| --- | --- |
| Successful adapter updates | 64 |
| Informative policy updates | 60 |
| Training loop including validation/saves | 22.1 seconds |
| PyTorch peak allocated GPU memory | 433.7 MiB |
| Initial held-out value MSE, random 5D heads | 0.06672 |
| Best trained checkpoint, step 32 | 0.06835 MSE; 278.0 clipped cp MAE |
| Latest checkpoint, step 64 | 0.07620 MSE; 291.4 clipped cp MAE |
| Best-checkpoint policy top-1 on nontrivial held-out prefixes | 20/52 (38.46%) |
| Uniform candidate policy on those same prefixes | 7.01% expected top-1 |

The best trained checkpoint **did not beat the initial value MSE**. Short-run
losses were unstable; the initial head also predicts values near zero, which
can look competitive on these small bootstrap positions. These results do not
show a strength gain. `best.pt` means lowest validation MSE among eligible
trained checkpoints, not superior play or successful model promotion.

The policy evaluation visited all 91 held-out prefixes and excluded 39 forced
SUBMIT decisions from accuracy. Cross-entropy was 1.902 versus 2.789 for a uniform
candidate distribution. Only one held-out target was a temporal move, and there
were no nontrivial SUBMIT choices, so this does not establish either capability
across diverse positions.

The actual inference service used the step-32 checkpoint to return a legal
time-travel escape and a legal two-component turn. Every principal-variation
turn was replay-validated, with policy calls active and finite values. CPU and
CUDA training, ordinary resume, and an unfrozen-block update were exercised.
Detailed local evidence is in `artifacts/lc0/validation-training.log`,
`evaluation.json`, and `runtime-smoke.json`. Generated artifacts are git-ignored.

All 476 JavaScript tests and 71 Python tests passed, including the optional test
against the supplied file. A separate short self-play cycle generated two search
targets, resumed training, and ran the promotion arena. Its two capped arena
games stayed unfinished, and promotion was correctly refused. That cycle used a
copy of the checkpoint; its report is under `artifacts/lc0/selfplay-smoke/`.

Before claiming improvement, run longer training with varied seeds, compare
against a randomly initialized spatial-body ablation, and complete enough paired
matches against the existing engine. The current miniature holdout is a
development diagnostic, not an Elo benchmark.

## Implementation map

- `neural/lc0_weights.py`: bounded protobuf reader, tensor decoding, inventory.
- `neural/lc0_backbone.py`: strict spatial weight transfer and T3 forward pass.
- `neural/lc0_encoding.py`: board selection, 5D planes, masks, SUBMIT encoding.
- `neural/lc0_model.py`: square/board adapters, timeline transformer, value/policy.
- `neural/lc0_train.py`: curriculum/replay training, fine-tuning, checkpoints.
- `neural/lc0_evaluate.py`: value and per-prefix policy metrics.
- `scripts/lc0-data.js`: legal curriculum generation and source-group split.
- `scripts/lc0-smoke.js`: real runtime and complete-turn legality verification.

Checkpoint loading uses `weights_only=True`, validates shape/dtype/finite values
before allocating the model, and rejects untrained checkpoints. Writes are
atomic. Checkpoints include all imported weights, adapter weights, optimizer,
source hash, tensor mapping, context settings, and training provenance; the
original `.pb.gz` is not needed to resume or play.
