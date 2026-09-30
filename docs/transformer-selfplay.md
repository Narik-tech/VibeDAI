# Transformer self-play

The runner repeatedly plays the current transformer against itself, adds value
targets to a bounded replay buffer, trains a candidate on the local GPU, and
tests it against the incumbent. Only a passing candidate replaces the selected
checkpoint. The current analysis checkpoint is used by **Transformer · 800k**
in the UI. Classical search and the
existing `npm run selfplay` diagnostic are separate.

## Run it

### Browser controls and game review

Start the local server with `npm start` and open **Training** in the app header
at `http://127.0.0.1:5173/training`. The local transformer Python environment
is required to start training. **Self-play · 800k model** uses the existing
analysis checkpoint at `artifacts/transformer/model.pt`.

Fresh 20M, 20M self-play, and Leela are archived and hidden from the UI.
Existing local 20M runs are in `archive/models-20260930/fresh20m`, and LCZero
artifacts are in `archive/models-20260930/lc0`. The archive's `manifest.json`
records original paths and checkpoint hashes. CLI workflows remain available with
explicit archived paths.

The page runs one training job at a time. It shows the 800k model's self-play
history; saved games can be reviewed without starting a model.

Adjust the parameters before starting a run. Settings cover cycle count
(`0` means continuous), self-play games, independent self-play and arena concurrency,
turn limits, search budgets,
training updates, batch size, training context, learning rate, replay capacity, exploration,
device, and the paired promotion gate. Changes apply to the next run.
Browser preferences are saved locally; each cycle also records its exact
settings in `iteration.json` and `report.json`.

The run view shows its current phase, retained cycle reports,
promotion decisions, and training log. **Stop run** cooperatively stops the
active job and preserves saved artifacts and completed records. Closing a browser tab
does not stop training; stopping the server requests cancellation. An external
command-line runner is detected through its lock and must be stopped from its
own terminal.

Select a retained cycle and then a self-play or arena game. Step through
complete player turns to inspect the boards and move trace. Review does not
change the live analysis game. Search scores are from White's perspective;
unfinished games remain unfinished. History includes cycles created from the
command line in the default run directory. Retention still follows
`keepIterations`, so older cycles and their games disappear when pruned.

### Command line

Run from the project directory in PowerShell. This workstation already has the
CUDA environment and a trained checkpoint. For a fresh installation, first run
`npm run transformer:setup`, `npm run transformer:data`, and
`node scripts/transformer.js train --width 128 --heads 4 --layers 4 --feedforward 384`.
Self-play requires a trained model to start; skip data generation and training
when the active checkpoint already exists.

One cycle with the default settings:

```powershell
node scripts/transformer-selfplay.js --iterations 1 --device cuda
```

Continuous operation until **Ctrl+C**:

```powershell
node scripts/transformer-selfplay.js --iterations 0 --device cuda
```

To play up to four self-play games concurrently:

```powershell
node scripts/transformer-selfplay.js --iterations 0 --device cuda --game-concurrency 4
```

The Training page exposes the same **Concurrent games** setting. It accepts
1–8 and defaults to 1; actual concurrency is capped by `--games`. Each active
game searches in its own CPU worker, sharing one loaded inference model. The
inference queue drops cancelled work before sending it to Python and keeps at
most one request in flight. This avoids multiplying GPU model memory. Training
updates and the paired promotion arena still run after generation as separate phases.
Inference defaults to one position per batch for models with at least 10M
parameters, including the 20M model; smaller checkpoints retain batches of 16.
The inference context remains 4,096 tokens independently of the training limit.

To also play up to four arena games concurrently:

```powershell
node scripts/transformer-selfplay.js --iterations 0 --device cuda --game-concurrency 4 --arena-concurrency 4
```

The Training page exposes **Concurrent arena games** under **Arena & promotion**.
`--arena-concurrency` accepts 1–8 and defaults to 1, independently of self-play
concurrency. It limits active games, rather than pairs, and is capped by the
number of scheduled arena games. Each active search uses a CPU worker. All
arena games share one candidate inference runtime and one separate incumbent
runtime, each with its own queue; increasing arena concurrency does not load
another model copy for every game. Starts and color-swapped pairs are scheduled
in the same order at every concurrency setting. Saved game numbers and arena
report order remain stable even if games finish out of order; live events also
report `completedGames`.

Start with 2 or 4 when CPU capacity permits. More workers use more CPU and
memory; speedup depends on the positions and inference load. In both phases, rules validation,
exploration and terminal certification share the coordinator, and search time
limits still use wall time, so contention can change cutoff timing and reduce
completed search depth. Arena pairing and equal per-move budgets are preserved,
but results can vary when concurrent games compete for resources.
Lower concurrency if timeouts increase.

For continuous operation with automatic device selection (CUDA when available),
the shortcut requires no forwarded arguments:

```powershell
npm run transformer:selfplay:continuous
```

Use the direct `node` commands when setting options. PowerShell's `npm.ps1`
wrapper can strip forwarded flag names, leaving values such as `0 cuda` for
the runner. Calling `node scripts/transformer-selfplay.js` directly avoids
that argument-forwarding issue.

Rerun the same command to continue using the saved replay buffer and current
accepted model. Iteration numbers and seeds advance. Each cycle starts training
from the incumbent; rejected candidates are retained for inspection but are not
used as the next incumbent. An interrupted cycle is recorded and a new cycle
starts on the next invocation; partially completed training is not resumed.

A longer run with more games and a larger acceptance sample:

```powershell
node scripts/transformer-selfplay.js --iterations 0 --device cuda --games 16 --plies 64 --steps 1000 --batch-size 32 --nodes 40000 --time-ms 5000 --arena-pairs 12 --min-pairs 8 --arena-plies 128
```

These settings can take substantial time. Legal move generation runs on the
CPU, so low GPU utilization during games is expected. Training uses the saved
checkpoint's architecture and CUDA mixed precision. The batch-32 example above
was intended for the previous small model. Start with `--batch-size 1 --max-tokens 512`
for a 20M checkpoint and measure memory before increasing either setting.
`--max-tokens` accepts integers from 16 to 4096 and limits the board history
encoded for each training example. Lower limits reduce training memory and
discard more distant board history when a position exceeds the limit. The command-line
default stays at 4096.
Self-play resumes the saved
architecture; first [train a fresh 20M checkpoint](transformer.md#training-the-20m-model)
to use the larger model. Use `--device cpu` without CUDA.
Avoid latency-sensitive UI analysis while training if response time matters.

For an isolated short pipeline check, copy the active checkpoint first:

```powershell
New-Item -ItemType Directory -Force artifacts/transformer/selfplay-check | Out-Null
Copy-Item artifacts/transformer/model.pt artifacts/transformer/selfplay-check/active.pt
node scripts/transformer-selfplay.js --iterations 1 --device cuda --checkpoint artifacts/transformer/selfplay-check/active.pt --run-dir artifacts/transformer/selfplay-check/run --games 2 --plies 4 --steps 16 --depth 1 --nodes 3000 --time-ms 1000 --arena-pairs 1 --min-pairs 1 --arena-plies 4 --replay-size 128
```

Short checks usually fail the promotion gate because games are unfinished or
tied. They verify the pipeline, not playing strength. Use a new check directory
if you want to preserve an earlier check's copied checkpoint.

## Defaults and learning targets

| Setting | Default |
| --- | --- |
| Cycles per invocation | 1; `--iterations 0` runs continuously |
| Self-play | 8 games, 40 complete turns/game |
| Concurrent self-play games | 1; `--game-concurrency` accepts 1–8 |
| Search per turn | Depth 2, 20,000 work units, 3 seconds |
| Terminal certification | Separate 3-second budget (`--terminal-time-ms`), plus `--terminal-work` |
| Exploration | 20% probability during the first 12 turns |
| Training | 500 additional updates, batch 16, learning rate 0.0001 |
| Training context | 4,096 tokens; `--max-tokens` accepts 16–4,096 |
| Replay | At most 8,192 unique full-history positions |
| Training weights | Equal total weight per self-play game represented in replay |
| Arena | 8 distinct starts, 2 games/start with colors swapped, 80 turns/game |
| Concurrent arena games | 1; `--arena-concurrency` accepts 1–8 independently of self-play |
| Promotion | At least 4 completed distinct pairs; candidate score at least 55% |
| Retention | Latest 5 iteration folders, replay, latest report and previous model |

See all options with `node scripts/transformer-selfplay.js --help`.
`--depth 0` enables dynamic depth for both self-play and arena games. Each search
starts with a ceiling of one turn and raises it when leading contenders have
selected evaluations, initial reply coverage, and continuations reaching the
ceiling. The ceiling can grow to 64 under the same per-turn time and
work limits. The training screen's **Search depth** setting also accepts zero.

`--plies` counts submitted full player turns, including turns requiring moves
on multiple boards. Exploration shuffles component order with the game's seeded
random stream, then samples from up to 32 legal complete turns. It is not uniform
over the entire 5D action space. Coverage reports count actual temporal,
compound, and timeline-creating turns. Seeds control
case rotation and exploration. Each game's exploration has its own random
stream derived from the cycle seed and game index, independent of completion
order; this changes the old shared random stream for later games. Wall-clock
limits and CUDA kernels can still affect reproducibility. Game files keep their
scheduled numbers even when games finish out of order, while live events also
report `completedGames`. Replay receives samples in scheduled game order.

The model has a value head. This is search distillation with outcome targets,
not an AlphaZero policy/MCTS implementation. Every training score is relative
to White. A completed root search supplies `v = tanh(score / 1000)`:

- A certified finished game supplies `z = +1` for a White win, `-1` for a Black
  win, or `0` for stalemate. The default target is `0.5*v + 0.5*z`.
- A capped, timed-out, or otherwise unfinished game supplies only its completed
  search targets. It has no invented win/loss/draw target.
- Incomplete/nonfinite searches supply no target. An illegal action, bad PV,
  input mutation, or model error discards that game's samples and stops the cycle
  before training.

`--outcome-weight` changes the finished-game mixture. Targets are converted back
to the trainer's centipawn JSONL format. Search scores belong to the searched
root even when exploration plays another turn. Records preserve both actions,
the checkpoint hash, seed, game result and target provenance. Every played
action and PV is checked against the full rules. Only full-rules terminal
certification establishes checkmate/stalemate; there is no score-based
adjudication or repetition draw.

The first replay buffer also samples `artifacts/transformer/training.jsonl`.
Use `--seed-data FILE` for another seed dataset, or `--seed-data none` to learn
only from self-play. Later cycles read existing replay instead. New unique
positions reserve up to half the buffer; historical positions fill the other
share, and spare capacity is filled when one source is small. Deduplication
includes side to move, promotions and the full multiverse history. Duplicate
positions retain up to 64 distinct certified game outcomes separately from the
latest search and policy labels. Outcomes are deduplicated by game provenance,
then averaged using their recorded outcome weights and blended with the latest
search value. An unfinished game cannot erase retained outcome evidence.
Bootstrap teacher labels from incomplete searches receive weight 0.1; completed
teacher searches receive weight 1. The trainer applies the same 0.1 fallback to
older teacher rows marked `completed: false` without an explicit weight; explicit
weights are preserved. JSONL is streamed with a 4 MiB
per-record limit; malformed examples fail the update without partial replacement.

After deduplication, arena exclusions and buffer trimming, the replay assigns
equal total training weight to every self-play game that still has samples.
For `S` retained self-play samples from `G` games, a game with `n` retained
samples gives each sample weight `S / (G * n)`. A short finished game and a
long unfinished game therefore have the same total weight; targets themselves
are unchanged. The total self-play weight remains `S`, preserving its overall
balance with teacher data. Games with no retained samples cannot contribute.

Weights are rebuilt on every replay update, including old replay files whose
self-play rows already have `gameId` (scoped by run and iteration provenance).
Legacy rows without a recoverable game ID
keep their existing weight (default 1); the report counts these as
`replay.weighting.ungroupedSamples`. Teacher rows retain their existing weights;
when absent, the trainer uses the confidence defaults described above.
Explicit weights must be finite and positive.
The trainer uses weighted squared error, normalized by the mean weight across
the whole dataset, so weighting still works with batch size 1. Validation
metrics remain unweighted per-position measurements. No unfinished result is
converted to a draw or loss by this balancing.

## Acceptance and the UI

Default self-play starts come from `examples/matches/training.json`; arena starts
come from `examples/matches/validation.json`, now containing 25 distinct starts.
The separate frozen `examples/matches/transformer-test.json` contains 24 starts
across six fixture families for final evaluation; it is never used for repeated
promotion. Exact overlapping starts are rejected between training, promotion
and final-test suites. Exact arena/test positions are excluded from bootstrap
generation, seed, existing and newly generated replay. These checks cannot remove
similar histories or undo exposure in an older checkpoint. Use `--suite` and `--arena-suite` for custom JSON
files in the same fixture format.

Candidate and incumbent play identical arena starts with colors swapped and
equal search limits. Both games in a pair must finish legally with certified
results and meaningful play. Duplicate or already-terminal starts do not count.
An unfinished game excludes its whole pair; any invalid game vetoes promotion.
The candidate must exceed 50% and meet `--promotion-score`, after at least
`--min-pairs` complete distinct pairs. The default threshold is 55%.

`decision.strengthAssessment` is reported separately from promotion. It uses
distinct color-swapped pair means and a conservative 95% bounded-mean interval,
requires at least 20 pairs, and remains inconclusive when pairs are unfinished,
excluded or invalid, or the interval contains 50%. Its independence and
representative-start assumptions are not guaranteed by a deterministic fixture
suite or repeated candidate selection. It also reports the score range obtained
if every unscored pair favored either engine; this is a sensitivity calculation,
not an adjudicated result.

Arena reports include `summary.completion` and `decision.completion`, also
saved in `latest.json` under `arena`. These show certified game completion,
complete-pair and eligible-pair rates alongside the candidate score, with
counts, denominators, unfinished reasons and a candidate-color breakdown.
Rates use all attempted games or recorded pairs, including unfinished/invalid
ones; they are `null` when nothing was attempted. Skipped starts are listed
separately. Completion rates describe the joint match, not which engine caused
a game to stop. They are diagnostic metrics and do not change the promotion gate.

If too few pairs finish, inspect the recorded reasons and increase `--arena-plies`,
`--nodes`, `--time-ms`, or `--terminal-time-ms` according to the recorded limit,
or supply suitable nonterminal miniature starts. A legal
move retained when an arena search reaches its time or node budget still plays,
including an unscored fallback from an incomplete iteration. Its search record
preserves the cutoff reason and completion status. If no move is available, the
game stays unfinished unless terminal verification proves checkmate or stalemate.
Terminal verification limits also leave games unfinished. The separate fixed-work
match benchmark still stops on time cutoffs. Do not count unfinished games as
draws to force acceptance.

This is a conservative operational gate, not a guarantee of improvement or an
Elo estimate. A small or repeatedly used selection suite can be overfit. The
existing bootstrap checkpoint has already seen at least one default arena start
in teacher training; excluding it from future replay cannot undo that exposure.
Use separate unseen suites for independent strength measurement. Falling replay
loss alone does not establish stronger play.

Promotion preserves the previous model and atomically replaces
`artifacts/transformer/model.pt` by default. The server reloads changed weights
on the next transformer analysis; the Classical option remains available.
Custom `--checkpoint` paths must match the server's `TRANSFORMER_CHECKPOINT` if
you want those weights in the UI. Generation and arena use immutable checkpoint
snapshots. Two runners cannot share either a run directory or active checkpoint.
An external checkpoint change during a cycle prevents promotion; avoid manually
training over the active file while this loop is running.

## Files, stopping and recovery

Under `artifacts/transformer/selfplay/` by default:

- `latest.json`: most recent cycle, data counts, hashes and acceptance decision.
- `replay.jsonl`: bounded training buffer; retained across invocations.
- `run.json`: run identity and next iteration number.
- `previous-model.pt`: previous incumbent after the latest promotion.
- `iteration-00000001/` and subsequent folders: immutable incumbent,
  trained candidate, game traces, sample JSONL files, `train.log`, the exact
  Python command, `arena.json` and `report.json`.

Only the latest `--keep-iterations` managed folders remain. Copy any candidate
or report elsewhere before it ages out. An untouched unrelated folder is not
part of retention. Runs and checkpoints are ignored by Git.

Ctrl+C closes owned inference/search/training processes, saves an interrupted
report, and releases locks. Existing replay survives; completed game files from
an interrupted generation phase remain in its folder but are not automatically
merged on restart. Training or arena failures leave the active model intact.

A forced process kill or machine crash can leave `.selfplay.lock` in the run
directory and/or `model.pt.selfplay-lock/`. Opening the Training page or starting
the runner automatically recovers locks whose recorded owner is confirmed to
have exited. It archives each old lock as `.selfplay.lock.recovered-...`, marks
unfinished iteration reports `interrupted`, and repairs `latest.json`. Completed
games, replay data, checkpoints and the next iteration number are preserved;
the next run starts a new iteration rather than resuming an unfinished game.
Training does not restart automatically.

Recovery checks both the run and checkpoint locks, and preserves live,
unverifiable, malformed or symbolic-link owners. A recovery interrupted after
claiming an old lock also requires manual review. In those cases the error gives
the lock path: verify no runner is active before removing only that named lock.
An abrupt crash immediately after atomic model publication can leave promotion
uncertain; the recovered report explains this. Compare the active model hash
with the recorded candidate hash and inspect its saved arena decision and
`previous.pt`.

To roll back after stopping the loop and current analysis:

```powershell
Copy-Item -Force artifacts/transformer/selfplay/previous-model.pt artifacts/transformer/model.pt
```

To inspect progress from another PowerShell window:

```powershell
Get-Content artifacts/transformer/selfplay/latest.json
Get-Content artifacts/transformer/selfplay/iteration-00000001/train.log -Tail 10
```

The runner prints JSON events after each game and phase, plus a heartbeat during
long training. Search runs in terminable workers with a hard safety deadline;
exploration and full-rules terminal verification use cooperative work/time caps.
Stopping prevents new games from starting, signals all active searches, and
waits for their workers to exit before releasing the run locks. Completed game
records are saved as they finish, including during a cooperative stop.
If a trainer exits while another process still holds its output pipes, the
runner drains output for up to one second before finishing the phase. A trainer
that ignores a stop signal receives a forced termination after two seconds.
Worker exits are tied to their invocation tokens, so the server can recover its
own failed worker without treating the still-running server PID as training.

## Verification

`node --test` covers game legality, outcomes for both colors, target blending,
unfinished games, exploration, invalid-result rejection, cancellation, replay
deduplication/mixing, atomic checkpoint replacement, locks, and the paired gate.
Concurrency tests cover bounded overlap, stable game and arena pair order,
per-game random streams, cancellation, failed record writes and shared inference
queues with separate candidate and incumbent runtimes.
Training UI tests also cover parameter validation, concurrent starts, cooperative
stop and shutdown, external runner locks, history browsing, and read-only legal
replay. These tests use temporary artifacts and fake workers. The optional
`node scripts/browser-training-smoke.cjs` checks the browser controls and mobile
layout with isolated fixtures; it requires Playwright (or `PLAYWRIGHT_MODULE`)
and an installed browser (`CHROME_PATH` can select one).

The implementation was also exercised on this machine's CUDA GPU through actual
self-play, candidate training, rejection of unfinished/tied arenas, and a resumed
run using persisted replay. Four short CUDA cycles completed, plus a deliberate
interruption during training followed by successful restart; retention and
recovery from an empty interrupted iteration folder were checked. None of the
smoke candidates passed promotion, so the active UI checkpoint was preserved.
These are functional checks, not strength claims.
