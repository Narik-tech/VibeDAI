# Rust classical engine

The app and command line use the native engine in `rust/classical-engine`. Node
retains the HTTP server, notation, position editing, and neural engine runtimes.
`src/rust-engine.js` transfers the complete position history to a Rust subprocess
and streams its search results back as newline-delimited JSON.

## Build and run

Install Node.js 22+ and the stable Rust toolchain from <https://rustup.rs>. On
Windows, use the C++ build tools required by the Rust MSVC toolchain. Then run:

```sh
npm ci
npm run build:classical
npm start
```

The build script uses Cargo's release profile and the checked-in lockfile. The
executable is `rust/classical-engine/target/release/classical-engine` (`.exe` on
Windows). `npm start`, `npm run analyze`, `npm run benchmark:classical`, and
`npm test` run an incremental build first. Direct commands such as
`node src/cli.js --time 5 --json` use the existing
executable. Set `CLASSICAL_ENGINE_PATH` to run a different compatible executable.
A missing executable is an explicit setup error.
If Cargo is absent from `PATH`, the build script also recognizes an existing
project-local toolchain under `artifacts/rust-toolchain/{cargo,rustup}`. It sets
the toolchain environment only for that build and does not change your shell.

```sh
npm run test:rust
npm test
```

## JavaScript API and cancellation

```js
import { analyze } from './src/rust-engine.js';
import { createPosition } from './src/rules.js';

let cancelled = false;
const result = await analyze(createPosition(), {
  timeMs: 5000,
  maxNodes: 2_000_000,
  maxDepth: 8,
  heuristics: { materialWeight: 1.2 },
  shouldStop: () => cancelled,
  onProgress: progress => console.log(progress.depth, progress.score),
});
```

The adapter validates heuristic overrides using the same schema as the UI and
sends the complete effective profile. `timeMs: 0` removes the time limit;
depth and node limits still apply. Progress and final results contain
`backend: 'rust'`. Scores use White's perspective and actions retain the existing
raw coordinate format. Each request owns one subprocess, closed before its promise
resolves. Cancellation is cooperative, with a bounded grace period; process
failure rejects the promise with diagnostic information.

Pass `threads: 1` through `16` to select the CPU thread count, including the
coordinator. The engine completes a serial depth-one warmup before dispatching
root branches to persistent native workers. Each worker retains its private
cache across iterations; their combined cache allocation stays within the
requested memory and entry limits. All threads share the deadline, cancellation
signal, and an atomic search/generation work budget.

`limits.threads` records the requested setting and `threadsUsed` reports actual
worker activation. Short, depth-one, or terminal searches can use only one
thread. Root batches accept results in generation order. Work distribution and
interrupted results can still depend on scheduling; more threads do not
guarantee a speedup on every position.

## Native protocol

Write one request line to stdin:

```json
{"type":"analyze","position":{"board":[],"action":0},"options":{"timeMs":5000,"maxDepth":8}}
```

The example shows the envelope; supply a real position, including its entire
sparse timeline history. The process writes zero or more
`{"type":"progress","result":{...}}` lines, followed by
`{"type":"result","result":{...}}`. Errors use
`{"type":"error","error":"message"}`. Stderr is reserved for diagnostics.
Send `{"type":"cancel"}` on another input line to stop the search. Closing stdin
also requests cancellation, so an abandoned caller does not leave search running.

Diagnostic requests for move generation, action generation, application, and
evaluation are available for cross-language regression tests. They do not replace
the public notation and game-session APIs.

## Reference implementation and inspiration

`src/search.js` and `src/parallel-search.js` remain available as JavaScript
reference APIs. Existing training teachers and older benchmark scripts that
import those modules continue to use JavaScript. `npm run benchmark:classical`
uses Rust by default; pass `--engine src/search.js` to select the reference, or
`--baseline src/search.js` to compare them. Historical JavaScript performance
numbers in `docs/performance.md` are not measurements of the Rust engine.
The native benchmark's wall-clock measurements include process startup and
transport; the result's `elapsedMs` measures the native search itself.

## Validation and initial measurements

The native regression tests compare exact move arrays, immutable successor
histories, complete-turn successor sets, and evaluation breakdowns with the
JavaScript implementation. They cover all 12 piece families, both colors,
unmoved flags, sparse/even timeline layouts, special moves, custom weights,
partial turns, all 15 tactical fixtures, and serial/parallel resource limits.
Subprocess tests also verify cancellation and cleanup after worker termination.

Run the native integration tests after building with:

```sh
node --test test/rust-*.test.js
```

On this Windows checkout (Node 22.15.1, Rust 1.99.1), three alternating
fixed-depth runs on October 16, 2026 produced these median wall times:

| Position | Depth / quiescence | JavaScript | Rust |
| --- | --- | ---: | ---: |
| Standard | 4 / 2 | 110 ms | 127 ms |
| Opening | 2 / 2 | 22 ms | 43 ms |
| Two timelines | 2 / 1 | 404 ms | 246 ms |
| Temporal | 2 / 1 | 151 ms | 82 ms |

All four comparisons completed their requested horizons with matching scores;
every returned turn and PV passed JavaScript legality validation. The Rust
measurements include launching a fresh native process. Its current evaluator
recomputes board features instead of reproducing every JavaScript evaluation
cache, and search work counts differ. These measurements show a gain on the
multiverse fixtures and a regression on smaller fixtures; they do not establish
a general speedup or stronger play. Reproduce with:

```sh
node scripts/benchmark-classical.js --baseline src/search.js --mode depth --repeat 3 --warmup 1 --depth-time-ms 10000 --json
```

## Inspiration

The migration draws architectural inspiration from
[FirePlank/5d-chess-engine](https://github.com/FirePlank/5d-chess-engine), including
a native search core with an explicit position representation. This project's
complete-turn search policy, position format, evaluation settings, and Node
application contract remain defined by this repository. Full history participates
in move legality and search identity; frontier-only positions are insufficient
for 5D chess.
