import childProcess from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { normalizeHeuristics } from './heuristics.js';

export const MATE_SCORE = 100_000;
export const DEFAULT_CLASSICAL_ENGINE_PATH = fileURLToPath(new URL(
  `../rust/classical-engine/target/release/classical-engine${process.platform === 'win32' ? '.exe' : ''}`,
  import.meta.url,
));

const finite = (value, fallback, min, max) => Number.isFinite(Number(value))
  ? Math.max(min, Math.min(max, Number(value))) : fallback;

/** Search complete turns in the native engine. Callbacks remain in Node; only
 * JSON position data and validated search settings cross the process boundary.
 */
export async function analyze(position, options = {}) {
  const configured = normalizeHeuristics(options.heuristics);
  const quiescenceDepth = options.quiescenceDepth === undefined ? configured.quiescenceDepth
    : Math.floor(finite(options.quiescenceDepth, 2, 0, 8));
  const heuristics = quiescenceDepth === configured.quiescenceDepth ? configured
    : normalizeHeuristics({ ...configured, quiescenceDepth });
  const timeMs = finite(options.timeMs, 3000, 0, 3_600_000);
  const settings = {
    timeMs,
    unlimitedTime: options.unlimitedTime === true || timeMs === 0,
    maxDepth: Math.floor(finite(options.maxDepth, 8, 1, 64)),
    maxNodes: Math.floor(finite(options.maxNodes, 2_000_000, 0, 1_000_000_000)),
    threads: Math.floor(finite(options.threads, 1, 1, 16)),
    cacheMemoryMb: finite(options.cacheMemoryMb, 128, 0, 4096),
    maxTableEntries: Math.floor(finite(options.maxTableEntries, 100_000, 0, 1_000_000)),
    mateProbeWork: Math.floor(finite(options.mateProbeWork, 4096, 0, 1_000_000)),
    quiescenceDepth,
    heuristics,
  };
  if (options.quiescenceWarmup !== undefined) settings.quiescenceWarmup = options.quiescenceWarmup !== false;
  // Serialize before starting a child so invalid/circular input cannot orphan it.
  const request = JSON.stringify({ type: 'analyze', position, options: settings }) + '\n';
  const executable = process.env.CLASSICAL_ENGINE_PATH || DEFAULT_CLASSICAL_ENGINE_PATH;

  return new Promise((resolve, reject) => {
    const child = childProcess.spawn(executable, [], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
    let settled = false, result, stdout = '', stderr = '', cancellationSent = false;
    let pollTimer, stopTimer, exitTimer;
    const cleanup = () => {
      clearInterval(pollTimer);
      clearTimeout(stopTimer);
      clearTimeout(exitTimer);
      process.off('exit', kill);
    };
    const kill = () => { if (child.exitCode === null) child.kill(); };
    const fail = error => {
      if (settled) return;
      settled = true;
      cleanup();
      kill();
      reject(error);
    };
    function receive(line) {
      if (!line.trim() || settled) return;
      try {
        const message = JSON.parse(line);
        if (message.type === 'error') throw new Error(String(message.error || 'Native classical search failed.'));
        if (!['result', 'progress'].includes(message.type) || !message.result || typeof message.result !== 'object') {
          throw new Error('The Rust classical engine returned an invalid protocol message.');
        }
        if (message.type === 'progress') {
          options.onProgress?.({ ...message.result, backend: 'rust' });
        } else {
          if (result) throw new Error('The Rust classical engine returned more than one result.');
          result = { ...message.result, backend: 'rust' };
          clearInterval(pollTimer);
          clearTimeout(stopTimer);
          child.stdin.end();
          // A result ends this one-shot session; do not retain a stuck subprocess.
          exitTimer = setTimeout(kill, 1000);
          exitTimer.unref();
        }
      } catch (error) {
        fail(error instanceof SyntaxError
          ? new Error('The Rust classical engine returned invalid JSON.', { cause: error }) : error);
      }
    }
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', chunk => {
      stdout += chunk;
      let end;
      while ((end = stdout.indexOf('\n')) !== -1) {
        const line = stdout.slice(0, end);
        stdout = stdout.slice(end + 1);
        receive(line);
      }
    });
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-8192); });
    child.on('error', error => fail(error.code === 'ENOENT'
      ? new Error('Rust classical engine not found. Install Rust and run "npm run build:classical" (or set CLASSICAL_ENGINE_PATH).', { cause: error })
      : new Error(`Cannot start Rust classical engine: ${error.message}`, { cause: error })));
    child.stdin.on('error', error => {
      // A child can close stdin immediately after writing its final message.
      // Let close report a missing result if that was an unexpected exit.
      if (error.code !== 'EPIPE' && error.code !== 'ERR_STREAM_DESTROYED') fail(error);
    });
    child.on('close', (code, signal) => {
      if (stdout.trim()) receive(stdout);
      if (settled) return;
      settled = true;
      cleanup();
      if (result) resolve(result);
      else reject(new Error(`Rust classical engine exited without a result (${signal || code}).${stderr.trim() ? ` ${stderr.trim()}` : ''}`));
    });
    const checkCancellation = () => {
      try {
        if (settled || result || cancellationSent || !options.shouldStop?.()) return;
        cancellationSent = true;
        child.stdin.write('{"type":"cancel"}\n');
        stopTimer = setTimeout(() => fail(new Error('Rust classical engine did not stop within the cancellation grace period.')), 2000);
        stopTimer.unref();
      } catch (error) { fail(error); }
    };
    process.once('exit', kill);
    child.stdin.write(request);
    pollTimer = setInterval(checkCancellation, 20);
    pollTimer.unref();
    checkCancellation();
  });
}
