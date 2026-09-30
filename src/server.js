import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { randomUUID } from 'node:crypto';
import { GameSession } from './session.js';
import { TransformerRuntime, listEngines, forwardInference } from './transformer-runtime.js';
import { LeelaRuntime } from './leela-runtime.js';
import { isNeuralEngine } from './leela-config.js';
import { TrainingManager } from './training-manager.js';
import { HEURISTIC_SETTINGS, DEFAULT_HEURISTICS, normalizeHeuristics } from './heuristics.js';
import { inspectEvaluation } from './evaluate.js';

const PUBLIC = new URL('../public/', import.meta.url);
const staticFiles = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/heuristics.js', ['heuristics.js', 'text/javascript; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
  ['/training', ['training.html', 'text/html; charset=utf-8']],
  ['/training.html', ['training.html', 'text/html; charset=utf-8']],
  ['/training.js', ['training.js', 'text/javascript; charset=utf-8']],
  ['/training.css', ['training.css', 'text/css; charset=utf-8']],
]);

function numericOption(value, fallback, min, max, name) {
  const number = value === undefined ? fallback : Number(value);
  const validType = value === undefined || typeof value === 'number' || (typeof value === 'string' && value.trim() !== '');
  if (!validType || !Number.isFinite(number) || !Number.isInteger(number) || number < min || number > max) {
    throw new Error(`${name} must be an integer between ${min} and ${max}.`);
  }
  return number;
}

async function readBody(req) {
  if (!req.headers['content-type']?.startsWith('application/json')) {
    throw new Error('Requests must use application/json.');
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 512 * 1024) throw new Error('Request exceeds the 512 KB limit.');
    chunks.push(chunk);
  }
  const value = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected a JSON object.');
  return value;
}

export function createApp({ transformerRuntime = new TransformerRuntime(), leelaRuntime = new LeelaRuntime(),
  trainingManager = new TrainingManager() } = {}) {
  const game = new GameSession();
  const jobs = new Map();
  let shuttingDown = false;
  const stopJobs = () => {
    for (const job of jobs.values()) {
      if (job.status === 'running') Atomics.store(job.cancelled, 0, 1);
    }
  };
  const publicJob = job => ({
    jobId: job.id, status: job.status, revision: job.revision,
    progress: job.progress, result: job.result, error: job.error,
  });
  const send = (res, status, value) => {
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(value));
  };

  const server = http.createServer(async (req, res) => {
    try {
      const host = req.headers.host ?? '';
      if (!/^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(host)) {
        return send(res, 403, { error: 'This service accepts local requests only.' });
      }
      if (req.headers.origin && ![`http://${host}`, `https://${host}`].includes(req.headers.origin)) {
        return send(res, 403, { error: 'Cross-origin requests are not allowed.' });
      }
      const url = new URL(req.url, `http://${host}`);
      res.setHeader('X-Content-Type-Options', 'nosniff');
      res.setHeader('Referrer-Policy', 'no-referrer');
      res.setHeader('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'");
      if (req.method === 'GET' && staticFiles.has(url.pathname)) {
        const [name, type] = staticFiles.get(url.pathname);
        const content = await readFile(new URL(name, PUBLIC));
        res.writeHead(200, { 'Content-Type': type, 'Cache-Control': 'no-cache' });
        return res.end(content);
      }
      if (req.method === 'GET' && url.pathname === '/api/game') return send(res, 200, game.snapshot());
      if (req.method === 'GET' && url.pathname === '/api/heuristics') {
        return send(res, 200, { settings: HEURISTIC_SETTINGS, defaults: DEFAULT_HEURISTICS });
      }
      if (req.method === 'GET' && url.pathname === '/api/engines') return send(res, 200, listEngines(transformerRuntime));
      if (req.method === 'GET' && url.pathname === '/api/training') return send(res, 200, await trainingManager.snapshot());
      const trainingGame = /^\/api\/training\/iterations\/([^/]+)\/games\/([^/]+)$/.exec(url.pathname);
      if (req.method === 'GET' && trainingGame) {
        return send(res, 200, await trainingManager.getGame(trainingGame[1], trainingGame[2], url.searchParams.get('ply') ?? 0));
      }
      const trainingIteration = /^\/api\/training\/iterations\/([^/]+)$/.exec(url.pathname);
      if (req.method === 'GET' && trainingIteration) return send(res, 200, await trainingManager.getIteration(trainingIteration[1]));
      const jobMatch = /^\/api\/analysis\/([a-zA-Z0-9-]+)(\/stop)?$/.exec(url.pathname);
      if (jobMatch) {
        const job = jobs.get(jobMatch[1]);
        if (!job) return send(res, 404, { error: 'Analysis job not found.' });
        if (req.method === 'POST' && jobMatch[2]) {
          await readBody(req);
          Atomics.store(job.cancelled, 0, 1);
          return send(res, 200, publicJob(job));
        }
        if (req.method === 'GET' && !jobMatch[2]) return send(res, 200, publicJob(job));
      }
      if (req.method !== 'POST') return send(res, 404, { error: 'Not found.' });
      const body = await readBody(req);
      switch (url.pathname) {
        case '/api/evaluate': {
          game.assertRevision(body.revision);
          const heuristics = normalizeHeuristics(body.heuristics);
          return send(res, 200, {
            revision: game.revision, heuristics,
            evaluation: inspectEvaluation(game.position, heuristics),
          });
        }
        case '/api/training/fresh/start':
          if (Object.keys(body).some(key => key !== 'options')) throw new Error('Only training options may be supplied.');
          return send(res, 202, await trainingManager.startFresh(body.options));
        case '/api/training/start':
          if (Object.keys(body).some(key => key !== 'options')) throw new Error('Only training options may be supplied.');
          return send(res, 202, await trainingManager.start(body.options));
        case '/api/training/stop':
          if (Object.keys(body).length) throw new Error('Stopping training does not accept options.');
          return send(res, 200, await trainingManager.stop());
        case '/api/new':
          if (body.variant !== undefined && !game.chess.variants.some(v => v.shortName === body.variant && v.shortName !== 'custom')) {
            throw new Error('Unknown board variant.');
          }
          game.reset({ variant: body.variant });
          stopJobs();
          break;
        case '/api/import':
          if (typeof body.pgn !== 'string' || !body.pgn.trim()) throw new Error('Enter a 5DPGN game or 5DFEN position.');
          game.reset({ pgn: body.pgn });
          stopJobs();
          break;
        case '/api/move':
          game.assertRevision(body.revision);
          game.move(body.move);
          stopJobs();
          break;
        case '/api/submit':
          game.assertRevision(body.revision);
          game.submit();
          stopJobs();
          break;
        case '/api/undo':
          game.undo();
          stopJobs();
          break;
        case '/api/analyze': {
          const engine = body.engine ?? 'classical';
          if (!['classical', 'transformer', 'leela'].includes(engine)) throw new Error('Unknown engine. Choose classical, transformer, or leela.');
          const neural = isNeuralEngine(engine);
          const heuristics = neural ? undefined : normalizeHeuristics(body.heuristics);
          const runtime = engine === 'leela' ? leelaRuntime : transformerRuntime;
          const timeMs = numericOption(body.timeMs, 3000, 0, 120000, 'Think time');
          if (timeMs > 0 && timeMs < 50) throw new Error('Think time must be 0 (infinite) or an integer between 50 and 120000.');
          const options = {
            engine,
            timeMs,
            unlimitedTime: timeMs === 0,
            maxDepth: numericOption(body.maxDepth, 8, neural ? 0 : 1, neural ? 64 : 16, 'Depth'),
            maxNodes: numericOption(body.maxNodes, 2000000, 1, 1000000000, 'Node budget'),
            threads: numericOption(body.threads, 1, 1, 16, 'Search threads'),
            cacheMemoryMb: numericOption(body.cacheMemoryMb, 128, 0, 4096, 'Cache memory'),
            maxTableEntries: 1000000,
            quiescenceDepth: numericOption(body.quiescenceDepth, heuristics?.quiescenceDepth ?? 2, 0, 8, 'Quiescence depth'),
            ...(heuristics ? { heuristics } : {}),
          };
          const revision = game.revision;
          const modelInfo = neural ? await runtime.start() : null;
          if (shuttingDown) throw new Error('The local server is shutting down.');
          if (res.destroyed) return;
          game.assertRevision(revision);
          stopJobs();
          // A bounded cache retains completed results for Play best and inspection.
          for (const [id, old] of jobs) {
            if (jobs.size < 12) break;
            if (old.status !== 'running') jobs.delete(id);
          }
          if ([...jobs.values()].filter(j => j.status === 'running').length >= 2) {
            return send(res, 429, { error: 'Previous searches are stopping. Try again in a moment.' });
          }
          const cancelled = new Int32Array(new SharedArrayBuffer(4));
          const job = { id: randomUUID(), status: 'running', revision: game.revision, cancelled };
          const worker = new Worker(new URL('./worker.js', import.meta.url), {
            workerData: { position: game.position, options, model: modelInfo?.model, cancelBuffer: cancelled.buffer },
          });
          job.worker = worker;
          jobs.set(job.id, job);
          // Finite searches also guard non-interruptible upstream move generation.
          const hardDeadline = options.unlimitedTime ? undefined : setTimeout(() => {
            if (job.status !== 'running') return;
            job.result = job.progress ? { ...job.progress, stoppedReason: 'hard-time-limit' } : undefined;
            job.status = job.result?.bestAction ? 'done' : 'cancelled';
            job.error = 'Hard time limit reached; showing the last completed search result.';
            void worker.terminate();
          }, options.timeMs + 5000);
          job.deadline = hardDeadline;
          hardDeadline?.unref();
          worker.on('message', message => {
            if (job.status !== 'running') return;
            if (message.type === 'evaluate' || message.type === 'policy') { void forwardInference(worker, runtime, message); return; }
            if (message.type === 'progress') job.progress = message.result;
            if (message.type === 'result') { job.result = message.result; job.status = 'done'; clearTimeout(hardDeadline); }
            if (message.type === 'error') { job.error = message.error; job.status = 'error'; clearTimeout(hardDeadline); }
          });
          worker.on('error', error => { if (job.status === 'running') { job.error = error.message; job.status = 'error'; } clearTimeout(hardDeadline); });
          worker.on('exit', code => {
            clearTimeout(hardDeadline);
            if (job.status === 'running') { job.status = 'error'; job.error = `Search worker exited (${code}).`; }
          });
          return send(res, 202, { jobId: job.id });
        }
        case '/api/play': {
          game.assertRevision(body.revision);
          const job = jobs.get(body.jobId);
          if (!job || job.revision !== game.revision) throw new Error('This analysis belongs to a different position.');
          if (job.status !== 'done' || !Array.isArray(job.result?.bestAction)) throw new Error('No completed legal recommendation is available.');
          game.play(job.result.bestAction);
          stopJobs();
          break;
        }
        default: return send(res, 404, { error: 'Not found.' });
      }
      return send(res, 200, game.snapshot());
    } catch (error) {
      if (!res.headersSent) send(res, error.statusCode ?? 400, { error: error.message });
      else res.end();
    }
  });
  const shutdown = () => {
    if (shuttingDown) return;
    shuttingDown = true;
    for (const job of jobs.values()) {
      Atomics.store(job.cancelled, 0, 1);
      clearTimeout(job.deadline);
      if (job.status === 'running') job.status = 'cancelled';
      void job.worker.terminate();
    }
    transformerRuntime.close();
    leelaRuntime.close();
    void trainingManager.close();
  };
  // `close` waits for active HTTP requests. Reject pending model startup first,
  // so an analysis awaiting startup cannot keep server shutdown open for a minute.
  const close = server.close;
  server.close = function (...args) { shutdown(); return close.apply(this, args); };
  server.on('close', shutdown);
  return server;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const port = Number(process.env.PORT ?? 5173);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('PORT must be a valid local port.');
  const app = createApp();
  app.listen(port, '127.0.0.1', () => console.log(`Vibe-D AI is ready at http://127.0.0.1:${app.address().port}`));
  app.on('error', error => { console.error(error.message); process.exitCode = 1; });
  const shutdown = () => { app.close(); app.closeAllConnections(); };
  process.once('SIGINT', shutdown);
  process.once('SIGTERM', shutdown);
}
