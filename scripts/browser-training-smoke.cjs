// Optional browser integration check. Uses a temporary history and fake training
// worker: running this check cannot train or replace the user's checkpoint.
// Set PLAYWRIGHT_MODULE / CHROME_PATH to use an existing development runtime.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');

async function createHistory(root) {
  const { createPosition, parseMove, validateAction, formatAction, positionKey } = await import('../src/rules.js');
  const initialPosition = createPosition();
  let position = initialPosition;
  const moves = [];
  for (const [ply, notation] of ['e4', 'e5'].entries()) {
    const action = [parseMove(position, notation)];
    const next = validateAction(position, action);
    moves.push({ ply, color: position.action % 2, action, notation: formatAction(position, action),
      beforeKey: positionKey(position), afterKey: positionKey(next),
      exploration: { explored: ply === 0 },
      search: { score: ply ? -20 : 35, completed: true, depth: 2, nodes: 100, stoppedReason: 'depth' } });
    position = next;
  }
  const game = { gameId: 'browser-smoke', startId: 'standard', result: 'UNFINISHED', valid: true,
    reason: 'ply-limit', plies: moves.length, samples: 2, initialPosition, finalPosition: position, moves };
  // The upper timeline is far ahead; the present board is on a lower row.
  // Review must scroll to an actual board rather than showing an empty cell.
  const sparsePosition = { ...initialPosition, board: [
    [...Array(20).fill(null), initialPosition.board[0][0]], [initialPosition.board[0][0]],
  ] };
  const folder = path.join(root, 'iteration-00000001');
  await fs.mkdir(folder, { recursive: true });
  const report = { runId: 'browser-smoke', iteration: 1, status: 'complete', promoted: false,
    startedAt: '2026-09-23T12:00:00.000Z', finishedAt: '2026-09-23T12:01:00.000Z',
    options: { games: 2, steps: 16, batchSize: 8, arenaPairs: 2, arenaConcurrency: 2, minPairs: 1 },
    selfplay: { games: 1, finished: 0, unfinished: 1, samples: 2 }, replay: { samples: 2 },
    arena: { decision: { promote: false, reason: 'insufficient-complete-distinct-pairs',
      candidateScore: null, eligiblePairs: 0, minPairs: 1 } } };
  await Promise.all([
    fs.writeFile(path.join(folder, 'report.json'), JSON.stringify(report)),
    fs.writeFile(path.join(folder, 'iteration.json'), JSON.stringify(report)),
    fs.writeFile(path.join(root, 'latest.json'), JSON.stringify(report)),
    fs.writeFile(path.join(folder, 'selfplay-001.json'), JSON.stringify(game)),
    fs.writeFile(path.join(folder, 'selfplay-002.json'), JSON.stringify({ ...game,
      gameId: 'sparse', startId: 'sparse-timelines', moves: [], plies: 0, samples: 0,
      initialPosition: sparsePosition, finalPosition: sparsePosition })),
    fs.writeFile(path.join(folder, 'arena-001.json'), JSON.stringify({ ...game, gameId: undefined, startId: undefined,
      caseId: 'standard-arena', aColor: 1, engines: { A: 'candidate', B: 'incumbent' } })),
    fs.writeFile(path.join(folder, 'train.log'), '{"event":"train","step":16,"loss":0.12}\n'),
  ]);
  return game;
}

(async () => {
  const { createApp } = await import('../src/server.js');
  const { TrainingManager } = await import('../src/training-manager.js');
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'vibe-training-browser-'));
  const runDir = path.join(directory, 'selfplay');
  const game = await createHistory(runDir);
  const started = [];
  const modes = [];
  const workers = [];
  let available = true;
  const manager = new TrainingManager({ runDir, checkpoint: path.join(directory, 'model.pt'),
    python: path.join(directory, 'python'),
    availability: async ({ mode }) => ({ available:mode === 'fresh20m' || available, reason:available ? undefined : 'Test environment needs a trained checkpoint.' }),
    workerFactory: args => {
      started.push(args.options);
      modes.push(args.mode);
      const worker = new EventEmitter();
      workers.push(worker);
      worker.postMessage = message => {
        if (message.type === 'stop') setTimeout(() => {
          worker.emit('message', { type: 'complete', state: 'interrupted' });
          worker.emit('exit', 0);
        }, 30);
      };
      worker.terminate = async () => { worker.emit('exit', 1); return 1; };
      setTimeout(() => worker.emit('message', { type: 'event', event: args.mode === 'fresh20m' ? {
        event:'data-progress',samples:8,total:64,
      } : { event: 'selfplay-start', iteration: 2, device: 'cpu', trainedSteps: 16 } }), 20);
      return worker;
    },
  });
  const server = createApp({ trainingManager: manager });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  const liveGameBefore = await (await fetch(`${base}/api/game`)).json();
  let browser;
  try {
    browser = await chromium.launch({ headless: true,
      ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    await page.addInitScript(() => {
      if (!localStorage.getItem('vibe-d-ai.training-settings.v1')) localStorage.setItem('vibe-d-ai.training-settings.v1',JSON.stringify({batchSize:8,steps:16}));
    });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`${base}/training`);
    await page.getByRole('heading', { name: /training/i }).first().waitFor();
    await page.locator('#param-samples').waitFor();
    await page.waitForFunction(() => !document.getElementById('start-training').disabled);
    assert.equal(await page.locator('#training-mode').inputValue(), 'fresh20m');
    assert.equal(await page.locator('#param-batchSize').inputValue(), '1', 'Fresh defaults must ignore legacy self-play settings.');
    assert.equal(await page.locator('#param-maxTokens').inputValue(), '512');
    assert.equal(await page.locator('#param-samples').inputValue(), '4096');
    assert.match(await page.locator('#model-summary').innerText(), /20,000,257/);
    assert.equal(started.length, 0, 'Opening the page must not start training.');
    await page.locator('#training-mode').selectOption('selfplay20m');
    assert.equal(await page.locator('#start-training').isDisabled(), true, '20M self-play needs a completed fresh model.');
    await page.locator('#training-mode').selectOption('fresh20m');
    await page.locator('#param-samples').fill('0');
    await page.locator('#start-training').click();
    assert.equal(started.length, 0, 'Invalid fresh settings must not start training.');
    await page.locator('#param-samples').fill('64');
    await page.locator('#param-steps').fill('3');
    await page.locator('#param-steps').blur();
    await page.reload();
    await page.waitForFunction(() => !document.getElementById('start-training').disabled);
    assert.equal(await page.locator('#param-samples').inputValue(), '64');
    assert.equal(await page.locator('#param-steps').inputValue(), '3');
    await page.locator('#start-training').click();
    await page.waitForFunction(() => !document.getElementById('stop-training').disabled);
    assert.equal(started.length, 1);
    assert.equal(modes[0], 'fresh20m');
    assert.equal(started[0].samples, 64);
    assert.equal(started[0].batchSize, 1);
    assert.equal(started[0].maxTokens, 512);
    assert.equal(await page.locator('#training-mode').isDisabled(), true, 'One worker locks all run modes.');
    await page.locator('#refresh-training').click();
    await page.locator('#run-detail').filter({ hasText: /8.*positions/ }).waitFor();
    workers[0].emit('message', { type:'event', event:{event:'training-progress',step:2,loss:.2345} });
    await page.locator('#refresh-training').click();
    await page.locator('#run-detail').filter({ hasText: /update 2.*loss 0\.2345/ }).waitFor();
    assert.equal(await page.locator('#current-iteration').innerText(), '2');
    assert.equal(await page.locator('.pipeline [data-phase="training"]').getAttribute('aria-current'), 'step');
    await page.locator('#stop-training').click();
    await page.locator('#run-state').filter({ hasText: /interrupted/i }).waitFor();
    await page.waitForFunction(() => !document.getElementById('training-mode').disabled);
    // Publish an isolated fake checkpoint to exercise selection without running Python.
    await fs.mkdir(started[0].runDir, { recursive:true });
    await Promise.all([
      fs.writeFile(started[0].checkpoint, 'fake 20M checkpoint'),
      fs.writeFile(path.join(started[0].runDir, 'training.jsonl'), '{}\n'),
      fs.writeFile(path.join(started[0].runDir, 'report.json'), JSON.stringify({
        id:started[0].runId,status:'complete',startedAt:new Date().toISOString(),finishedAt:new Date().toISOString(),
        parameters:20000257,trainedSteps:3,loss:.12,
      })),
    ]);
    await page.locator('#refresh-training').click();
    await page.locator('#model20m-state').filter({ hasText: 'READY FOR SELF-PLAY' }).waitFor();
    assert.match(await page.locator('#fresh-runs').innerText(), /20,000,257 parameters/);
    assert.match(await page.locator('#fresh-runs').innerText(), /model\.pt/);
    await page.locator('#training-mode').selectOption('selfplay20m');
    assert.equal(await page.locator('#param-batchSize').inputValue(), '1');
    assert.equal(await page.locator('#param-maxTokens').inputValue(), '512');
    await page.locator('#start-training').click();
    await page.waitForFunction(() => !document.getElementById('stop-training').disabled);
    assert.equal(started.length, 2);
    assert.equal(started[1].model, '20m');
    assert.equal(started[1].checkpoint, started[0].checkpoint);
    await page.locator('#stop-training').click();
    await page.locator('#run-state').filter({ hasText: /interrupted/i }).waitFor();
    await page.waitForFunction(() => !document.getElementById('training-mode').disabled);
    await page.locator('#training-mode').selectOption('current');
    assert.equal(await page.locator('#param-batchSize').inputValue(), '8');
    const previousRuns = started.length;
    await page.locator('.game-item').first().waitFor();
    await page.locator('.game-item[data-game-id="selfplay-001"]').click();
    await page.locator('#review-content').waitFor({ state: 'visible' });
    await page.locator('#ply-label').filter({ hasText: '0 / 2' }).waitFor();
    assert.ok(await page.locator('#review-grid .board-card').count());
    assert.equal(await page.locator('#first-ply').isDisabled(), true);
    await page.locator('#next-ply').click();
    await page.locator('#ply-label').filter({ hasText: '1 / 2' }).waitFor();
    await page.locator('#last-ply').click();
    await page.locator('#ply-label').filter({ hasText: '2 / 2' }).waitFor();
    assert.equal(await page.locator('#next-ply').isDisabled(), true);
    await page.locator('#first-ply').click();
    await page.locator('#ply-label').filter({ hasText: '0 / 2' }).waitFor();
    await page.locator('#game-kind').selectOption('arena');
    assert.equal(await page.locator('.game-item').count(), 1);
    await page.locator('.game-item').click();
    await page.locator('#review-game-title').filter({ hasText: 'standard-arena' }).waitFor();
    await page.locator('#game-outcome').selectOption('finished');
    assert.equal(await page.locator('.game-item').count(), 0, 'Unfinished games must not be listed as finished.');
    await page.locator('#game-outcome').selectOption('all');
    await page.locator('#game-kind').selectOption('all');
    await page.locator('#game-search').fill('missing-case');
    assert.equal(await page.locator('.game-item').count(), 0);
    await page.locator('#game-search').fill('');
    assert.equal(await page.locator('.game-item').count(), 3);
    await page.locator('.game-item[data-game-id="selfplay-002"]').click();
    await page.locator('#review-game-title').filter({ hasText: 'sparse-timelines' }).waitFor();
    await page.waitForFunction(() => {
      const viewport = document.getElementById('review-viewport').getBoundingClientRect();
      return [...document.querySelectorAll('#review-grid .board-card')].some(card => {
        const rect = card.getBoundingClientRect();
        return Math.min(rect.right, viewport.right) - Math.max(rect.left, viewport.left) > 100
          && Math.min(rect.bottom, viewport.bottom) - Math.max(rect.top, viewport.top) > 100;
      });
    });

    const fillParameter = async (name, value) => {
      const field = page.locator(`#param-${name}`);
      const details = page.locator('details').filter({ has: field });
      for (let index = 0; index < await details.count(); index++) {
        if (!(await details.nth(index).evaluate(node => node.open))) await details.nth(index).locator('summary').first().click();
      }
      await field.fill(value);
      await field.blur();
    };
    await fillParameter('games', '0');
    await page.locator('#start-training').click();
    assert.equal(started.length, previousRuns, 'Invalid configuration cannot start training.');
    await fillParameter('games', '3');
    assert.equal(await page.locator('#param-arenaConcurrency').inputValue(), '1');
    await fillParameter('arenaConcurrency', '9');
    await page.locator('#start-training').click();
    assert.equal(started.length, previousRuns, 'Invalid arena concurrency cannot start training.');
    await fillParameter('arenaConcurrency', '4');
    await fillParameter('steps', '17');
    await fillParameter('learningRate', '0.0005');
    await page.reload();
    await page.waitForFunction(() => !document.getElementById('start-training').disabled);
    await page.locator('#training-mode').selectOption('current');
    assert.equal(await page.locator('#param-games').inputValue(), '3');
    assert.equal(await page.locator('#param-arenaConcurrency').inputValue(), '4');
    assert.equal(await page.locator('#param-steps').inputValue(), '17');
    assert.equal(Number(await page.locator('#param-learningRate').inputValue()), 0.0005);
    await page.locator('#start-training').click();
    await page.waitForFunction(() => !document.getElementById('stop-training').disabled);
    assert.equal(started.length, previousRuns + 1);
    assert.equal(started[previousRuns].games, 3);
    assert.equal(started[previousRuns].arenaConcurrency, 4);
    assert.equal(started[previousRuns].steps, 17);
    assert.equal(started[previousRuns].learningRate, 0.0005);
    assert.equal(await page.locator('#start-training').isDisabled(), true);
    await page.locator('#stop-training').click();
    await page.locator('#run-state').filter({ hasText: /interrupted/i }).waitFor();
    await page.waitForFunction(() => !document.getElementById('start-training').disabled);
    await page.locator('#reuse-parameters').click();
    assert.equal(await page.locator('#param-steps').inputValue(), '16');
    assert.equal(await page.locator('#param-arenaConcurrency').inputValue(), '2');
    assert.equal(started.length, previousRuns + 1, 'Reusing parameters must not start a run.');
    await page.locator('.game-item[data-game-id="selfplay-001"]').click();
    await page.locator('#review-content').waitFor({ state: 'visible' });
    await page.locator('#training-mode').selectOption('fresh20m');
    await fs.mkdir('artifacts', { recursive: true });
    await page.screenshot({ path: 'artifacts/training-desktop.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'Training workspace must fit mobile width.');
    await page.screenshot({ path: 'artifacts/training-mobile.png', fullPage: true });
    await page.locator('#training-mode').selectOption('current');
    available = false;
    await page.locator('#refresh-training').click();
    await page.locator('#run-hint').filter({ hasText: /trained checkpoint/i }).waitFor();
    assert.equal(await page.locator('#start-training').isDisabled(), true);
    await page.locator('#training-mode').selectOption('fresh20m');
    assert.equal(await page.locator('#start-training').isDisabled(), false, 'Fresh training must remain available without a current checkpoint.');
    assert.equal(game.moves.length, 2);
    assert.deepEqual(await (await fetch(`${base}/api/game`)).json(), liveGameBefore);
    assert.deepEqual(errors, []);
    console.log('Training browser smoke passed: fresh 20M defaults, separate saved settings, validation, fresh start/stop and progress, checkpoint history and 20M selection, self-play/arena replay, filters, parameter reuse, availability, read-only review, 390px layout.');
    console.log('Screenshots: artifacts/training-desktop.png, artifacts/training-mobile.png');
  } finally {
    if (browser) await browser.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    await manager.close();
    const safeParent = path.resolve(os.tmpdir());
    assert.equal(path.dirname(path.resolve(directory)), safeParent);
    assert.ok(path.basename(directory).startsWith('vibe-training-browser-'));
    await fs.rm(directory, { recursive: true, force: true });
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
