// Optional integration test, using its own game server and browser profile.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');

(async () => {
  const { createApp } = await import('../src/server.js');
  const server = createApp();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [], searches = [], evaluations = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (request.method() !== 'POST') return;
      const path = new URL(request.url()).pathname;
      if (path === '/api/analyze') searches.push(request.postDataJSON());
      if (path === '/api/evaluate') evaluations.push(request.postDataJSON());
    });
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const current = () => page.waitForFunction(() => document.getElementById('heuristics-status').textContent.startsWith('Current position'));
    await current();
    assert.equal(await page.locator('#heuristics-components .heuristic-component').count(), 6);
    assert.ok(await page.locator('.heuristic-control').count() >= 40);
    assert.ok(await page.locator('[data-feature]').count() > 20);
    assert.equal(await page.locator('#heuristics-total').textContent(), '0 cp');

    await page.locator('#notation-button').click();
    await page.locator('#pgn-input').fill('[Size "4x4"]\n[3k/4/4/KQ2:0:1:w]');
    await page.locator('#import-pgn').click();
    await current();
    await page.locator('.heuristics-group > summary').filter({ hasText: 'Piece values' }).click();
    await page.locator('#heuristic-queenValue-number').fill('1500');
    await current();
    assert.equal(await page.locator('#heuristic-queenValue').inputValue(), '1500');
    assert.equal(await page.locator('[data-component="material"] strong').textContent(), '+1,500 cp');
    assert.equal(await page.locator('[data-feature="queenValue"] strong').textContent(), '+1,500 cp');
    assert.equal(evaluations.at(-1).heuristics.queenValue, 1500);
    await page.reload(); await current();
    assert.equal(await page.locator('#heuristic-queenValue-number').inputValue(), '1500');
    await page.locator('#heuristic-materialWeight').focus();
    await page.locator('#heuristic-materialWeight').press('Home');
    await current();
    assert.equal(await page.locator('#heuristic-materialWeight-number').inputValue(), '0');
    assert.equal(await page.locator('[data-component="material"] strong').textContent(), '0 cp');
    await page.locator('#heuristics-reset').click(); await current();
    assert.equal(await page.locator('#heuristic-queenValue-number').inputValue(), '1150');
    assert.equal(await page.locator('#heuristics-profile').textContent(), 'DEFAULT WEIGHTS');

    await page.locator('#new-button').click(); await current();
    await page.locator('.heuristics-group > summary').filter({ hasText: 'Piece values' }).click();
    await page.locator('#heuristic-queenValue-number').fill('1500'); await current();
    await page.locator('.heuristics-group > summary').filter({ hasText: 'Search' }).click();
    await page.locator('#heuristic-quiescenceDepth-number').fill('0'); await current();
    await page.locator('#search-depth').selectOption('1');
    await page.locator('#time-budget').selectOption('1000');
    await page.locator('#node-budget').fill('100000');
    const started = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/analyze');
    await page.locator('#analyze-button').click();
    const { jobId } = await (await started).json();
    await page.waitForFunction(() => !document.getElementById('play-button').disabled);
    const job = await page.request.get(new URL(`/api/analysis/${jobId}`, page.url()).href).then(response => response.json());
    assert.equal(job.result.heuristics.queenValue, 1500);
    assert.equal(job.result.limits.quiescenceDepth, 0);
    assert.equal(searches.at(-1).heuristics.queenValue, 1500);
    await page.locator('#heuristic-queenValue-number').fill('1600'); await current();
    assert.equal(await page.locator('#play-button').isDisabled(), true, 'A changed profile invalidates the recommendation.');

    await page.locator('#time-budget').selectOption('0');
    await page.locator('#search-depth').selectOption('16');
    await page.locator('#node-budget').fill('1000000000');
    await page.locator('#analyze-button').click();
    await page.waitForFunction(() => document.getElementById('analyze-label').textContent === 'Stop analysis');
    await page.locator('#heuristic-queenValue-number').fill('1700'); await current();
    assert.equal(await page.locator('#play-button').isDisabled(), true);
    assert.equal(await page.locator('#analyze-label').textContent(), 'Analyze position');
    await page.locator('#engine-select').selectOption('transformer');
    assert.equal(await page.locator('#heuristics-content').isHidden(), true);
    assert.equal(await page.locator('#heuristics-engine-note').isVisible(), true);
    await page.locator('#engine-select').selectOption('classical'); await current();
    assert.equal(await page.locator('#heuristic-queenValue-number').inputValue(), '1700');

    await page.locator('#time-budget').selectOption('1000');
    await page.locator('#search-depth').selectOption('1');
    await page.locator('#node-budget').fill('1000');
    await page.locator('#ai-side').selectOption('black');
    const beforeMove = evaluations.at(-1).revision;
    const movedEvaluation = page.waitForResponse(response => new URL(response.url()).pathname === '/api/evaluate'
      && response.request().postDataJSON().revision > beforeMove);
    await page.getByRole('button', { name: '(0L T1) e2, white pawn', exact: true }).click();
    await page.getByRole('button', { name: '(0L T1) e4, empty, available destination', exact: true }).click();
    await movedEvaluation;
    await current();
    assert.ok(evaluations.at(-1).revision > beforeMove, 'Partial-turn moves refresh the current-position evaluation.');
    await page.locator('#submit-button').click();
    await page.waitForFunction(() => document.getElementById('history-count').textContent === '2 turns', null, { timeout: 15000 });
    await current();
    assert.equal(searches.at(-1).heuristics.queenValue, 1700, 'Automatic replies use the tuned profile.');
    const afterReply = evaluations.at(-1).revision;
    const undoneEvaluation = page.waitForResponse(response => new URL(response.url()).pathname === '/api/evaluate'
      && response.request().postDataJSON().revision > afterReply);
    await page.locator('#undo-button').click(); await undoneEvaluation; await current();
    assert.ok(evaluations.at(-1).revision > afterReply, 'Undo refreshes the breakdown.');

    await fs.mkdir('artifacts', { recursive: true });
    await page.locator('.heuristics-group > summary').filter({ hasText: 'Piece values' }).click();
    await page.locator('.heuristics-group > summary').filter({ hasText: 'Search' }).click();
    await page.locator('#heuristics-panel').evaluate(element => element.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: 'artifacts/heuristics-desktop.png' });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'Mobile page must not overflow.');
    for (const id of ['heuristic-materialWeight-number', 'heuristic-activityWeight-number']) {
      const bounds = await page.locator(`#${id}`).boundingBox();
      assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= 390, `${id} fits a mobile viewport.`);
    }
    await page.locator('#heuristics-panel').evaluate(element => element.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: 'artifacts/heuristics-mobile.png' });
    await page.locator('.heuristics-tuning-heading').evaluate(element => element.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: 'artifacts/heuristics-mobile-controls.png' });
    assert.deepEqual(errors, []);
    console.log('Heuristics browser smoke passed: detailed contributions, live tuning, defaults, persistence, worker propagation, stale recommendation invalidation, running-search cancellation, engine switching, automatic replies, position refresh, and mobile layout.');
  } finally {
    if (browser) await browser.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
