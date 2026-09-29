// Optional browser integration check. Install Playwright separately or provide
// PLAYWRIGHT_MODULE and CHROME_PATH for an existing development runtime.
// Uses its own ephemeral server, so the user's game is never modified.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');

(async () => {
  const { createApp } = await import('../src/server.js');
  const { DEFAULT_HEURISTICS } = await import('../src/heuristics.js');
  const server = createApp();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  let browser;
  try {
    browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/analyze') requests.push(request.postDataJSON());
    });
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.locator('#connection.online').waitFor();

    for (const value of ['0', '', '1.5', '1000000001']) {
      await page.locator('#node-budget').fill(value);
      const before = requests.length;
      await page.locator('#analyze-button').click();
      assert.equal(await page.locator('#node-budget').evaluate(input => input.validity.valid), false);
      await page.waitForTimeout(150);
      assert.equal(requests.length, before, `Invalid node budget ${JSON.stringify(value)} must not start analysis.`);
    }

    await page.locator('#node-budget').fill('50');
    await page.locator('#cache-memory').selectOption('16');
    await page.locator('#search-threads').selectOption('2');
    await page.locator('#time-budget').selectOption('60000');
    await page.locator('#search-depth').selectOption('12');
    await page.reload();
    await page.locator('#connection.online').waitFor();
    for (const [id, expected] of Object.entries({ 'node-budget': '50', 'cache-memory': '16', 'search-threads': '2', 'time-budget': '60000', 'search-depth': '12' })) {
      assert.equal(await page.locator(`#${id}`).inputValue(), expected, `${id} must persist on reload.`);
    }
    await page.locator('#engine-select').selectOption('transformer');
    await page.waitForFunction(() => document.getElementById('search-threads').disabled);
    assert.equal(await page.locator('#cache-memory').isDisabled(), true);
    assert.match(await page.locator('#search-threads-help').textContent(), /Classical search only/);
    await page.locator('#engine-select').selectOption('classical');
    await page.waitForFunction(() => !document.getElementById('search-threads').disabled);
    assert.equal(await page.locator('#search-threads').inputValue(), '2', 'Switching engines preserves classical thread selection.');

    async function analyze() {
      const responsePromise = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/analyze');
      await page.locator('#analyze-button').click();
      const response = await responsePromise;
      assert.equal(response.status(), 202);
      const { jobId } = await response.json();
      await page.waitForFunction(() => document.getElementById('analyze-label').textContent === 'Stop analysis');
      assert.equal(await page.evaluate(() => ['time-budget', 'search-depth', 'node-budget', 'cache-memory', 'search-threads'].every(id => document.getElementById(id).disabled)), true, 'Resource settings must be disabled while analysis runs.');
      await page.waitForFunction(() => document.getElementById('analyze-label').textContent === 'Analyze position', null, { timeout: 15000 });
      const job = await page.request.get(new URL(`/api/analysis/${jobId}`, page.url()).href).then(response => response.json());
      assert.equal(job.status, 'done');
      return job.result;
    }

    const limited = await analyze();
    assert.deepEqual(requests.at(-1), { engine: 'classical', timeMs: 60000, maxDepth: 12, maxNodes: 50, cacheMemoryMb: 16, threads: 2, heuristics: DEFAULT_HEURISTICS });
    assert.equal(limited.limits.threads, 2);
    assert.equal(limited.limits.maxNodes, 50);
    assert.equal(limited.limits.cacheMemoryMb, 16);
    assert.equal(limited.stoppedReason, 'nodes');
    assert.ok(limited.nodes <= 50, 'Search must respect its node budget.');
    assert.ok(limited.cacheMemoryBytes <= 16 * 1048576, 'Retained cache must respect its memory budget.');
    assert.equal(await page.locator('#stat-node-budget').textContent(), 'Max nodes: 50');
    assert.match(await page.locator('#stat-cache').textContent(), /^Cache: ≈[\d.]+ \/ 16 MiB$/);
    assert.match(await page.locator('#analysis-note').textContent(), /Node limit reached/);
    await fs.mkdir('artifacts', { recursive: true });
    await page.screenshot({ path: 'artifacts/resources-desktop.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'Page must fit a 390px mobile viewport.');
    for (const id of ['node-budget', 'cache-memory', 'search-threads']) {
      const bounds = await page.locator(`#${id}`).boundingBox();
      assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= 390, `${id} must fit mobile width.`);
    }
    await page.screenshot({ path: 'artifacts/resources-mobile.png', fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1100 });

    // Changing the next search's settings must preserve the completed result's budget.
    await page.locator('#node-budget').fill('200');
    await page.locator('#cache-memory').selectOption('32');
    assert.equal(await page.locator('#stat-node-budget').textContent(), 'Max nodes: 50');
    assert.match(await page.locator('#stat-cache').textContent(), /\/ 16 MiB$/);
    await page.locator('#cache-memory').selectOption('0');
    const uncached = await analyze();
    assert.equal(requests.at(-1).cacheMemoryMb, 0);
    assert.equal(uncached.limits.cacheMemoryMb, 0);
    assert.equal(uncached.cacheMemoryBytes, 0);
    assert.equal(await page.locator('#stat-cache').textContent(), 'Cache: off');
    assert.equal(await page.locator('#stat-node-budget').textContent(), 'Max nodes: 200');

    await page.locator('#node-budget').fill('700');
    await page.locator('#cache-memory').selectOption('32');
    await page.locator('#time-budget').selectOption('1000');
    await page.locator('#search-depth').selectOption('2');
    await page.locator('#ai-side').selectOption('black');
    await page.getByRole('button', { name: '(0L T1) e2, white pawn', exact: true }).click();
    await page.getByRole('button', { name: '(0L T1) e4, empty, available destination', exact: true }).click();
    await page.waitForFunction(() => !document.getElementById('submit-button').disabled);
    await page.locator('#submit-button').click();
    await page.locator('#history-count').filter({ hasText: '2 turns' }).waitFor({ timeout: 15000 });
    assert.deepEqual(requests.at(-1), { engine: 'classical', timeMs: 1000, maxDepth: 2, maxNodes: 700, cacheMemoryMb: 32, threads: 2, heuristics: DEFAULT_HEURISTICS }, 'Automatic engine replies must use the chosen resource settings.');
    assert.equal(await page.locator('#turn-label').textContent(), 'White to play');
    assert.deepEqual(errors, []);
    console.log('Resource browser smoke passed: node validation, persisted settings including threads, submitted limits, controls disabled during analysis/transformer selection, node-limit result, retained result budgets, cache off, automatic reply, 390px mobile width.');
    console.log('Screenshots: artifacts/resources-desktop.png, artifacts/resources-mobile.png');
  } finally {
    if (browser) await browser.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
