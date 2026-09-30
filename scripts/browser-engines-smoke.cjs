// Optional UI regression check using Playwright. Transformer availability and
// inference metadata are mocked; legal search/play is backed by an isolated server.
// This verifies UI routing, not model quality or GPU inference.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');

(async () => {
  const { createApp } = await import('../src/server.js');
  const server = createApp();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  let browser;
  try {
    browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [], requests = [], neuralJobs = new Set();
    let available = false, holdSearch = true, innerCandidateLimit = 64, legacySearch = false, releasePoll, startedPoll, stoppedSearch, finishedPoll;
    const pollStarted = new Promise(resolve => { startedPoll = resolve; });
    const searchStopped = new Promise(resolve => { stoppedSearch = resolve; });
    const pollFinished = new Promise(resolve => { finishedPoll = resolve; });
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/engines', route => route.fulfill({ json: { engines: [
      {id:'classical', name:'Classical search', available:true},
      {id:'transformer', name:'Transformer', available, status:available ? 'unloaded' : 'setup-required'},
    ] } }));
    await page.route('**/api/analyze', async route => {
      const options = route.request().postDataJSON();
      requests.push(options);
      if (options.engine !== 'transformer') return route.continue();
      if (holdSearch) return route.fulfill({ status:202, json:{jobId:'mock-transformer-pending'} });
      // A real classical job supplies a legal turn for the UI-only neural mock.
      const response = await route.fetch({ postData:JSON.stringify({...options, engine:'classical'}) });
      const data = await response.json();
      neuralJobs.add(data.jobId);
      await route.fulfill({ response, json:data });
    });
    await page.route('**/api/analysis/**', async route => {
      const path = new URL(route.request().url()).pathname;
      if (path === '/api/analysis/mock-transformer-pending/stop') {
        await route.fulfill({ json:{status:'cancelled'} });
        stoppedSearch();
        return;
      }
      if (path === '/api/analysis/mock-transformer-pending') {
        await new Promise(resolve => { releasePoll = resolve; startedPoll(); });
        await route.fulfill({ json:{status:'done', result:{score:999, bestAction:[], notation:'Stale transformer result'}} });
        finishedPoll();
        return;
      }
      if (!neuralJobs.has(path.split('/').at(-1))) return route.continue();
      const response = await route.fetch();
      const data = await response.json();
      if (data.result) Object.assign(data.result, {
        engine:'transformer', searchPolicy:legacySearch ? 'transformer-bounded-beam' : 'transformer-ranked-depth',
        expansionRank:3, depth:5, pvDepth:2,
        depthStats:[{depth:1,candidates:60,trueEvaluations:4,searchedMoves:4,topCandidateRank:5}, {depth:2,candidates:0,trueEvaluations:3,searchedMoves:null,topCandidateRank:null}],
        contextTruncated:true, frontierTruncated:true,
        limits:{...data.result.limits, candidateLimit:64, innerCandidateLimit, ...(legacySearch ? {beamWidth:4} : {})},
        model:{device:'cuda:0', config:{max_tokens:1024}},
      });
      await route.fulfill({ response, json:data });
    });

    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.locator('#connection.online').waitFor();
    assert.equal(await page.locator('#engine-select').inputValue(), 'classical');
    assert.deepEqual(await page.locator('#engine-select option').evaluateAll(options => options.map(option => option.value)), ['classical', 'transformer']);
    assert.equal(await page.locator('#search-depth option[value="64"]').evaluate(option => option.disabled), true);
    await page.locator('#engine-select').selectOption('transformer');
    await page.locator('#engine-readiness').filter({hasText:'setup required'}).waitFor();
    assert.equal(await page.locator('#analyze-button').isDisabled(), true);
    assert.equal(await page.locator('#cache-memory').isDisabled(), true);
    await page.locator('#search-depth').selectOption('64');
    await page.locator('#transformer-setup summary').click();
    assert.match(await page.locator('#transformer-setup').textContent(), /npm run transformer:setup/);
    assert.match(await page.locator('#engine-description').textContent(), /strength is unmeasured/);
    await page.reload();
    await page.locator('#connection.online').waitFor();
    assert.equal(await page.locator('#engine-select').inputValue(), 'transformer', 'Selected engine persists on reload.');
    assert.equal(await page.locator('#search-depth').inputValue(), '64', 'Deep transformer limit persists on reload.');
    assert.equal(await page.locator('#analyze-button').isDisabled(), true);
    await page.locator('#engine-select').selectOption('classical');
    await page.waitForFunction(() => !document.getElementById('analyze-button').disabled);
    assert.equal(await page.locator('#cache-memory').isDisabled(), false);
    assert.equal(await page.locator('#search-depth').inputValue(), '16', 'Classical selection keeps its supported depth limit.');

    available = true;
    await page.locator('#refresh-engines').click();
    await page.waitForFunction(() => !document.getElementById('refresh-engines').disabled);
    await page.locator('#engine-select').selectOption('transformer');
    await page.locator('#engine-readiness').filter({hasText:'Checkpoint ready'}).waitFor();
    await page.locator('#analyze-button').click();
    await pollStarted;
    assert.equal(requests.at(-1).engine, 'transformer');
    assert.equal(requests.at(-1).cacheMemoryMb, 0);
    assert.equal(await page.locator('#engine-select').isDisabled(), false, 'Engine can be switched during analysis.');
    await page.locator('#engine-select').selectOption('classical');
    await searchStopped;
    releasePoll();
    await pollFinished;
    await page.waitForTimeout(200);
    assert.equal(await page.locator('#eval-score').textContent(), '—', 'Cancelled engine results cannot repopulate the UI.');
    assert.equal(await page.locator('#best-move').isVisible(), false);
    assert.equal(await page.locator('#play-button').isDisabled(), true);

    holdSearch = false;
    await page.locator('#engine-select').selectOption('transformer');
    await page.locator('#time-budget').selectOption('1000');
    await page.locator('#search-depth').selectOption('2');
    await page.locator('#node-budget').fill('700');
    await page.locator('#analyze-button').click();
    await page.waitForFunction(() => !document.getElementById('play-button').disabled, null, {timeout:15000});
    assert.match(await page.locator('#stat-cache').textContent(), /cuda:0/);
    const note = await page.locator('#analysis-note').textContent();
    assert.match(note, /Transformer · 800k ranked depth search; up to 64 candidate turns per position/);
    assert.match(note, /Shared evaluated ranks: 3/);
    assert.match(note, /Searched moves by depth: 1: 4 · 2: no candidates/);
    assert.match(await page.locator('#stat-depth').getAttribute('title'), /Deepest true evaluation: 5 turns.*Current best line: 2 turns/);
    assert.doesNotMatch(note, /no full depth completed/);
    assert.doesNotMatch(note, /best 4 deepened/);
    assert.match(note, /1024 tokens/);
    assert.match(note, /Historical context was truncated/);
    assert.match(note, /current-board features were omitted/);
    innerCandidateLimit = 16;
    await page.locator('#analyze-button').click();
    await page.locator('#analysis-note').filter({hasText:'Transformer · 800k ranked depth search; up to 64 root / 16 reply candidate turns'}).waitFor();
    await page.waitForFunction(() => !document.getElementById('play-button').disabled);
    legacySearch = true;
    await page.locator('#analyze-button').click();
    await page.locator('#analysis-note').filter({hasText:'Selective transformer search; up to 64 root / 16 reply candidate turns; best 4 deepened'}).waitFor();
    legacySearch = false;
    await page.locator('#engine-select').selectOption('classical');
    await page.waitForFunction(() => document.getElementById('best-move').hidden);
    assert.equal(await page.locator('#play-button').isDisabled(), true, 'Switching clears completed recommendations.');

    await page.locator('#engine-select').selectOption('transformer');
    await page.locator('#ai-side').selectOption('black');
    await page.getByRole('button', {name:'(0L T1) e2, white pawn', exact:true}).click();
    await page.getByRole('button', {name:'(0L T1) e4, empty, available destination', exact:true}).click();
    await page.waitForFunction(() => !document.getElementById('submit-button').disabled);
    await page.locator('#submit-button').click();
    await page.locator('#history-count').filter({hasText:'2 turns'}).waitFor({timeout:15000});
    assert.equal(requests.at(-1).engine, 'transformer', 'Automatic replies use the selected engine.');
    assert.equal(await page.locator('#turn-label').textContent(), 'White to play');
    await page.locator('#ai-side').selectOption('off');
    available = false;
    await page.locator('#refresh-engines').click();
    await page.locator('#engine-readiness').filter({hasText:'setup required'}).waitFor();
    await page.locator('#transformer-setup summary').click();
    await fs.mkdir('artifacts', {recursive:true});
    await page.screenshot({path:'artifacts/engines-desktop.png', fullPage:true});
    await page.setViewportSize({width:390, height:844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'Engine UI fits mobile width.');
    await page.screenshot({path:'artifacts/engines-mobile.png', fullPage:true});
    assert.deepEqual(errors, []);
    console.log('Engine UI smoke passed: readiness, setup, persistence, routing, cancellation, stale results, metadata, automatic replies, mobile width. Neural inference is mocked.');
  } finally {
    if (browser) await browser.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
