#!/usr/bin/env node
/**
 * Runs the WebGPU test-suite (tests/gpu) in headless Chromium.
 *
 * Uses Playwright's Chromium with WebGPU enabled. On machines without a GPU,
 * Chromium falls back to SwiftShader (a CPU Vulkan implementation), so the
 * suite exercises the real WebGPU/WGSL code paths everywhere; on real
 * hardware run `npm run dev` and open /tests/gpu/index.html instead.
 *
 *   node scripts/run-gpu-tests.mjs [--grep substring] [--headed]
 *
 * Env: CHROMIUM_PATH to use a specific browser binary.
 */
import { createServer } from 'vite';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  ({ chromium } = await import('playwright'));
}

const args = process.argv.slice(2);
const grepIdx = args.indexOf('--grep');
const grep = grepIdx >= 0 ? args[grepIdx + 1] : null;
const headed = args.includes('--headed');

const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error', clearScreen: false });
await server.listen();
const addr = server.httpServer.address();
const url = `http://127.0.0.1:${addr.port}/tests/gpu/index.html${grep ? `?grep=${encodeURIComponent(grep)}` : ''}`;

const browser = await chromium.launch({
  headless: !headed,
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
});
let exitCode = 1;
try {
  const page = await browser.newPage();
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning') console.log(`[browser ${m.type()}] ${m.text()}`);
  });
  page.on('pageerror', (e) => console.log(`[page error] ${e.message}`));
  await page.goto(url);
  const t0 = Date.now();
  let printed = 0;
  for (;;) {
    const state = await page.evaluate(() => window.__gpuTests && { done: window.__gpuTests.done, results: window.__gpuTests.results, adapter: window.__gpuTests.adapter, fatal: window.__gpuTests.fatal });
    if (state?.adapter && printed === 0) console.log(`WebGPU adapter: ${state.adapter}\n`);
    if (state?.done) {
      if (state.fatal) {
        console.error(`FATAL: ${state.fatal}`);
        break;
      }
      for (const r of state.results) {
        console.log(`${r.ok ? '  ✓' : '  ✗'} ${r.name} (${r.ms.toFixed(0)} ms)${r.notes.length ? `\n      ${r.notes.join('\n      ')}` : ''}`);
        if (!r.ok) console.log(`      ${r.error}`);
      }
      const failed = state.results.filter((r) => !r.ok).length;
      console.log(`\n${state.results.length - failed} passed, ${failed} failed (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
      exitCode = failed || state.results.length === 0 ? 1 : 0;
      break;
    }
    printed = 1;
    if (Date.now() - t0 > 15 * 60 * 1000) {
      console.error('timeout');
      break;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
} finally {
  await browser.close();
  await server.close();
}
process.exit(exitCode);
