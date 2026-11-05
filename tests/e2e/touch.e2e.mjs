#!/usr/bin/env node
/**
 * End-to-end test of the iPad / Apple Pencil input model, driving the real
 * app in headless Chromium with synthetic pen and multi-touch input via the
 * Chrome DevTools Protocol (Input.dispatchMouseEvent with pointerType 'pen'
 * and force; Input.dispatchTouchEvent for fingers and palms).
 *
 *   node tests/e2e/touch.e2e.mjs [--headed]
 *
 * Runs the app with ?present=0 (headless SwiftShader cannot present WebGPU
 * canvases and read back at the same time; see README).
 */
import { createServer } from 'vite';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  ({ chromium } = await import('playwright'));
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const server = await createServer({ root, server: { port: 0, host: '127.0.0.1' }, logLevel: 'error', clearScreen: false });
await server.listen();
const port = server.httpServer.address().port;
const browser = await chromium.launch({
  headless: !process.argv.includes('--headed'),
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const context = await browser.newContext({ viewport: { width: 1180, height: 820 }, hasTouch: true, deviceScaleFactor: 2 });
// real iPads report 5 touch points; Playwright's touch emulation reports 1
await context.addInitScript(() => Object.defineProperty(Navigator.prototype, 'maxTouchPoints', { get: () => 5 }));
const page = await context.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
const cdp = await context.newCDPSession(page);

let failed = 0;
let passed = 0;
async function check(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (e) {
    failed++;
    console.log(`  ✗ ${name}\n      ${e.message}`);
  }
}
function assert(c, msg) {
  if (!c) throw new Error(msg);
}

const state = () =>
  page.evaluate(() => {
    const a = window.gradientPaint;
    return {
      depth: a.canvas.undoDepth,
      hsv: [...a.paintHsv],
      view: { ...a.view },
      penSeen: a.penSeen,
      brush: a.brushSize,
      painting: a.painting,
      current: a.current ? { n: a.current.stroke.points.length, p: [...a.current.stroke.pressures], x0: a.current.stroke.points[0]?.[0] } : null,
      mouseInside: a.mouseInside,
      pick: a.pickLatch,
      clone: a.clonePoint,
      mode: a.mode,
    };
  });
const frames = (n = 4) => page.evaluate((n) => new Promise((r) => { let k = 0; const f = () => (++k >= n ? r() : requestAnimationFrame(f)); requestAnimationFrame(f); }), n);

await page.goto(`http://127.0.0.1:${port}/?size=257x193&present=0`);
await page.evaluate(() => localStorage.clear());
await page.reload();
await page.waitForFunction(() => window.gradientPaint?.solver, null, { timeout: 60000 });
await frames();
const box = await page.locator('#overlay').boundingBox();
const v0 = (await state()).view;
const at = (x, y) => [box.x + v0.ox + x * v0.scale, box.y + v0.oy + y * v0.scale];

// --- input primitives -------------------------------------------------------
async function pen(type, [x, y], force = 0.5, buttons = 1) {
  await cdp.send('Input.dispatchMouseEvent', {
    type, x, y, button: type === 'mouseMoved' && !buttons ? 'none' : 'left', buttons, pointerType: 'pen', force, clickCount: 1, tiltX: 20, tiltY: 0,
  });
}
async function penStroke(points, forces) {
  await pen('mousePressed', at(...points[0]), forces[0]);
  for (let i = 1; i < points.length; i++) {
    const [ax, ay] = points[i - 1];
    const [bx, by] = points[i];
    for (let k = 1; k <= 8; k++) {
      const f = forces[i - 1] + ((forces[i] - forces[i - 1]) * k) / 8;
      await pen('mouseMoved', at(ax + ((bx - ax) * k) / 8, ay + ((by - ay) * k) / 8), f);
    }
  }
  await pen('mouseReleased', at(...points[points.length - 1]), 0, 0);
}
const tp = (id, [x, y], r = 8) => ({ x, y, id, radiusX: r, radiusY: r, force: 0 });
async function touches(type, pts) {
  await cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts });
}
async function fingerDrag(from, to, r = 8) {
  await touches('touchStart', [tp(1, at(...from), r)]);
  for (let k = 1; k <= 8; k++) await touches('touchMove', [tp(1, at(from[0] + ((to[0] - from[0]) * k) / 8, from[1] + ((to[1] - from[1]) * k) / 8), r)]);
  await touches('touchEnd', []);
}

console.log('iPad / Apple Pencil input');

await check('touch device detected: dock visible with touch hints', async () => {
  const r = await page.evaluate(() => ({ touch: navigator.maxTouchPoints, dock: getComputedStyle(document.getElementById('dock')).display, finger: document.getElementById('btn-finger').textContent }));
  assert(r.touch > 1, `maxTouchPoints ${r.touch}`);
  assert(r.dock !== 'none', 'dock hidden');
  assert(r.finger === 'finger: auto (draw)', r.finger);
});

await check('before any Pencil use, a finger draws', async () => {
  await fingerDrag([40, 40], [200, 60]);
  await frames();
  const s = await state();
  assert(s.depth === 1 && s.current && s.current.n > 100, `depth ${s.depth}, points ${s.current?.n}`);
  assert(s.current.p.every((p) => Math.abs(p - 0.5) < 1e-9), 'fingers record neutral pressure');
});

await check('Pencil hover shows the brush cursor without drawing', async () => {
  await pen('mouseMoved', at(120, 120), 0, 0);
  await frames(2);
  const s = await state();
  assert(s.mouseInside && !s.painting && s.depth === 1, JSON.stringify({ inside: s.mouseInside, painting: s.painting }));
});

await check('Pencil draws and records pressure per sample', async () => {
  await penStroke([[30, 100], [130, 100], [230, 110]], [0.1, 0.5, 1.0]);
  await frames();
  const s = await state();
  assert(s.penSeen, 'pen not detected');
  assert(s.depth === 2, `depth ${s.depth}`);
  const p = s.current.p;
  assert(Math.min(...p) < 0.25 && Math.max(...p) > 0.8, `pressure range ${Math.min(...p)}..${Math.max(...p)}`);
  assert(p[Math.floor(p.length / 4)] < p[Math.floor((3 * p.length) / 4)], 'pressure should rise along the stroke');
});

await check('pressure changes what is painted (strength mode)', async () => {
  // light vs firm horizontal strokes; compare the integrated step across each
  await page.evaluate(() => {
    const a = window.gradientPaint;
    while (a.canvas.undoDepth) a.undo(true); // back to a blank canvas
  });
  await penStroke([[20, 60], [235, 60]], [0.15, 0.15]);
  await penStroke([[20, 150], [235, 150]], [0.9, 0.9]);
  await page.evaluate(async () => {
    const a = window.gradientPaint;
    a.cyclesPerFrame = 8;
  });
  await frames(30);
  const r = await page.evaluate(async () => {
    const a = window.gradientPaint;
    const d = await a.solver.readSolution();
    const v = (x, y) => d[(y * a.W + x) * 4];
    return { light: v(128, 50) - v(128, 70), firm: v(128, 140) - v(128, 160) };
  });
  assert(r.light > 0.005 && r.firm > 3 * r.light, `light ${r.light.toFixed(4)}, firm ${r.firm.toFixed(4)}`);
});

await check('after the Pencil, a finger picks the colour instead of drawing', async () => {
  await page.waitForTimeout(500); // past the palm-rejection grace period
  const before = await state();
  await fingerDrag([150, 150], [190, 120]);
  await frames();
  const s = await state();
  assert(s.depth === before.depth, `finger drew (depth ${before.depth} -> ${s.depth})`);
  assert(s.hsv[1] > 0.2 && s.hsv[0] !== before.hsv[0], `hsv ${s.hsv}`);
});

await check('palms and touches during Pencil use are ignored', async () => {
  const before = await state();
  await fingerDrag([60, 60], [120, 90], 45); // large contact = palm
  await pen('mousePressed', at(30, 30), 0.5);
  await fingerDrag([60, 60], [120, 90]); // finger while the Pencil is down
  await pen('mouseReleased', at(30, 30), 0, 0);
  await frames();
  const s = await state();
  assert(JSON.stringify(s.hsv) === JSON.stringify(before.hsv), 'palm/finger changed the colour');
  await page.waitForTimeout(500);
});

await check('two fingers pinch-zoom and pan without painting', async () => {
  const before = await state();
  const c = at(128, 96);
  await touches('touchStart', [tp(1, [c[0] - 50, c[1]]), tp(2, [c[0] + 50, c[1]])]);
  for (let k = 1; k <= 10; k++) await touches('touchMove', [tp(1, [c[0] - 50 - 5 * k, c[1] + 2 * k]), tp(2, [c[0] + 50 + 5 * k, c[1] + 2 * k])]);
  await touches('touchEnd', []);
  await frames();
  const s = await state();
  assert(Math.abs(s.view.scale / before.view.scale - 2) < 0.05, `scale ${before.view.scale} -> ${s.view.scale}`);
  assert(s.depth === before.depth, 'gesture painted');
  await page.evaluate(() => window.gradientPaint.fitView());
});

await check('a second finger cancels a stroke the first finger started', async () => {
  await page.evaluate(() => (window.gradientPaint.fingerMode = 'draw'));
  try {
    const before = await state();
    await touches('touchStart', [tp(1, at(40, 40))]);
    for (let k = 1; k <= 4; k++) await touches('touchMove', [tp(1, at(40 + 10 * k, 40))]);
    assert((await state()).depth === before.depth + 1, 'first finger should have started a stroke');
    await touches('touchMove', [tp(1, at(80, 40)), tp(2, at(140, 80))]);
    await touches('touchEnd', []);
    await frames();
    const s = await state();
    assert(s.depth === before.depth, `expected the stroke (only) to be removed: depth ${before.depth} -> ${s.depth}`);
  } finally {
    await page.evaluate(() => (window.gradientPaint.fingerMode = 'auto'));
  }
});

await check('a quick two-finger tap undoes', async () => {
  const before = await state();
  await touches('touchStart', [tp(1, at(100, 100)), tp(2, at(160, 100))]);
  await touches('touchEnd', []);
  await frames();
  const s = await state();
  assert(s.depth === before.depth - 1, `depth ${before.depth} -> ${s.depth}`);
});

await check('dock: Pick latch makes the next Pencil press set the clone source', async () => {
  await page.keyboard.press('ArrowUp'); // Clone
  await page.click('#btn-pick');
  assert((await state()).pick === 'pick', 'latch not armed');
  await pen('mousePressed', at(64, 64), 0.5);
  await pen('mouseReleased', at(64, 64), 0, 0);
  const s = await state();
  assert(s.mode === 'clone' && s.clone && Math.abs(s.clone[0] - 64) < 1 && s.pick === null, JSON.stringify(s.clone));
  await page.waitForTimeout(450);
  await penStroke([[150, 150], [220, 150]], [0.5, 0.5]);
  assert((await state()).depth >= 1, 'clone stroke');
});

await check('dock: Move drags the current stroke', async () => {
  const before = await state();
  await page.click('#btn-move');
  await pen('mousePressed', at(150, 150), 0.5);
  for (let k = 1; k <= 5; k++) await pen('mouseMoved', at(150, 150 - 8 * k), 0.5);
  await pen('mouseReleased', at(150, 110), 0, 0);
  await frames();
  const s = await state();
  assert(s.depth === before.depth, 'moving must not create a stroke');
  assert(Math.abs(s.current.x0 - before.current.x0) < 1e-6, 'x unchanged');
  const ys = await page.evaluate(() => window.gradientPaint.current.stroke.points[0][1]);
  assert(ys < 115, `stroke y ${ys}`);
});

await check('dock: size slider sets the brush size', async () => {
  const track = await page.locator('#size-slider .track').boundingBox();
  await page.touchscreen.tap(track.x + track.width / 2, track.y + 4);
  const big = (await state()).brush;
  await page.touchscreen.tap(track.x + track.width / 2, track.y + track.height - 2);
  const small = (await state()).brush;
  assert(big > 150 && small < 1, `${big} / ${small}`);
});

await check('settings persist (finger & pressure modes, Pencil seen)', async () => {
  await page.click('#btn-pressure');
  const label = await page.textContent('#btn-pressure');
  await page.reload();
  await page.waitForFunction(() => window.gradientPaint?.solver);
  const r = await page.evaluate(() => ({ p: window.gradientPaint.pressureMode, seen: window.gradientPaint.penSeen, label: document.getElementById('btn-finger').textContent }));
  assert(label === 'pressure: size' && r.p === 'size' && r.seen && r.label === 'finger: auto (pick)', JSON.stringify(r));
});

await check('no page errors', async () => assert(errors.length === 0, errors.join('\n')));

console.log(`\n${passed} passed, ${failed} failed`);
await browser.close();
await server.close();
process.exit(failed ? 1 : 0);
