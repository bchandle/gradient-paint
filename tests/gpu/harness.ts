/**
 * A tiny in-browser test harness for the WebGPU suite. Results are rendered
 * into the page and exposed on `window.__gpuTests` for the headless runner
 * (scripts/run-gpu-tests.mjs). Open tests/gpu/index.html in any
 * WebGPU-capable browser to run the suite on real hardware.
 */

export interface TestResult {
  name: string;
  ok: boolean;
  ms: number;
  error?: string;
  notes: string[];
}

interface TestCase {
  name: string;
  fn: (ctx: TestContext) => Promise<void> | void;
}

export interface TestContext {
  device: GPUDevice;
  note(msg: string): void;
}

const tests: TestCase[] = [];
let prefix: string[] = [];

export function describe(name: string, body: () => void): void {
  prefix.push(name);
  body();
  prefix.pop();
}

export function test(name: string, fn: TestCase['fn']): void {
  tests.push({ name: [...prefix, name].join(' › '), fn });
}

export class AssertionError extends Error {}

export function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new AssertionError(msg);
}

export function assertLess(a: number, b: number, what: string): void {
  if (!(a < b)) throw new AssertionError(`${what}: expected ${a} < ${b}`);
}

export function assertClose(actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number, what: string, relative = false): number {
  if (actual.length !== expected.length) throw new AssertionError(`${what}: length ${actual.length} != ${expected.length}`);
  let worst = 0;
  let worstAt = -1;
  let scale = 1;
  if (relative) {
    scale = 0;
    for (let i = 0; i < expected.length; i++) scale = Math.max(scale, Math.abs(expected[i]));
    scale = Math.max(scale, 1e-30);
  }
  for (let i = 0; i < actual.length; i++) {
    const d = Math.abs(actual[i] - expected[i]) / scale;
    if (!(d <= worst)) {
      worst = d;
      worstAt = i;
    }
  }
  if (!(worst <= tol)) {
    throw new AssertionError(
      `${what}: max ${relative ? 'relative ' : ''}error ${worst.toExponential(3)} > ${tol} at index ${worstAt} (got ${actual[worstAt]}, expected ${expected[worstAt]})`,
    );
  }
  return worst;
}

export async function runAll(device: GPUDevice, out: HTMLElement): Promise<TestResult[]> {
  const results: TestResult[] = [];
  const filter = new URLSearchParams(location.search).get('grep');
  for (const t of tests) {
    if (filter && !t.name.includes(filter)) continue;
    const notes: string[] = [];
    const t0 = performance.now();
    const row = document.createElement('div');
    row.className = 'row running';
    row.textContent = `… ${t.name}`;
    out.appendChild(row);
    let error: string | undefined;
    device.pushErrorScope('validation');
    try {
      await t.fn({ device, note: (m) => notes.push(m) });
      await device.queue.onSubmittedWorkDone();
    } catch (e) {
      error = e instanceof Error ? `${e.message}` : String(e);
    }
    const gpuErr = await device.popErrorScope();
    if (gpuErr && !error) error = `WebGPU validation error: ${gpuErr.message}`;
    const r: TestResult = { name: t.name, ok: !error, ms: performance.now() - t0, error, notes };
    results.push(r);
    row.className = `row ${r.ok ? 'pass' : 'fail'}`;
    row.textContent = `${r.ok ? '✓' : '✗'} ${t.name} (${r.ms.toFixed(0)} ms)${notes.length ? ' — ' + notes.join('; ') : ''}${error ? '\n    ' + error : ''}`;
  }
  return results;
}
