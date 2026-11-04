import { requestGpuDevice } from '../../src/solver/gpu';
import { runAll, TestResult } from './harness';
import './solver.gpu';
import './paint.gpu';

declare global {
  interface Window {
    __gpuTests?: { done: boolean; results: TestResult[]; adapter: string; fatal?: string };
  }
}

async function main() {
  const state: NonNullable<Window['__gpuTests']> = { done: false, results: [], adapter: '' };
  window.__gpuTests = state;
  const summary = document.getElementById('summary')!;
  try {
    const device = await requestGpuDevice();
    device.addEventListener('uncapturederror', (e) => console.error('uncaptured WebGPU error', (e as GPUUncapturedErrorEvent).error.message));
    const info = device.adapterInfo;
    state.adapter = info ? `${info.vendor} ${info.architecture} ${info.description}`.trim() : 'unknown';
    document.getElementById('adapter')!.textContent = `adapter: ${state.adapter}`;
    state.results = await runAll(device, document.getElementById('out')!);
    const failed = state.results.filter((r) => !r.ok).length;
    summary.textContent = `${state.results.length - failed} passed, ${failed} failed`;
    summary.className = failed ? 'fail' : 'pass';
  } catch (e) {
    state.fatal = e instanceof Error ? e.message : String(e);
    summary.textContent = `fatal: ${state.fatal}`;
    summary.className = 'fail';
  }
  state.done = true;
}
main();
