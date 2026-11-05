/**
 * WebGPU solver tests: every kernel against the CPU reference, full V-cycles
 * against the reference solver, and convergence of the GPU integrator.
 */
import {
  createFloatTexture,
  GpuSolver,
  KernelName,
  KernelParams,
  makeDispatch,
  makeParamsBuffer,
  readBuffer,
  runDispatch,
} from '../../src/solver/gpu';
import { REDUCE_GROUPS } from '../../src/solver/kernels';
import {
  channelMeans,
  computeRhs,
  Field,
  gradientsOf,
  interpolate,
  ReferenceSolver,
  rmsDifferenceModuloMean,
  residual,
  restrict,
  smooth,
} from '../../src/solver/reference';
import { buildLevels, Coarsening } from '../../src/solver/spec';
import { exactSolve, randomField, syntheticImage } from '../unit/exact';
import { assert, assertClose, assertLess, describe, test } from './harness';

// ---------------------------------------------------------------------------
// helpers

/** Round a float64 field to f32 so CPU and GPU see identical inputs. */
const f32 = (f: Field) => new Field(f.width, f.height, new Float32Array(f.data));

function upload(device: GPUDevice, f: Field): GPUBuffer {
  const buf = device.createBuffer({
    size: f.data.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, new Float32Array(f.data));
  return buf;
}

function emptyBuf(device: GPUDevice, w: number, h: number): GPUBuffer {
  return device.createBuffer({
    size: Math.max(16, w * h * 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

async function runKernel(
  device: GPUDevice,
  name: KernelName,
  bindings: Partial<Record<0 | 1 | 2, GPUBuffer | GPUTextureView>>,
  params: KernelParams,
  groups: [number, number],
): Promise<void> {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  runDispatch(pass, makeDispatch(device, name, bindings, makeParamsBuffer(device, params), groups));
  pass.end();
  device.queue.submit([enc.finish()]);
}

const g16 = (w: number, h: number): [number, number] => [Math.ceil(w / 16), Math.ceil(h / 16)];

async function readField(device: GPUDevice, buf: GPUBuffer, w: number, h: number): Promise<Float32Array> {
  return (await readBuffer(device, buf, w * h * 16)).slice(0, w * h * 4);
}

const SIZES: [number, number][] = [
  [2, 2], [3, 2], [5, 7], [16, 16], [17, 17], [33, 20], [64, 48], [100, 37], [129, 65],
];

// ---------------------------------------------------------------------------

describe('kernels vs CPU reference', () => {
  test('rhs (Eq. 6) from gradient textures', async ({ device }) => {
    for (const [W, H] of SIZES) {
      const gx = f32(randomField(W - 1, H, W));
      const gy = f32(randomField(W, H - 1, H + 100));
      const gxt = createFloatTexture(device, W - 1, H, new Float32Array(gx.data));
      const gyt = createFloatTexture(device, W, H - 1, new Float32Array(gy.data));
      const out = emptyBuf(device, W, H);
      await runKernel(device, 'rhs', { 0: gxt.createView(), 1: gyt.createView(), 2: out }, { w: W, h: H }, g16(W, H));
      assertClose(await readField(device, out, W, H), computeRhs(gx, gy, W, H).data, 1e-6, `rhs ${W}x${H}`);
    }
  });

  test('residual on every level stencil', async ({ device }) => {
    for (const [W, H] of SIZES) {
      for (const lev of buildLevels(W, H).slice(0, 4)) {
        const w = lev.width;
        const h = lev.height;
        const f = f32(randomField(w, h, 1));
        const r = f32(randomField(w, h, 2));
        const out = emptyBuf(device, w, h);
        await runKernel(device, 'residual', { 0: upload(device, f), 1: upload(device, r), 2: out }, { w, h, middle: lev.middle, edge: lev.edge, corner: lev.corner }, g16(w, h));
        assertClose(await readField(device, out, w, h), residual(f, r, lev).data, 2e-6, `residual ${w}x${h} L${lev.index}`);
      }
    }
  });

  test('magic-x Jacobi smoother on every level stencil', async ({ device }) => {
    for (const [W, H] of SIZES) {
      for (const lev of buildLevels(W, H).slice(0, 4)) {
        const w = lev.width;
        const h = lev.height;
        const f = f32(randomField(w, h, 3));
        const r = f32(randomField(w, h, 4));
        for (const x of [lev.xA, lev.xB]) {
          const out = emptyBuf(device, w, h);
          await runKernel(device, 'smooth', { 0: upload(device, f), 1: upload(device, r), 2: out }, { w, h, middle: lev.middle, edge: lev.edge, corner: lev.corner, x }, g16(w, h));
          assertClose(await readField(device, out, w, h), smooth(f, r, lev, x).data, 2e-6, `smooth ${w}x${h} L${lev.index}`);
        }
      }
    }
  });

  test('restriction (paper and padded coarsening)', async ({ device }) => {
    for (const mode of ['paper', 'padded'] as Coarsening[]) {
      for (const [W, H] of SIZES) {
        const [l0, l1] = buildLevels(W, H, mode);
        const f = f32(randomField(W, H, 5));
        const out = emptyBuf(device, l1.width, l1.height);
        await runKernel(device, 'restrict', { 0: upload(device, f), 2: out }, { w: W, h: H, cw: l1.width, ch: l1.height }, g16(l1.width, l1.height));
        assertClose(await readField(device, out, l1.width, l1.height), restrict(f, l1.width, l1.height).data, 2e-6, `restrict ${mode} ${l0.width}x${l0.height}`);
      }
    }
  });

  test('interpolation and interpolate-add', async ({ device }) => {
    for (const mode of ['paper', 'padded'] as Coarsening[]) {
      for (const [W, H] of SIZES) {
        const [, l1] = buildLevels(W, H, mode);
        const c = f32(randomField(l1.width, l1.height, 6));
        const add = f32(randomField(W, H, 7));
        const out = emptyBuf(device, W, H);
        const params = { w: W, h: H, cw: l1.width, ch: l1.height };
        await runKernel(device, 'interpolate', { 0: upload(device, c), 2: out }, params, g16(W, H));
        assertClose(await readField(device, out, W, H), interpolate(c, W, H).data, 1e-6, `interpolate ${mode} ${W}x${H}`);
        await runKernel(device, 'interpolateAdd', { 0: upload(device, c), 1: upload(device, add), 2: out }, params, g16(W, H));
        assertClose(await readField(device, out, W, H), interpolate(c, W, H, add).data, 2e-6, `interpolateAdd ${mode} ${W}x${H}`);
      }
    }
  });

  test('exact mean reduction + recentering', async ({ device }) => {
    for (const [W, H] of [[2, 2], [7, 3], [300, 200], [1031, 517]]) {
      const f = f32(randomField(W, H, 8));
      for (let i = 0; i < f.data.length; i += 4) f.data[i] += 3; // non-zero mean in red
      const src = upload(device, f);
      const partial = emptyBuf(device, REDUCE_GROUPS, 1);
      const offset = emptyBuf(device, 1, 1);
      const out = emptyBuf(device, W, H);
      const target = [0.25, -0.5, 0.75];
      const tgt = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(tgt, 0, new Float32Array([...target, 0]));
      await runKernel(device, 'reducePartial', { 0: src, 2: partial }, { w: W, h: H }, [REDUCE_GROUPS, 1]);
      await runKernel(device, 'reduceFinal', { 0: partial, 1: tgt, 2: offset }, { w: W, h: H }, [1, 1]);
      await runKernel(device, 'recenter', { 0: src, 1: offset, 2: out }, { w: W, h: H }, g16(W, H));
      const m = channelMeans(f);
      const off = await readField(device, offset, 1, 1);
      assertClose(Array.from(off).slice(0, 3), target.map((t, c) => t - m[c]), 1e-5, `offset ${W}x${H}`);
      const res = new Field(W, H, await readField(device, out, W, H));
      assertClose(channelMeans(res).slice(0, 3), target, 1e-5, `recentred mean ${W}x${H}`);
    }
  });
});

describe('full V-cycle vs reference solver', () => {
  test('matches CPU V-cycles step by step', async ({ device, note }) => {
    let worst = 0;
    for (const mode of ['paper', 'padded'] as Coarsening[]) {
      for (const [W, H] of [[2, 2], [5, 7], [33, 33], [64, 48], [100, 37], [130, 66]]) {
        const img = syntheticImage(W, H, W + H);
        const { gx, gy } = gradientsOf(img);
        const g = { gx: f32(gx), gy: f32(gy) };
        const gpu = new GpuSolver(device, W, H, { coarsening: mode });
        gpu.setGradientTextures(createFloatTexture(device, W - 1, H, g.gx.toFloat32()), createFloatTexture(device, W, H - 1, g.gy.toFloat32()));
        const cpu = new ReferenceSolver(W, H, mode);
        cpu.setGradients(g.gx, g.gy);
        const mean = channelMeans(img);
        cpu.targetMean = mean;
        gpu.setTargetMean(mean);
        const init = f32(randomField(W, H, 3, 0.5));
        cpu.f = init.clone();
        gpu.writeSolution(init.toFloat32());
        gpu.run(1, true);
        cpu.vcycle();
        worst = Math.max(worst, assertClose(await gpu.readSolution(), cpu.f.data, 5e-5, `1 cycle ${mode} ${W}x${H}`));
        gpu.run(3);
        for (let k = 0; k < 3; k++) cpu.vcycle();
        worst = Math.max(worst, assertClose(await gpu.readSolution(), cpu.f.data, 5e-5, `4 cycles ${mode} ${W}x${H}`));
        gpu.destroy();
      }
    }
    note(`max |gpu - cpu| = ${worst.toExponential(2)}`);
  });
});

// ---------------------------------------------------------------------------

async function gpuConvergence(device: GPUDevice, W: number, H: number, cycles: number) {
  const img = syntheticImage(W, H, 42);
  const { gx, gy } = gradientsOf(img);
  const s = new GpuSolver(device, W, H);
  s.setGradientTextures(createFloatTexture(device, W - 1, H, gx.toFloat32()), createFloatTexture(device, W, H - 1, gy.toFloat32()));
  s.setTargetMean(channelMeans(img));
  s.writeSolution(randomField(W, H, 9, 0.5).toFloat32());
  const errs: number[] = [];
  s.run(0, true);
  const initial = rmsDifferenceModuloMean(new Field(W, H, await s.readSolution()), img);
  for (let k = 0; k < cycles; k++) {
    s.run(1);
    errs.push(rmsDifferenceModuloMean(new Field(W, H, await s.readSolution()), img));
  }
  s.destroy();
  return { errs, initial };
}

describe('GPU convergence', () => {
  for (const [W, H] of [[257, 257], [320, 240], [513, 300]] as [number, number][]) {
    test(`${W}x${H} conservative field: random start -> < 2e-5 RMS, monotone`, async ({ device, note }) => {
      const { errs, initial } = await gpuConvergence(device, W, H, 16);
      note(errs.filter((_, i) => i % 3 === 0).map((e) => e.toExponential(1)).join(' '));
      // strictly decreasing until the f32 noise floor (~3e-6) is reached
      for (let k = 1; k < 16 && errs[k - 1] > 2e-5; k++) assertLess(errs[k], errs[k - 1], `error must decrease at cycle ${k}`);
      assertLess(errs[errs.length - 1], 2e-5, 'final RMS error');
      assertLess(errs[0], initial / 3, 'recognisable after one cycle (Fig. 12)');
      assertLess(errs[1], initial / 10, 'nearly perfect after two cycles (Fig. 12)');
    });
  }

  test('convergence factor on 65x65 matches the paper (rho ~= 0.34)', async ({ device, note }) => {
    const W = 65;
    const s = new GpuSolver(device, W, W);
    s.writeRhs(new Float32Array(W * W * 4)); // homogeneous problem
    s.setTargetMean([0, 0, 0]);
    let x = randomField(W, W, 77).toFloat32();
    const norm = (a: Float32Array) => Math.sqrt(a.reduce((acc, v, i) => (i % 4 === 3 ? acc : acc + v * v), 0));
    const ratios: number[] = [];
    for (let k = 0; k < 20; k++) {
      const n0 = norm(x);
      x = x.map((v) => v / n0);
      s.writeSolution(x);
      s.run(1);
      x = await s.readSolution();
      ratios.push(norm(x));
    }
    const tail = ratios.slice(10);
    const rho = Math.exp(tail.reduce((a, r) => a + Math.log(r), 0) / tail.length);
    note(`rho = ${rho.toFixed(3)}`);
    assert(rho > 0.2 && rho < 0.4, `rho ${rho} outside [0.2, 0.4]`);
    s.destroy();
  });

  test('edited (non-conservative) field converges to the exact least-squares solution', async ({ device, note }) => {
    const W = 65;
    const H = 48;
    const img = syntheticImage(W, H, 2);
    const g = gradientsOf(img);
    const nx = randomField(W - 1, H, 3, 0.2);
    const ny = randomField(W, H - 1, 4, 0.2);
    for (let i = 0; i < g.gx.data.length; i++) g.gx.data[i] += nx.data[i];
    for (let i = 0; i < g.gy.data.length; i++) g.gy.data[i] += ny.data[i];
    const gx = f32(g.gx);
    const gy = f32(g.gy);
    const exact = exactSolve(computeRhs(gx, gy, W, H));
    const s = new GpuSolver(device, W, H);
    s.setGradientTextures(createFloatTexture(device, W - 1, H, gx.toFloat32()), createFloatTexture(device, W, H - 1, gy.toFloat32()));
    s.run(30, true);
    const err = rmsDifferenceModuloMean(new Field(W, H, await s.readSolution()), exact);
    note(`rms vs exact = ${err.toExponential(2)}`);
    assertLess(err, 1e-5, 'rms error vs exact LS solution');
    s.destroy();
  });

  test('megapixel canvas (1025x1025): converges and reports timing', async ({ device, note }) => {
    const W = 1025;
    const img = syntheticImage(W, W, 5);
    const { gx, gy } = gradientsOf(img);
    const s = new GpuSolver(device, W, W);
    s.setGradientTextures(createFloatTexture(device, W - 1, W, gx.toFloat32()), createFloatTexture(device, W, W - 1, gy.toFloat32()));
    s.setTargetMean(channelMeans(img));
    s.run(12, true);
    const err = rmsDifferenceModuloMean(new Field(W, W, await s.readSolution()), img);
    const ms = await s.benchmark(5);
    note(`rms ${err.toExponential(2)}, ${ms.toFixed(1)} ms / V-cycle on this adapter`);
    assertLess(err, 1e-4, 'rms error after 12 cycles');
    s.destroy();
  });
});
