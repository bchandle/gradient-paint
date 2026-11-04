/**
 * WebGPU implementation of the multigrid integrator (VCycle.cpp).
 *
 * One call to encodeVCycle() records the whole V-cycle (~5 dispatches per
 * level + 3 for re-centering) into a compute pass; all bind groups are built
 * up front, so the per-frame cost is just command encoding.
 */
import {
  clearKernel,
  interpolateAddKernelSrc,
  interpolateKernelSrc,
  recenterKernel,
  reduceFinalKernel,
  reducePartialKernel,
  REDUCE_GROUPS,
  residualKernel,
  restrictKernel,
  rhsKernel,
  smoothKernel,
  WG,
} from './kernels';
import { buildLevels, CHANNELS, Coarsening, LevelSpec } from './spec';

export type KernelName =
  | 'rhs'
  | 'residual'
  | 'smooth'
  | 'restrict'
  | 'interpolate'
  | 'interpolateAdd'
  | 'reducePartial'
  | 'reduceFinal'
  | 'recenter'
  | 'clear';

const KERNEL_SOURCES: Record<KernelName, string> = {
  rhs: rhsKernel,
  residual: residualKernel,
  smooth: smoothKernel,
  restrict: restrictKernel,
  interpolate: interpolateKernelSrc,
  interpolateAdd: interpolateAddKernelSrc,
  reducePartial: reducePartialKernel,
  reduceFinal: reduceFinalKernel,
  recenter: recenterKernel,
  clear: clearKernel,
};

const pipelineCache = new WeakMap<GPUDevice, Map<KernelName, GPUComputePipeline>>();

export function getPipeline(device: GPUDevice, name: KernelName): GPUComputePipeline {
  let cache = pipelineCache.get(device);
  if (!cache) pipelineCache.set(device, (cache = new Map()));
  let p = cache.get(name);
  if (!p) {
    const module = device.createShaderModule({ label: `gp-${name}`, code: KERNEL_SOURCES[name] });
    p = device.createComputePipeline({ label: `gp-${name}`, layout: 'auto', compute: { module, entryPoint: 'main' } });
    cache.set(name, p);
  }
  return p;
}

export interface KernelParams {
  w: number;
  h: number;
  cw?: number;
  ch?: number;
  middle?: number;
  edge?: number;
  corner?: number;
  x?: number;
}

export function makeParamsBuffer(device: GPUDevice, p: KernelParams): GPUBuffer {
  const buf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const ab = new ArrayBuffer(32);
  const u = new Uint32Array(ab);
  const f = new Float32Array(ab);
  u[0] = p.w;
  u[1] = p.h;
  u[2] = p.cw ?? 0;
  u[3] = p.ch ?? 0;
  f[4] = p.middle ?? 0;
  f[5] = p.edge ?? 0;
  f[6] = p.corner ?? 0;
  f[7] = p.x ?? 0;
  device.queue.writeBuffer(buf, 0, ab);
  return buf;
}

/** A prepared dispatch: pipeline + bind group + workgroup counts. */
export interface Dispatch {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  groups: [number, number];
  label: string;
}

/** Bindings are given by slot number (0,1,2 = data, 3 = params is added automatically). */
export function makeDispatch(
  device: GPUDevice,
  name: KernelName,
  bindings: Partial<Record<0 | 1 | 2, GPUBuffer | GPUTextureView>>,
  params: GPUBuffer,
  groups: [number, number],
): Dispatch {
  const pipeline = getPipeline(device, name);
  const entries: GPUBindGroupEntry[] = [];
  for (const [k, v] of Object.entries(bindings)) {
    if (!v) continue;
    entries.push({ binding: Number(k), resource: v instanceof GPUBuffer ? { buffer: v } : v });
  }
  // reduceFinal uses binding 1 for a uniform and has no params use of w/h beyond count
  entries.push({ binding: 3, resource: { buffer: params } });
  const bindGroup = device.createBindGroup({ label: name, layout: pipeline.getBindGroupLayout(0), entries });
  return { pipeline, bindGroup, groups, label: name };
}

export function runDispatch(pass: GPUComputePassEncoder, d: Dispatch): void {
  pass.setPipeline(d.pipeline);
  pass.setBindGroup(0, d.bindGroup);
  pass.dispatchWorkgroups(d.groups[0], d.groups[1]);
}

const groups2d = (w: number, h: number): [number, number] => [Math.ceil(w / WG), Math.ceil(h / WG)];

export interface GpuSolverOptions {
  coarsening?: Coarsening;
}

export class GpuSolver {
  readonly levels: LevelSpec[];
  /** per-level solution / correction buffers ("f" in VCycle.cpp); a[0] holds the current image */
  readonly a: GPUBuffer[] = [];
  /** per-level scratch ("f_temp") */
  readonly b: GPUBuffer[] = [];
  /** per-level right-hand sides */
  readonly rhs: GPUBuffer[] = [];
  private readonly partial: GPUBuffer;
  private readonly offset: GPUBuffer;
  private readonly targetBuf: GPUBuffer;
  private readonly owned: GPUBuffer[] = [];
  private readonly cycle: Dispatch[] = [];
  private rhsDispatch: Dispatch | null = null;
  private readonly level0Params: GPUBuffer;
  private targetMeanValue: [number, number, number] = [0, 0, 0];

  constructor(
    readonly device: GPUDevice,
    readonly width: number,
    readonly height: number,
    opts: GpuSolverOptions = {},
  ) {
    if (width < 2 || height < 2) throw new Error('GpuSolver needs at least a 2x2 grid');
    this.levels = buildLevels(width, height, opts.coarsening ?? 'padded');
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const mk = (label: string, n: number, u = usage) => {
      const buf = device.createBuffer({ label, size: Math.max(16, n * CHANNELS * 4), usage: u });
      this.owned.push(buf);
      return buf;
    };
    for (const l of this.levels) {
      const n = l.width * l.height;
      this.a.push(mk(`f${l.index}`, n));
      this.b.push(mk(`ftemp${l.index}`, n));
      this.rhs.push(mk(`rhs${l.index}`, n));
    }
    this.partial = mk('partial', REDUCE_GROUPS);
    this.offset = mk('offset', 1);
    this.targetBuf = mk('targetMean', 1, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.setTargetMean([0, 0, 0]);
    this.level0Params = this.paramsFor(0);
    this.buildCycle();
  }

  get solution(): GPUBuffer {
    return this.a[0];
  }

  get targetMean(): [number, number, number] {
    return this.targetMeanValue;
  }

  /** Per-channel mean the solution is re-centred to after each cycle (Section 6: the solution is only defined up to a constant). */
  setTargetMean(m: ArrayLike<number>): void {
    this.targetMeanValue = [m[0], m[1], m[2]];
    this.device.queue.writeBuffer(this.targetBuf, 0, new Float32Array([m[0], m[1], m[2], 0]));
  }

  private paramsFor(l: number, extra: Partial<KernelParams> = {}): GPUBuffer {
    const lev = this.levels[l];
    const buf = makeParamsBuffer(this.device, {
      w: lev.width,
      h: lev.height,
      middle: lev.middle,
      edge: lev.edge,
      corner: lev.corner,
      ...extra,
    });
    this.owned.push(buf);
    return buf;
  }

  private buildCycle(): void {
    const d = this.device;
    const L = this.levels;
    const last = L.length - 1;
    const seq = this.cycle;
    // down: residual on the finest grid, then restrict
    seq.push(makeDispatch(d, 'residual', { 0: this.a[0], 1: this.rhs[0], 2: this.b[0] }, this.level0Params, groups2d(L[0].width, L[0].height)));
    for (let l = 0; l < last; l++) {
      const src = l === 0 ? this.b[0] : this.rhs[l];
      const p = this.paramsFor(l, { cw: L[l + 1].width, ch: L[l + 1].height });
      seq.push(makeDispatch(d, 'restrict', { 0: src, 2: this.rhs[l + 1] }, p, groups2d(L[l + 1].width, L[l + 1].height)));
    }
    // coarsest: zero correction
    seq.push(makeDispatch(d, 'clear', { 2: this.a[last] }, this.paramsFor(last), groups2d(L[last].width, L[last].height)));
    // up: interpolate, then two magic-x Jacobi smooths. Final result of each level lands in a[l]
    // (level 0: b[0], which re-centering then writes back into a[0]).
    for (let l = last - 1; l >= 0; l--) {
      const lev = L[l];
      const g = groups2d(lev.width, lev.height);
      const pi = this.paramsFor(l, { cw: L[l + 1].width, ch: L[l + 1].height });
      const pA = this.paramsFor(l, { x: lev.xA });
      const pB = this.paramsFor(l, { x: lev.xB });
      if (l === 0) {
        seq.push(makeDispatch(d, 'interpolateAdd', { 0: this.a[1], 1: this.a[0], 2: this.b[0] }, pi, g));
        seq.push(makeDispatch(d, 'smooth', { 0: this.b[0], 1: this.rhs[0], 2: this.a[0] }, pA, g));
        seq.push(makeDispatch(d, 'smooth', { 0: this.a[0], 1: this.rhs[0], 2: this.b[0] }, pB, g));
      } else {
        seq.push(makeDispatch(d, 'interpolate', { 0: this.a[l + 1], 2: this.a[l] }, pi, g));
        seq.push(makeDispatch(d, 'smooth', { 0: this.a[l], 1: this.rhs[l], 2: this.b[l] }, pA, g));
        seq.push(makeDispatch(d, 'smooth', { 0: this.b[l], 1: this.rhs[l], 2: this.a[l] }, pB, g));
      }
    }
    // re-center: b[0] -> a[0]
    seq.push(makeDispatch(d, 'reducePartial', { 0: this.b[0], 2: this.partial }, this.level0Params, [REDUCE_GROUPS, 1]));
    seq.push(makeDispatch(d, 'reduceFinal', { 0: this.partial, 1: this.targetBuf, 2: this.offset }, this.level0Params, [1, 1]));
    seq.push(makeDispatch(d, 'recenter', { 0: this.b[0], 1: this.offset, 2: this.a[0] }, this.level0Params, groups2d(L[0].width, L[0].height)));
  }

  /** Bind the gradient textures (rgba32float, (W-1)xH and Wx(H-1)) that feed the RHS. */
  setGradientTextures(gx: GPUTexture, gy: GPUTexture): void {
    if (gx.width !== this.width - 1 || gx.height !== this.height || gy.width !== this.width || gy.height !== this.height - 1) {
      throw new Error('gradient textures do not match the solver size');
    }
    this.rhsDispatch = makeDispatch(
      this.device,
      'rhs',
      { 0: gx.createView(), 1: gy.createView(), 2: this.rhs[0] },
      this.level0Params,
      groups2d(this.width, this.height),
    );
  }

  /** Recompute the finest RHS from the gradient textures ("gradient_changed"). */
  encodeRhs(pass: GPUComputePassEncoder): void {
    if (!this.rhsDispatch) throw new Error('no gradient textures bound');
    runDispatch(pass, this.rhsDispatch);
  }

  /** Record one full V-cycle (plus re-centering) into a compute pass. */
  encodeVCycle(pass: GPUComputePassEncoder): void {
    for (const d of this.cycle) runDispatch(pass, d);
  }

  /** Convenience: submit `cycles` V-cycles (optionally recomputing the RHS first). */
  run(cycles: number, recomputeRhs = false): void {
    const enc = this.device.createCommandEncoder({ label: 'gp-solve' });
    const pass = enc.beginComputePass();
    if (recomputeRhs) this.encodeRhs(pass);
    for (let i = 0; i < cycles; i++) this.encodeVCycle(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  writeSolution(data: Float32Array): void {
    this.device.queue.writeBuffer(this.a[0], 0, data);
  }

  writeRhs(data: Float32Array): void {
    this.device.queue.writeBuffer(this.rhs[0], 0, data);
  }

  readSolution(): Promise<Float32Array> {
    return readBuffer(this.device, this.a[0]);
  }

  /** Time `cycles` V-cycles (ms per cycle), waiting for the GPU to finish. */
  async benchmark(cycles = 50): Promise<number> {
    this.run(2);
    await this.device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let i = 0; i < cycles; i++) this.run(1);
    await this.device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / cycles;
  }

  destroy(): void {
    for (const b of this.owned) b.destroy();
    this.owned.length = 0;
  }
}

export async function readBuffer(device: GPUDevice, src: GPUBuffer, size = src.size): Promise<Float32Array> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

/** Upload rgba32float data into a new texture usable as a gradient field. */
export function createFloatTexture(device: GPUDevice, width: number, height: number, data?: Float32Array, extraUsage = 0): GPUTexture {
  const tex = device.createTexture({
    size: [width, height],
    format: 'rgba32float',
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.RENDER_ATTACHMENT |
      extraUsage,
  });
  if (data) device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: width * 16 }, [width, height]);
  return tex;
}

/** Request an adapter/device with storage limits large enough for big canvases. */
export async function requestGpuDevice(): Promise<GPUDevice> {
  if (!('gpu' in navigator) || !navigator.gpu) throw new Error('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
    },
  });
}
