/**
 * GPU side of painting: the gradient images being edited (gx, gy), undo
 * snapshots, stroke rasterisation with the five blend modes, and edge capture.
 *
 * Port of GeneralPaint::render_stroke / grab_current_edge and the
 * gp_blend_*.glsl / gp_edge_gradient.glsl shaders.
 */
import { createFloatTexture } from '../solver/gpu';
import { BLEND_INDEX, BlendMode } from './blend';
import { strokeNormal, VERTEX_FLOATS } from './geometry';
import type { Vec2 } from './stroke';

/** Texture address mode per axis: clamp-to-edge or clamp-to-border(black). */
export const Address = { Edge: 0, Black: 1 } as const;
export type Address = (typeof Address)[keyof typeof Address];

export interface BrushSource {
  /** brush gradient along the stroke tangent (rgb, alpha) */
  tan: GPUTexture;
  /** brush gradient along the stroke normal (rgb, alpha) */
  perp: GPUTexture;
  tanAddress: [Address, Address];
  perpAddress: [Address, Address];
  /** ignore the brush textures' alpha (true for gradient images, i.e. the clone brush) */
  opaque?: boolean;
}

export interface CapturedEdge {
  tan: GPUTexture;
  perp: GPUTexture;
  /** number of samples along the edge (texture width of `perp`) */
  steps: number;
  /** half-height of the captured band */
  radius: number;
}

const strokeShader = /* wgsl */ `
struct U {
  dim: u32, blend: u32,
  tanX: u32, tanY: u32, perpX: u32, perpY: u32,
  opaque: u32, _pad: u32,
  tw: f32, th: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var baseGx: texture_2d<f32>;
@group(0) @binding(2) var baseGy: texture_2d<f32>;
@group(0) @binding(3) var brushTan: texture_2d<f32>;
@group(0) @binding(4) var brushPerp: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) t: vec2f,
  @location(2) @interpolate(flat) n: vec2f,
};

@vertex
fn vs(@location(0) p: vec2f, @location(1) depth: f32, @location(2) uv: vec2f,
      @location(3) t: vec2f, @location(4) n: vec2f) -> VOut {
  // gx(i, j) lives at (i + 1, j + 0.5) and gy(i, j) at (i + 0.5, j + 1): shift geometry by half a pixel
  let off = select(vec2f(0.0, 0.5), vec2f(0.5, 0.0), u.dim == 0u);
  let tc = p - off;
  var o: VOut;
  o.pos = vec4f(tc.x / u.tw * 2.0 - 1.0, 1.0 - tc.y / u.th * 2.0, depth, 1.0);
  o.uv = uv;
  o.t = t;
  o.n = n;
  return o;
}

fn texel(tex: texture_2d<f32>, q: vec2i, mx: u32, my: u32) -> vec4f {
  let d = vec2i(textureDimensions(tex));
  var x = q.x;
  var y = q.y;
  if (x < 0 || x >= d.x) {
    if (mx == 1u) { return vec4f(0.0); }
    x = clamp(x, 0, d.x - 1);
  }
  if (y < 0 || y >= d.y) {
    if (my == 1u) { return vec4f(0.0); }
    y = clamp(y, 0, d.y - 1);
  }
  return textureLoad(tex, vec2i(x, y), 0);
}

// GL-style bilinear lookup: tc in texel units, texel centres at k + 0.5
fn bilinear(tex: texture_2d<f32>, tc: vec2f, mx: u32, my: u32) -> vec4f {
  let s = tc - vec2f(0.5);
  let fl = floor(s);
  let f = s - fl;
  let q = vec2i(fl);
  let a = mix(texel(tex, q, mx, my), texel(tex, q + vec2i(1, 0), mx, my), f.x);
  let b = mix(texel(tex, q + vec2i(0, 1), mx, my), texel(tex, q + vec2i(1, 1), mx, my), f.x);
  return mix(a, b, f.y);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let ij = vec2i(floor(in.pos.xy));
  let fij = vec2f(ij);
  // background gradient at this sample: own component exact, other one interpolated
  var gx: vec3f;
  var gy: vec3f;
  if (u.dim == 0u) {
    gx = textureLoad(baseGx, ij, 0).rgb;
    gy = bilinear(baseGy, fij + vec2f(1.0, 0.0), 0u, 0u).rgb;
  } else {
    gx = bilinear(baseGx, fij + vec2f(0.0, 1.0), 0u, 0u).rgb;
    gy = textureLoad(baseGy, ij, 0).rgb;
  }
  // brush gradient in the stroke frame, alpha-attenuated, rotated into the image frame
  let tan = bilinear(brushTan, in.uv - vec2f(0.5, 0.0), u.tanX, u.tanY);
  let perp = bilinear(brushPerp, in.uv - vec2f(0.0, 0.5), u.perpX, u.perpY);
  // brush gradients are attenuated by their own alpha; gradient images (clone source) are opaque
  let bt = tan.rgb * select(tan.a, 1.0, u.opaque == 1u);
  let bp = perp.rgb * select(perp.a, 1.0, u.opaque == 1u);
  let bx = in.t.x * bt + in.n.x * bp;
  let by = in.t.y * bt + in.n.y * bp;

  var ox: vec3f;
  var oy: vec3f;
  switch (u.blend) {
    case 0u: { // add (Eq. 1)
      ox = gx + bx;
      oy = gy + by;
    }
    case 1u: { // directional (Eq. 4)
      var len2 = gx * gx + gy * gy;
      len2 += select(vec3f(0.0), vec3f(1.0), len2 < vec3f(1e-6));
      let amt = max((bx * gx + by * gy) / len2, vec3f(-1.0));
      ox = gx + gx * amt;
      oy = gy + gy * amt;
    }
    case 2u, 3u: { // max (Eq. 2) / min
      let lb = bx * bx + by * by;
      let lg = gx * gx + gy * gy;
      let pick = select((lb < lg), (lb > lg), u.blend == 2u);
      ox = select(gx, bx, pick);
      oy = select(gy, by, pick);
    }
    default: { // over (Eq. 3)
      ox = bx;
      oy = by;
    }
  }
  return vec4f(select(oy, ox, u.dim == 0u), 1.0);
}
`;

const edgeSampleShader = /* wgsl */ `
struct P { w: u32, h: u32, steps: u32, rows: u32, rad: f32, _a: f32, _b: f32, _c: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pts: array<vec2f>;
@group(0) @binding(2) var<storage, read> img: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> samples: array<vec4f>;

fn px(x: i32, y: i32) -> vec4f {
  let cx = clamp(x, 0, i32(p.w) - 1);
  let cy = clamp(y, 0, i32(p.h) - 1);
  return img[u32(cy) * p.w + u32(cx)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.steps || id.y >= p.rows) { return; }
  let at = pts[id.x];
  let nx = pts[id.x + 1u];
  let d = nx - at;
  let t = d / max(length(d), 1e-12);
  let perp = p.rad * vec2f(t.y, -t.x);
  // quad i of the original spans the band at the midpoint of segment i; row r is at offset (r+0.5)/rad - 1
  let pos = 0.5 * (at + nx) + ((f32(id.y) + 0.5) / p.rad - 1.0) * perp;
  let s = pos - vec2f(0.5);
  let fl = floor(s);
  let f = s - fl;
  let q = vec2i(fl);
  let v = mix(mix(px(q.x, q.y), px(q.x + 1, q.y), f.x), mix(px(q.x, q.y + 1), px(q.x + 1, q.y + 1), f.x), f.y);
  samples[id.y * p.steps + id.x] = v;
}
`;

const edgeGradientShader = /* wgsl */ `
struct P { w: u32, h: u32, steps: u32, rows: u32, rad: f32, _a: f32, _b: f32, _c: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(3) var<storage, read> samples: array<vec4f>;
@group(0) @binding(4) var perpTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var tanTex: texture_storage_2d<rgba32float, write>;

fn S(i: u32, r: u32) -> vec3f { return samples[r * p.steps + i].rgb; }

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let r = id.y;
  if (i < p.steps && r + 1u < p.rows) {
    textureStore(perpTex, vec2u(i, r), vec4f(S(i, r + 1u) - S(i, r), 1.0));
  }
  if (i + 1u < p.steps && r < p.rows) {
    textureStore(tanTex, vec2u(i, r), vec4f(S(i + 1u, r) - S(i, r), 1.0));
  }
}
`;

interface Snapshot {
  gx: GPUTexture;
  gy: GPUTexture;
}

export interface GradientCanvasOptions {
  /** upper bound on GPU memory used by undo snapshots */
  historyBytes?: number;
}

export class GradientCanvas {
  readonly gx: GPUTexture;
  readonly gy: GPUTexture;
  private readonly origGx: GPUTexture;
  private readonly origGy: GPUTexture;
  private readonly depth: [GPUTexture, GPUTexture];
  private readonly pipeline: GPURenderPipeline;
  private readonly uniforms: [GPUBuffer, GPUBuffer];
  private vertexBuffer: GPUBuffer | null = null;
  private readonly history: Snapshot[] = [];
  private readonly pool: Snapshot[] = [];
  private readonly maxSnapshots: number;
  /** gradient-brush profile textures (1x4 tangent, 1x3 normal), see render_paint_stroke */
  readonly paintTan: GPUTexture;
  readonly paintPerp: GPUTexture;
  private readonly edgeSamplePipeline: GPUComputePipeline;
  private readonly edgeGradientPipeline: GPUComputePipeline;

  constructor(
    readonly device: GPUDevice,
    readonly width: number,
    readonly height: number,
    origGx?: Float32Array<ArrayBuffer>,
    origGy?: Float32Array<ArrayBuffer>,
    opts: GradientCanvasOptions = {},
  ) {
    const W = width;
    const H = height;
    this.origGx = createFloatTexture(device, W - 1, H, origGx);
    this.origGy = createFloatTexture(device, W, H - 1, origGy);
    this.gx = createFloatTexture(device, W - 1, H);
    this.gy = createFloatTexture(device, W, H - 1);
    const enc = device.createCommandEncoder();
    enc.copyTextureToTexture({ texture: this.origGx }, { texture: this.gx }, [W - 1, H]);
    enc.copyTextureToTexture({ texture: this.origGy }, { texture: this.gy }, [W, H - 1]);
    device.queue.submit([enc.finish()]);

    const snapshotBytes = ((W - 1) * H + W * (H - 1)) * 16;
    this.maxSnapshots = Math.max(2, Math.floor((opts.historyBytes ?? 384 * 1024 * 1024) / snapshotBytes));

    const mkDepth = (w: number, h: number) =>
      device.createTexture({ size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depth = [mkDepth(W - 1, H), mkDepth(W, H - 1)];

    const module = device.createShaderModule({ label: 'gp-stroke', code: strokeShader });
    this.pipeline = device.createRenderPipeline({
      label: 'gp-stroke',
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: VERTEX_FLOATS * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x2' },
              { shaderLocation: 1, offset: 8, format: 'float32' },
              { shaderLocation: 2, offset: 12, format: 'float32x2' },
              { shaderLocation: 3, offset: 20, format: 'float32x2' },
              { shaderLocation: 4, offset: 28, format: 'float32x2' },
            ],
          },
        ],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba32float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthCompare: 'less', depthWriteEnabled: true },
    });
    const mkU = () => device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniforms = [mkU(), mkU()];

    this.paintTan = createFloatTexture(device, 1, 4, new Float32Array(16));
    this.paintPerp = createFloatTexture(device, 1, 3, new Float32Array(12));

    this.edgeSamplePipeline = device.createComputePipeline({
      label: 'gp-edge-sample',
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: edgeSampleShader }), entryPoint: 'main' },
    });
    this.edgeGradientPipeline = device.createComputePipeline({
      label: 'gp-edge-gradient',
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: edgeGradientShader }), entryPoint: 'main' },
    });
  }

  // -------------------------------------------------------------------------
  // undo history

  private allocSnapshot(): Snapshot {
    return this.pool.pop() ?? { gx: createFloatTexture(this.device, this.width - 1, this.height), gy: createFloatTexture(this.device, this.width, this.height - 1) };
  }

  private copyInto(enc: GPUCommandEncoder, gx: GPUTexture, gy: GPUTexture, dstGx: GPUTexture, dstGy: GPUTexture): void {
    enc.copyTextureToTexture({ texture: gx }, { texture: dstGx }, [this.width - 1, this.height]);
    enc.copyTextureToTexture({ texture: gy }, { texture: dstGy }, [this.width, this.height - 1]);
  }

  get undoDepth(): number {
    return this.history.length;
  }

  /** The state the current stroke is drawn on top of ("gx_src" in the original). */
  get source(): Snapshot | null {
    return this.history[this.history.length - 1] ?? null;
  }

  /** Save the current gradients; the next stroke is rendered over this snapshot. */
  pushSnapshot(enc: GPUCommandEncoder): void {
    const s = this.allocSnapshot();
    this.copyInto(enc, this.gx, this.gy, s.gx, s.gy);
    this.history.push(s);
    while (this.history.length > this.maxSnapshots) this.pool.push(this.history.shift()!);
  }

  /** Restore the most recent snapshot (removing it). Returns false if there is nothing to undo. */
  undo(enc: GPUCommandEncoder): boolean {
    const s = this.history.pop();
    if (!s) return false;
    this.copyInto(enc, s.gx, s.gy, this.gx, this.gy);
    this.pool.push(s);
    return true;
  }

  /** Back to the original image's gradients (undoable). */
  revert(enc: GPUCommandEncoder): void {
    this.pushSnapshot(enc);
    this.copyInto(enc, this.origGx, this.origGy, this.gx, this.gy);
  }

  /** Replace the "original" gradients (e.g. after loading an image). Clears history. */
  setOriginal(gx: Float32Array<ArrayBuffer>, gy: Float32Array<ArrayBuffer>): void {
    const q = this.device.queue;
    q.writeTexture({ texture: this.origGx }, gx, { bytesPerRow: (this.width - 1) * 16 }, [this.width - 1, this.height]);
    q.writeTexture({ texture: this.origGy }, gy, { bytesPerRow: this.width * 16 }, [this.width, this.height - 1]);
    const enc = this.device.createCommandEncoder();
    this.copyInto(enc, this.origGx, this.origGy, this.gx, this.gy);
    q.submit([enc.finish()]);
    while (this.history.length) this.pool.push(this.history.pop()!);
  }

  // -------------------------------------------------------------------------
  // brushes

  /** Gradient brush: a tent of half-width radius/2 across the stroke, peak 1/size, tinted by colour. */
  paintBrush(color: [number, number, number], size: number): BrushSource {
    const [r, g, b] = color;
    this.device.queue.writeTexture(
      { texture: this.paintPerp },
      new Float32Array([r, g, b, 0, r, g, b, 1 / size, r, g, b, 0]),
      { bytesPerRow: 16 },
      [1, 3],
    );
    return { tan: this.paintTan, perp: this.paintPerp, tanAddress: [Address.Edge, Address.Black], perpAddress: [Address.Edge, Address.Black] };
  }

  edgeBrush(edge: CapturedEdge): BrushSource {
    return { tan: edge.tan, perp: edge.perp, tanAddress: [Address.Edge, Address.Black], perpAddress: [Address.Edge, Address.Black] };
  }

  /** Clone brush: copies gradients from the stroke's source snapshot at an offset. */
  cloneBrush(): BrushSource {
    const src = this.source;
    if (!src) throw new Error('cloneBrush needs an active stroke snapshot');
    return { tan: src.gx, perp: src.gy, tanAddress: [Address.Black, Address.Edge], perpAddress: [Address.Edge, Address.Black], opaque: true };
  }

  /**
   * Redraw the current stroke: restore the source snapshot, then rasterise the
   * stroke geometry into gx and gy with the given brush and blend mode.
   */
  renderStroke(enc: GPUCommandEncoder, geometry: Float32Array<ArrayBuffer>, brush: BrushSource, mode: BlendMode): void {
    const src = this.source;
    if (!src) throw new Error('renderStroke needs a snapshot (call pushSnapshot when the stroke starts)');
    this.copyInto(enc, src.gx, src.gy, this.gx, this.gy);
    const count = geometry.length / VERTEX_FLOATS;
    if (count === 0) return;
    if (!this.vertexBuffer || this.vertexBuffer.size < geometry.byteLength) {
      this.vertexBuffer?.destroy();
      this.vertexBuffer = this.device.createBuffer({
        size: Math.max(4096, 2 ** Math.ceil(Math.log2(geometry.byteLength))),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.vertexBuffer, 0, geometry);
    const targets = [this.gx, this.gy];
    for (const dim of [0, 1] as const) {
      const t = targets[dim];
      const ab = new ArrayBuffer(48);
      const u32 = new Uint32Array(ab);
      const f32 = new Float32Array(ab);
      u32[0] = dim;
      u32[1] = BLEND_INDEX[mode];
      u32[2] = brush.tanAddress[0];
      u32[3] = brush.tanAddress[1];
      u32[4] = brush.perpAddress[0];
      u32[5] = brush.perpAddress[1];
      u32[6] = brush.opaque ? 1 : 0;
      f32[8] = t.width;
      f32[9] = t.height;
      this.device.queue.writeBuffer(this.uniforms[dim], 0, ab);
      const bind = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniforms[dim] } },
          { binding: 1, resource: src.gx.createView() },
          { binding: 2, resource: src.gy.createView() },
          { binding: 3, resource: brush.tan.createView() },
          { binding: 4, resource: brush.perp.createView() },
        ],
      });
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: t.createView(), loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: this.depth[dim].createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bind);
      pass.setVertexBuffer(0, this.vertexBuffer, 0, geometry.byteLength);
      pass.draw(count);
      pass.end();
    }
  }

  // -------------------------------------------------------------------------
  // edge capture (grab_current_edge)

  /**
   * Sample the current image in a band of half-width `radius` along `points`
   * and store its along-stroke (tan) and across-stroke (perp) differences.
   * `image` is the solver's solution buffer (array<vec4f>, W x H).
   */
  captureEdge(image: GPUBuffer, points: Vec2[], radius: number): CapturedEdge | null {
    const maxDim = this.device.limits.maxTextureDimension2D;
    if (points.length > maxDim) points = points.slice(0, maxDim);
    const steps = points.length - 1;
    const rows = 2 * radius;
    if (steps < 2 || radius < 1) return null;
    const d = this.device;
    const pts = d.createBuffer({ size: points.length * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(pts, 0, new Float32Array(points.flat()) as Float32Array<ArrayBuffer>);
    const samples = d.createBuffer({ size: steps * rows * 16, usage: GPUBufferUsage.STORAGE });
    const params = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const ab = new ArrayBuffer(32);
    new Uint32Array(ab).set([this.width, this.height, steps, rows]);
    new Float32Array(ab)[4] = radius;
    d.queue.writeBuffer(params, 0, ab);
    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    const perp = d.createTexture({ size: [steps, rows - 1], format: 'rgba32float', usage });
    const tan = d.createTexture({ size: [steps - 1, rows], format: 'rgba32float', usage });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.edgeSamplePipeline);
    pass.setBindGroup(
      0,
      d.createBindGroup({
        layout: this.edgeSamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: pts } },
          { binding: 2, resource: { buffer: image } },
          { binding: 3, resource: { buffer: samples } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(steps / 16), Math.ceil(rows / 16));
    pass.setPipeline(this.edgeGradientPipeline);
    pass.setBindGroup(
      0,
      d.createBindGroup({
        layout: this.edgeGradientPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 3, resource: { buffer: samples } },
          { binding: 4, resource: perp.createView() },
          { binding: 5, resource: tan.createView() },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(steps / 16), Math.ceil(rows / 16));
    pass.end();
    d.queue.submit([enc.finish()]);
    pts.destroy();
    samples.destroy();
    params.destroy();
    return { tan, perp, steps, radius };
  }

  destroy(): void {
    for (const t of [this.gx, this.gy, this.origGx, this.origGy, ...this.depth, this.paintTan, this.paintPerp]) t.destroy();
    for (const s of [...this.history, ...this.pool]) {
      s.gx.destroy();
      s.gy.destroy();
    }
    this.vertexBuffer?.destroy();
    for (const u of this.uniforms) u.destroy();
  }
}

/** Read an rgba32float texture back to the CPU (rows unpadded). */
export async function readTexture(device: GPUDevice, tex: GPUTexture): Promise<Float32Array<ArrayBuffer>> {
  const w = tex.width;
  const h = tex.height;
  const bytesPerRow = Math.ceil((w * 16) / 256) * 256;
  const buf = device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Float32Array(buf.getMappedRange());
  const out = new Float32Array(w * h * 4);
  const rowFloats = bytesPerRow / 4;
  for (let y = 0; y < h; y++) out.set(src.subarray(y * rowFloats, y * rowFloats + w * 4), y * w * 4);
  buf.unmap();
  buf.destroy();
  return out;
}

export { strokeNormal };
