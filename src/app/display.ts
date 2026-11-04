/**
 * Draws a solver field (array<vec4f>) to the canvas with a pan/zoom view
 * transform. Port of the gp_scale_bias / gp_compare / grad_view shaders.
 */

export const ViewMode = { Image: 0, Signed: 1, Compare: 2 } as const;
export type ViewMode = (typeof ViewMode)[keyof typeof ViewMode];

export interface DisplayParams {
  /** device pixels per image pixel */
  scale: number;
  /** device-pixel position of the image's top-left corner */
  offsetX: number;
  offsetY: number;
  mode: ViewMode;
  /** dimensions of the field being shown, and of the image it is drawn over */
  srcW: number;
  srcH: number;
  imgW: number;
  imgH: number;
  /** contrast is applied around `mean`; `bias` is added afterwards (exposure / colour adapt) */
  mean: [number, number, number];
  gain: number;
  bias: [number, number, number];
}

const shader = /* wgsl */ `
struct D {
  scale: f32, ox: f32, oy: f32, mode: f32,
  srcW: f32, srcH: f32, imgW: f32, imgH: f32,
  mean: vec3f, gain: f32,
  bias: vec3f, _p: f32,
};
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read> orig: array<vec4f>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn fetch(q: vec2f) -> vec3f {
  // image coords -> source texel (sources may be coarser / smaller than the image)
  let s = q * vec2f(d.srcW / d.imgW, d.srcH / d.imgH);
  let ij = vec2u(clamp(vec2i(floor(s)), vec2i(0), vec2i(i32(d.srcW) - 1, i32(d.srcH) - 1)));
  let k = ij.y * u32(d.srcW) + ij.x;
  var v = src[k].rgb;
  let m = u32(d.mode);
  if (m == 2u) {
    let qi = vec2u(clamp(vec2i(floor(q)), vec2i(0), vec2i(i32(d.imgW) - 1, i32(d.imgH) - 1)));
    return abs(v - orig[qi.y * u32(d.imgW) + qi.x].rgb) * d.gain;
  }
  if (m == 1u) {
    return v * d.gain + vec3f(0.5);
  }
  return (v - d.mean) * d.gain + d.mean + d.bias;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = (pos.xy - vec2f(d.ox, d.oy)) / d.scale;
  if (q.x < 0.0 || q.y < 0.0 || q.x >= d.imgW || q.y >= d.imgH) {
    // backdrop
    let c = select(0.105, 0.125, ((u32(pos.x) / 12u + u32(pos.y) / 12u) & 1u) == 1u);
    return vec4f(c, c, c, 1.0);
  }
  var c: vec3f;
  if (d.scale >= 1.0) {
    c = fetch(q);
  } else {
    // minified: average a 2x2 grid of samples over the pixel footprint
    let h = 0.25 / d.scale;
    c = 0.25 * (fetch(q + vec2f(-h, -h)) + fetch(q + vec2f(h, -h)) + fetch(q + vec2f(-h, h)) + fetch(q + vec2f(h, h)));
  }
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

const texToBufShader = /* wgsl */ `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outb: array<vec4f>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dim = textureDimensions(tex);
  if (id.x >= dim.x || id.y >= dim.y) { return; }
  outb[id.y * dim.x + id.x] = textureLoad(tex, vec2i(id.xy), 0);
}
`;

export class Display {
  private readonly pipeline: GPURenderPipeline;
  private readonly uniform: GPUBuffer;
  private readonly texPipeline: GPUComputePipeline;
  private bindCache = new Map<string, GPUBindGroup>();
  private ids = new WeakMap<object, number>();
  private nextId = 1;

  constructor(
    private readonly device: GPUDevice,
    readonly format: GPUTextureFormat,
  ) {
    const module = device.createShaderModule({ label: 'gp-display', code: shader });
    this.pipeline = device.createRenderPipeline({
      label: 'gp-display',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.texPipeline = device.createComputePipeline({
      label: 'gp-tex-to-buf',
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: texToBufShader }), entryPoint: 'main' },
    });
  }

  private id(o: object): number {
    let i = this.ids.get(o);
    if (!i) this.ids.set(o, (i = this.nextId++));
    return i;
  }

  /** Copy an rgba32float texture into a field buffer (for viewing gradient images). */
  encodeTextureToBuffer(pass: GPUComputePassEncoder, tex: GPUTexture, buf: GPUBuffer): void {
    const key = `t${this.id(tex)}-${this.id(buf)}`;
    let bg = this.bindCache.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this.texPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: tex.createView() },
          { binding: 1, resource: { buffer: buf } },
        ],
      });
      this.bindCache.set(key, bg);
    }
    pass.setPipeline(this.texPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(tex.width / 16), Math.ceil(tex.height / 16));
  }

  encode(enc: GPUCommandEncoder, target: GPUTextureView, src: GPUBuffer, orig: GPUBuffer, p: DisplayParams): void {
    const f = new Float32Array(16);
    f.set([p.scale, p.offsetX, p.offsetY, p.mode, p.srcW, p.srcH, p.imgW, p.imgH, ...p.mean, p.gain, ...p.bias, 0]);
    this.device.queue.writeBuffer(this.uniform, 0, f);
    const key = `d${this.id(src)}-${this.id(orig)}`;
    let bg = this.bindCache.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: orig } },
        ],
      });
      this.bindCache.set(key, bg);
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0.1, 0.1, 0.1, 1] }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }

  /** Drop cached bind groups (call when buffers are replaced). */
  reset(): void {
    this.bindCache.clear();
  }
}
