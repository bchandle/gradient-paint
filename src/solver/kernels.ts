/**
 * WGSL compute kernels for the multigrid integrator. Each one is a direct
 * port of a GLSL fragment shader from the original implementation
 * (dist/shaders/vc_*.glsl, mg_rhs.glsl); see reference.ts for the
 * equivalent CPU code that the GPU test-suite compares against.
 *
 * All fields are array<vec4f> (rgb + unused), row-major, y down.
 */

export const WG = 16; // 2D workgroup edge
export const REDUCE_WG = 256; // 1D reduction workgroup size
export const REDUCE_GROUPS = 256; // number of partial sums in the first reduction stage

const params = /* wgsl */ `
struct Params {
  w: u32, h: u32,        // grid this kernel writes (or the fine grid for restrict)
  cw: u32, ch: u32,      // the other grid (coarse for restrict/interpolate)
  middle: f32, edge: f32, corner: f32, x: f32,  // L_h stencil + magic relaxation weight
};
@group(0) @binding(3) var<uniform> p: Params;
`;

/** mg_rhs.glsl — f = gx(x) - gx(x-1) + gy(y) - gy(y-1), clamp-to-black. */
export const rhsKernel = /* wgsl */ `
${params}
@group(0) @binding(0) var gx: texture_2d<f32>;
@group(0) @binding(1) var gy: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outb: array<vec4f>;

fn GX(x: i32, y: i32) -> vec4f {
  if (x < 0 || x >= i32(p.w) - 1) { return vec4f(0.0); }
  return textureLoad(gx, vec2i(x, y), 0);
}
fn GY(x: i32, y: i32) -> vec4f {
  if (y < 0 || y >= i32(p.h) - 1) { return vec4f(0.0); }
  return textureLoad(gy, vec2i(x, y), 0);
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.w || id.y >= p.h) { return; }
  let x = i32(id.x);
  let y = i32(id.y);
  var r = (GX(x, y) - GX(x - 1, y)) + (GY(x, y) - GY(x, y - 1));
  r.w = 0.0;
  outb[id.y * p.w + id.x] = r;
}
`;

const stencilHelpers = /* wgsl */ `
@group(0) @binding(0) var<storage, read> f: array<vec4f>;
@group(0) @binding(1) var<storage, read> rhs: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outb: array<vec4f>;

// clamp-to-edge fetch (Neumann boundary)
fn F(x: i32, y: i32) -> vec4f {
  let cx = clamp(x, 0, i32(p.w) - 1);
  let cy = clamp(y, 0, i32(p.h) - 1);
  return f[u32(cy) * p.w + u32(cx)];
}
`;

/** vc_residual.glsl — r = rhs - L_h f */
export const residualKernel = /* wgsl */ `
${params}
${stencilHelpers}
@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.w || id.y >= p.h) { return; }
  let x = i32(id.x);
  let y = i32(id.y);
  let i = id.y * p.w + id.x;
  let n4 = (F(x + 1, y) + F(x - 1, y)) + (F(x, y + 1) + F(x, y - 1));
  let d4 = (F(x + 1, y + 1) + F(x + 1, y - 1)) + (F(x - 1, y + 1) + F(x - 1, y - 1));
  var r = rhs[i] - (p.middle * f[i] + p.edge * n4 + p.corner * d4);
  r.w = 0.0;
  outb[i] = r;
}
`;

/** vc_iter.glsl — u <- (rhs - (L_h - (m - x) I) u) / (m - x) */
export const smoothKernel = /* wgsl */ `
${params}
${stencilHelpers}
@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.w || id.y >= p.h) { return; }
  let x = i32(id.x);
  let y = i32(id.y);
  let i = id.y * p.w + id.x;
  let n4 = (F(x + 1, y) + F(x - 1, y)) + (F(x, y + 1) + F(x, y - 1));
  let d4 = (F(x + 1, y + 1) + F(x + 1, y - 1)) + (F(x - 1, y + 1) + F(x - 1, y - 1));
  var u = (rhs[i] - p.edge * n4 - p.corner * d4 - p.x * f[i]) / (p.middle - p.x);
  u.w = 0.0;
  outb[i] = u;
}
`;

/** vc_restrict.glsl — 3x3 [1/4 1/2 1/4; 1/2 1 1/2; 1/4 1/2 1/4] around fine (2i, 2j), clamp-to-black. */
export const restrictKernel = /* wgsl */ `
${params}
@group(0) @binding(0) var<storage, read> fine: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outb: array<vec4f>;

fn FB(x: i32, y: i32) -> vec4f {
  if (x < 0 || y < 0 || x >= i32(p.w) || y >= i32(p.h)) { return vec4f(0.0); }
  return fine[u32(y) * p.w + u32(x)];
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.cw || id.y >= p.ch) { return; }
  let x = 2 * i32(id.x);
  let y = 2 * i32(id.y);
  let row0 = 0.5 * FB(x - 1, y - 1) + FB(x, y - 1) + 0.5 * FB(x + 1, y - 1);
  let row1 = 0.5 * FB(x - 1, y) + FB(x, y) + 0.5 * FB(x + 1, y);
  let row2 = 0.5 * FB(x - 1, y + 1) + FB(x, y + 1) + 0.5 * FB(x + 1, y + 1);
  var r = 0.5 * row0 + row1 + 0.5 * row2;
  r.w = 0.0;
  outb[id.y * p.cw + id.x] = r;
}
`;

function interpolateKernel(add: boolean): string {
  return /* wgsl */ `
${params}
@group(0) @binding(0) var<storage, read> coarse: array<vec4f>;
${add ? '@group(0) @binding(1) var<storage, read> addb: array<vec4f>;' : ''}
@group(0) @binding(2) var<storage, read_write> outb: array<vec4f>;

fn C(x: u32, y: u32) -> vec4f {
  return coarse[min(y, p.ch - 1u) * p.cw + min(x, p.cw - 1u)];
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.w || id.y >= p.h) { return; }
  let x0 = id.x >> 1u;
  let y0 = id.y >> 1u;
  let tx = f32(id.x & 1u) * 0.5;
  let ty = f32(id.y & 1u) * 0.5;
  let top = mix(C(x0, y0), C(x0 + 1u, y0), tx);
  let bot = mix(C(x0, y0 + 1u), C(x0 + 1u, y0 + 1u), tx);
  let i = id.y * p.w + id.x;
  var v = mix(top, bot, ty)${add ? ' + addb[i]' : ''};
  v.w = 0.0;
  outb[i] = v;
}
`;
}
/** vc_interpolate.glsl — bilinear prolongation, clamp-to-edge. */
export const interpolateKernelSrc = interpolateKernel(false);
/** vc_interpolate_add.glsl — prolongation added to the current fine solution. */
export const interpolateAddKernelSrc = interpolateKernel(true);

/**
 * Re-centering, stage 1: per-workgroup partial sums (grid-stride loop +
 * shared-memory tree). Replaces the original's 64x64 bilinear averaging
 * pyramid with an exact mean.
 */
export const reducePartialKernel = /* wgsl */ `
${params}
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> partial: array<vec4f>;
var<workgroup> sh: array<vec4f, ${REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u) {
  let n = p.w * p.h;
  let stride = ${REDUCE_WG * REDUCE_GROUPS}u;
  var acc = vec4f(0.0);
  for (var i = wg.x * ${REDUCE_WG}u + li; i < n; i += stride) {
    acc += src[i];
  }
  sh[li] = acc;
  workgroupBarrier();
  for (var s = ${REDUCE_WG / 2}u; s > 0u; s >>= 1u) {
    if (li < s) { sh[li] += sh[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { partial[wg.x] = sh[0]; }
}
`;

/** Re-centering, stage 2: total -> offset = target - mean (written to offset[0]). */
export const reduceFinalKernel = /* wgsl */ `
${params}
struct Target { mean: vec4f };
@group(0) @binding(0) var<storage, read> partial: array<vec4f>;
@group(0) @binding(1) var<uniform> tgt: Target;
@group(0) @binding(2) var<storage, read_write> offset: array<vec4f>;
var<workgroup> sh: array<vec4f, ${REDUCE_GROUPS}>;

@compute @workgroup_size(${REDUCE_GROUPS})
fn main(@builtin(local_invocation_index) li: u32) {
  sh[li] = partial[li];
  workgroupBarrier();
  for (var s = ${REDUCE_GROUPS / 2}u; s > 0u; s >>= 1u) {
    if (li < s) { sh[li] += sh[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let mean = sh[0] / f32(p.w * p.h);
    var o = tgt.mean - mean;
    o.w = 0.0;
    offset[0] = o;
  }
}
`;

/** vc_recenter_sub.glsl — out = src + offset */
export const recenterKernel = /* wgsl */ `
${params}
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read> offset: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outb: array<vec4f>;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.w || id.y >= p.h) { return; }
  let i = id.y * p.w + id.x;
  outb[i] = src[i] + offset[0];
}
`;

/** Zero a buffer (coarsest-level correction). */
export const clearKernel = /* wgsl */ `
${params}
@group(0) @binding(2) var<storage, read_write> outb: array<vec4f>;
@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.w || id.y >= p.h) { return; }
  outb[id.y * p.w + id.x] = vec4f(0.0);
}
`;
