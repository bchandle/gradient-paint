/**
 * CPU reference implementation of the multigrid integrator.
 *
 * This is a direct, readable transcription of VCycle.cpp + its GLSL shaders,
 * in float64. It is the oracle that the WebGPU implementation is tested
 * against, kernel by kernel, and it is what the convergence tests analyse.
 */
import { buildLevels, CHANNELS, Coarsening, LevelSpec } from './spec';

export class Field {
  readonly data: Float64Array;
  constructor(
    readonly width: number,
    readonly height: number,
    data?: ArrayLike<number>,
  ) {
    this.data = new Float64Array(width * height * CHANNELS);
    if (data) {
      if (data.length !== this.data.length) {
        throw new Error(`Field ${width}x${height} expects ${this.data.length} values, got ${data.length}`);
      }
      this.data.set(data);
    }
  }
  static fill(width: number, height: number, fn: (x: number, y: number, c: number) => number): Field {
    const f = new Field(width, height);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++)
        for (let c = 0; c < CHANNELS; c++) f.data[(y * width + x) * CHANNELS + c] = fn(x, y, c);
    return f;
  }
  clone(): Field {
    return new Field(this.width, this.height, this.data);
  }
  idx(x: number, y: number): number {
    return (y * this.width + x) * CHANNELS;
  }
  /** value with clamp-to-edge addressing */
  edge(x: number, y: number, c: number): number {
    const cx = x < 0 ? 0 : x >= this.width ? this.width - 1 : x;
    const cy = y < 0 ? 0 : y >= this.height ? this.height - 1 : y;
    return this.data[(cy * this.width + cx) * CHANNELS + c];
  }
  /** value with clamp-to-border(0) addressing */
  black(x: number, y: number, c: number): number {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height) return 0;
    return this.data[(y * this.width + x) * CHANNELS + c];
  }
  toFloat32(): Float32Array<ArrayBuffer> {
    return new Float32Array(this.data);
  }
}

/** Forward-difference gradients of an image (what GeneralPaint does on load). */
export function gradientsOf(u: Field): { gx: Field; gy: Field } {
  const { width: W, height: H } = u;
  const gx = Field.fill(W - 1, H, (x, y, c) => (c === 3 ? 0 : u.edge(x + 1, y, c) - u.edge(x, y, c)));
  const gy = Field.fill(W, H - 1, (x, y, c) => (c === 3 ? 0 : u.edge(x, y + 1, c) - u.edge(x, y, c)));
  return { gx, gy };
}

/** Eq. 6, mg_rhs.glsl: f = gx(x) - gx(x-1) + gy(y) - gy(y-1), clamp-to-black. */
export function computeRhs(gx: Field, gy: Field, W: number, H: number): Field {
  if (gx.width !== W - 1 || gx.height !== H || gy.width !== W || gy.height !== H - 1) {
    throw new Error('gradient field dimensions do not match image');
  }
  return Field.fill(W, H, (x, y, c) =>
    c === 3 ? 0 : gx.black(x, y, c) - gx.black(x - 1, y, c) + gy.black(x, y, c) - gy.black(x, y - 1, c),
  );
}

/** Apply the L_h stencil to f (clamp-to-edge neighbours). */
export function applyStencil(f: Field, lev: Pick<LevelSpec, 'middle' | 'edge' | 'corner'>, x: number, y: number, c: number): number {
  const n4 = f.edge(x + 1, y, c) + f.edge(x - 1, y, c) + f.edge(x, y + 1, c) + f.edge(x, y - 1, c);
  const d4 = f.edge(x + 1, y + 1, c) + f.edge(x + 1, y - 1, c) + f.edge(x - 1, y + 1, c) + f.edge(x - 1, y - 1, c);
  return lev.middle * f.edge(x, y, c) + lev.edge * n4 + lev.corner * d4;
}

/** vc_residual.glsl: r = rhs - L_h f */
export function residual(f: Field, rhs: Field, lev: LevelSpec): Field {
  return Field.fill(f.width, f.height, (x, y, c) => (c === 3 ? 0 : rhs.black(x, y, c) - applyStencil(f, lev, x, y, c)));
}

/**
 * vc_restrict.glsl (bilinear-hardware version): coarse(i,j) = sum of
 * [1/4 1/2 1/4; 1/2 1 1/2; 1/4 1/2 1/4] around fine(2i, 2j), clamp-to-black.
 */
export function restrict(fine: Field, cw: number, ch: number): Field {
  const w = [0.5, 1, 0.5];
  return Field.fill(cw, ch, (i, j, c) => {
    if (c === 3) return 0;
    let s = 0;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) s += w[dx + 1] * w[dy + 1] * fine.black(2 * i + dx, 2 * j + dy, c);
    return s;
  });
}

/**
 * vc_interpolate(_add).glsl: bilinear prolongation. Fine pixel x samples the
 * coarse grid at x/2 (clamp-to-edge), optionally adding an existing field.
 */
export function interpolate(coarse: Field, fw: number, fh: number, add?: Field): Field {
  return Field.fill(fw, fh, (x, y, c) => {
    if (c === 3) return 0;
    const x0 = x >> 1;
    const y0 = y >> 1;
    const tx = (x & 1) * 0.5;
    const ty = (y & 1) * 0.5;
    const a = coarse.edge(x0, y0, c);
    const b = coarse.edge(x0 + 1, y0, c);
    const d = coarse.edge(x0, y0 + 1, c);
    const e = coarse.edge(x0 + 1, y0 + 1, c);
    const v = (1 - ty) * ((1 - tx) * a + tx * b) + ty * ((1 - tx) * d + tx * e);
    return add ? v + add.data[add.idx(x, y) + c] : v;
  });
}

/**
 * vc_iter.glsl: modified Jacobi step  u <- (f - (L_h - (m - x) I) u) / (m - x)
 */
export function smooth(f: Field, rhs: Field, lev: LevelSpec, magic: number): Field {
  const inv = 1 / (lev.middle - magic);
  return Field.fill(f.width, f.height, (x, y, c) => {
    if (c === 3) return 0;
    const n4 = f.edge(x + 1, y, c) + f.edge(x - 1, y, c) + f.edge(x, y + 1, c) + f.edge(x, y - 1, c);
    const d4 = f.edge(x + 1, y + 1, c) + f.edge(x + 1, y - 1, c) + f.edge(x - 1, y + 1, c) + f.edge(x - 1, y - 1, c);
    return inv * (rhs.data[rhs.idx(x, y) + c] - lev.edge * n4 - lev.corner * d4 - magic * f.edge(x, y, c));
  });
}

export function channelMeans(f: Field): number[] {
  const sums = [0, 0, 0, 0];
  for (let i = 0; i < f.data.length; i += CHANNELS) for (let c = 0; c < CHANNELS; c++) sums[c] += f.data[i + c];
  const n = f.width * f.height;
  return sums.map((s) => s / n);
}

/** Shift each channel so its mean equals target[c] (the solution is defined up to a constant). */
export function recenter(f: Field, target: ArrayLike<number>): Field {
  const m = channelMeans(f);
  const out = f.clone();
  for (let i = 0; i < out.data.length; i += CHANNELS)
    for (let c = 0; c < 3; c++) out.data[i + c] += target[c] - m[c];
  return out;
}

/**
 * One V-cycle exactly as VCycle::update: no pre-smoothing, residual only on
 * the finest grid (coarser grids start from a zero guess so their residual
 * equals their RHS), zero on the coarsest grid, then interpolate + two
 * magic-x Jacobi smooths on the way up.
 */
export class ReferenceSolver {
  readonly levels: LevelSpec[];
  /** current solution (finest level) */
  f: Field;
  /** RHS for each level (level 0 = div g; deeper = restricted residuals) */
  readonly rhs: Field[];
  /** per-level corrections from the most recent cycle (debug/inspection) */
  readonly corr: Field[];
  /** per-channel mean the solution is re-centred to after every cycle (null disables) */
  targetMean: number[] | null = [0, 0, 0, 0];

  constructor(
    readonly width: number,
    readonly height: number,
    coarsening: Coarsening = 'padded',
  ) {
    this.levels = buildLevels(width, height, coarsening);
    this.f = new Field(width, height);
    this.rhs = this.levels.map((l) => new Field(l.width, l.height));
    this.corr = this.levels.map((l) => new Field(l.width, l.height));
  }

  setGradients(gx: Field, gy: Field): void {
    this.rhs[0] = computeRhs(gx, gy, this.width, this.height);
  }

  setRhs(rhs: Field): void {
    this.rhs[0] = rhs.clone();
  }

  vcycle(): void {
    const L = this.levels;
    // down
    for (let l = 0; l + 1 < L.length; l++) {
      const src = l === 0 ? residual(this.f, this.rhs[0], L[0]) : this.rhs[l];
      this.rhs[l + 1] = restrict(src, L[l + 1].width, L[l + 1].height);
    }
    // coarsest: zero
    const last = L.length - 1;
    this.corr[last] = new Field(L[last].width, L[last].height);
    // up
    for (let l = last - 1; l >= 0; l--) {
      let u = interpolate(this.corr[l + 1], L[l].width, L[l].height, l === 0 ? this.f : undefined);
      u = smooth(u, this.rhs[l], L[l], L[l].xA);
      u = smooth(u, this.rhs[l], L[l], L[l].xB);
      this.corr[l] = u;
    }
    this.f = L.length > 1 ? this.corr[0] : this.f;
    if (this.targetMean) this.f = recenter(this.f, this.targetMean);
  }
}

// ---------------------------------------------------------------------------
// Error metrics (Fig. 12)

/** RMS over pixels & RGB channels of (L_1 u - f). */
export function rmsResidual(u: Field, rhs: Field): number {
  const lev = buildLevels(u.width, u.height)[0];
  let s = 0;
  let n = 0;
  for (let y = 0; y < u.height; y++)
    for (let x = 0; x < u.width; x++)
      for (let c = 0; c < 3; c++) {
        const r = applyStencil(u, lev, x, y, c) - rhs.data[rhs.idx(x, y) + c];
        s += r * r;
        n++;
      }
  return Math.sqrt(s / n);
}

/** RMS over pixels & RGB channels of (G^x - d/dx u). */
export function rmsGradientError(u: Field, gx: Field): number {
  let s = 0;
  let n = 0;
  for (let y = 0; y < gx.height; y++)
    for (let x = 0; x < gx.width; x++)
      for (let c = 0; c < 3; c++) {
        const d = gx.data[gx.idx(x, y) + c] - (u.edge(x + 1, y, c) - u.edge(x, y, c));
        s += d * d;
        n++;
      }
  return Math.sqrt(s / n);
}

/** RMS difference over RGB after removing each channel's mean from both. */
export function rmsDifferenceModuloMean(a: Field, b: Field): number {
  const ma = channelMeans(a);
  const mb = channelMeans(b);
  let s = 0;
  let n = 0;
  for (let i = 0; i < a.data.length; i += CHANNELS)
    for (let c = 0; c < 3; c++) {
      const d = a.data[i + c] - ma[c] - (b.data[i + c] - mb[c]);
      s += d * d;
      n++;
    }
  return Math.sqrt(s / n);
}
