/**
 * Exact least-squares integration for small grids, used as ground truth.
 *
 * Solves  L_1 u = f  (5-point Neumann Laplacian, singular with constant null
 * space) per channel by conjugate gradients on the SPD operator -L_1
 * restricted to zero-mean vectors, in float64, to a tight tolerance.
 */
import { applyStencil, Field } from '../../src/solver/reference';
import { buildLevels, CHANNELS } from '../../src/solver/spec';

export function exactSolve(rhs: Field, tol = 1e-12, maxIter = 20000): Field {
  const { width: W, height: H } = rhs;
  const n = W * H;
  const lev = buildLevels(W, H)[0];
  const out = new Field(W, H);
  const tmp = new Field(W, H);
  const applyNegL = (v: Float64Array, res: Float64Array) => {
    for (let i = 0; i < n; i++) tmp.data[i * CHANNELS] = v[i];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) res[y * W + x] = -applyStencil(tmp, lev, x, y, 0);
  };
  const dot = (a: Float64Array, b: Float64Array) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  };
  const demean = (a: Float64Array) => {
    let m = 0;
    for (let i = 0; i < n; i++) m += a[i];
    m /= n;
    for (let i = 0; i < n; i++) a[i] -= m;
  };
  for (let c = 0; c < 3; c++) {
    const b = new Float64Array(n);
    for (let i = 0; i < n; i++) b[i] = -rhs.data[i * CHANNELS + c];
    demean(b); // project onto range (compatibility condition)
    const x = new Float64Array(n);
    const r = b.slice();
    const p = r.slice();
    const Ap = new Float64Array(n);
    let rr = dot(r, r);
    const stop = tol * tol * Math.max(dot(b, b), 1e-300);
    for (let it = 0; it < maxIter && rr > stop; it++) {
      applyNegL(p, Ap);
      const alpha = rr / dot(p, Ap);
      for (let i = 0; i < n; i++) {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
      }
      demean(r);
      const rr2 = dot(r, r);
      const beta = rr2 / rr;
      rr = rr2;
      for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    }
    demean(x);
    for (let i = 0; i < n; i++) out.data[i * CHANNELS + c] = x[i];
  }
  return out;
}

/** Deterministic PRNG (mulberry32) so tests are reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A smooth-ish synthetic "photo" with edges, for realistic test gradients. */
export function syntheticImage(W: number, H: number, seed = 1): Field {
  const r = rng(seed);
  const blobs = Array.from({ length: 6 }, () => ({ x: r() * W, y: r() * H, s: 0.1 + r() * 0.3, c: [r(), r(), r()] }));
  return Field.fill(W, H, (x, y, c) => {
    if (c === 3) return 0;
    let v = 0.3 + 0.2 * Math.sin((x / W) * 6.1 + c) * Math.cos((y / H) * 4.3);
    for (const b of blobs) {
      const d = Math.hypot(x - b.x, y - b.y) / (b.s * Math.max(W, H));
      v += d < 1 ? 0.3 * b.c[c] : 0; // hard edges
    }
    return v + 0.02 * (r() - 0.5); // a little texture
  });
}

export function randomField(W: number, H: number, seed: number, amp = 1): Field {
  const r = rng(seed);
  return Field.fill(W, H, (_x, _y, c) => (c === 3 ? 0 : amp * (r() * 2 - 1)));
}
