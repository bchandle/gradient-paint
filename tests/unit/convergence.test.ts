/**
 * Convergence of the multigrid integrator (CPU reference), following the
 * evaluation in Section 7 / Fig. 12 of the paper.
 */
import { describe, expect, it } from 'vitest';
import {
  channelMeans,
  computeRhs,
  Field,
  gradientsOf,
  ReferenceSolver,
  rmsDifferenceModuloMean,
  rmsGradientError,
  rmsResidual,
} from '../../src/solver/reference';
import { Coarsening } from '../../src/solver/spec';
import { exactSolve, randomField, rng, syntheticImage } from './exact';

/** Asymptotic per-cycle error reduction, measured by power iteration on the homogeneous problem. */
function convergenceFactor(W: number, H: number, coarsening: Coarsening, cycles = 30): number {
  const s = new ReferenceSolver(W, H, coarsening);
  s.setRhs(new Field(W, H)); // f = 0, so the iterate *is* the error
  s.targetMean = [0, 0, 0, 0]; // project out the null space (constants)
  s.f = randomField(W, H, 1234);
  const norm = (f: Field) => Math.sqrt(f.data.reduce((a, v, i) => (i % 4 === 0 ? a + v * v : a), 0));
  let prev = norm(s.f);
  const ratios: number[] = [];
  for (let k = 0; k < cycles; k++) {
    s.vcycle();
    const n = norm(s.f);
    ratios.push(n / prev);
    // renormalise to avoid underflow
    for (let i = 0; i < s.f.data.length; i++) s.f.data[i] /= n;
    prev = 1;
  }
  const tail = ratios.slice(cycles - 10);
  return Math.exp(tail.reduce((a, r) => a + Math.log(r), 0) / tail.length);
}

describe('convergence factor (Section 7: rho ~= 0.34 on 65x65)', () => {
  it('reproduces the paper on a 65x65 grid', () => {
    for (const mode of ['paper', 'padded'] as const) {
      const rho = convergenceFactor(65, 65, mode);
      expect(rho).toBeGreaterThan(0.2);
      expect(rho).toBeLessThan(0.4);
    }
  });

  it('stays fast on other 2^k+1 grids', () => {
    for (const [W, H] of [[33, 33], [129, 129], [129, 65]]) {
      expect(convergenceFactor(W, H, 'padded')).toBeLessThan(0.4);
    }
  });

  it('padded coarsening keeps general (even / odd) sizes converging well', () => {
    for (const [W, H] of [[64, 64], [100, 75], [128, 96], [99, 64], [160, 90]]) {
      const padded = convergenceFactor(W, H, 'padded');
      expect(padded).toBeLessThan(0.56);
    }
  });

  it('padded coarsening beats the original scheme on even sizes', () => {
    for (const [W, H] of [[64, 64], [96, 72]]) {
      expect(convergenceFactor(W, H, 'padded')).toBeLessThan(convergenceFactor(W, H, 'paper') - 0.05);
    }
  });

  it('still converges (slowly) for elongated images', () => {
    expect(convergenceFactor(17, 129, 'padded')).toBeLessThan(0.65);
    expect(convergenceFactor(256, 32, 'padded')).toBeLessThan(0.7);
    expect(convergenceFactor(400, 7, 'padded')).toBeLessThan(0.97);
  });
});

describe('integrating conservative gradient fields recovers the image', () => {
  const sizes: [number, number][] = [
    [2, 2], [3, 5], [16, 16], [17, 17], [65, 65], [64, 48], [100, 37], [129, 120], [257, 257],
  ];
  for (const [W, H] of sizes) {
    it(`${W}x${H}: from a random guess to < 1e-4 RMS in 20 cycles`, () => {
      const img = syntheticImage(W, H, W * 31 + H);
      const { gx, gy } = gradientsOf(img);
      const s = new ReferenceSolver(W, H);
      s.setGradients(gx, gy);
      s.targetMean = channelMeans(img);
      s.f = randomField(W, H, 99, 0.5);
      for (let k = 0; k < 20; k++) s.vcycle();
      expect(rmsDifferenceModuloMean(s.f, img)).toBeLessThan(1e-4);
      // the target mean is restored exactly (white balance / recentering)
      const m = channelMeans(s.f);
      for (let c = 0; c < 3; c++) expect(m[c]).toBeCloseTo(s.targetMean![c], 10);
    });
  }

  it('error decreases monotonically every cycle (padded coarsening)', () => {
    for (const [W, H] of [[65, 65], [64, 64], [256, 200], [100, 37]]) {
      const img = syntheticImage(W, H, 3);
      const { gx, gy } = gradientsOf(img);
      const s = new ReferenceSolver(W, H);
      s.setGradients(gx, gy);
      s.targetMean = channelMeans(img);
      s.f = randomField(W, H, 7, 0.5);
      let prev = rmsDifferenceModuloMean(s.f, img);
      for (let k = 0; k < 12; k++) {
        s.vcycle();
        const e = rmsDifferenceModuloMean(s.f, img);
        expect(e).toBeLessThan(prev);
        prev = e;
      }
    }
  });

  it('Fig. 12: recognisable after one cycle, nearly perfect after two (257x257, random start)', () => {
    const W = 257;
    const img = syntheticImage(W, W, 11);
    const { gx, gy } = gradientsOf(img);
    const s = new ReferenceSolver(W, W);
    s.setGradients(gx, gy);
    s.targetMean = channelMeans(img);
    const r = rng(5);
    s.f = Field.fill(W, W, (_x, _y, c) => (c === 3 ? 0 : r())); // u = random in [0,1], like the paper
    const res0 = rmsResidual(s.f, s.rhs[0]);
    const gxe0 = rmsGradientError(s.f, gx);
    s.vcycle();
    const res1 = rmsResidual(s.f, s.rhs[0]);
    s.vcycle();
    const res2 = rmsResidual(s.f, s.rhs[0]);
    const gxe2 = rmsGradientError(s.f, gx);
    expect(res0).toBeGreaterThan(1); // the paper's starting RMS is ~1.3
    expect(res1).toBeLessThan(res0 / 5);
    expect(res2).toBeLessThan(res0 / 40);
    expect(gxe2).toBeLessThan(gxe0 / 10);
  });
});

describe('integrating non-conservative (edited) gradient fields', () => {
  for (const [W, H] of [[9, 9], [33, 20], [48, 48], [65, 65]]) {
    it(`${W}x${H}: converges to the exact least-squares solution`, () => {
      // a painted field: an image's gradients plus arbitrary random strokes (curl != 0)
      const img = syntheticImage(W, H, 2);
      const g = gradientsOf(img);
      const nx = randomField(W - 1, H, 3, 0.2);
      const ny = randomField(W, H - 1, 4, 0.2);
      for (let i = 0; i < g.gx.data.length; i++) g.gx.data[i] += nx.data[i];
      for (let i = 0; i < g.gy.data.length; i++) g.gy.data[i] += ny.data[i];
      const rhs = computeRhs(g.gx, g.gy, W, H);
      const exact = exactSolve(rhs);
      const s = new ReferenceSolver(W, H);
      s.setRhs(rhs);
      for (let k = 0; k < 30; k++) s.vcycle();
      expect(rmsDifferenceModuloMean(s.f, exact)).toBeLessThan(1e-7);
      // and its residual is (numerically) zero, unlike the gradient error
      expect(rmsResidual(s.f, rhs)).toBeLessThan(1e-7);
      expect(rmsGradientError(s.f, g.gx)).toBeGreaterThan(1e-3);
    });
  }

  it('the least-squares solution minimises the gradient-domain energy', () => {
    const W = 20;
    const H = 14;
    const gx = randomField(W - 1, H, 21);
    const gy = randomField(W, H - 1, 22);
    const s = new ReferenceSolver(W, H);
    s.setGradients(gx, gy);
    for (let k = 0; k < 40; k++) s.vcycle();
    const energy = (u: Field) => {
      let e = 0;
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          for (let c = 0; c < 3; c++) {
            if (x + 1 < W) e += (u.edge(x + 1, y, c) - u.edge(x, y, c) - gx.data[gx.idx(x, y) + c]) ** 2;
            if (y + 1 < H) e += (u.edge(x, y + 1, c) - u.edge(x, y, c) - gy.data[gy.idx(x, y) + c]) ** 2;
          }
      return e;
    };
    const e0 = energy(s.f);
    const r = rng(3);
    for (let t = 0; t < 20; t++) {
      const p = s.f.clone();
      const i = Math.floor(r() * W * H) * 4 + Math.floor(r() * 3);
      p.data[i] += (r() - 0.5) * 1e-2;
      expect(energy(p)).toBeGreaterThan(e0);
    }
  });
});

describe('solver invariants', () => {
  it('the result does not depend on a constant offset of the initial guess', () => {
    const W = 40;
    const H = 30;
    const img = syntheticImage(W, H, 8);
    const { gx, gy } = gradientsOf(img);
    const run = (offset: number) => {
      const s = new ReferenceSolver(W, H);
      s.setGradients(gx, gy);
      s.f = randomField(W, H, 1);
      for (let i = 0; i < s.f.data.length; i++) if (i % 4 !== 3) s.f.data[i] += offset;
      s.vcycle();
      s.vcycle();
      return s.f;
    };
    const a = run(0);
    const b = run(17.5);
    for (let i = 0; i < a.data.length; i++) expect(a.data[i]).toBeCloseTo(b.data[i], 9);
  });

  it('a V-cycle is affine in (guess, rhs): zero rhs and zero guess stay zero', () => {
    const s = new ReferenceSolver(33, 21);
    s.vcycle();
    expect(s.f.data.every((v) => v === 0)).toBe(true);
  });

  it('channels are integrated independently', () => {
    const W = 24;
    const H = 18;
    const img = syntheticImage(W, H, 4);
    const { gx, gy } = gradientsOf(img);
    // zero out green & blue gradients: those channels must stay flat
    for (const g of [gx, gy]) for (let i = 0; i < g.data.length; i += 4) g.data[i + 1] = g.data[i + 2] = 0;
    const s = new ReferenceSolver(W, H);
    s.setGradients(gx, gy);
    s.f = randomField(W, H, 3);
    for (let k = 0; k < 25; k++) s.vcycle();
    for (let i = 0; i < s.f.data.length; i += 4) {
      expect(Math.abs(s.f.data[i + 1])).toBeLessThan(1e-6);
      expect(Math.abs(s.f.data[i + 2])).toBeLessThan(1e-6);
    }
  });
});
