/**
 * Correctness of the individual multigrid building blocks (CPU reference).
 */
import { describe, expect, it } from 'vitest';
import {
  applyStencil,
  channelMeans,
  computeRhs,
  Field,
  gradientsOf,
  interpolate,
  recenter,
  residual,
  restrict,
  smooth,
} from '../../src/solver/reference';
import { buildLevels, coarserSize, coarserSizeWith, magicX, stencil } from '../../src/solver/spec';
import { randomField, syntheticImage } from './exact';

const maxAbs = (a: Field, b: Field) => {
  let m = 0;
  for (let i = 0; i < a.data.length; i++) m = Math.max(m, Math.abs(a.data[i] - b.data[i]));
  return m;
};

describe('grid hierarchy', () => {
  it('halves 2^k+1 grids exactly down to 1x1', () => {
    expect(buildLevels(513, 513).map((l) => l.width)).toEqual([513, 257, 129, 65, 33, 17, 9, 5, 3, 2, 1]);
  });

  it('matches VCycle.cpp sizes in paper mode', () => {
    expect(buildLevels(64, 10, 'paper').map((l) => [l.width, l.height])).toEqual([
      [64, 10], [32, 5], [16, 3], [8, 2], [4, 1], [2, 1], [1, 1],
    ]);
  });

  it('padded mode only differs for even sizes and always covers the fine grid', () => {
    for (let n = 1; n < 300; n++) {
      const c = coarserSizeWith(n, 'padded');
      if (n % 2 === 1) expect(c).toBe(coarserSize(n));
      if (n > 2) expect(2 * (c - 1)).toBeGreaterThanOrEqual(n - 1); // last coarse node reaches the last fine node
      if (n >= 2) expect(c).toBeLessThan(n);
    }
  });

  it('terminates for every shape', () => {
    for (const [w, h] of [[1, 1], [2, 1], [1, 2], [2, 2], [1000, 3], [3, 1000], [1920, 1080]]) {
      const levels = buildLevels(w, h);
      const last = levels[levels.length - 1];
      expect([last.width, last.height]).toEqual([1, 1]);
      expect(levels.length).toBeLessThan(40);
    }
  });

  it('assigns spacing h = 2^level', () => {
    buildLevels(100, 60).forEach((l, i) => expect(l.h).toBe(2 ** i));
  });
});

describe('stencil coefficients (Eq. 8)', () => {
  it('L_1 is the 5-point Laplacian', () => {
    const s = stencil(1);
    expect(s.middle).toBeCloseTo(-4, 12);
    expect(s.edge).toBeCloseTo(1, 12);
    expect(s.corner).toBeCloseTo(0, 12);
  });

  it('annihilates constants at every level (row sums are zero)', () => {
    for (let l = 0; l < 12; l++) {
      const s = stencil(2 ** l);
      expect(s.middle + 4 * s.edge + 4 * s.corner).toBeCloseTo(0, 12);
    }
  });

  it('tends to the (1/3)[1 1 1; 1 -8 1; 1 1 1] stencil for large h', () => {
    const s = stencil(2 ** 20);
    expect(s.middle).toBeCloseTo(-8 / 3, 9);
    expect(s.edge).toBeCloseTo(1 / 3, 9);
    expect(s.corner).toBeCloseTo(1 / 3, 9);
  });

  it('uses the magic relaxation constants from Fig. 9', () => {
    const m1 = magicX(1);
    expect(m1.xA).toBeCloseTo(-2.1532 + 1.507 + 0.5882, 12);
    expect(m1.xB).toBeCloseTo(0.1138 + 0.9529 + 1.5065, 12);
    const m4 = magicX(4);
    expect(m4.xA).toBeCloseTo(-2.1532 + 1.507 / 4 + 0.5882 / 16, 12);
  });

  it('keeps the smoother well defined (m - x != 0)', () => {
    for (const l of buildLevels(4097, 4097)) {
      expect(Math.abs(l.middle - l.xA)).toBeGreaterThan(0.5);
      expect(Math.abs(l.middle - l.xB)).toBeGreaterThan(0.5);
    }
  });
});

describe('right-hand side (Eq. 6)', () => {
  it('is the Neumann Laplacian of the image when gradients are conservative', () => {
    for (const [W, H] of [[2, 2], [7, 5], [31, 40]]) {
      const u = syntheticImage(W, H, W);
      const { gx, gy } = gradientsOf(u);
      const rhs = computeRhs(gx, gy, W, H);
      const lev = buildLevels(W, H)[0];
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          for (let c = 0; c < 3; c++) expect(rhs.data[rhs.idx(x, y) + c]).toBeCloseTo(applyStencil(u, lev, x, y, c), 12);
    }
  });

  it('sums to zero for any gradient field (compatibility condition)', () => {
    const W = 23;
    const H = 17;
    const rhs = computeRhs(randomField(W - 1, H, 1), randomField(W, H - 1, 2), W, H);
    for (let c = 0; c < 3; c++) expect(channelMeans(rhs)[c]).toBeCloseTo(0, 13);
  });

  it('rejects mismatched gradient dimensions', () => {
    expect(() => computeRhs(new Field(5, 5), new Field(5, 4), 5, 5)).toThrow();
  });
});

describe('restriction and interpolation (Eq. 7)', () => {
  it('restricts an impulse to the [1/4 1/2 1/4; 1/2 1 1/2; 1/4 1/2 1/4] stencil', () => {
    const fine = new Field(9, 9);
    // a single impulse at fine (4,4) contributes to coarse (2,2) with weight 1,
    fine.data[fine.idx(4, 4)] = 1;
    let c = restrict(fine, 5, 5);
    expect(c.data[c.idx(2, 2)]).toBe(1);
    // ... impulse at (5,4) -> coarse (2,2),(3,2) with weight 1/2 each,
    fine.data.fill(0);
    fine.data[fine.idx(5, 4)] = 1;
    c = restrict(fine, 5, 5);
    expect([c.data[c.idx(2, 2)], c.data[c.idx(3, 2)]]).toEqual([0.5, 0.5]);
    // ... impulse at (5,5) -> four coarse nodes with weight 1/4 each.
    fine.data.fill(0);
    fine.data[fine.idx(5, 5)] = 1;
    c = restrict(fine, 5, 5);
    expect([c.data[c.idx(2, 2)], c.data[c.idx(3, 2)], c.data[c.idx(2, 3)], c.data[c.idx(3, 3)]]).toEqual([0.25, 0.25, 0.25, 0.25]);
  });

  it('is "four times larger than the standard" restriction: interior mass is multiplied by 4', () => {
    const fine = Field.fill(17, 17, (x, y, c) => (c === 0 && x > 2 && x < 14 && y > 2 && y < 14 ? 1 : 0));
    const c = restrict(fine, 9, 9);
    const sum = (f: Field) => f.data.filter((_, i) => i % 4 === 0).reduce((a, b) => a + b, 0);
    expect(sum(c)).toBeCloseTo(sum(fine), 12); // each fine value is spread with total weight 1 ...
    expect(c.data[c.idx(4, 4)]).toBe(4); // ... so coarse values are 4x the local fine value
  });

  it('interpolates bilinear functions exactly on 2^k+1 grids', () => {
    const coarse = Field.fill(9, 5, (x, y, c) => (c === 3 ? 0 : 1 + 2 * x - 3 * y + 0.5 * x * y));
    const fine = interpolate(coarse, 17, 9);
    for (let y = 0; y < 9; y++)
      for (let x = 0; x < 17; x++) {
        const X = x / 2;
        const Y = y / 2;
        expect(fine.data[fine.idx(x, y)]).toBeCloseTo(1 + 2 * X - 3 * Y + 0.5 * X * Y, 12);
      }
  });

  it('reproduces constants on every grid size (clamp-to-edge)', () => {
    for (const [fw, fh] of [[16, 10], [17, 9], [2, 3]]) {
      const levels = buildLevels(fw, fh);
      const coarse = Field.fill(levels[1].width, levels[1].height, (_x, _y, c) => (c === 3 ? 0 : 0.7));
      const fine = interpolate(coarse, fw, fh);
      for (let i = 0; i < fine.data.length; i += 4) expect(fine.data[i]).toBeCloseTo(0.7, 12);
    }
  });

  it('interpolate-add adds the existing fine solution', () => {
    const coarse = randomField(5, 5, 3);
    const base = randomField(9, 9, 4);
    const a = interpolate(coarse, 9, 9, base);
    const b = interpolate(coarse, 9, 9);
    for (let i = 0; i < a.data.length; i++) if (i % 4 !== 3) expect(a.data[i]).toBeCloseTo(b.data[i] + base.data[i], 12);
  });
});

describe('residual and smoother', () => {
  const W = 21;
  const H = 13;
  const u = syntheticImage(W, H, 5);
  const { gx, gy } = gradientsOf(u);
  const rhs = computeRhs(gx, gy, W, H);
  const lev = buildLevels(W, H)[0];

  it('residual of the exact solution is zero', () => {
    expect(maxAbs(residual(u, rhs, lev), new Field(W, H))).toBeLessThan(1e-12);
  });

  it('the exact solution is a fixed point of both magic-x Jacobi steps', () => {
    expect(maxAbs(smooth(u, rhs, lev, lev.xA), u)).toBeLessThan(1e-12);
    expect(maxAbs(smooth(u, rhs, lev, lev.xB), u)).toBeLessThan(1e-12);
  });

  it('a fixed point of the smoother on every level is a solution of L_h u = f', () => {
    for (const l of buildLevels(33, 33).slice(1, 4)) {
      const f = randomField(l.width, l.height, l.index);
      const r = randomField(l.width, l.height, 10 + l.index);
      const s = smooth(f, r, l, l.xA);
      // Jacobi identity: (m - x) s = r - (L - (m - x)) f  =>  checks the stencil algebra
      for (let y = 0; y < l.height; y++)
        for (let x = 0; x < l.width; x++) {
          const lhs = (l.middle - l.xA) * s.data[s.idx(x, y)];
          const rhsv = r.data[r.idx(x, y)] - (applyStencil(f, l, x, y, 0) - (l.middle - l.xA) * f.data[f.idx(x, y)]);
          expect(lhs).toBeCloseTo(rhsv, 10);
        }
    }
  });

  it('recenter sets per-channel means and preserves gradients', () => {
    const f = randomField(W, H, 9);
    const g = recenter(f, [0.1, 0.2, 0.3, 0]);
    const m = channelMeans(g);
    expect(m[0]).toBeCloseTo(0.1, 12);
    expect(m[1]).toBeCloseTo(0.2, 12);
    expect(m[2]).toBeCloseTo(0.3, 12);
    const d0 = f.data[f.idx(3, 3)] - f.data[f.idx(4, 3)];
    const d1 = g.data[g.idx(3, 3)] - g.data[g.idx(4, 3)];
    expect(d1).toBeCloseTo(d0, 12);
  });
});
