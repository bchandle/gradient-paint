import { describe, expect, it } from 'vitest';
import { blend, BLEND_MODES } from '../../src/paint/blend';
import { buildStrokeGeometry, DEPTH_CENTER, DEPTH_EDGE, strokeNormal, VERTEX_FLOATS } from '../../src/paint/geometry';
import { hsvToRgb, rgbToHsv } from '../../src/paint/hsv';
import { gaussianKernel, SmoothStroke, Stroke, Vec2 } from '../../src/paint/stroke';

describe('SmoothStroke', () => {
  it('resamples input at unit arc-length steps', () => {
    const s = new SmoothStroke([1]); // identity kernel: no smoothing
    s.submit([0, 0]);
    s.submit([10.5, 0]);
    s.submit([10.5, 5.2]);
    const pts = s.points;
    for (let i = 1; i < pts.length; i++) {
      expect(Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])).toBeLessThanOrEqual(1 + 1e-9);
    }
    expect(pts.length).toBe(15); // start + 15 unit steps, the newest sample still queued
    expect(pts[10]).toEqual([10, 0]);
  });

  it('smoothing preserves straight lines (kernel sums to 1) and lags by half the kernel', () => {
    const k = gaussianKernel();
    expect(k.length).toBe(9);
    expect(k.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    const s = new SmoothStroke(k);
    for (let x = 0; x <= 30; x += 0.37) s.submit([x, 5]);
    for (const p of s.points) expect(p[1]).toBeCloseTo(5, 12);
    expect(s.points[0][0]).toBeCloseTo(4, 9); // centre of the first 9 samples 0..8
    expect(s.points.length).toBe(30 - 8 + 1 - 1);
  });

  it('rounds corners', () => {
    const s = new SmoothStroke(gaussianKernel());
    for (let x = 0; x <= 20; x++) s.submit([x, 0]);
    for (let y = 1; y <= 20; y++) s.submit([20, y]);
    const near = s.points.reduce((m, p) => Math.min(m, Math.hypot(p[0] - 20, p[1])), Infinity);
    expect(near).toBeGreaterThan(0.5); // never reaches the sharp corner
  });

  it('versions change on edits (renderers use this as the dirty flag)', () => {
    const s = new Stroke();
    const v0 = s.version;
    s.submit([1, 1]);
    s.translate(2, 3);
    expect(s.points).toEqual([[3, 4]]);
    expect(s.version).toBeGreaterThan(v0 + 1);
  });
});

describe('stroke geometry', () => {
  const line = (n: number, y = 10): Vec2[] => Array.from({ length: n }, (_, i) => [5 + i, y] as Vec2);
  const verts = (g: Float32Array) => Array.from({ length: g.length / VERTEX_FLOATS }, (_, i) => Array.from(g.subarray(i * VERTEX_FLOATS, (i + 1) * VERTEX_FLOATS)));

  it('emits a joint blob + segment ribbon (18 vertices) per segment', () => {
    const g = buildStrokeGeometry(line(6), { radius: 3, texRadius: 2, stepsAlong: 1 });
    expect(g.length).toBe(5 * 18 * VERTEX_FLOATS);
    expect(buildStrokeGeometry(line(1), { radius: 3, texRadius: 2, stepsAlong: 1 }).length).toBe(0);
  });

  it('puts the centre line at depth 0.1 and the outer edge at 0.9, |offset| = radius', () => {
    const r = 3;
    for (const v of verts(buildStrokeGeometry(line(4), { radius: r, texRadius: 2, stepsAlong: 1 }))) {
      const off = Math.abs(v[1] - 10);
      if (Math.abs(v[2] - DEPTH_CENTER) < 1e-6) expect(off).toBeCloseTo(0, 6);
      else {
        expect(v[2]).toBeCloseTo(DEPTH_EDGE, 6);
        expect(off).toBeCloseTo(r, 6);
      }
    }
  });

  it('maps the cross-section to brush v in [0, 2R] with R on the centre line, along the normal', () => {
    const R = 5;
    for (const v of verts(buildStrokeGeometry(line(4), { radius: 3, texRadius: R, stepsAlong: 1 }))) {
      const [, y, , , tv, tx, ty, nx, ny] = v;
      expect([tx, ty]).toEqual([1, 0]);
      expect([nx, ny]).toEqual(strokeNormal([1, 0]));
      // moving along +n changes v towards 2R
      const across = (y - 10) * ny;
      expect(tv).toBeCloseTo(R + (across / 3) * R, 5);
    }
  });

  it('normal is the counter-clockwise perpendicular as seen on screen (matches the original y-up convention)', () => {
    // heading right on screen, CCW-perpendicular points up the screen (negative y in image space)
    expect(strokeNormal([1, 0])).toEqual([0, -1]);
    expect(strokeNormal([0, 1])).toEqual([1, -0]);
  });

  it('loops edge-brush texture coordinates every `stepsAlong` pixels', () => {
    const g = verts(buildStrokeGeometry(line(40), { radius: 3, texRadius: 4, stepsAlong: 10 }));
    const us = g.map((v) => v[3]);
    // like the original, playback wraps one step early (when the *next* segment would overflow)
    expect(Math.min(...us)).toBeGreaterThanOrEqual(-1 - 1e-6);
    expect(Math.max(...us)).toBeLessThanOrEqual(10 + 1e-6);
    // segment 12 starts at along = 12 mod 10 = 2 (wraps before exceeding the texture)
    const seg12 = g[12 * 18 + 6];
    expect(seg12[3]).toBeCloseTo(2, 5);
  });

  it('clone geometry uses image-space texture coordinates and an identity frame', () => {
    const g = verts(buildStrokeGeometry(line(3), { radius: 2, texRadius: 0, stepsAlong: 0, cloneOffset: [7, -3] }));
    for (const v of g) {
      expect(v[3]).toBeCloseTo(v[0] + 7, 5);
      expect(v[4]).toBeCloseTo(v[1] - 3, 5);
      expect(v.slice(5, 9)).toEqual([1, 0, 0, 1]);
      expect(v[9]).toBe(1); // default strength
    }
  });
});

describe('blend modes (Section 5)', () => {
  it('add sums (Eq. 1)', () => expect(blend('add', [1, 2], [0.5, -1])).toEqual([1.5, 1]));
  it('over replaces (Eq. 3)', () => expect(blend('over', [1, 2], [0.5, -1])).toEqual([0.5, -1]));
  it('max keeps the longer vector (Eq. 2)', () => {
    expect(blend('max', [1, 2], [0.5, -1])).toEqual([1, 2]);
    expect(blend('max', [0.1, 0.1], [0.5, -1])).toEqual([0.5, -1]);
  });
  it('min keeps the shorter vector', () => {
    expect(blend('min', [1, 2], [0.5, -1])).toEqual([0.5, -1]);
    expect(blend('min', [0.1, 0.1], [0.5, -1])).toEqual([0.1, 0.1]);
  });
  it('directional scales the background by 1 + b.g / g.g (Eq. 4)', () => {
    const [x, y] = blend('dir', [2, 0], [1, 5]);
    expect([x, y]).toEqual([2 * (1 + 2 / 4), 0]);
    // never changes direction, never creates gradients where there were none
    expect(blend('dir', [2, 0], [-100, 0])).toEqual([0, 0]);
    expect(blend('dir', [0, 0], [3, 3])).toEqual([0, 0]);
  });
  it('all modes leave the background alone under a zero brush except over/min', () => {
    for (const m of BLEND_MODES) {
      const r = blend(m, [0.3, -0.2], [0, 0]);
      if (m === 'over' || m === 'min') expect(r).toEqual([0, 0]);
      else expect(r).toEqual([0.3, -0.2]);
    }
  });
});

describe('HSV', () => {
  it('round-trips', () => {
    for (const c of [[1, 0, 0], [0.2, 0.5, 0.9], [0.5, 0.5, 0.5], [0, 0, 0], [0.9, 0.9, 0.1]] as [number, number, number][]) {
      const back = hsvToRgb(rgbToHsv(c));
      for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(c[i], 9);
    }
  });
  it('negative value inverts the colour (used to paint reversed gradients)', () => {
    expect(hsvToRgb([0, 0, -0.5])).toEqual([-0.5, -0.5, -0.5]);
    expect(hsvToRgb([1, 1, 1])).toEqual(hsvToRgb([0, 1, 1]));
  });
});
