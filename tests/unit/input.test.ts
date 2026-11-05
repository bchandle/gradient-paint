import { describe, expect, it } from 'vitest';
import {
  isPalm,
  isTwoFingerTap,
  PALM_SIZE,
  pinchView,
  PolicyState,
  pressureProfile,
  pressureResponse,
  routePointer,
} from '../../src/app/input-policy';
import { buildStrokeGeometry, VERTEX_FLOATS } from '../../src/paint/geometry';
import { gaussianKernel, SmoothStroke, Vec2 } from '../../src/paint/stroke';

const base: PolicyState = { fingerMode: 'auto', penSeen: false, penActive: false, pickLatch: false, spaceDown: false, isMac: false };
const pen = { pointerType: 'pen', button: 0 };
const finger = { pointerType: 'touch', button: 0, width: 20, height: 20 };
const mouse = (button: number, extra = {}) => ({ pointerType: 'mouse', button, ...extra });

describe('pointer routing', () => {
  it('mouse keeps the original buttons: left draws, right configures, middle pans', () => {
    expect(routePointer(mouse(0), base)).toBe('draw');
    expect(routePointer(mouse(2), base)).toBe('configure');
    expect(routePointer(mouse(1), base)).toBe('pan');
    expect(routePointer(mouse(0, { altKey: true }), base)).toBe('configure');
    expect(routePointer(mouse(0, { ctrlKey: true }), { ...base, isMac: true })).toBe('configure');
    expect(routePointer(mouse(0, { ctrlKey: true }), base)).toBe('draw');
    expect(routePointer(mouse(0), { ...base, spaceDown: true })).toBe('pan');
  });

  it('the Pencil always draws (and ignores the eraser button)', () => {
    expect(routePointer(pen, base)).toBe('draw');
    expect(routePointer(pen, { ...base, penSeen: true, fingerMode: 'configure' })).toBe('draw');
    expect(routePointer({ pointerType: 'pen', button: 5 }, base)).toBe('ignore');
  });

  it('fingers draw until a Pencil is seen, then configure (auto mode)', () => {
    expect(routePointer(finger, base)).toBe('draw');
    expect(routePointer(finger, { ...base, penSeen: true })).toBe('configure');
    expect(routePointer(finger, { ...base, penSeen: true, fingerMode: 'draw' })).toBe('draw');
    expect(routePointer(finger, { ...base, fingerMode: 'configure' })).toBe('configure');
  });

  it('the Pick latch turns any press into a configure action', () => {
    expect(routePointer(pen, { ...base, pickLatch: true })).toBe('configure');
    expect(routePointer(finger, { ...base, pickLatch: true })).toBe('configure');
    expect(routePointer(mouse(0), { ...base, pickLatch: true })).toBe('configure');
  });

  it('rejects palms and touches while the Pencil is down', () => {
    expect(isPalm({ width: PALM_SIZE, height: 10 })).toBe(true);
    expect(isPalm({ width: 30, height: 30 })).toBe(false);
    expect(isPalm({})).toBe(false);
    expect(routePointer({ ...finger, width: 90, height: 70 }, base)).toBe('ignore');
    expect(routePointer(finger, { ...base, penActive: true })).toBe('ignore');
    expect(routePointer(pen, { ...base, penActive: true })).toBe('draw');
  });
});

describe('pressure', () => {
  it('maps a normal Pencil press (0.5) to the mouse strength (1)', () => {
    expect(pressureResponse(0.5)).toBe(1);
    expect(pressureResponse(1)).toBe(2);
    expect(pressureResponse(0.01)).toBe(0.05);
    expect(pressureResponse(0)).toBe(1); // unknown pressure
    expect(pressureResponse(0.25)).toBe(0.5);
  });

  it('strength mode scales the gradient; size mode scales width and keeps the gradient brush tone', () => {
    const p = [0.25, 0.5, 1];
    expect(pressureProfile(p, 'off', 4, true)).toEqual({});
    expect(pressureProfile(p, 'strength', 4, true)).toEqual({ strengths: [0.5, 1, 2] });
    const size = pressureProfile(p, 'size', 4, true);
    expect(size.radii).toEqual([2, 4, 6]);
    // tone of a gradient-brush stroke is proportional to width/brush size, so compensate
    size.radii!.forEach((r, i) => expect(r * size.strengths![i]).toBeCloseTo(4, 12));
    expect(pressureProfile(p, 'size', 4, false)).toEqual({ radii: [2, 4, 6] });
  });

  it('SmoothStroke resamples and smooths pressure alongside position', () => {
    const s = new SmoothStroke([1]);
    s.submit([0, 0], 0);
    s.submit([10, 0], 1); // pressure ramps linearly along the segment
    s.points.forEach((pt, i) => expect(s.pressures[i]).toBeCloseTo(pt[0] / 10, 9));
    const g = new SmoothStroke(gaussianKernel());
    for (let x = 0; x <= 30; x++) g.submit([x, 0], 0.7);
    expect(g.pressures.length).toBe(g.points.length);
    g.pressures.forEach((p) => expect(p).toBeCloseTo(0.7, 9));
    g.translate(5, 5);
    expect(g.pressures.length).toBe(g.points.length);
  });

  it('geometry carries per-point strength and radius', () => {
    const pts: Vec2[] = [[0, 10], [1, 10], [2, 10]];
    const g = buildStrokeGeometry(pts, { radius: 3, texRadius: 2, stepsAlong: 1, radii: [1, 2, 4], strengths: [0.1, 0.5, 2] });
    const verts = Array.from({ length: g.length / VERTEX_FLOATS }, (_, i) => g.subarray(i * VERTEX_FLOATS, (i + 1) * VERTEX_FLOATS));
    for (const v of verts) {
      const i = Math.round(v[0]); // points sit at x = 0, 1, 2
      expect(v[9]).toBeCloseTo([0.1, 0.5, 2][i], 6);
      const off = Math.abs(v[1] - 10);
      if (off > 1e-6) expect(off).toBeCloseTo([1, 2, 4][i], 5);
    }
    // default strength is 1
    const d = buildStrokeGeometry(pts, { radius: 3, texRadius: 2, stepsAlong: 1 });
    for (let k = 9; k < d.length; k += VERTEX_FLOATS) expect(d[k]).toBe(1);
  });
});

describe('two-finger gestures', () => {
  const v0 = { scale: 2, ox: 100, oy: 50 };
  it('pure pan keeps the scale and moves by the midpoint shift', () => {
    expect(pinchView(v0, [0, 0], [100, 0], [30, 20], [130, 20])).toEqual({ scale: 2, ox: 130, oy: 70 });
  });
  it('pinch keeps the image point under the fingers fixed', () => {
    const v = pinchView(v0, [200, 200], [300, 200], [150, 200], [350, 200]);
    expect(v.scale).toBe(4);
    const img = (x: number, y: number, view: typeof v0) => [(x - view.ox) / view.scale, (y - view.oy) / view.scale];
    expect(img(250, 200, v)).toEqual(img(250, 200, v0));
  });
  it('clamps zoom', () => {
    expect(pinchView(v0, [0, 0], [100, 0], [0, 0], [100000, 0]).scale).toBe(64);
  });
  it('distinguishes a tap from a gesture', () => {
    expect(isTwoFingerTap(150, 4)).toBe(true);
    expect(isTwoFingerTap(500, 4)).toBe(false);
    expect(isTwoFingerTap(150, 40)).toBe(false);
  });
});
