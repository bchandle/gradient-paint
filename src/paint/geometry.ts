/**
 * Stroke geometry, ported from GeneralPaint::render_stroke.
 *
 * A stroke is drawn as a ribbon of quads (one per segment) plus a triangle
 * "blob" at every joint. The centre line gets depth 0.1 and the outer edges
 * 0.9, so with a LESS depth test each pixel takes its value from the nearest
 * part of the stroke only, and overlapping pieces never double-apply.
 *
 * Every vertex carries texture coordinates into the brush (the "along, across"
 * stroke-local frame, or source-image coordinates for the clone brush) and the
 * local frame (tangent t, normal n) that maps brush gradients into the image.
 */
import type { Vec2 } from './stroke';

/** floats per vertex: pos.xy, depth, uv.xy, t.xy, n.xy */
export const VERTEX_FLOATS = 9;
export const DEPTH_CENTER = 0.1;
export const DEPTH_EDGE = 0.9;

export interface StrokeGeometryOptions {
  /** half-width of the ribbon in image pixels (brush_size) */
  radius: number;
  /** brush texture half-height (texture v runs 0 .. 2*texRadius across the stroke) */
  texRadius: number;
  /** brush texture length; <= 1 means "no along variation" (gradient brush) */
  stepsAlong: number;
  /** clone brush: texture coordinates are image positions + offset, frame is identity */
  cloneOffset?: Vec2 | null;
}

/** Stroke normal: the original's perpendicular() (CCW rotation), expressed in y-down coordinates. */
export function strokeNormal(t: Vec2): Vec2 {
  return [t[1], -t[0]];
}

function normalize(v: Vec2): Vec2 {
  const l = Math.hypot(v[0], v[1]);
  return l > 0 ? [v[0] / l, v[1] / l] : [1, 0];
}

export function buildStrokeGeometry(points: Vec2[], opts: StrokeGeometryOptions): Float32Array<ArrayBuffer> {
  const n = points.length;
  if (n < 2) return new Float32Array(0);
  const clone = opts.cloneOffset ?? null;
  const R = opts.texRadius;
  const r = opts.radius;
  const out: number[] = [];

  // frame of the current primitive
  let ft: Vec2 = [1, 0];
  let fn: Vec2 = [0, 1];
  const setFrameFromNormal = (perp: Vec2) => {
    if (clone) {
      ft = [1, 0];
      fn = [0, 1];
    } else {
      fn = perp;
      ft = [-perp[1], perp[0]]; // inverse of strokeNormal
    }
  };
  let tu = 0;
  let tv = 0;
  const vert = (p: Vec2, depth: number) => {
    const u = clone ? p[0] + clone[0] : tu;
    const v = clone ? p[1] + clone[1] : tv;
    out.push(p[0], p[1], depth, u, v, ft[0], ft[1], fn[0], fn[1]);
  };
  const tex = (u: number, v: number) => {
    tu = u;
    tv = v;
  };
  const add = (a: Vec2, s: number, d: Vec2): Vec2 => [a[0] + s * d[0], a[1] + s * d[1]];

  let prevPerp = strokeNormal(normalize([points[1][0] - points[0][0], points[1][1] - points[0][1]]));
  let along = 0;
  const steps = opts.stepsAlong;

  for (let i = 1; i < n; i++) {
    const cur = points[i - 1];
    const next = points[i];
    const nextPerp = strokeNormal(normalize([next[0] - cur[0], next[1] - cur[1]]));
    const curPerp = normalize([nextPerp[0] + prevPerp[0], nextPerp[1] + prevPerp[1]]);
    let len = Math.hypot(next[0] - cur[0], next[1] - cur[1]);
    if (steps <= 1) {
      along = 0.5;
      len = 0;
    }

    // joint blob: two triangles fanning from cur between the previous and next normals
    setFrameFromNormal(curPerp);
    tex(along, 2 * R);
    const bp0 = add(cur, r, nextPerp);
    const bp1 = add(cur, r, prevPerp);
    vert(bp0, DEPTH_EDGE);
    vert(bp1, DEPTH_EDGE);
    tex(along, R);
    vert(cur, DEPTH_CENTER);
    tex(along, 0);
    vert(add(cur, -r, nextPerp), DEPTH_EDGE);
    vert(add(cur, -r, prevPerp), DEPTH_EDGE);
    tex(along, R);
    vert(cur, DEPTH_CENTER);

    // segment: two quads (outer edge -> centre line), each as two triangles
    setFrameFromNormal(nextPerp);
    for (const side of [1, -1]) {
      const v0 = side > 0 ? 2 * R : 0;
      const q: [Vec2, number, number, number][] = [
        [add(cur, side * r, nextPerp), along, v0, DEPTH_EDGE],
        [add(next, side * r, nextPerp), along + len, v0, DEPTH_EDGE],
        [next, along + len, R, DEPTH_CENTER],
        [cur, along, R, DEPTH_CENTER],
      ];
      for (const k of [0, 1, 2, 0, 2, 3]) {
        tex(q[k][1], q[k][2]);
        vert(q[k][0], q[k][3]);
      }
    }

    along += len;
    if (along + len >= steps) along -= steps;
    prevPerp = nextPerp;
  }
  return new Float32Array(out);
}
