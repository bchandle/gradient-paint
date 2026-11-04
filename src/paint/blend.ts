/**
 * Blend modes (Section 5). The GPU versions live in painter.ts; this CPU
 * model documents the equations and is what the tests check the GPU against.
 *
 * All operations are per colour channel on 2D gradient vectors.
 */
export const BLEND_MODES = ['add', 'dir', 'max', 'min', 'over'] as const;
export type BlendMode = (typeof BLEND_MODES)[number];

export const BLEND_INDEX: Record<BlendMode, number> = { add: 0, dir: 1, max: 2, min: 3, over: 4 };

export const BLEND_DESCRIPTIONS: Record<BlendMode, string> = {
  add: 'additive — accumulates strokes (Eq. 1)',
  dir: 'directional — enhances background gradients that agree with the brush (Eq. 4)',
  max: 'maximum — keeps the stronger gradient (Eq. 2)',
  min: 'minimum — keeps the weaker gradient',
  over: 'over — replaces the background (Eq. 3)',
};

export type G2 = [number, number];

/** Blend one channel: base gradient (gx, gy) and brush gradient (bx, by) -> new gradient. */
export function blend(mode: BlendMode, base: G2, brush: G2): G2 {
  const [gx, gy] = base;
  const [bx, by] = brush;
  switch (mode) {
    case 'add':
      return [gx + bx, gy + by];
    case 'over':
      return [bx, by];
    case 'max':
    case 'min': {
      const lb = bx * bx + by * by;
      const lg = gx * gx + gy * gy;
      const pickBrush = mode === 'max' ? lb > lg : lb < lg;
      return pickBrush ? [bx, by] : [gx, gy];
    }
    case 'dir': {
      let len2 = gx * gx + gy * gy;
      if (len2 < 1e-6) len2 += 1; // zero-length gets 1
      // "protect from reversing edges": the scale factor never goes below 0
      const amt = Math.max((bx * gx + by * gy) / len2, -1);
      return [gx + gx * amt, gy + gy * amt];
    }
  }
}
