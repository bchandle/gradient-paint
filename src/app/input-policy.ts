/**
 * Pure input policy: which action a pointer performs, palm rejection, the
 * pen pressure response, and pinch/pan maths. Kept free of DOM state so it
 * can be unit-tested.
 *
 * The original program had two mouse buttons: left draws, right configures
 * (colour wheel / clone source / example edge). On a tablet:
 *  - the Pencil (pointerType 'pen') draws, and hovers to show the cursor;
 *  - a finger configures once a Pencil has been seen ("auto"), otherwise it
 *    draws, so the app still works with fingers only;
 *  - two fingers pan & pinch-zoom, and a quick two-finger tap undoes;
 *  - the on-screen "Pick" latch turns the next press into a configure action
 *    for any pointer.
 */
import type { Vec2 } from '../paint/stroke';

export type PointerAction = 'draw' | 'configure' | 'pan' | 'ignore';
export type FingerMode = 'auto' | 'draw' | 'configure';
export type PressureMode = 'strength' | 'size' | 'off';

export const FINGER_MODES: FingerMode[] = ['auto', 'draw', 'configure'];
export const PRESSURE_MODES: PressureMode[] = ['strength', 'size', 'off'];

/** Touches this large (CSS px) are treated as a resting palm. */
export const PALM_SIZE = 60;
/** Touches are ignored for this long after the Pencil was last down. */
export const PEN_GRACE_MS = 400;
/** A two-finger gesture shorter than this, that barely moved, is a tap (undo). */
export const TAP_MS = 300;
export const TAP_SLOP_PX = 12;

export interface PointerInfo {
  pointerType: string;
  button: number;
  altKey?: boolean;
  ctrlKey?: boolean;
  width?: number;
  height?: number;
}

export interface PolicyState {
  fingerMode: FingerMode;
  /** a Pencil has been used in this browser */
  penSeen: boolean;
  /** the Pencil is currently touching, or lifted less than PEN_GRACE_MS ago */
  penActive: boolean;
  /** the on-screen Pick latch is on */
  pickLatch: boolean;
  spaceDown: boolean;
  isMac: boolean;
}

export function isPalm(p: Pick<PointerInfo, 'width' | 'height'>): boolean {
  return Math.max(p.width ?? 0, p.height ?? 0) >= PALM_SIZE;
}

/** Decide what a pointerdown does. Multi-finger gestures are handled separately by the caller. */
export function routePointer(p: PointerInfo, s: PolicyState): PointerAction {
  if (p.pointerType === 'touch') {
    if (isPalm(p) || s.penActive) return 'ignore';
    if (s.pickLatch) return 'configure';
    const fingersDraw = s.fingerMode === 'draw' || (s.fingerMode === 'auto' && !s.penSeen);
    return fingersDraw ? 'draw' : 'configure';
  }
  if (p.button === 1 || (p.button === 0 && s.spaceDown)) return 'pan';
  if (p.button === 2 || (p.button === 0 && (p.altKey || (p.ctrlKey && s.isMac)))) return 'configure';
  if (p.button !== 0) return 'ignore'; // eraser end, extra buttons
  return s.pickLatch ? 'configure' : 'draw';
}

/**
 * Pen pressure -> stroke factor. A normal Pencil stroke (~0.5) behaves like a
 * mouse (1.0); light touches fade towards `min`, hard presses go up to `max`.
 * Pressure 0 while pressed means "unknown" (some hardware) and maps to 1.
 */
export function pressureResponse(pressure: number, min = 0.05, max = 2): number {
  if (!(pressure > 0)) return 1;
  return Math.min(max, Math.max(min, pressure / 0.5));
}

/**
 * Per-point radius and strength for a stroke, from its pen pressures.
 *  - strength: gradient magnitude follows pressure, width fixed.
 *  - size: width follows pressure; for the gradient brush the strength is
 *    compensated (step across a gradient-brush stroke is proportional to its
 *    width / brush size) so pressure changes softness, not tone.
 */
export function pressureProfile(
  pressures: ArrayLike<number>,
  mode: PressureMode,
  radius: number,
  keepToneForSize: boolean,
): { radii?: number[]; strengths?: number[] } {
  if (mode === 'off') return {};
  const n = pressures.length;
  if (mode === 'strength') return { strengths: Array.from({ length: n }, (_, i) => pressureResponse(pressures[i])) };
  const radii: number[] = [];
  const strengths: number[] = [];
  for (let i = 0; i < n; i++) {
    const f = pressureResponse(pressures[i], 0.15, 1.5);
    radii.push(Math.max(0.5, radius * f));
    strengths.push(keepToneForSize ? radius / radii[i] : 1);
  }
  return keepToneForSize ? { radii, strengths } : { radii };
}

export interface View {
  scale: number;
  ox: number;
  oy: number;
}

/**
 * Two-finger pan + pinch: the image point under the starting midpoint stays
 * under the current midpoint, scaled by the change in finger distance.
 */
export function pinchView(start: View, a0: Vec2, b0: Vec2, a1: Vec2, b1: Vec2, minScale = 0.05, maxScale = 64): View {
  const d0 = Math.max(1, Math.hypot(b0[0] - a0[0], b0[1] - a0[1]));
  const d1 = Math.max(1, Math.hypot(b1[0] - a1[0], b1[1] - a1[1]));
  const m0: Vec2 = [(a0[0] + b0[0]) / 2, (a0[1] + b0[1]) / 2];
  const m1: Vec2 = [(a1[0] + b1[0]) / 2, (a1[1] + b1[1]) / 2];
  const scale = Math.min(maxScale, Math.max(minScale, (start.scale * d1) / d0));
  const ix = (m0[0] - start.ox) / start.scale;
  const iy = (m0[1] - start.oy) / start.scale;
  return { scale, ox: m1[0] - ix * scale, oy: m1[1] - iy * scale };
}

/** Did a finished two-finger gesture look like a tap? */
export function isTwoFingerTap(durationMs: number, maxMovePx: number): boolean {
  return durationMs <= TAP_MS && maxMovePx <= TAP_SLOP_PX;
}
