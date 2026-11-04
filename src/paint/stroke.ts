/**
 * Port of Stroke / SmoothStroke (Stroke.hpp/.cpp): raw pointer samples are
 * resampled at a fixed arc-length step and smoothed with a small kernel.
 * Points are in image coordinates (pixels, y down).
 */
export type Vec2 = [number, number];

export class Stroke {
  points: Vec2[] = [];
  /** incremented on every change so renderers know to redraw */
  version = 0;

  clear(): void {
    this.points = [];
    this.version++;
  }

  submit(p: Vec2): void {
    this.points.push([p[0], p[1]]);
    this.version++;
  }

  translate(dx: number, dy: number, from: Vec2[] = this.points): void {
    this.points = from.map(([x, y]) => [x + dx, y + dy]);
    this.version++;
  }

  touch(): void {
    this.version++;
  }
}

/** Normalised Gaussian used by GeneralPaint for paint & example-edge strokes ("a tad smoother"). */
export function gaussianKernel(): number[] {
  const k: number[] = [];
  for (let x = -4; x <= 4; x++) {
    const v = x / 2;
    k.push(Math.exp(-v * v));
  }
  const s = k.reduce((a, b) => a + b, 0);
  return k.map((v) => v / s);
}

export const DEFAULT_KERNEL = [0.25, 0.5, 0.25];

export class SmoothStroke extends Stroke {
  private last: Vec2 = [0, 0];
  private along = 0;
  private queue: Vec2[] = [];

  constructor(
    public kernel: number[] = DEFAULT_KERNEL,
    public step = 1,
  ) {
    super();
  }

  clear(): void {
    this.queue = [];
    super.clear();
  }

  submit(p: Vec2): void {
    if (this.points.length === 0 && this.queue.length === 0) {
      this.queue.push([p[0], p[1]]);
      this.last = [p[0], p[1]];
      this.along = 0;
    }
    for (;;) {
      const dx = p[0] - this.last[0];
      const dy = p[1] - this.last[1];
      const len = Math.hypot(dx, dy);
      if (!(this.along + len > this.step)) break;
      const t = (this.step - this.along) / len;
      this.last = [this.last[0] + dx * t, this.last[1] + dy * t];
      this.queue.push(this.last);
      this.along = 0;
      if (this.queue.length >= this.kernel.length) {
        let x = 0;
        let y = 0;
        for (let i = 0; i < this.kernel.length; i++) {
          x += this.kernel[i] * this.queue[i][0];
          y += this.kernel[i] * this.queue[i][1];
        }
        super.submit([x, y]);
        this.queue.shift();
      }
    }
  }
}
