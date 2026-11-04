/**
 * Shared specification of the multigrid integrator from
 * "Real-Time Gradient-Domain Painting" (McCann & Pollard, SIGGRAPH 2008).
 *
 * Both the CPU reference implementation (reference.ts) and the WebGPU
 * implementation (gpu/solver.ts) derive their grid hierarchy and stencil
 * coefficients from here, so the two cannot drift apart.
 *
 * Conventions (shared by everything in this project):
 *  - Images are W x H, row-major, y pointing DOWN, pixel (x, y) has its
 *    center at (x + 0.5, y + 0.5) in continuous image coordinates.
 *  - Every field stores 4 floats per pixel (r, g, b, unused) so the CPU
 *    layout matches the GPU's array<vec4<f32>> layout exactly.
 *  - gx(x, y) = u(x+1, y) - u(x, y) for x in [0, W-2]   ((W-1) x H values)
 *  - gy(x, y) = u(x, y+1) - u(x, y) for y in [0, H-2]   (W x (H-1) values)
 *  - The solver solves L u = f with Neumann (zero-derivative) boundaries,
 *    where f = div(g) (Eq. 6), i.e. the least-squares integral of (gx, gy).
 */

export const CHANNELS = 4;

export interface LevelSpec {
  /** level index; 0 is the full-resolution grid */
  index: number;
  width: number;
  height: number;
  /** grid spacing h = 2^index */
  h: number;
  /** L_h stencil (Eq. 8): middle, edge and corner weights */
  middle: number;
  edge: number;
  corner: number;
  /** magic Jacobi relaxation parameters for the two post-smoothing passes (Fig. 9) */
  xA: number;
  xB: number;
}

/** Size of the next-coarser grid: vertex-centred halving, as in VCycle.cpp. */
export function coarserSize(n: number): number {
  return Math.floor((n - 1) / 2) + 1;
}

/** Stencil coefficients of L_h for grid spacing h (Eq. 8). */
export function stencil(h: number): { middle: number; edge: number; corner: number } {
  const h2 = h * h;
  return {
    middle: (-8 * h2 - 4) / (3 * h2),
    edge: (h2 + 2) / (3 * h2),
    corner: (h2 - 1) / (3 * h2),
  };
}

/** The two magic relaxation constants x_{h0}, x_{h1} from Fig. 9. */
export function magicX(h: number): { xA: number; xB: number } {
  return {
    xA: -2.1532 + 1.507 / h + 0.5882 / (h * h),
    xB: 0.1138 + 0.9529 / h + 1.5065 / (h * h),
  };
}

/**
 * How coarse grid sizes are chosen.
 *  - 'paper':  n -> floor((n-1)/2)+1, exactly VCycle.cpp. Ideal for n = 2^k+1,
 *              but for even n the last fine pixel has no coarse partner, which
 *              slows convergence to ~0.6/cycle (the "increase in error ... for
 *              certain image sizes" the paper mentions under Fig. 12).
 *  - 'padded': n -> ceil((n-1)/2)+1. Identical for odd n; for even n the
 *              coarse grid gets one extra node overhanging the fine domain,
 *              which restores ~0.45-0.5/cycle. This is the default.
 */
export type Coarsening = 'paper' | 'padded';

export function coarserSizeWith(n: number, mode: Coarsening): number {
  if (mode === 'paper' || n <= 2) return coarserSize(n);
  return Math.ceil((n - 1) / 2) + 1;
}

/** The full grid hierarchy: halve until both dimensions are < 2 (1x1), like VCycle::init. */
export function buildLevels(width: number, height: number, mode: Coarsening = 'padded'): LevelSpec[] {
  if (!(width >= 1 && height >= 1) || !Number.isInteger(width) || !Number.isInteger(height)) {
    throw new Error(`invalid grid size ${width}x${height}`);
  }
  const levels: LevelSpec[] = [];
  let w = width;
  let h = height;
  for (;;) {
    const index = levels.length;
    const spacing = 2 ** index;
    levels.push({ index, width: w, height: h, h: spacing, ...stencil(spacing), ...magicX(spacing) });
    if (w < 2 && h < 2) break;
    w = coarserSizeWith(w, mode);
    h = coarserSizeWith(h, mode);
  }
  return levels;
}

/** Restriction / prolongation 1D weights: P = R = [1/2 1 1/2] (x) [1/2 1 1/2] (Eq. 7). */
export const RESTRICT_WEIGHTS = [0.5, 1, 0.5] as const;
