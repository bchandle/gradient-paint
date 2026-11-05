/**
 * The Gradient Paint application: state, frame loop and input handling.
 * Mirrors GeneralPaint.cpp's interaction model:
 *
 *   left mouse ............ paint / clone / draw edge copy
 *   right mouse ........... configure (colour selector / clone source / pick example edge)
 *   up/down ............... brush (Paint, Clone, Edge)
 *   left/right ............ blend mode (add, dir, max, min, over)
 *   wheel ................. brush size
 *   backspace ............. delete current stroke; shift+backspace: revert
 *   g ..................... drag the current stroke around (live re-integration)
 */
import { BLEND_DESCRIPTIONS, BLEND_MODES, BlendMode } from '../paint/blend';
import { buildStrokeGeometry } from '../paint/geometry';
import { hsvToRgb, RGB } from '../paint/hsv';
import { CapturedEdge, GradientCanvas, readTexture } from '../paint/painter';
import { DEFAULT_KERNEL, gaussianKernel, SmoothStroke, Vec2 } from '../paint/stroke';
import { GpuSolver } from '../solver/gpu';
import { Display, ViewMode } from './display';
import { channelMeans, imageGradients, loadImageFile, savePNG } from './image-io';
import {
  FINGER_MODES,
  FingerMode,
  isTwoFingerTap,
  PEN_GRACE_MS,
  pinchView,
  PolicyState,
  PRESSURE_MODES,
  PressureMode,
  pressureProfile,
  routePointer,
  View,
} from './input-policy';
import { drawOverlay, OverlayState } from './overlay';

export type Mode = 'paint' | 'clone' | 'edge';
export const MODES: Mode[] = ['paint', 'clone', 'edge'];
export const MODE_NAMES: Record<Mode, string> = { paint: 'Paint', clone: 'Clone', edge: 'Edge' };

const COLOR_RATE = 0.99; // colour-adapt EMA rate (ColorRate)
const COLOR_SEL_RADIUS = 100; // CSS px (ColorSelRad)
/** iPads / phones: tighter memory budgets (and touch-oriented hints) */
export const IS_TOUCH_DEVICE = typeof navigator !== 'undefined' && navigator.maxTouchPoints > 1;
export const MAX_PIXELS = IS_TOUCH_DEVICE ? 2.4e6 : 4.2e6;
const HISTORY_BYTES = (IS_TOUCH_DEVICE ? 128 : 384) * 1024 * 1024;
const CYCLE_CHOICES = [1, 2, 4, 8];

interface ViewSource {
  name: string;
  w: number;
  h: number;
  mode: ViewMode;
  buffer?: GPUBuffer;
  texture?: GPUTexture;
}

export interface UiHooks {
  update(app: App): void;
  toast(msg: string, ms?: number): void;
}

export class App {
  // GPU objects
  solver!: GpuSolver;
  canvas!: GradientCanvas;
  private display: Display;
  private context: GPUCanvasContext | null = null;
  /** set if GPU->CPU readback fails (some headless/virtual GPUs); disables features that need it */
  readbackBroken = false;
  private fOrig!: GPUBuffer;
  private viewBuf!: GPUBuffer;
  private views: ViewSource[] = [];

  // image
  W = 0;
  H = 0;
  imageName = 'blank';
  mean: RGB = [0.5, 0.5, 0.5];

  // tools
  mode: Mode = 'paint';
  blend: BlendMode = 'add';
  brushSize = 4;
  paintHsv: RGB = [0, 0, 0.5];
  readonly strokes = {
    paint: new SmoothStroke(gaussianKernel()),
    clone: new SmoothStroke(DEFAULT_KERNEL),
    edge: new SmoothStroke(DEFAULT_KERNEL),
  };
  readonly example = new SmoothStroke(gaussianKernel());
  current: { mode: Mode; stroke: SmoothStroke } | null = null;
  clonePoint: Vec2 | null = null;
  edge: CapturedEdge | null = null;
  private edgeId = 0;
  edgePreview: { perp: ImageData; tan: ImageData } | null = null;

  // interaction
  fingerMode: FingerMode = 'auto';
  pressureMode: PressureMode = 'strength';
  penSeen = false;
  pickLatch: 'pick' | 'value' | null = null;
  moveLatch = false;
  private penDown = false;
  private lastPenUp = -Infinity;
  private colorValueMode = false;
  private strokeIsPen = false;
  private dragByPress = false;
  private actionPointer: number | null = null;
  private action: { cancel?: () => void } | null = null;
  private touches = new Map<number, { start: Vec2; pos: Vec2; t: number }>();
  gesture: { ids: [number, number]; startView: View; a0: Vec2; b0: Vec2; t0: number; maxMove: number } | null = null;
  painting = false;
  colorSelecting = false;
  edgeSelecting = false;
  dragging: { start: Vec2; points: Vec2[] } | null = null;
  panning: { x: number; y: number; ox: number; oy: number } | null = null;
  spaceDown = false;
  shiftDown = false;
  mouse: Vec2 = [-1e9, -1e9]; // CSS px within the stage
  mouseInside = false;
  private wheelAccum = 0;

  // display / debug
  view = { scale: 1, ox: 0, oy: 0 };
  viewIndex = 0;
  help = false;
  showStrokeLine = false;
  colorAdapt = false;
  superScale = false;
  compare = false;
  singleStep = false;
  private stepOnce = false;
  cyclesPerFrame = 2;
  private adaptAcc: RGB = [0.5, 0.5, 0.5];
  adaptBias = 0;
  private sampling = false;
  private sampleBuf: GPUBuffer;

  // bookkeeping
  private gradDirty = true;
  private lastRenderKey = '';
  fps = 0;
  private lastGpuFrame = performance.now();
  private inFlight = 0;
  private dpr = 1;
  private readonly overlayCtx: CanvasRenderingContext2D;
  private disposed = false;

  constructor(
    readonly device: GPUDevice,
    readonly gpuCanvas: HTMLCanvasElement,
    readonly overlayCanvas: HTMLCanvasElement,
    readonly stage: HTMLElement,
    private readonly ui: UiHooks,
    /** false: don't present to the canvas (diagnostics / automated QA, see README) */
    present = true,
  ) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    if (present) {
      this.context = gpuCanvas.getContext('webgpu')!;
      this.context.configure({ device, format, alphaMode: 'opaque' });
    }
    this.display = new Display(device, format);
    this.overlayCtx = overlayCanvas.getContext('2d')!;
    this.sampleBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    new ResizeObserver(() => this.resize()).observe(stage);
    this.resize();
    this.loadSettings();
    this.installInput();
  }

  // ---------------------------------------------------------------------------
  // setup

  get paintColor(): RGB {
    return hsvToRgb(this.paintHsv);
  }

  newCanvas(W: number, H: number): void {
    const rgba = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) rgba.set([0.5, 0.5, 0.5, 0], i * 4);
    this.setup(W, H, rgba, 'blank');
  }

  async openFile(file: File): Promise<void> {
    try {
      const img = await loadImageFile(file, MAX_PIXELS);
      this.setup(img.width, img.height, img.rgba, img.name);
      if (img.scaledFrom) this.ui.toast(`Scaled ${img.scaledFrom[0]}×${img.scaledFrom[1]} down to ${img.width}×${img.height} to fit the GPU budget.`, 4000);
    } catch (e) {
      this.ui.toast(`Could not open image: ${e instanceof Error ? e.message : e}`, 5000);
    }
  }

  private setup(W: number, H: number, rgba: Float32Array<ArrayBuffer>, name: string): void {
    this.solver?.destroy();
    this.canvas?.destroy();
    this.fOrig?.destroy();
    this.viewBuf?.destroy();
    this.edge?.tan.destroy();
    this.edge?.perp.destroy();
    this.W = W;
    this.H = H;
    this.imageName = name;
    const { gx, gy } = imageGradients(rgba, W, H);
    this.mean = channelMeans(rgba, W, H);
    this.canvas = new GradientCanvas(this.device, W, H, gx, gy, { historyBytes: HISTORY_BYTES });
    this.solver = new GpuSolver(this.device, W, H);
    this.solver.setGradientTextures(this.canvas.gx, this.canvas.gy);
    this.solver.setTargetMean(this.mean);
    this.solver.writeSolution(rgba);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.fOrig = this.device.createBuffer({ size: rgba.byteLength, usage });
    this.device.queue.writeBuffer(this.fOrig, 0, rgba);
    this.viewBuf = this.device.createBuffer({ size: W * H * 16, usage });
    this.display.reset();
    this.views = [
      { name: 'f', w: W, h: H, mode: ViewMode.Image, buffer: this.solver.a[0] },
      { name: 'gx', w: W - 1, h: H, mode: ViewMode.Signed, texture: this.canvas.gx },
      { name: 'gy', w: W, h: H - 1, mode: ViewMode.Signed, texture: this.canvas.gy },
    ];
    this.solver.levels.forEach((l, i) => {
      if (i > 0) this.views.push({ name: `f${i + 1}: ${l.width}×${l.height}`, w: l.width, h: l.height, mode: ViewMode.Signed, buffer: this.solver.a[i] });
      this.views.push({ name: `rhs${i + 1}: ${l.width}×${l.height}`, w: l.width, h: l.height, mode: ViewMode.Signed, buffer: this.solver.rhs[i] });
    });
    this.viewIndex = 0;
    this.current = null;
    this.edge = null;
    this.edgePreview = null;
    this.example.clear();
    this.clonePoint = null;
    this.dragging = null;
    this.adaptBias = 0;
    this.gradDirty = true;
    this.lastRenderKey = '';
    this.fitView();
    this.ui.update(this);
  }

  // ---------------------------------------------------------------------------
  // view transform

  private resize(): void {
    const r = this.stage.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * this.dpr));
    const h = Math.max(1, Math.round(r.height * this.dpr));
    for (const c of [this.gpuCanvas, this.overlayCanvas]) {
      if (c.width !== w || c.height !== h) {
        c.width = w;
        c.height = h;
      }
    }
  }

  stageSize(): [number, number] {
    const r = this.stage.getBoundingClientRect();
    return [r.width, r.height];
  }

  fitView(): void {
    if (!this.W) return;
    const [sw, sh] = this.stageSize();
    let s = Math.min((sw - 24) / this.W, (sh - 24) / this.H);
    if (s >= 1) s = Math.floor(s); // integer zoom like the original, for crisp pixels
    s = Math.max(s, 0.05);
    this.view = { scale: s, ox: Math.round((sw - this.W * s) / 2), oy: Math.round((sh - this.H * s) / 2) };
  }

  zoomAt(factor: number, cx: number, cy: number): void {
    const s = Math.min(64, Math.max(0.05, this.view.scale * factor));
    const ix = (cx - this.view.ox) / this.view.scale;
    const iy = (cy - this.view.oy) / this.view.scale;
    this.view = { scale: s, ox: cx - ix * s, oy: cy - iy * s };
  }

  toImage([x, y]: Vec2): Vec2 {
    return [(x - this.view.ox) / this.view.scale, (y - this.view.oy) / this.view.scale];
  }

  toScreen([x, y]: Vec2): Vec2 {
    return [this.view.ox + x * this.view.scale, this.view.oy + y * this.view.scale];
  }

  // ---------------------------------------------------------------------------
  // actions

  setMode(m: Mode): void {
    if (this.painting || this.edgeSelecting || this.colorSelecting) return;
    this.mode = m;
    this.current = null;
    this.dragging = null;
    this.ui.update(this);
  }

  cycleMode(d: number): void {
    this.setMode(MODES[(MODES.indexOf(this.mode) + d + MODES.length) % MODES.length]);
  }

  setBlend(b: BlendMode): void {
    this.blend = b; // the current stroke re-renders with the new mode (render key changes)
    this.ui.update(this);
  }

  cycleBlend(d: number): void {
    this.setBlend(BLEND_MODES[(BLEND_MODES.indexOf(this.blend) + d + BLEND_MODES.length) % BLEND_MODES.length]);
  }

  /** Wheel semantics from the original: below 10px step by 1, above that by 10%. */
  stepBrush(bigger: boolean): void {
    let s = this.brushSize;
    if (s < 10) s = Math.trunc(s + 0.5 + (bigger ? 1 : -1));
    else s *= bigger ? 1.1 : 0.9;
    this.brushSize = Math.min(400, Math.max(0.5, s));
    this.ui.update(this);
  }

  private submitNow(fn: (enc: GPUCommandEncoder) => void): void {
    const enc = this.device.createCommandEncoder();
    fn(enc);
    this.device.queue.submit([enc.finish()]);
    this.gradDirty = true;
    this.lastRenderKey = '';
  }

  undo(silent = false): void {
    let ok = false;
    this.submitNow((enc) => (ok = this.canvas.undo(enc)));
    this.current = null;
    this.dragging = null;
    this.moveLatch = false;
    if (!ok && !silent) this.ui.toast('Nothing to undo');
    this.ui.update(this);
  }

  deleteCurrentStroke(): void {
    if (this.current) this.undo();
  }

  revert(): void {
    this.submitNow((enc) => this.canvas.revert(enc));
    this.current = null;
    this.dragging = null;
    this.ui.toast('Reverted to the original gradients (Ctrl+Z to undo)');
    this.ui.update(this);
  }

  private readbackFailed(e: unknown): void {
    if (this.readbackBroken) return;
    this.readbackBroken = true;
    this.colorAdapt = false;
    console.warn('GPU readback failed:', e);
    this.ui.toast('This GPU/browser cannot read results back from the GPU: saving, colour adapt and the edge preview are unavailable.', 6000);
  }

  async save(): Promise<void> {
    let data: Float32Array;
    try {
      data = await this.solver.readSolution();
    } catch (e) {
      this.readbackFailed(e);
      return;
    }
    const gain = this.superScale ? 10 : 1;
    const base = this.imageName.replace(/\.[^.]+$/, '') || 'canvas';
    await savePNG(data, this.W, this.H, (v, c) => (v - this.mean[c]) * gain + this.mean[c] + this.adaptBias, `${base}-gradient-paint.png`);
  }

  async benchmark(): Promise<void> {
    this.ui.toast('Timing the integrator…');
    let ms: number;
    try {
      ms = await this.solver.benchmark(30);
    } catch (e) {
      this.ui.toast(`Benchmark failed: ${e instanceof Error ? e.message : e}`);
      return;
    }
    const mp = (this.W * this.H) / 1e6;
    this.ui.toast(`${ms.toFixed(2)} ms per V-cycle at ${this.W}×${this.H} (${mp.toFixed(2)} MP) — ${(1000 / ms).toFixed(0)} cycles/s`, 6000);
  }

  randomizeGuess(zero: boolean): void {
    const n = this.W * this.H;
    const f = new Float32Array(n * 4);
    if (!zero) for (let i = 0; i < n * 4; i++) f[i] = i % 4 === 3 ? 0 : Math.random();
    this.solver.writeSolution(f);
  }

  cycleView(d: number): void {
    this.viewIndex = (this.viewIndex + d + this.views.length) % this.views.length;
    this.ui.update(this);
  }

  get viewName(): string {
    return this.views[this.viewIndex]?.name ?? 'f';
  }

  // ---------------------------------------------------------------------------
  // strokes

  private startStroke(pos: Vec2, pressure = 0.5): boolean {
    if (this.mode === 'clone' && !this.clonePoint) {
      this.ui.toast(`Set the clone source first (${this.configureHint()})`);
      return false;
    }
    if (this.mode === 'edge' && !this.edge) {
      this.ui.toast(`Capture an edge first: drag along one in the image (${this.configureHint()})`);
      return false;
    }
    // the previous stroke may have changed since the last frame (fast pens, quick dots):
    // render it before snapshotting, or its latest samples would be lost
    this.flushStroke();
    this.submitNow((enc) => this.canvas.pushSnapshot(enc));
    const stroke = this.strokes[this.mode];
    stroke.clear();
    stroke.submit(pos, pressure);
    this.current = { mode: this.mode, stroke };
    this.painting = true;
    this.ui.update(this);
    return true;
  }

  private grabEdge(): void {
    let rad = Math.trunc(this.brushSize);
    if (rad < 2) rad = 2;
    if (rad < this.brushSize) rad += 1;
    const edge = this.canvas.captureEdge(this.solver.a[0], this.example.points, rad);
    if (!edge) return;
    this.edge?.tan.destroy();
    this.edge?.perp.destroy();
    this.edge = edge;
    this.edgeId++;
    this.lastRenderKey = '';
    void this.buildEdgePreview(edge);
  }

  private async buildEdgePreview(edge: CapturedEdge): Promise<void> {
    const toImage = (data: Float32Array, w: number, h: number) => {
      const id = new ImageData(w, h);
      for (let i = 0; i < w * h; i++) {
        for (let c = 0; c < 3; c++) id.data[i * 4 + c] = Math.max(0, Math.min(255, (data[i * 4 + c] * 10 + 0.5) * 255));
        id.data[i * 4 + 3] = 255;
      }
      return id;
    };
    if (this.readbackBroken) return;
    let perp: Float32Array;
    let tan: Float32Array;
    try {
      [perp, tan] = await Promise.all([readTexture(this.device, edge.perp), readTexture(this.device, edge.tan)]);
    } catch (e) {
      this.readbackFailed(e);
      return;
    }
    if (this.edge !== edge) return;
    this.edgePreview = { perp: toImage(perp, edge.perp.width, edge.perp.height), tan: toImage(tan, edge.tan.width, edge.tan.height) };
  }

  private flushStroke(): void {
    if (!this.current) return;
    const key = this.renderKey();
    if (key === this.lastRenderKey) return;
    const enc = this.device.createCommandEncoder({ label: 'flush-stroke' });
    this.renderCurrentStroke(enc);
    this.device.queue.submit([enc.finish()]);
    this.lastRenderKey = key;
    this.gradDirty = true;
  }

  private renderKey(): string {
    if (!this.current) return '';
    const c = this.current;
    return [c.mode, c.stroke.version, c.stroke.points.length, this.blend, this.brushSize, this.paintHsv.join(','), this.clonePoint?.join(','), this.edgeId, this.strokeIsPen && this.pressureMode].join('|');
  }

  private renderCurrentStroke(enc: GPUCommandEncoder): void {
    const c = this.current!;
    const pts = c.stroke.points;
    // pen pressure -> per-point strength or width
    const pressure = this.strokeIsPen ? pressureProfile(c.stroke.pressures, this.pressureMode, this.brushSize, c.mode === 'paint') : {};
    let geometry: Float32Array<ArrayBuffer>;
    let brush;
    if (c.mode === 'paint') {
      geometry = buildStrokeGeometry(pts, { radius: this.brushSize, texRadius: 2, stepsAlong: 1, ...pressure });
      brush = this.canvas.paintBrush(this.paintColor, this.brushSize);
    } else if (c.mode === 'edge') {
      if (!this.edge) return;
      geometry = buildStrokeGeometry(pts, { radius: this.brushSize, texRadius: this.edge.radius, stepsAlong: this.edge.steps, ...pressure });
      brush = this.canvas.edgeBrush(this.edge);
    } else {
      if (!this.clonePoint || pts.length === 0) return;
      const off: Vec2 = [this.clonePoint[0] - pts[0][0], this.clonePoint[1] - pts[0][1]];
      geometry = buildStrokeGeometry(pts, { radius: this.brushSize, texRadius: 0, stepsAlong: 0, cloneOffset: off, ...pressure });
      brush = this.canvas.cloneBrush();
    }
    this.canvas.renderStroke(enc, geometry, brush, this.blend);
  }

  // ---------------------------------------------------------------------------
  // frame

  start(): void {
    const loop = (t: number) => {
      if (this.disposed) return;
      this.frame(t);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  private frame(t: number): void {
    if (!this.solver) return;
    // back-pressure: never queue more than two frames of GPU work (slow GPUs, or when not presenting)
    if (this.inFlight >= 2) {
      this.drawOverlay();
      return;
    }
    const gpuDt = t - this.lastGpuFrame;
    if (gpuDt > 0) this.fps = this.fps * 0.9 + (1000 / gpuDt) * 0.1;
    this.lastGpuFrame = t;

    const enc = this.device.createCommandEncoder({ label: 'frame' });
    if (this.current) {
      const key = this.renderKey();
      if (key !== this.lastRenderKey) {
        this.renderCurrentStroke(enc);
        this.lastRenderKey = key;
        this.gradDirty = true;
      }
    }
    const view = this.views[this.viewIndex];
    const pass = enc.beginComputePass({ label: 'integrate' });
    if (this.gradDirty) {
      this.solver.encodeRhs(pass);
      this.gradDirty = false;
    }
    if (!this.singleStep || this.stepOnce) {
      const n = this.singleStep ? 1 : this.cyclesPerFrame;
      for (let i = 0; i < n; i++) this.solver.encodeVCycle(pass);
      this.stepOnce = false;
    }
    if (view.texture) this.display.encodeTextureToBuffer(pass, view.texture, this.viewBuf);
    pass.end();

    const s = this.dpr * this.view.scale;
    if (this.context) this.display.encode(enc, this.context.getCurrentTexture().createView(), view.texture ? this.viewBuf : view.buffer!, this.fOrig, {
      scale: s,
      offsetX: this.view.ox * this.dpr,
      offsetY: this.view.oy * this.dpr,
      mode: this.compare && view.mode === ViewMode.Image ? ViewMode.Compare : view.mode,
      srcW: view.w,
      srcH: view.h,
      imgW: this.W,
      imgH: this.H,
      mean: this.mean,
      gain: this.superScale ? 10 : 1,
      bias: [this.adaptBias, this.adaptBias, this.adaptBias],
    });

    // colour adapt: sample the image under the cursor (asynchronously)
    let sampled = false;
    if (this.colorAdapt && !this.sampling && this.mouseInside && !this.readbackBroken) {
      const [ix, iy] = this.toImage(this.mouse).map(Math.floor);
      const x = Math.min(this.W - 1, Math.max(0, ix));
      const y = Math.min(this.H - 1, Math.max(0, iy));
      enc.copyBufferToBuffer(this.solver.a[0], (y * this.W + x) * 16, this.sampleBuf, 0, 16);
      sampled = true;
    }
    this.device.queue.submit([enc.finish()]);
    this.inFlight++;
    this.device.queue.onSubmittedWorkDone().then(
      () => this.inFlight--,
      () => this.inFlight--,
    );
    if (sampled) {
      this.sampling = true;
      this.sampleBuf.mapAsync(GPUMapMode.READ).then(() => {
        const v = new Float32Array(this.sampleBuf.getMappedRange().slice(0));
        this.sampleBuf.unmap();
        this.sampling = false;
        if (!this.colorAdapt) return;
        for (let c = 0; c < 3; c++) this.adaptAcc[c] = this.adaptAcc[c] * COLOR_RATE + v[c] * (1 - COLOR_RATE);
        const gain = this.superScale ? 10 : 1;
        const shown = [0, 1, 2].map((c) => (this.adaptAcc[c] - this.mean[c]) * gain + this.mean[c]);
        this.adaptBias = 0.5 - (shown[0] + shown[1] + shown[2]) / 3;
      }, (e) => {
        this.sampling = false;
        this.readbackFailed(e);
      });
    }

    this.drawOverlay();
  }

  private drawOverlay(): void {
    const st: OverlayState = {
      dpr: this.dpr,
      mouse: this.mouse,
      mouseInside: this.mouseInside && !this.panning && !this.gesture,
      brushRadiusPx: this.brushSize * this.view.scale,
      toScreen: (p) => this.toScreen(p),
      mode: this.mode,
      clonePoint: this.clonePoint,
      cloneSource:
        this.mode === 'clone' && this.painting && this.clonePoint && this.current?.stroke.points.length
          ? [
              this.mouse[0] + (this.clonePoint[0] - this.current.stroke.points[0][0]) * this.view.scale,
              this.mouse[1] + (this.clonePoint[1] - this.current.stroke.points[0][1]) * this.view.scale,
            ]
          : null,
      strokeLine: this.showStrokeLine && this.current ? this.current.stroke.points : null,
      exampleEdge: this.mode === 'edge' ? this.example.points : null,
      edgePreview: this.mode === 'edge' ? this.edgePreview : null,
      edgePreviewScale: Math.max(1, Math.min(4, Math.round(this.view.scale))),
      colorSelector: this.colorSelecting ? { hsv: this.paintHsv, valueMode: this.colorValueMode || this.shiftDown, radius: COLOR_SEL_RADIUS } : null,
    };
    drawOverlay(this.overlayCtx, st);
  }

  // ---------------------------------------------------------------------------
  // input

  private stagePos(e: { clientX: number; clientY: number }): Vec2 {
    const r = this.stage.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  /** Raw pen pressure; mice and fingers report "normal" (0.5, which maps to strength 1). */
  private pressureOf(e: PointerEvent): number {
    return e.pointerType === 'pen' && e.pressure > 0 ? e.pressure : 0.5;
  }

  private penActive(): boolean {
    return this.penDown || performance.now() - this.lastPenUp < PEN_GRACE_MS;
  }

  private policy(): PolicyState {
    return {
      fingerMode: this.fingerMode,
      penSeen: this.penSeen,
      penActive: this.penActive(),
      pickLatch: this.pickLatch !== null,
      spaceDown: this.spaceDown,
      isMac: /Mac|iPhone|iPad/.test(navigator.platform),
    };
  }

  /** The configure ("right button") action of the current mode. */
  private configure(pos: Vec2, shift: boolean): void {
    if (this.dragging && this.current) {
      // configuring cancels a drag
      this.current.stroke.translate(0, 0, this.dragging.points);
      this.dragging = null;
    }
    if (this.mode === 'paint') {
      const prev: RGB = [...this.paintHsv];
      this.colorSelecting = true;
      this.colorValueMode = shift || this.pickLatch === 'value';
      this.action = { cancel: () => (this.paintHsv = prev) };
    } else if (this.mode === 'clone') {
      const prev = this.clonePoint;
      this.clonePoint = pos;
      this.action = { cancel: () => (this.clonePoint = prev) };
    } else if (shift && this.example.points.length) {
      this.example.submit(pos);
      this.grabEdge();
    } else {
      const prevPts = this.example.points;
      const prevP = this.example.pressures;
      this.example.clear();
      this.example.submit(pos);
      this.edgeSelecting = true;
      this.action = {
        cancel: () => {
          this.edgeSelecting = false;
          this.example.restore(prevPts, prevP);
        },
      };
    }
    this.pickLatch = null; // the on-screen Pick latch is one-shot
    this.ui.update(this);
  }

  /** Abandon whatever the action pointer started (used when a second finger turns it into a pinch). */
  private cancelAction(): void {
    this.action?.cancel?.();
    this.action = null;
    this.painting = false;
    this.colorSelecting = false;
    this.edgeSelecting = false;
    this.panning = null;
    if (this.actionPointer !== null && this.overlayCanvas.hasPointerCapture(this.actionPointer)) this.overlayCanvas.releasePointerCapture(this.actionPointer);
    this.actionPointer = null;
    this.ui.update(this);
  }

  private installInput(): void {
    const el = this.overlayCanvas;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerenter', (e) => {
      if (e.pointerType !== 'touch') this.mouseInside = true;
    });
    el.addEventListener('pointerleave', () => (this.mouseInside = false));
    // keep Safari from turning touches into page zoom, selection or callouts
    for (const type of ['gesturestart', 'gesturechange', 'gestureend', 'dblclick', 'selectstart']) {
      document.addEventListener(type, (e) => {
        if (!(e.target instanceof HTMLInputElement)) e.preventDefault();
      }, { passive: false });
    }
    el.addEventListener('touchstart', (e) => e.preventDefault(), { passive: false });

    el.addEventListener('pointerdown', (e) => {
      if (!this.solver) return;
      const p = this.stagePos(e);
      if (e.pointerType === 'pen') {
        this.penDown = true;
        if (!this.penSeen) {
          this.penSeen = true;
          this.saveSettings();
          if (this.fingerMode === 'auto') this.ui.toast('Apple Pencil detected: the Pencil draws, a finger picks colours / sources, two fingers pan, zoom and tap to undo.', 6000);
        }
      }
      if (e.pointerType === 'touch') {
        this.touches.set(e.pointerId, { start: p, pos: p, t: performance.now() });
        if (this.touches.size >= 2) {
          if (!this.gesture && this.touches.size === 2 && !this.penActive()) {
            this.cancelAction();
            const [[ia, ta], [ib, tb]] = [...this.touches.entries()];
            // a tap is measured from when/where each finger first touched, so a stroke
            // cancelled by a second finger never also counts as a two-finger tap (undo)
            const moved = Math.max(Math.hypot(ta.pos[0] - ta.start[0], ta.pos[1] - ta.start[1]), Math.hypot(tb.pos[0] - tb.start[0], tb.pos[1] - tb.start[1]));
            this.gesture = { ids: [ia, ib], startView: { ...this.view }, a0: ta.pos, b0: tb.pos, t0: Math.min(ta.t, tb.t), maxMove: moved };
          }
          return;
        }
      }
      if (this.actionPointer !== null && this.actionPointer !== e.pointerId) return; // one action at a time
      const act = routePointer(e, this.policy());
      if (act === 'ignore') return;
      el.setPointerCapture(e.pointerId);
      this.actionPointer = e.pointerId;
      this.mouse = p;
      if (e.pointerType !== 'touch' || act === 'draw') this.mouseInside = true;
      this.shiftDown = e.shiftKey;
      const pos = this.toImage(p);
      this.action = {};
      if (act === 'pan') {
        this.panning = { x: p[0], y: p[1], ox: this.view.ox, oy: this.view.oy };
        e.preventDefault();
        return;
      }
      if (act === 'configure') {
        this.configure(pos, e.shiftKey);
        return;
      }
      if (this.dragging && !this.dragByPress) {
        this.dragging = null; // drop (keyboard 'g' drag)
      } else if (this.moveLatch && this.current) {
        this.dragging = { start: pos, points: this.current.stroke.points.map((q) => [q[0], q[1]] as Vec2) };
        this.dragByPress = true;
      } else if (e.shiftKey && this.current && this.current.mode === this.mode) {
        this.current.stroke.submit(pos, this.pressureOf(e)); // straight line to the cursor
        this.painting = true;
      } else {
        const isPen = e.pointerType === 'pen';
        if (this.startStroke(pos, this.pressureOf(e))) {
          this.strokeIsPen = isPen;
          this.action = { cancel: () => this.undo(true) };
        }
      }
    });

    el.addEventListener('pointermove', (e) => {
      const p = this.stagePos(e);
      if (e.pointerType === 'touch') {
        const t = this.touches.get(e.pointerId);
        if (t) t.pos = p;
        if (this.gesture) {
          const g = this.gesture;
          if (g.ids.includes(e.pointerId)) {
            const a = this.touches.get(g.ids[0])?.pos ?? g.a0;
            const b = this.touches.get(g.ids[1])?.pos ?? g.b0;
            g.maxMove = Math.max(g.maxMove, Math.hypot(a[0] - g.a0[0], a[1] - g.a0[1]), Math.hypot(b[0] - g.b0[0], b[1] - g.b0[1]));
            this.view = pinchView(g.startView, g.a0, g.b0, a, b);
            this.ui.update(this);
          }
          return;
        }
      }
      if (this.actionPointer !== null && e.pointerId !== this.actionPointer) return;
      if (this.actionPointer === null && e.pointerType === 'touch') return; // e.g. the finger left over from a pinch
      const old = this.mouse;
      this.mouse = p;
      if (e.pointerType !== 'touch') this.mouseInside = true; // includes Pencil hover
      this.shiftDown = e.shiftKey;
      const pos = this.toImage(p);
      if (this.panning) {
        this.view.ox = this.panning.ox + p[0] - this.panning.x;
        this.view.oy = this.panning.oy + p[1] - this.panning.y;
        return;
      }
      if (this.dragging && this.current) {
        const [sx, sy] = this.dragging.start;
        this.current.stroke.translate(pos[0] - sx, pos[1] - sy, this.dragging.points);
      } else if (this.colorSelecting) {
        this.moveColorSelector(p[0] - old[0], p[1] - old[1], this.colorValueMode || e.shiftKey);
      } else if (this.painting && this.current) {
        // coalesced events carry every sample (240 Hz for the Pencil), with its own pressure
        const evs = typeof e.getCoalescedEvents === 'function' ? e.getCoalescedEvents() : [];
        for (const ce of evs.length ? evs : [e]) this.current.stroke.submit(this.toImage(this.stagePos(ce)), this.pressureOf(ce));
      }
      if (this.edgeSelecting) this.example.submit(pos);
    });

    const end = (e: PointerEvent) => {
      if (e.pointerType === 'pen') {
        this.penDown = false;
        this.lastPenUp = performance.now();
      }
      if (e.pointerType === 'touch') {
        this.touches.delete(e.pointerId);
        const g = this.gesture;
        if (g) {
          if (this.touches.size === 0) {
            this.gesture = null;
            if (e.type === 'pointerup' && isTwoFingerTap(performance.now() - g.t0, g.maxMove)) {
              this.undo();
              this.ui.toast('Undo');
            }
          }
          return;
        }
      }
      if (e.pointerId !== this.actionPointer) return;
      this.actionPointer = null;
      if (e.type === 'pointercancel') {
        this.cancelAction();
        return;
      }
      this.action = null;
      if (e.pointerType === 'touch') this.mouseInside = false;
      if (this.panning) {
        this.panning = null;
        return;
      }
      this.painting = false;
      if (this.dragByPress) {
        this.dragging = null;
        this.dragByPress = false;
        this.moveLatch = false;
      }
      if (this.colorSelecting) this.colorSelecting = false;
      if (this.edgeSelecting) {
        this.edgeSelecting = false;
        this.grabEdge();
      }
      this.ui.update(this);
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);

    el.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
        if (e.ctrlKey || e.metaKey) {
          this.zoomAt(Math.exp(-dy * 0.002), ...this.stagePos(e));
          this.ui.update(this);
          return;
        }
        this.wheelAccum += dy;
        while (Math.abs(this.wheelAccum) >= 40) {
          // as in the original: wheel up -> smaller brush, wheel down -> bigger
          this.stepBrush(this.wheelAccum > 0);
          this.wheelAccum -= Math.sign(this.wheelAccum) * 40;
        }
      },
      { passive: false },
    );

    window.addEventListener('keydown', (e) => this.onKey(e));
    window.addEventListener('keyup', (e) => {
      if (e.key === ' ') this.spaceDown = false;
      if (e.key === 'Shift') this.shiftDown = false;
    });
  }

  // ---------------------------------------------------------------------------
  // on-screen controls (touch)

  /** Arm the one-shot Pick latch: the next press configures instead of drawing ('value' = value slider). */
  setPickLatch(kind: 'pick' | 'value' | null): void {
    this.pickLatch = this.pickLatch === kind ? null : kind;
    this.ui.update(this);
  }

  toggleMoveLatch(): void {
    if (!this.current) {
      this.ui.toast('Nothing to move: draw a stroke first');
      this.moveLatch = false;
    } else this.moveLatch = !this.moveLatch;
    this.ui.update(this);
  }

  setBrushSize(v: number): void {
    this.brushSize = Math.min(400, Math.max(0.5, v));
    this.ui.update(this);
  }

  cycleFingerMode(): void {
    this.fingerMode = FINGER_MODES[(FINGER_MODES.indexOf(this.fingerMode) + 1) % FINGER_MODES.length];
    this.saveSettings();
    this.ui.update(this);
  }

  cyclePressureMode(): void {
    this.pressureMode = PRESSURE_MODES[(PRESSURE_MODES.indexOf(this.pressureMode) + 1) % PRESSURE_MODES.length];
    this.saveSettings();
    this.ui.update(this);
  }

  /** Whether a finger currently draws (for UI hints). */
  get fingersDraw(): boolean {
    return this.fingerMode === 'draw' || (this.fingerMode === 'auto' && !this.penSeen);
  }

  private configureHint(): string {
    return IS_TOUCH_DEVICE ? (this.fingersDraw ? 'use Pick' : 'tap with a finger, or use Pick') : 'right-click, or use Pick';
  }

  private loadSettings(): void {
    try {
      const s = JSON.parse(localStorage.getItem('gp-settings') ?? '{}');
      if (FINGER_MODES.includes(s.fingerMode)) this.fingerMode = s.fingerMode;
      if (PRESSURE_MODES.includes(s.pressureMode)) this.pressureMode = s.pressureMode;
      this.penSeen = !!s.penSeen;
    } catch {
      /* storage unavailable */
    }
  }

  private saveSettings(): void {
    try {
      localStorage.setItem('gp-settings', JSON.stringify({ fingerMode: this.fingerMode, pressureMode: this.pressureMode, penSeen: this.penSeen }));
    } catch {
      /* storage unavailable */
    }
  }

  /** Relative drag on an HSV wheel (shift: value slider in [-1, 1]), as in the original. */
  private moveColorSelector(dx: number, dy: number, valueMode: boolean): void {
    const hsv = this.paintHsv;
    const R = COLOR_SEL_RADIUS;
    if (valueMode) {
      hsv[2] = Math.max(-1, Math.min(1, hsv[2] + dx / R));
    } else {
      let x = Math.cos(hsv[0] * 2 * Math.PI) * hsv[1] * R + dx;
      let y = Math.sin(hsv[0] * 2 * Math.PI) * hsv[1] * R - dy; // screen y is down
      const len = Math.hypot(x, y);
      if (len > R) {
        x *= R / len;
        y *= R / len;
      }
      let h = Math.atan2(y, x) / (2 * Math.PI);
      if (h < 0) h += 1;
      hsv[0] = h;
      hsv[1] = Math.min(1, Math.hypot(x, y) / R);
    }
    this.ui.update(this);
  }

  private onKey(e: KeyboardEvent): void {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.closest('dialog'))) return;
    if (!this.solver) return;
    const k = e.key;
    const mod = e.ctrlKey || e.metaKey;
    this.shiftDown = e.shiftKey;
    let handled = true;
    if (mod && (k === 'z' || k === 'Z')) this.undo();
    else if (mod && (k === 's' || k === 'S')) void this.save();
    else if (mod && (k === 'o' || k === 'O')) document.getElementById('open-file')?.click();
    else if (mod) handled = false;
    else if (k === 'F1' || k === '?' || k === 'h') {
      this.help = !this.help;
      this.ui.update(this);
    } else if (k === 'Escape') {
      if (this.dragging && this.current) this.current.stroke.translate(0, 0, this.dragging.points);
      this.dragging = null;
      this.help = false;
      this.ui.update(this);
    } else if (k === ' ') this.spaceDown = true;
    else if (k === 'ArrowUp') this.cycleMode(1);
    else if (k === 'ArrowDown') this.cycleMode(-1);
    else if (k === 'ArrowLeft') this.cycleBlend(-1);
    else if (k === 'ArrowRight') this.cycleBlend(1);
    else if (k === 'Backspace' || k === 'Delete') {
      if (e.shiftKey) this.revert();
      else this.deleteCurrentStroke();
    } else if (k === 'g' && this.current) {
      this.dragging = { start: this.toImage(this.mouse), points: this.current.stroke.points.map((p) => [p[0], p[1]] as Vec2) };
    } else if (k === 't') this.showStrokeLine = !this.showStrokeLine;
    else if (k === 'a') {
      this.colorAdapt = !this.colorAdapt;
      if (this.colorAdapt) {
        // seed the running average so that the current exposure is unchanged
        const gain = this.superScale ? 10 : 1;
        const avgMean = (this.mean[0] + this.mean[1] + this.mean[2]) / 3;
        this.adaptAcc = this.mean.map((m) => m + (0.5 - this.adaptBias - avgMean) / gain) as RGB;
      }
      this.ui.toast(`Colour adapt ${this.colorAdapt ? 'on' : 'off'}`);
    } else if (k === 'A') {
      this.superScale = !this.superScale;
      this.ui.toast(`10× contrast ${this.superScale ? 'on' : 'off'}`);
    } else if (k === 'x') {
      this.adaptBias = 0;
      this.colorAdapt = false;
      this.superScale = false;
      this.ui.toast('Exposure reset');
    } else if (k === 'c') {
      this.compare = !this.compare;
      this.ui.toast(`Compare to original ${this.compare ? 'on' : 'off'}`);
    } else if (k === 's') {
      this.singleStep = true;
      this.stepOnce = true;
      this.ui.toast('Single-step: one V-cycle per press (S to resume)');
    } else if (k === 'S') {
      this.singleStep = false;
      this.ui.toast('Running');
    } else if (k === 'd') void this.save();
    else if (k === 'r') this.randomizeGuess(false);
    else if (k === 'R') this.randomizeGuess(true);
    else if (k === 'e') void this.benchmark();
    else if (k === 'PageUp') this.cycleView(1);
    else if (k === 'PageDown') this.cycleView(-1);
    else if (k === '[') this.stepBrush(false);
    else if (k === ']') this.stepBrush(true);
    else if (k === 'v') {
      this.cyclesPerFrame = CYCLE_CHOICES[(CYCLE_CHOICES.indexOf(this.cyclesPerFrame) + 1) % CYCLE_CHOICES.length];
      this.ui.toast(`${this.cyclesPerFrame} V-cycle${this.cyclesPerFrame > 1 ? 's' : ''} per frame`);
    } else if (k === '0' || k === 'f') this.fitView();
    else if (k === '1') {
      const [sw, sh] = this.stageSize();
      this.zoomAt(1 / this.view.scale, sw / 2, sh / 2);
    } else if (k === '=' || k === '+') this.zoomAt(1.25, ...this.mouseOrCenter());
    else if (k === '-' || k === '_') this.zoomAt(0.8, ...this.mouseOrCenter());
    else if (k === 'o') document.getElementById('open-file')?.click();
    else if (k === 'n') document.getElementById('btn-new')?.click();
    else handled = false;
    if (handled) {
      e.preventDefault();
      this.ui.update(this);
    }
  }

  private mouseOrCenter(): Vec2 {
    if (this.mouseInside) return this.mouse;
    const [sw, sh] = this.stageSize();
    return [sw / 2, sh / 2];
  }

  blendDescription(): string {
    return BLEND_DESCRIPTIONS[this.blend];
  }
}
