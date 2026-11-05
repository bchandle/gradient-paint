import { App, IS_TOUCH_DEVICE, MAX_PIXELS, MODE_NAMES, MODES } from './app/app';
import { BLEND_MODES } from './paint/blend';
import { requestGpuDevice } from './solver/gpu';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

function toast(msg: string, ms = 2200): void {
  const host = $('toasts');
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  host.appendChild(el);
  while (host.children.length > 3) host.firstElementChild!.remove();
  setTimeout(() => {
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 350);
  }, ms);
}

function fatal(title: string, detail: string): void {
  const el = $('fatal');
  el.hidden = false;
  el.innerHTML = '';
  const box = document.createElement('div');
  const h = document.createElement('h2');
  h.textContent = title;
  const p = document.createElement('p');
  p.textContent = detail;
  box.append(h, p);
  el.appendChild(box);
}

function colorCss(c: number[]): string {
  // negative colours paint the complementary edge; show their magnitude
  return `rgb(${c.map((v) => Math.round(Math.min(1, Math.abs(v)) * 255)).join(',')})`;
}

function buildUi(app: App) {
  const modes = $('modes');
  for (const m of MODES) {
    const b = document.createElement('button');
    b.textContent = MODE_NAMES[m];
    b.dataset.mode = m;
    b.onclick = () => app.setMode(m);
    modes.appendChild(b);
  }
  const blends = $('blends');
  for (const m of BLEND_MODES) {
    const b = document.createElement('button');
    b.textContent = m;
    b.dataset.blend = m;
    b.onclick = () => app.setBlend(m);
    blends.appendChild(b);
  }
  $('btn-undo').onclick = () => app.undo();
  $('btn-revert').onclick = () => app.revert();
  $('btn-save').onclick = () => void app.save();
  $('btn-help').onclick = () => {
    app.help = !app.help;
    update(app);
  };
  $('help').onclick = (e) => {
    if ((e.target as HTMLElement).tagName !== 'A') {
      app.help = false;
      update(app);
    }
  };
  const fileInput = $<HTMLInputElement>('open-file');
  $('btn-open').onclick = () => fileInput.click();
  fileInput.onchange = () => {
    const f = fileInput.files?.[0];
    if (f) void app.openFile(f);
    fileInput.value = '';
  };

  // new-canvas dialog
  const dlg = $<HTMLDialogElement>('dlg-new');
  const wIn = $<HTMLInputElement>('new-w');
  const hIn = $<HTMLInputElement>('new-h');
  $('btn-new').onclick = () => {
    wIn.value = String(app.W || 1025);
    hIn.value = String(app.H || 769);
    dlg.showModal();
  };
  dlg.querySelectorAll<HTMLButtonElement>('[data-size]').forEach((b) => {
    b.onclick = () => {
      const [w, h] = b.dataset.size === 'fit' ? fitSize(app) : b.dataset.size!.split('x').map(Number);
      wIn.value = String(w);
      hIn.value = String(h);
    };
  });
  dlg.addEventListener('close', () => {
    if (dlg.returnValue !== 'ok') return;
    const w = Math.round(Number(wIn.value));
    const h = Math.round(Number(hIn.value));
    if (!(w >= 2 && h >= 2 && w <= 4096 && h <= 4096 && w * h <= MAX_PIXELS)) {
      toast(`Canvas must be between 2×2 and 4096×4096 (at most ${(MAX_PIXELS / 1e6).toFixed(1)} megapixels on this device).`, 4000);
      return;
    }
    app.newCanvas(w, h);
  });

  // drag & drop
  let depth = 0;
  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    if (++depth === 1) document.body.classList.add('dragover');
  });
  window.addEventListener('dragleave', () => {
    if (--depth === 0) document.body.classList.remove('dragover');
  });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    document.body.classList.remove('dragover');
    const f = e.dataTransfer?.files?.[0];
    if (f) void app.openFile(f);
  });

  setInterval(() => {
    $('st-fps').textContent = `${app.fps.toFixed(0)} fps`;
  }, 500);

  // dock (touch-friendly controls)
  $('btn-pick').onclick = () => app.setPickLatch('pick');
  $('btn-value').onclick = () => app.setPickLatch('value');
  $('btn-move').onclick = () => app.toggleMoveLatch();
  $('btn-fit').onclick = () => {
    app.fitView();
    update(app);
  };
  $('btn-finger').onclick = () => app.cycleFingerMode();
  $('btn-pressure').onclick = () => app.cyclePressureMode();
  $('st-color').onclick = () => {
    if (app.mode === 'paint') app.setPickLatch('pick');
  };
  installSizeSlider(app);
}

/** Brush size on a logarithmic scale, 0.5 .. 200 px. */
const SIZE_MIN = 0.5;
const SIZE_MAX = 200;
const sizeToT = (s: number) => Math.log(Math.min(SIZE_MAX, Math.max(SIZE_MIN, s)) / SIZE_MIN) / Math.log(SIZE_MAX / SIZE_MIN);
const tToSize = (t: number) => {
  const s = SIZE_MIN * Math.pow(SIZE_MAX / SIZE_MIN, Math.min(1, Math.max(0, t)));
  return s < 10 ? Math.max(SIZE_MIN, Math.round(s * 2) / 2) : Math.round(s);
};

function installSizeSlider(app: App): void {
  const el = $('size-slider');
  const track = el.querySelector('.track') as HTMLElement;
  const set = (e: PointerEvent) => {
    const r = track.getBoundingClientRect();
    app.setBrushSize(tToSize(1 - (e.clientY - r.top) / r.height));
  };
  el.addEventListener('pointerdown', (e) => {
    el.setPointerCapture(e.pointerId);
    set(e);
  });
  el.addEventListener('pointermove', (e) => {
    if (el.hasPointerCapture(e.pointerId)) set(e);
  });
  el.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') app.stepBrush(true);
    else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') app.stepBrush(false);
    else return;
    e.preventDefault();
    e.stopPropagation();
  });
}

function fitSize(app: App): [number, number] {
  const [sw, sh] = app.stageSize();
  const clamp = (v: number) => Math.max(64, Math.min(2048, Math.floor(v)));
  return [clamp(sw - 24), clamp(sh - 24)];
}

function update(app: App): void {
  $('help').hidden = !app.help;
  document.querySelectorAll<HTMLButtonElement>('#modes button').forEach((b) => b.classList.toggle('on', b.dataset.mode === app.mode));
  document.querySelectorAll<HTMLButtonElement>('#blends button').forEach((b) => b.classList.toggle('on', b.dataset.blend === app.blend));
  $('hud-view').textContent = app.viewName === 'f' && !app.compare ? '' : app.compare && app.viewName === 'f' ? 'f − original' : app.viewName;
  $('hud-mode').textContent = `${MODE_NAMES[app.mode]}/${app.blend}`;
  $('st-size').innerHTML = `<b>${app.W}×${app.H}</b> ${escapeHtml(app.imageName)}`;
  $('st-brush').innerHTML = `brush <b>${app.brushSize < 10 ? app.brushSize.toFixed(0) : app.brushSize.toFixed(1)}</b> px`;
  const c = app.paintColor;
  const color = $('st-color');
  color.hidden = app.mode !== 'paint';
  (color.querySelector('.swatch') as HTMLElement).style.background = colorCss(c);
  (color.lastElementChild as HTMLElement).textContent = app.paintHsv[2] < 0 ? 'inverted' : '';
  $('st-blend').textContent = app.blendDescription();
  $('st-cycles').innerHTML = `<b>${app.cyclesPerFrame}</b> V-cycle${app.cyclesPerFrame > 1 ? 's' : ''}/frame${app.singleStep ? ' (single-step)' : ''}`;
  $('st-zoom').textContent = `${Math.round(app.view.scale * 100)}%`;

  // dock
  const t = sizeToT(app.brushSize);
  const slider = $('size-slider');
  (slider.querySelector('.fill') as HTMLElement).style.height = `${t * 100}%`;
  (slider.querySelector('.knob') as HTMLElement).style.top = `${(1 - t) * 100}%`;
  (slider.querySelector('.value') as HTMLElement).textContent = app.brushSize < 10 ? app.brushSize.toFixed(1).replace(/\.0$/, '') : app.brushSize.toFixed(0);
  slider.setAttribute('aria-valuenow', String(app.brushSize));
  const pick = $('btn-pick');
  pick.textContent = app.mode === 'paint' ? 'Colour' : app.mode === 'clone' ? 'Source' : 'Capture';
  pick.classList.toggle('on', app.pickLatch === 'pick');
  const value = $('btn-value');
  value.hidden = app.mode !== 'paint';
  value.classList.toggle('on', app.pickLatch === 'value');
  $('btn-move').classList.toggle('on', app.moveLatch);
  $('btn-finger').textContent = `finger: ${app.fingerMode === 'auto' ? (app.fingersDraw ? 'auto (draw)' : 'auto (pick)') : app.fingerMode === 'draw' ? 'draws' : 'picks'}`;
  $('btn-pressure').textContent = `pressure: ${app.pressureMode}`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
}

async function main() {
  let device: GPUDevice;
  try {
    device = await requestGpuDevice();
  } catch (e) {
    fatal(
      'WebGPU is not available',
      `${e instanceof Error ? e.message : e} Gradient Paint runs its integrator on the GPU with WebGPU — try a current version of Chrome, Edge, Safari or Firefox, and make sure hardware acceleration is enabled.`,
    );
    return;
  }
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') fatal('The GPU device was lost', `${info.message} Reload the page to continue.`);
  });
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message));

  const params = new URLSearchParams(location.search);
  const app = new App(device, $<HTMLCanvasElement>('gpu'), $<HTMLCanvasElement>('overlay'), $('stage'), { update, toast }, params.get('present') !== '0');
  buildUi(app);
  (window as unknown as { gradientPaint: App }).gradientPaint = app;

  // initial canvas: ?size=WxH, else fill the window
  const q = params.get('size');
  const m = q?.match(/^(\d+)x(\d+)$/);
  const [w, h] = m ? [Math.max(2, Math.min(4096, +m[1])), Math.max(2, Math.min(4096, +m[2]))] : fitSize(app);
  app.newCanvas(w, h);
  app.start();
  let seen = false;
  try {
    seen = !!localStorage.getItem('gp-seen-help');
    localStorage.setItem('gp-seen-help', '1');
  } catch {
    /* storage unavailable: just show the hint */
  }
  if (!seen)
    toast(
      IS_TOUCH_DEVICE
        ? 'Draw with the Pencil or a finger · Colour in the dock picks a colour · two fingers pan, zoom and tap to undo'
        : 'Left-drag to paint gradients · right-drag to pick a colour · F1 for help',
      7000,
    );
}

void main();
