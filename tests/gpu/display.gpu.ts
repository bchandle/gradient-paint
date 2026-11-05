/**
 * Display shader: renders to an offscreen texture (no canvas, so this also
 * runs where headless browsers can't present WebGPU canvases) and checks the
 * view transform and tone mapping against the expected pixel values.
 */
import { Display, DisplayParams, ViewMode } from '../../src/app/display';
import { assert, describe, test } from './harness';

async function render(device: GPUDevice, src: Float32Array<ArrayBuffer>, orig: Float32Array<ArrayBuffer>, outW: number, outH: number, p: DisplayParams): Promise<Uint8Array> {
  const display = new Display(device, 'rgba8unorm');
  const mk = (d: Float32Array<ArrayBuffer>) => {
    const b = device.createBuffer({ size: d.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, d);
    return b;
  };
  const tex = device.createTexture({ size: [outW, outH], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const bytesPerRow = Math.ceil((outW * 4) / 256) * 256;
  const out = device.createBuffer({ size: bytesPerRow * outH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  display.encode(enc, tex.createView(), mk(src), mk(orig), p);
  enc.copyTextureToBuffer({ texture: tex }, { buffer: out, bytesPerRow }, [outW, outH]);
  device.queue.submit([enc.finish()]);
  await out.mapAsync(GPUMapMode.READ);
  const raw = new Uint8Array(out.getMappedRange());
  const px = new Uint8Array(outW * outH * 4);
  for (let y = 0; y < outH; y++) px.set(raw.subarray(y * bytesPerRow, y * bytesPerRow + outW * 4), y * outW * 4);
  out.unmap();
  return px;
}

const base: Omit<DisplayParams, 'srcW' | 'srcH' | 'imgW' | 'imgH'> = {
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  mode: ViewMode.Image,
  mean: [0.5, 0.5, 0.5],
  gain: 1,
  bias: [0, 0, 0],
};

describe('display', () => {
  test('maps image pixels through pan/zoom with nearest sampling, backdrop outside', async ({ device }) => {
    // 4x3 image with distinct red values per pixel
    const W = 4;
    const H = 3;
    const img = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) img.set([i / 16, 0.25, 0.75, 0], i * 4);
    const px = await render(device, img, img, 20, 16, { ...base, scale: 3, offsetX: 2, offsetY: 1, srcW: W, srcH: H, imgW: W, imgH: H });
    const at = (x: number, y: number) => Array.from(px.subarray((y * 20 + x) * 4, (y * 20 + x) * 4 + 4));
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        for (const [dx, dy] of [[0, 0], [2, 2]]) {
          const [r, g, b] = at(2 + 3 * x + dx, 1 + 3 * y + dy);
          assert(Math.abs(r - Math.round((255 * (y * W + x)) / 16)) <= 1 && Math.abs(g - 64) <= 1 && Math.abs(b - 191) <= 1, `pixel (${x},${y}) -> ${r},${g},${b}`);
        }
    const [r, g, b] = at(0, 0);
    assert(r === g && g === b && r < 40, `backdrop ${r},${g},${b}`);
    const outside = at(19, 15);
    assert(outside[0] < 40, 'right/bottom outside is backdrop');
  });

  test('applies contrast about the mean, exposure bias and clamping', async ({ device }) => {
    const img = new Float32Array([0.6, 0.4, 0.5, 0, 2, -1, 0.5, 0]);
    const px = await render(device, img, img, 2, 1, { ...base, srcW: 2, srcH: 1, imgW: 2, imgH: 1, gain: 2, bias: [0.1, 0.1, 0.1] });
    // (v - 0.5) * 2 + 0.5 + 0.1
    assert(Math.abs(px[0] - 0.8 * 255) <= 1 && Math.abs(px[1] - 0.4 * 255) <= 1 && Math.abs(px[2] - 0.6 * 255) <= 1, `toned ${px.slice(0, 3)}`);
    assert(px[4] === 255 && px[5] === 0, 'clamped to [0,1]');
  });

  test('signed view (gradients) and compare view', async ({ device }) => {
    const img = new Float32Array([0.25, -0.25, 0, 0]);
    const orig = new Float32Array([0.0, 0.0, 0.5, 0]);
    const signed = await render(device, img, orig, 1, 1, { ...base, mode: ViewMode.Signed, srcW: 1, srcH: 1, imgW: 1, imgH: 1 });
    assert(Math.abs(signed[0] - 191) <= 1 && Math.abs(signed[1] - 64) <= 1 && Math.abs(signed[2] - 128) <= 1, `signed ${signed.slice(0, 3)}`);
    const cmp = await render(device, img, orig, 1, 1, { ...base, mode: ViewMode.Compare, srcW: 1, srcH: 1, imgW: 1, imgH: 1 });
    assert(Math.abs(cmp[0] - 64) <= 1 && Math.abs(cmp[1] - 64) <= 1 && Math.abs(cmp[2] - 128) <= 1, `compare ${cmp.slice(0, 3)}`);
  });

  test('shows coarse multigrid levels stretched over the image', async ({ device }) => {
    const lvl = new Float32Array([0.1, 0, 0, 0, 0.9, 0, 0, 0]); // 2x1 level over an 8x2 image
    const px = await render(device, lvl, lvl, 8, 2, { ...base, mode: ViewMode.Signed, srcW: 2, srcH: 1, imgW: 8, imgH: 2 });
    assert(Math.abs(px[0] - 0.6 * 255) <= 1 && Math.abs(px[(1 * 8 + 7) * 4] - 255) <= 1, `levels ${px[0]} ${px[(8 + 7) * 4]}`);
  });
});
