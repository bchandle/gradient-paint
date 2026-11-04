/** Loading and saving images. Pixel values are kept display-encoded in [0, 1], as in the original. */

export interface LoadedImage {
  width: number;
  height: number;
  /** rgba32 float, alpha unused */
  rgba: Float32Array<ArrayBuffer>;
  name: string;
  /** set when the image was downscaled to fit the size limit */
  scaledFrom?: [number, number];
}

/** Binary 8-bit PPM (P6), as the original accepted. */
export function parsePPM(buf: ArrayBuffer): { width: number; height: number; rgb: Uint8Array } | null {
  const bytes = new Uint8Array(buf);
  let pos = 0;
  const token = (): string => {
    for (;;) {
      while (pos < bytes.length && /\s/.test(String.fromCharCode(bytes[pos]))) pos++;
      if (bytes[pos] === 0x23) {
        while (pos < bytes.length && bytes[pos] !== 0x0a) pos++;
        continue;
      }
      break;
    }
    let t = '';
    while (pos < bytes.length && !/\s/.test(String.fromCharCode(bytes[pos]))) t += String.fromCharCode(bytes[pos++]);
    return t;
  };
  if (token() !== 'P6') return null;
  const width = parseInt(token(), 10);
  const height = parseInt(token(), 10);
  const max = parseInt(token(), 10);
  pos++; // single whitespace
  if (!(width > 0 && height > 0) || max !== 255 || pos + width * height * 3 > bytes.length) return null;
  return { width, height, rgb: bytes.subarray(pos, pos + width * height * 3) };
}

export async function loadImageFile(file: Blob & { name?: string }, maxPixels: number): Promise<LoadedImage> {
  const name = file.name ?? 'image';
  let source: CanvasImageSource;
  let w: number;
  let h: number;
  if (/\.ppm$/i.test(name)) {
    const ppm = parsePPM(await file.arrayBuffer());
    if (!ppm) throw new Error('Unsupported PPM (expected binary 8-bit P6).');
    const c = new OffscreenCanvas(ppm.width, ppm.height);
    const id = new ImageData(ppm.width, ppm.height);
    for (let i = 0; i < ppm.width * ppm.height; i++) {
      id.data[i * 4] = ppm.rgb[i * 3];
      id.data[i * 4 + 1] = ppm.rgb[i * 3 + 1];
      id.data[i * 4 + 2] = ppm.rgb[i * 3 + 2];
      id.data[i * 4 + 3] = 255;
    }
    c.getContext('2d')!.putImageData(id, 0, 0);
    source = c;
    w = ppm.width;
    h = ppm.height;
  } else {
    const bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    source = bmp;
    w = bmp.width;
    h = bmp.height;
  }
  const scale = Math.min(1, Math.sqrt(maxPixels / (w * h)));
  const W = Math.max(2, Math.round(w * scale));
  const H = Math.max(2, Math.round(h * scale));
  const canvas = new OffscreenCanvas(W, H);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, W, H);
  const data = ctx.getImageData(0, 0, W, H).data;
  const rgba = new Float32Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const a = data[i * 4 + 3] / 255;
    // composite transparent images over mid grey
    for (let c = 0; c < 3; c++) rgba[i * 4 + c] = (data[i * 4 + c] / 255) * a + 0.5 * (1 - a);
  }
  return { width: W, height: H, rgba, name, scaledFrom: scale < 1 ? [w, h] : undefined };
}

/** Forward-difference gradients of an rgba image: gx is (W-1) x H, gy is W x (H-1). */
export function imageGradients(img: Float32Array, W: number, H: number): { gx: Float32Array<ArrayBuffer>; gy: Float32Array<ArrayBuffer> } {
  const gx = new Float32Array((W - 1) * H * 4);
  const gy = new Float32Array(W * (H - 1) * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      for (let c = 0; c < 3; c++) {
        if (x + 1 < W) gx[(y * (W - 1) + x) * 4 + c] = img[i + 4 + c] - img[i + c];
        if (y + 1 < H) gy[(y * W + x) * 4 + c] = img[i + W * 4 + c] - img[i + c];
      }
      if (x + 1 < W) gx[(y * (W - 1) + x) * 4 + 3] = 1;
      if (y + 1 < H) gy[(y * W + x) * 4 + 3] = 1;
    }
  return { gx, gy };
}

export function channelMeans(img: Float32Array, W: number, H: number): [number, number, number] {
  const m = [0, 0, 0];
  for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) m[c] += img[i * 4 + c];
  return [m[0] / (W * H), m[1] / (W * H), m[2] / (W * H)];
}

export async function savePNG(
  rgba: Float32Array,
  W: number,
  H: number,
  transform: (v: number, c: number) => number,
  filename: string,
): Promise<void> {
  const id = new ImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) id.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(transform(rgba[i * 4 + c], c) * 255)));
    id.data[i * 4 + 3] = 255;
  }
  const canvas = new OffscreenCanvas(W, H);
  canvas.getContext('2d')!.putImageData(id, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
