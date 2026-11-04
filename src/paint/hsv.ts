/** HSV <-> RGB as in hsv_helpers.hpp. h in [0,1); s in [0,1]; v may be negative (inverted gradients). */
export type RGB = [number, number, number];

export function rgbToHsv([r, g, b]: RGB): RGB {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const v = max;
  if (max === 0) return [0, 0, 0];
  const delta = max - min;
  const s = delta / max;
  let h = 0;
  if (delta !== 0) {
    if (r === max) h = (g - b) / delta;
    else if (g === max) h = 2 + (b - r) / delta;
    else h = 4 + (r - g) / delta;
  }
  h *= 60;
  if (h < 0) h += 360;
  return [h / 360, s, v];
}

export function hsvToRgb([hIn, s, v]: RGB): RGB {
  let h = hIn * 360;
  if (h > 360) h -= 360;
  if (h < 0) h += 360;
  if (s === 0) return [v, v, v];
  h /= 60;
  let i = Math.floor(h);
  let f = h - i;
  if (i === 6) {
    i = 5;
    f = 1;
  }
  const p = v * (1 - s);
  const q = v * (1 - s * f);
  const t = v * (1 - s * (1 - f));
  switch (i) {
    case 0:
      return [v, t, p];
    case 1:
      return [q, v, p];
    case 2:
      return [p, v, t];
    case 3:
      return [p, q, v];
    case 4:
      return [t, p, v];
    default:
      return [v, p, q];
  }
}
