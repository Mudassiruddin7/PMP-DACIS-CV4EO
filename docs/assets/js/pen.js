/* Hand-drawn marks. Every function returns an SVG path string, seeded so a mark looks the same on every load. */
(function () {
  "use strict";

  function rng(seed) {
    let a = (seed | 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const f = (x) => (Math.round(x * 10) / 10).toString();

  /** An ellipse drawn with a pen: a bit more than one turn, slightly spiralling, radius wobbling. */
  function circle(cx, cy, rx, ry, seed) {
    const r = rng(seed || 3);
    const n = 30, turn = Math.PI * 2 * 1.1, a0 = -Math.PI * 0.55 + (r() - 0.5) * 0.5;
    let d = "";
    for (let i = 0; i <= n; i++) {
      const t = i / n, a = a0 + t * turn;
      const k = 1 + (r() - 0.5) * 0.045 + t * 0.07;
      d += (i ? "L" : "M") + f(cx + Math.cos(a) * rx * k) + " " + f(cy + Math.sin(a) * ry * k) + " ";
    }
    return d;
  }

  /** A curved arrow from (x1,y1) to (x2,y2). Returns {d, head} so the shaft and the head can share a stroke. */
  function arrow(x1, y1, x2, y2, opt) {
    opt = opt || {};
    const r = rng(opt.seed || 5), bend = opt.bend == null ? 0.22 : opt.bend, hs = opt.head || 11;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len;
    const c1x = x1 + dx * 0.3 + nx * len * bend + (r() - 0.5) * 4, c1y = y1 + dy * 0.3 + ny * len * bend + (r() - 0.5) * 4;
    const c2x = x1 + dx * 0.75 + nx * len * bend * 0.6, c2y = y1 + dy * 0.75 + ny * len * bend * 0.6;
    const d = "M" + f(x1) + " " + f(y1) + " C" + f(c1x) + " " + f(c1y) + " " + f(c2x) + " " + f(c2y) + " " + f(x2) + " " + f(y2);
    const ang = Math.atan2(y2 - c2y, x2 - c2x), sp = 0.52;
    const hx1 = x2 - Math.cos(ang - sp) * hs, hy1 = y2 - Math.sin(ang - sp) * hs;
    const hx2 = x2 - Math.cos(ang + sp) * hs, hy2 = y2 - Math.sin(ang + sp) * hs;
    const head = "M" + f(hx1) + " " + f(hy1) + " L" + f(x2) + " " + f(y2) + " L" + f(hx2) + " " + f(hy2);
    return { d: d, head: head };
  }

  /** A slightly wavy underline. */
  function underline(x1, x2, y, seed) {
    const r = rng(seed || 7), n = Math.max(2, Math.round((x2 - x1) / 38));
    let d = "M" + f(x1) + " " + f(y + (r() - 0.5) * 2);
    for (let i = 1; i <= n; i++) {
      const x = x1 + ((x2 - x1) * i) / n;
      d += " Q" + f(x - ((x2 - x1) / n) * 0.5) + " " + f(y + (r() - 0.5) * 5) + " " + f(x) + " " + f(y + (r() - 0.5) * 3);
    }
    return d;
  }

  /** A square bracket opening downwards (side 'down') or to the left, with hand wobble. */
  function bracket(x1, x2, y, depth, seed) {
    const r = rng(seed || 9), w = () => (r() - 0.5) * 1.6;
    return "M" + f(x1 + w()) + " " + f(y) + " Q" + f(x1 + w()) + " " + f(y + depth) + " " + f(x1 + 5) + " " + f(y + depth + w()) +
      " L" + f(x2 - 5) + " " + f(y + depth + w()) + " Q" + f(x2 + w()) + " " + f(y + depth) + " " + f(x2 + w()) + " " + f(y);
  }

  /** A pen cross for 'removed'. */
  function cross(cx, cy, s, seed) {
    const r = rng(seed || 11), j = () => (r() - 0.5) * 1.5;
    return "M" + f(cx - s + j()) + " " + f(cy - s + j()) + " L" + f(cx + s + j()) + " " + f(cy + s + j()) +
      " M" + f(cx + s + j()) + " " + f(cy - s + j()) + " L" + f(cx - s + j()) + " " + f(cy + s + j());
  }

  window.Pen = { rng: rng, circle: circle, arrow: arrow, underline: underline, bracket: bracket, cross: cross };
})();
