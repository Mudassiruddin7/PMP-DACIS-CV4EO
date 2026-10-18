/* Figures built from HTML and small SVGs: block bars, channel barcodes, partitions, the episode, cost, bands, tables. */
(function () {
  "use strict";
  const D = window.SITE_DATA, K = window.ChartKit;
  const { S, T, H, $, num, signed, mount, onResize, seg, dataTable, Tip } = K;
  const commas = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

  // ------------------------------------------------------------------------------------------ blocks: how many channels each block keeps
  function initBudget() {
    const cont = $("#chart-budget"); if (!cont) return;
    const P = D.profile, keys = ["100", "50", "30", "20", "10"], read = $("#read-budget");
    let b = "30", els = null;
    function update() {
      if (!els) return;
      const kept = P.kept[b], tot = kept.reduce((a, c) => a + c, 0), all = P.orig.reduce((a, c) => a + c, 0);
      kept.forEach((k, i) => {
        els.bar[i].style.transform = "scaleY(" + k / P.orig[i] + ")";
        els.cnt[i].textContent = String(k);
        els.pct[i].textContent = Math.round((100 * k) / P.orig[i]) + "%";
      });
      const params = b === "100" ? 11.1765 : P.params[b] * 11.1765;
      read.replaceChildren(H("b", { text: commas(tot) + " of " + commas(all) + " channels" }), document.createTextNode(" kept (" + Math.round((100 * tot) / all) + "%), carrying "),
        H("b", { text: num(params, 2) + " M of 11.18 M" }), document.createTextNode(" trunk parameters (" + b + "%). The first block keeps " + Math.round((100 * kept[0]) / P.orig[0]) + "% of its channels, the last " + Math.round((100 * kept[7]) / P.orig[7]) + "%."));
      cont.setAttribute("aria-label", "Channels kept per block at " + b + "% of the parameters: " + kept.join(", "));
    }
    mount(cont, (W) => {
      const Tp = 26, maxH = 190, B = 82, H_ = Tp + maxH + B, slot = W / 8, bw = Math.min(74, slot * 0.6);
      const svg = S("svg", { viewBox: "0 0 " + W + " " + H_, role: "presentation" }, cont);
      els = { bar: [], cnt: [], pct: [] };
      S("path", { d: "M0 " + (Tp + maxH) + "H" + W, class: "axis-l" }, svg);
      P.orig.forEach((o, i) => {
        const cx = slot * i + slot / 2, x = cx - bw / 2, h = (o / 512) * maxH, y = Tp + maxH - h;
        S("rect", { x: x, y: y, width: bw, height: h, style: "fill:url(#hatch);color:var(--ink-3);stroke:var(--ink-3);stroke-dasharray:4 3;stroke-width:1.2;opacity:.9" }, svg);
        const r = S("rect", { x: x, y: y, width: bw, height: h, style: "fill:var(--ink);transform-box:fill-box;transform-origin:50% 100%;transition:transform .45s cubic-bezier(.3,.7,.2,1)" }, svg);
        els.bar.push(r);
        els.pct.push(T("", { x: cx, y: y - 7, "text-anchor": "middle", class: "lbl" }, svg));
        T(slot < 70 ? String(i + 1) : "block " + (i + 1), { x: cx, y: Tp + maxH + 19, "text-anchor": "middle", class: "lbl" }, svg);
        els.cnt.push(T("", { x: cx, y: Tp + maxH + 44, "text-anchor": "middle", class: "lbl-d", style: "font-size:20px;font-weight:800" }, svg));
        T("of " + o, { x: cx, y: Tp + maxH + 62, "text-anchor": "middle", class: "lbl", style: "fill:var(--ink-3)" }, svg);
        const hit = S("rect", { x: slot * i, y: Tp, width: slot, height: maxH + B - 10, class: "hit", tabindex: 0, role: "img", "aria-label": "Block " + (i + 1) }, svg);
        Tip.bind(hit, () => ({ head: P.kept[b][i] + " of " + o + " channels", lines: [{ text: "block " + (i + 1) + " at " + b + "% of the parameters" }] }));
      });
      update();
    });
    seg($("#seg-budget"), keys.map((k) => ({ v: k, t: k + "%" })), b, (v) => { b = v; update(); });
    dataTable(cont, "budget", ["Block"].concat(P.orig.map((o, i) => String(i + 1))), [["channels before"].concat(P.orig.map(String))].concat(keys.map((k) => [k + "% kept"].concat(P.kept[k].map(String)))), "Data table: channels per block");
  }

  // ------------------------------------------------------------------------------------------ picks: the real kept channels of four scores
  function initPicks() {
    const rowsHost = $("#picks-rows"); if (!rowsHost) return;
    const jacHost = $("#picks-jac"), read = $("#read-picks"), PK = D.picks, ids = ["random", "l1", "taylor", "dacis"];
    const NAME = { random: "Random", l1: "Filter L1", taylor: "Taylor", dacis: "DACIS" };
    let blk = 4;
    function render() {
      rowsHost.replaceChildren();
      const n = PK.orig[blk];
      ids.forEach((id) => {
        const kept = new Set(PK.scores[id][blk]);
        let dk = "", dc = "";
        for (let i = 0; i < n; i++) { const x = i * 4 + 2; if (kept.has(i)) dk += "M" + x + " 2V32"; else dc += "M" + x + " 13V21"; }
        const row = H("div", { class: "barcode-row" });
        const nm = H("span", { class: "nm" }); const dot = H("i"); dot.style.background = K.SC_VAR[id]; nm.appendChild(dot); nm.appendChild(document.createTextNode(NAME[id]));
        const svg = S("svg", { viewBox: "0 0 " + n * 4 + " 34", preserveAspectRatio: "none", role: "img", "aria-label": NAME[id] + " kept " + kept.size + " of " + n + " channels in block " + (blk + 1) });
        S("path", { d: dc, style: "stroke:var(--rule);stroke-width:1.6;vector-effect:non-scaling-stroke" }, svg);
        S("path", { d: dk, style: "stroke:" + K.SC_VAR[id] + ";stroke-width:1.8;vector-effect:non-scaling-stroke" }, svg);
        row.appendChild(nm); row.appendChild(svg); row.appendChild(H("span", { class: "ac", text: num(PK.acc[id]) + "%" }));
        rowsHost.appendChild(row);
      });
      rowsHost.appendChild(H("p", { class: "readout", text: "Block " + (blk + 1) + " has " + n + " channels. Each score keeps " + PK.scores.random[blk].length + " of them. The number on the right is the score's 5-shot accuracy after recovery." }));
    }
    function jac() {
      const tb = H("table", { class: "jac" }), cap = H("caption", { text: "Overlap of what two scores kept" });
      tb.appendChild(cap);
      const hr = H("tr"); hr.appendChild(H("th")); ids.forEach((i) => hr.appendChild(H("th", { scope: "col", text: NAME[i] })));
      tb.appendChild(hr);
      ids.forEach((a) => {
        const tr = H("tr"); tr.appendChild(H("th", { scope: "row", text: NAME[a] }));
        ids.forEach((c) => {
          const v = PK.jaccard[a + "|" + c], td = H("td", { text: a === c ? "–" : num(v, 2) });
          if (a !== c) td.style.background = "color-mix(in srgb, var(--pen) " + Math.round(v * 120) + "%, transparent)";
          tr.appendChild(td);
        });
        tb.appendChild(tr);
      });
      jacHost.replaceChildren(tb);
    }
    // what two random picks would share by luck, from the real block sizes
    let inter = 0, uni = 0;
    PK.orig.forEach((o, i) => { const p = D.profile.kept["30"][i] / o; inter += o * p * p; uni += o * (2 * p - p * p); });
    const chance = inter / uni;
    const vsRandom = ["l1", "taylor", "dacis"].map((i) => PK.jaccard["random|" + i]);
    read.replaceChildren(document.createTextNode("Overlap with random choice is "), H("b", { text: num(Math.min.apply(null, vsRandom), 2) + " to " + num(Math.max.apply(null, vsRandom), 2) }),
      document.createTextNode(", about what two random picks would share by luck (" + num(chance, 2) + "). Taylor and DACIS agree most, "), H("b", { text: num(PK.jaccard["taylor|dacis"], 2) }),
      document.createTextNode(". Different channels, similar accuracy: after recovery the four land within "), H("b", { text: num(Math.max.apply(null, ids.map((i) => PK.acc[i])) - Math.min.apply(null, ids.map((i) => PK.acc[i]))) + " points" }),
      document.createTextNode(" of each other."));
    seg($("#seg-block"), PK.orig.map((o, i) => ({ v: i, t: String(i + 1) + " · " + o })), blk, (v) => { blk = v; render(); });
    render(); jac();
  }

  // ------------------------------------------------------------------------------------------ the 45 classes
  function initPartitions() {
    const grid = $("#pgrid"); if (!grid) return;
    const PT = D.partitions, list = $("#plist"); let p = 0;
    function render() {
      const sp = PT.splits[p], role = {};
      sp.base.forEach((c) => (role[c] = "base")); sp.val.forEach((c) => (role[c] = "val")); sp.novel.forEach((c) => (role[c] = "novel"));
      grid.replaceChildren();
      PT.classes.forEach((nm, i) => {
        const cell = H("div", { class: "pcell " + role[i], tabindex: 0, role: "img", "aria-label": nm + ": " + { base: "trains the network", val: "validation", novel: "held out for the test" }[role[i]], text: String(i + 1) });
        Tip.bind(cell, () => ({ head: nm, lines: [{ text: { base: "trains the network", val: "validation check", novel: "held out: the test" }[role[i]] }] }));
        grid.appendChild(cell);
      });
      const names = (a) => a.map((c) => PT.classes[c]).join(", ");
      list.replaceChildren();
      const add = (cls, title, body) => { const dt = H("dt"); dt.appendChild(H("span", { class: "key " + cls })); dt.appendChild(H("b", { text: title })); list.appendChild(dt); list.appendChild(H("dd", { text: body })); };
      add("novel", "Test, held out (15)", names(sp.novel));
      add("val", "Validation (5)", names(sp.val));
      add("base", "Train (25)", names(sp.base));
    }
    seg($("#seg-part"), PT.splits.map((s, i) => ({ v: i, t: String(i) })), p, (v) => { p = v; render(); });
    render();
  }

  // ------------------------------------------------------------------------------------------ one episode with real scenes
  function initEpisode() {
    const cols = $("#eps-cols"); if (!cols) return;
    const classes = [["airplane", "airplane"], ["industrial_area", "industrial area"], ["island", "island"], ["roundabout", "roundabout"], ["storage_tank", "storage tank"]];
    classes.forEach(([c, nm], i) => {
      const col = H("div", { class: "eps-col" }), stack = H("div", { class: "stack" });
      for (let k = 1; k <= 5; k++) stack.appendChild(H("img", { src: "assets/img/episode/" + c + "_s" + k + ".jpg", alt: "Support photo " + k + " of class " + nm, width: 192, height: 192, loading: "lazy" }));
      const pr = H("div", { class: "proto" + (c === "roundabout" ? " hit" : "") }, [H("i"), H("span", { text: nm })]);
      col.appendChild(stack); col.appendChild(pr); cols.appendChild(col);
    });
    const bus = $("#eps-bus");
    function draw() {
      bus.replaceChildren();
      const eb = bus.getBoundingClientRect(), q = $(".eps-q img").getBoundingClientRect(), dots = Array.from(document.querySelectorAll("#eps-cols .proto i"));
      if (eb.width < 220) return;
      const svg = S("svg", { viewBox: "0 0 " + eb.width + " " + eb.height }, bus);
      const qx = q.left + q.width / 2 - eb.left, railY = eb.height * 0.58, xs = dots.map((d) => { const r = d.getBoundingClientRect(); return r.left + r.width / 2 - eb.left; });
      const base = "fill:none;stroke:var(--ink-3);stroke-width:1.6;stroke-linecap:round";
      S("path", { d: "M" + Math.min.apply(null, xs) + " " + railY + "H" + Math.max.apply(null, xs), style: base }, svg);
      S("path", { d: "M" + qx + " " + eb.height + "V" + railY, style: base }, svg);
      xs.forEach((x, i) => {
        const hit = i === 3;
        S("path", { d: "M" + x + " " + railY + "V" + (hit ? 4 : 10), style: hit ? "fill:none;stroke:var(--pen);stroke-width:3;stroke-linecap:round" : base + ";stroke-dasharray:3 4" }, svg);
      });
      const hx = xs[3];
      S("path", { d: "M" + (hx - 7) + " 14L" + hx + " 3L" + (hx + 7) + " 14", style: "fill:none;stroke:var(--pen);stroke-width:3;stroke-linecap:round;stroke-linejoin:round" }, svg);
      const wide = hx + 190 < eb.width;
      T(wide ? "closest prototype wins" : "closest wins", { x: wide ? hx + 16 : hx - 16, y: railY - 8, "text-anchor": wide ? "start" : "end", class: "pen-t" }, svg);
    }
    onResize($("#episode"), () => requestAnimationFrame(draw));
    window.addEventListener("load", () => requestAnimationFrame(draw));
  }

  // ------------------------------------------------------------------------------------------ the PMP schedule: the dashed hand-off from M to the second cut
  function initPmp() {
    const ol = $("#pmp"); if (!ol) return;
    const wrap = ol.parentElement; wrap.style.position = "relative";
    let svg = null;
    function draw() {
      if (svg) svg.remove();
      const m = ol.querySelector("li.meta"), p2 = ol.querySelectorAll("li.cut")[1];
      if (!m || !p2) return;
      const wr = wrap.getBoundingClientRect(), a = m.getBoundingClientRect(), b = p2.getBoundingClientRect();
      if (Math.abs(a.top - b.top) > 4 || wr.width < 700) return;
      const x1 = a.left + a.width / 2 - wr.left, x2 = b.left + b.width / 2 - wr.left, y0 = a.bottom - wr.top + 6;
      svg = S("svg", { class: "pmp-note", viewBox: "0 0 " + wr.width + " 80", "aria-hidden": "true" }, wrap);
      svg.style.top = y0 + "px";
      const d = "M" + x1 + " 0 C" + x1 + " 46 " + x2 + " 46 " + x2 + " 4";
      S("path", { d: d, class: "pen-s", "stroke-dasharray": "6 6", filter: "url(#wob)" }, svg);
      S("path", { d: "M" + (x2 - 7) + " 14L" + x2 + " 2L" + (x2 + 7) + " 14", class: "pen-s", filter: "url(#wob)" }, svg);
      const note = T("the gradient norm of every filter, collected while the episodes run", { x: (x1 + x2) / 2, y: 70, "text-anchor": "middle", class: "pen-t" }, svg);
      const nb = note.getBoundingClientRect(), wb = wrap.getBoundingClientRect();
      if (nb.left < wb.left || nb.right > wb.right) note.textContent = "meta-gradient norms";
    }
    onResize(wrap, () => requestAnimationFrame(draw));
    window.addEventListener("load", () => requestAnimationFrame(draw));
  }

  // ------------------------------------------------------------------------------------------ operations against time
  function initLatency() {
    const host = $("#lat"); if (!host) return;
    const rows = D.cost.rows, dense = rows.find((r) => r.key === "dense"), read = $("#read-lat");
    let b = "30";
    const mk = (cls) => ({ k: H("div", { class: "k" }), tr: H("div", { class: "track " + cls }) });
    const a = mk("macs"), c = mk("cpu");
    [a, c].forEach((r) => { r.ghost = H("div", { class: "ghost" }); r.fill = H("div", { class: "fill" }); r.v = H("div", { class: "v" }); r.tr.appendChild(r.ghost); r.tr.appendChild(r.fill); r.tr.appendChild(r.v); host.appendChild(r.k); host.appendChild(r.tr); });
    a.k.appendChild(document.createTextNode("Multiply-accumulates")); a.k.appendChild(H("small", { text: "per image" }));
    c.k.appendChild(document.createTextNode("CPU time")); c.k.appendChild(H("small", { text: "one thread, batch 1" }));
    function update() {
      const r = rows.find((x) => x.key === "r18_" + b);
      a.fill.style.setProperty("--w", (100 - r.macs_red) + "%"); a.v.style.setProperty("--w", (100 - r.macs_red) + "%");
      c.fill.style.setProperty("--w", (100 - r.cpu_red) + "%"); c.v.style.setProperty("--w", (100 - r.cpu_red) + "%");
      a.v.replaceChildren(document.createTextNode(num(r.macs_g, 3) + " G "), H("i", { text: "−" + Math.round(r.macs_red) + "%" }));
      c.v.replaceChildren(document.createTextNode(num(r.cpu_ms) + " ms "), H("i", { text: "−" + Math.round(r.cpu_red) + "%" }));
      read.replaceChildren(document.createTextNode("Dense ResNet-18: " + num(dense.macs_g, 3) + " G, " + num(dense.cpu_ms) + " ms. Kept at " + b + "%: "), H("b", { text: Math.round(r.macs_red) + "% fewer operations" }),
        document.createTextNode(", "), H("b", { text: Math.round(r.cpu_red) + "% less time" }), document.createTextNode(". The gap stays at every budget we measured."));
    }
    seg($("#seg-lat"), ["50", "30", "20", "10"].map((k) => ({ v: k, t: k + "%" })), b, (v) => { b = v; update(); });
    update();
  }

  // ------------------------------------------------------------------------------------------ squares: area is work
  function initSquares() {
    const host = $("#squares"); if (!host) return;
    let ds = "resisc45";
    const ids = [["dense_ce", "ResNet-18, unpruned"], ["A_taylor_PMP_30", "ResNet-18 pruned to 30%"], ["MP_taylor_P_60", "MobileNetV2 pruned to 60%"], ["K_mobilenet_v3_small", "MobileNetV3-Small, never pruned"]];
    function render() {
      host.replaceChildren();
      const pts = D.tradeoff[ds], items = ids.map(([id, nm]) => ({ p: pts.find((p) => p.id === id), nm: nm })).filter((x) => x.p);
      const maxm = Math.max.apply(null, items.map((x) => x.p.macs)), base = Math.min(220, Math.max(130, host.getBoundingClientRect().width * 0.28));
      items.forEach(({ p, nm }) => {
        const s = Math.round(base * Math.sqrt(p.macs / maxm)), win = p.id === "K_mobilenet_v3_small";
        const sq = H("div", { class: "sq" + (win ? " win" : "") }), box = H("div", { class: "box" });
        box.style.setProperty("--s", s + "px");
        sq.appendChild(H("div", { class: "acc", text: num(p.m) + "%" })); sq.appendChild(box); sq.appendChild(H("div", { class: "t", text: nm }));
        sq.appendChild(H("div", { class: "m", text: num(p.macs, 3) + " G MACs · " + num(p.params, 2) + " M params" }));
        host.appendChild(sq);
      });
    }
    seg($("#seg-sq"), [{ v: "resisc45", t: "RESISC45" }, { v: "eurosat13", t: "EuroSAT-MS" }, { v: "plantvillage", t: "PlantVillage" }], ds, (v) => { ds = v; render(); });
    render();
    let w0 = 0; new ResizeObserver(() => { const w = Math.round(host.getBoundingClientRect().width); if (w !== w0) { w0 = w; render(); } }).observe(host);
  }

  // ------------------------------------------------------------------------------------------ one real scene: pins, spectra, twelve bands
  const SCENE = window.SCENE_DATA;
  const BAND_INFO = { B01: ["Coastal aerosol", 60], B02: ["Blue", 10], B03: ["Green", 10], B04: ["Red", 10], B05: ["Red edge 1", 20], B06: ["Red edge 2", 20],
    B07: ["Red edge 3", 20], B08: ["Near infrared", 10], B8A: ["Narrow near infrared", 20], B09: ["Water vapour", 60], B11: ["Shortwave infrared 1", 20], B12: ["Shortwave infrared 2", 20] };
  const SHORT_BAND = { B01: "Coastal", B02: "Blue", B03: "Green", B04: "Red", B05: "Red edge 1", B06: "Red edge 2", B07: "Red edge 3", B08: "NIR", B8A: "Narrow NIR", B09: "Water vapour", B11: "SWIR 1", B12: "SWIR 2" };

  function initScene() {
    const ov = $("#scene-svg"); if (!ov || !SCENE) return;
    const pins = SCENE.pins, N = SCENE.size_px, bi = (id) => SCENE.bands.findIndex((b) => b.id === id);
    let sticky = null, hover = null;
    const cur = () => (hover != null ? hover : sticky);
    const pinEls = [];

    // the size of one EuroSAT patch, drawn to scale
    const P = SCENE.patch_px, px0 = 130, py0 = 395;
    S("rect", { x: px0, y: py0, width: P, height: P, style: "fill:none;stroke:#fff;stroke-width:4" }, ov);
    S("rect", { x: px0 - 2, y: py0 - 2, width: P + 4, height: P + 4, style: "fill:none;stroke:rgba(0,0,0,.55);stroke-width:1.5" }, ov);
    T("one EuroSAT patch", { x: px0 + P + 12, y: py0 + 44, class: "patch-note" }, ov);

    pins.forEach((p, i) => {
      const x = p.x * N, y = p.y * N, left = p.x > 0.7 || p.name === "Po river";
      const g = S("g", { class: "pin", tabindex: 0, role: "button", "aria-pressed": "false", "aria-label": p.name + ", " + p.what }, ov);
      S("circle", { class: "ring", cx: x, cy: y, r: 17 }, g);
      S("circle", { class: "dot", cx: x, cy: y, r: 9, style: "fill:" + p.color }, g);
      T(p.name, { x: x + (left ? -26 : 26), y: y + 11, "text-anchor": left ? "end" : "start" }, g);
      g.addEventListener("pointerenter", () => { hover = i; sync(); });
      g.addEventListener("pointerleave", () => { hover = null; sync(); });
      g.addEventListener("click", () => { sticky = sticky === i ? null : i; sync(); });
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sticky = sticky === i ? null : i; sync(); } });
      g.addEventListener("focus", () => { hover = i; sync(); });
      g.addEventListener("blur", () => { hover = null; sync(); });
      pinEls.push(g);
    });

    const read = $("#read-spectra"), cont = $("#chart-spectra");
    const val = (p, id) => p.spectrum[bi(id)];
    function sync() {
      const c = cur();
      pinEls.forEach((g, i) => { g.classList.toggle("on", c === i); g.classList.toggle("off", c != null && c !== i); g.setAttribute("aria-pressed", String(sticky === i)); });
      if (c == null) {
        read.textContent = "Hover or tap a pin to follow one spot across the bands.";
      } else {
        const p = pins[c], r = val(p, "B04"), n = val(p, "B08"), s = val(p, "B12"), nd = (n - r) / (n + r);
        read.replaceChildren(H("b", { text: p.name }), document.createTextNode(" (" + p.what + "): red " + num(r, 2) + ", near infrared " + num(n, 2) + ", shortwave infrared " + num(s, 2) + ". Vegetation index (NIR \u2212 red) / (NIR + red) = "), H("b", { text: num(nd, 2) }), document.createTextNode("."));
      }
      chart.redraw();
    }

    // the spectra
    const chart = mount(cont, (W) => {
      const mobile = W < 520, L = mobile ? 38 : 46, R = mobile ? 66 : 92, Tp = 16, B = 62, Hh = Math.round(Math.max(300, Math.min(430, W * 0.76)));
      const wl = SCENE.bands.map((b) => b.wl), x0 = 400, x1 = 2250, ymax = 0.7;
      const X = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R), Y = (v) => Hh - B - (v / ymax) * (Hh - B - Tp);
      const svg = S("svg", { viewBox: "0 0 " + W + " " + Hh, role: "presentation" }, cont);
      const c = cur();
      S("rect", { x: X(400), y: Tp, width: X(700) - X(400), height: Hh - B - Tp, style: "fill:var(--accent-wash)" }, svg);
      [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].forEach((t) => {
        S("path", { d: "M" + L + " " + Y(t) + "H" + (W - R), class: t === 0 ? "axis-l" : "grid-l" }, svg);
        T(num(t, 1), { x: L - 7, y: Y(t) + 4, "text-anchor": "end", class: "lbl" }, svg);
      });
      [500, 1000, 1500, 2000].forEach((t) => { S("path", { d: "M" + X(t) + " " + Y(0) + "v5", class: "axis-l" }, svg); T(String(t), { x: X(t), y: Y(0) + 19, "text-anchor": "middle", class: "lbl" }, svg); });
      T("nm", { x: W - R + 8, y: Y(0) + 19, class: "lbl" }, svg);
      [["visible", 400, 700], ["near infrared", 700, 1000], ["shortwave infrared", 1000, 2250]].forEach(([nm, a0, a1], k) => {
        const ya = Y(0) + 30 + (k === 1 ? 14 : 0);
        S("path", { d: "M" + (X(a0) + 2) + " " + ya + "H" + (X(a1) - 2), style: "stroke:var(--axis);stroke-width:1.2" }, svg);
        T(nm, { x: (X(a0) + X(a1)) / 2, y: ya + 14, "text-anchor": "middle", class: "lbl", style: "font-size:" + (mobile ? 10 : 11) + "px" }, svg);
      });
      T("reflectance", { x: L, y: Tp - 4, class: "lbl" }, svg);
      const order = pins.map((p, i) => i).sort((i, j) => (i === c) - (j === c));   // the chosen curve is drawn last, on top
      const ends = [];
      order.forEach((i) => {
        const p = pins[i], dim = c != null && c !== i;
        const d = p.spectrum.map((v, k) => (k ? "L" : "M") + X(wl[k]) + " " + Y(v)).join(" ");
        S("path", { d: d, style: "fill:none;stroke:" + p.color + ";stroke-width:" + (c === i ? 4.2 : 2.6) + ";stroke-linejoin:round;stroke-linecap:round;opacity:" + (dim ? 0.25 : 1) }, svg);
        p.spectrum.forEach((v, k) => {
          S("circle", { cx: X(wl[k]), cy: Y(v), r: c === i ? 4.6 : 3.4, style: "fill:" + p.color + ";stroke:var(--surface);stroke-width:1.6;opacity:" + (dim ? 0.3 : 1) }, svg);
          const bid = SCENE.bands[k].id, hit = S("circle", { cx: X(wl[k]), cy: Y(v), r: 10, class: "hit", tabindex: -1 }, svg);
          Tip.bind(hit, () => ({ head: num(v, 3), lines: [{ color: p.color, text: p.name + " \u00b7 " + bid + ", " + wl[k] + " nm" }, { text: BAND_INFO[bid][0] + ", native pixel " + BAND_INFO[bid][1] + " m" }] }));
          hit.addEventListener("click", () => { sticky = sticky === i ? null : i; sync(); });
        });
        ends.push({ i: i, y: Y(p.spectrum[p.spectrum.length - 1]), dim: dim });
      });
      ends.sort((a, b) => a.y - b.y);
      for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 14) ends[k].y = ends[k - 1].y + 14;
      ends.forEach((e) => {
        const p = pins[e.i], lab = S("g", { style: "cursor:pointer;opacity:" + (e.dim ? 0.45 : 1) }, svg);
        S("circle", { cx: X(2190) + 14, cy: e.y, r: 4, style: "fill:" + p.color }, lab);
        T(p.name, { x: X(2190) + 23, y: e.y + 4, class: "lbl lbl-strong", style: "font-size:" + (mobile ? 10.5 : 12) + "px" }, lab);
        lab.addEventListener("click", () => { sticky = sticky === e.i ? null : e.i; sync(); });
      });
    });
    const rows = SCENE.bands.map((b, k) => [b.id + " (" + b.wl + " nm)"].concat(pins.map((p) => num(p.spectrum[k], 3))));
    dataTable(cont, "spectra", ["Band"].concat(pins.map((p) => p.name)), rows, "Data table: reflectance of the six spots");
    sync();
  }

  function initBands() {
    const host = $("#bands-sheet"); if (!host || !SCENE) return;
    SCENE.bands.forEach((b) => {
      const info = BAND_INFO[b.id], vis = ["B02", "B03", "B04"].includes(b.id);
      const t = H("figure", { class: "band-tile" + (vis ? " vis" : ""), tabindex: 0 });
      t.appendChild(H("img", { src: "assets/img/scene/" + b.id + ".jpg", alt: "Band " + b.id + ", " + info[0].toLowerCase() + ", " + b.wl + " nanometres, of the scene", width: 480, height: 480, loading: "lazy" }));
      t.appendChild(H("div", { class: "bn", text: b.id + " \u00b7 " + SHORT_BAND[b.id] })); t.appendChild(H("div", { class: "wl", text: b.wl + " nm \u00b7 " + info[1] + " m pixels" }));
      Tip.bind(t, () => ({ head: b.id + " \u00b7 " + info[0], lines: [{ text: b.wl + " nm, native pixel " + info[1] + " m" }, { text: vis ? "one of the three visible bands in the RGB control" : "only the 13-band network sees this one" }] }));
      host.appendChild(t);
    });
    const comps = $("#comps");
    [["true", "True colour", "B4, B3 and B2 as red, green and blue."], ["nir", "Near-infrared false colour", "B8, B4, B3. Living crops glow red, water goes black."],
      ["swir", "Shortwave-infrared mix", "B12, B8A, B4. Damp and dry ground separate."], ["ndvi", "Vegetation index", "(B8 \u2212 B4) / (B8 + B4). Dense crops dark green, bare ground pale, water blue."]].forEach(([f, t, d]) => {
      const fg = H("figure"); fg.appendChild(H("img", { src: "assets/img/scene/comp_" + f + ".jpg", alt: t + " composite of the scene", width: 640, height: 640, loading: "lazy" }));
      const cap = H("figcaption"); cap.appendChild(H("b", { text: t })); cap.appendChild(document.createTextNode(d)); fg.appendChild(cap); comps.appendChild(fg);
    });
    const brk = $("#band-bracket");
    function bracket() {
      brk.replaceChildren();
      const hr = host.getBoundingClientRect(), tiles = host.querySelectorAll(".band-tile");
      if (hr.width < 300 || tiles.length < 4) return;
      const a = tiles[1].getBoundingClientRect(), z = tiles[3].getBoundingClientRect();
      if (Math.abs(a.top - z.top) > 4) return;              // the three tiles wrapped onto two rows: the green outline says it alone
      brk.setAttribute("viewBox", "0 0 " + hr.width + " 46"); brk.setAttribute("preserveAspectRatio", "xMinYMin meet");
      S("path", { d: Pen.bracket(a.left - hr.left, z.right - hr.left, 42, -10, 6), class: "pen-s", filter: "url(#wob)" }, brk);
      T("the three visible bands: the RGB control", { x: a.left - hr.left, y: 22, class: "pen-t" }, brk);
    }
    onResize(host, () => requestAnimationFrame(bracket));
    window.addEventListener("load", () => requestAnimationFrame(bracket));
    // the two big numbers
    const B = D.bands, vs = $("#vs-bands");
    $("#b13").textContent = num(B.m13) + "%"; $("#b3").textContent = num(B.m3) + "%";
    [["All 13 bands", B.m13, "ahead in " + B.ahead + " of " + B.seeds.length + " partitions"], ["Three visible bands", B.m3, "ahead in " + (B.seeds.length - B.ahead) + " of " + B.seeds.length + " partitions"]].forEach(([who, v, sub]) => {
      const d = H("div", { class: "vs" }); d.appendChild(H("div", { class: "who", text: who })); d.appendChild(H("div", { class: "big", text: num(v) + "%" }));
      d.appendChild(H("div", { class: "sub", text: "5-shot, dense ResNet-18 \u00b7 " + sub })); vs.appendChild(d);
    });
    dataTable($("#comps"), "bands", ["Partition", "13 bands (%)", "RGB (%)", "difference"], B.seeds.map((s, i) => ["p" + s, num(B.thirteen[i]), num(B.rgb[i]), signed(B.thirteen[i] - B.rgb[i])]).concat([["mean", num(B.m13), num(B.m3), signed(B.diff) + " [" + signed(B.lo) + ", " + signed(B.hi) + "]"]]), "Data table: 13 bands against RGB");
  }

  // ------------------------------------------------------------------------------------------ the paper's table 1
  function initTable1() {
    const host = $("#tbl1"); if (!host) return;
    const T1 = D.table1, tb = H("table", { class: "tb" }), thead = H("thead");
    const r1 = H("tr"); r1.appendChild(H("th", { rowspan: 2, text: "Score at 30% of the parameters" }));
    T1.columns.forEach((c) => r1.appendChild(H("th", { colspan: 2, text: c, style: "text-align:center" })));
    const r2 = H("tr"); T1.columns.forEach(() => { r2.appendChild(H("th", { text: "P" })); r2.appendChild(H("th", { text: "PMP" })); });
    thead.appendChild(r1); thead.appendChild(r2); tb.appendChild(thead);
    const body = H("tbody"), dr = H("tr"); dr.appendChild(H("th", { scope: "row", text: T1.dense.label }));
    T1.columns.forEach((c) => dr.appendChild(H("td", { colspan: 2, text: num(T1.dense.cells[c].m) + "±" + num(T1.dense.cells[c].sd), style: "text-align:center" })));
    body.appendChild(dr);
    T1.rows.forEach((r) => {
      const tr = H("tr"); tr.appendChild(H("th", { scope: "row", text: r.label }));
      T1.columns.forEach((c) => ["P", "PMP"].forEach((s) => {
        const e = r.cells[c + "|" + s]; const td = H("td", { text: e ? num(e.m) + "±" + num(e.sd) : "–" });
        if (e && e.sig > 0) td.className = "up"; if (e && e.sig < 0) td.className = "dn"; tr.appendChild(td);
      }));
      body.appendChild(tr);
    });
    tb.appendChild(body); host.appendChild(tb);
    host.appendChild(H("p", { class: "foot", text: "Mean ± standard deviation over class partitions (ten for RESISC45 and EuroSAT-MS, five for the two leaf sets). ▲ or ▼: the paired difference to random choice has a 95% interval entirely above or below zero. A dash is a score that was not run. P: prune once and recover. PMP: prune, meta-train, prune." }));
  }

  K.initFigs = function () { initBudget(); initPicks(); initPartitions(); initEpisode(); initPmp(); initLatency(); initSquares(); initScene(); initBands(); initTable1(); };
})();
