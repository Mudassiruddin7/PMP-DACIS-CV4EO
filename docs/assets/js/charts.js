/* Statistical charts, drawn as plain SVG from window.SITE_DATA. No libraries. */
(function () {
  "use strict";
  const D = window.SITE_DATA;
  const NS = "http://www.w3.org/2000/svg";

  // ------------------------------------------------------------------------------------------ helpers
  const $ = (s, r) => (r || document).querySelector(s);
  function S(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function T(text, attrs, parent) { const e = S("text", attrs, parent); e.textContent = text; return e; }
  function H(tag, attrs, kids) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) { if (k === "text") e.textContent = attrs[k]; else if (k === "class") e.className = attrs[k]; else e.setAttribute(k, attrs[k]); }
    (kids || []).forEach((c) => e.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return e;
  }
  const num = (x, d) => { d = d == null ? 1 : d; const v = Math.abs(x) < 0.5 * Math.pow(10, -d) ? 0 : x; return v.toFixed(d).replace("-", "−"); };
  const signed = (x, d) => { d = d == null ? 1 : d; const s = num(x, d); return x > 0 && s !== num(0, d) ? "+" + s : s; };
  const sc = (lo, hi, a, b) => (v) => a + ((v - lo) / (hi - lo)) * (b - a);
  function niceTicks(lo, hi, n) {
    const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), nm = raw / mag;
    const step = (nm < 1.5 ? 1 : nm < 3 ? 2 : nm < 7 ? 5 : 10) * mag, t = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) t.push(Math.round(v / step) * step);
    return t;
  }
  const SC_LABEL = { random: "Random", l1: "Filter L1", bn_gamma: "BN scale", taylor: "Taylor", G: "Gradient", V: "Variance", D: "Fisher", dacis: "DACIS" };
  const SC_LONG = { random: "Random choice", l1: "Filter L1", bn_gamma: "BatchNorm scale", taylor: "Taylor", G: "Gradient (DACIS part)", V: "Variance (DACIS part)", D: "Fisher (DACIS part)", dacis: "DACIS" };
  const SC_VAR = { ink: "var(--ink)", random: "var(--c-random)", l1: "var(--c-l1)", bn_gamma: "var(--c-bn)", taylor: "var(--c-taylor)", G: "var(--c-dacis)", V: "var(--c-dacis)", D: "var(--c-dacis)", dacis: "var(--c-dacis)" };
  const SC_SHAPE = { random: "circle", l1: "square", bn_gamma: "diamond", taylor: "circle", dacis: "tri", G: "tri", V: "tri", D: "tri" };
  const DS_NAME = { resisc45: "RESISC45", eurosat13: "EuroSAT-MS", plantvillage: "PlantVillage", plantdoc: "PlantDoc" };

  /** a marker centred on (x,y): a ring underneath (surface colour) and the shape on top */
  function marker(parent, shape, x, y, r, score, opts) {
    opts = opts || {};
    const g = S("g", { transform: "translate(" + x + " " + y + ")" }, parent);
    const path = (rr) => {
      if (shape === "square") return "M" + -rr + " " + -rr + "h" + 2 * rr + "v" + 2 * rr + "h" + -2 * rr + "z";
      if (shape === "diamond") return "M0 " + -rr * 1.3 + "L" + rr * 1.3 + " 0L0 " + rr * 1.3 + "L" + -rr * 1.3 + " 0z";
      if (shape === "tri") return "M0 " + -rr * 1.25 + "L" + rr * 1.15 + " " + rr * 0.95 + "L" + -rr * 1.15 + " " + rr * 0.95 + "z";
      return "M" + -rr + " 0a" + rr + " " + rr + " 0 1 0 " + 2 * rr + " 0a" + rr + " " + rr + " 0 1 0 " + -2 * rr + " 0z";
    };
    if (!opts.hollow && !opts.noRing) S("path", { d: path(r + 2), style: "fill:var(--surface)" }, g);
    const m = S("path", { d: path(r) }, g);
    if (opts.hollow) m.setAttribute("style", "fill:var(--surface);stroke:" + SC_VAR[score] + ";stroke-width:2");
    else m.setAttribute("style", "fill:" + SC_VAR[score] + ";stroke:none" + (opts.opacity ? ";opacity:" + opts.opacity : ""));
    return g;
  }

  // ------------------------------------------------------------------------------------------ tooltip
  const Tip = (function () {
    const box = document.getElementById("tip");
    function show(c, x, y) {
      box.replaceChildren();
      if (c.head) box.appendChild(H("b", { text: c.head }));
      (c.lines || []).forEach((l) => {
        const s = H("span");
        if (l.color) { const i = H("i"); i.style.background = l.color; s.appendChild(i); }
        s.appendChild(document.createTextNode(l.text));
        box.appendChild(s);
      });
      box.classList.add("on");
      const w = box.offsetWidth, h = box.offsetHeight;
      let left = x + 16, top = y - h - 14;
      if (left + w > window.innerWidth - 8) left = x - w - 16;
      if (left < 8) left = 8;
      if (top < 8) top = y + 20;
      box.style.left = left + "px"; box.style.top = top + "px";
    }
    function hide() { box.classList.remove("on"); }
    function bind(el, fn) {
      const at = (e) => show(fn(), e.clientX, e.clientY);
      el.addEventListener("pointerenter", at);
      el.addEventListener("pointermove", at);
      el.addEventListener("pointerleave", hide);
      el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); show(fn(), r.left + r.width / 2, r.top); });
      el.addEventListener("blur", hide);
    }
    return { show: show, hide: hide, bind: bind };
  })();

  // ------------------------------------------------------------------------------------------ plumbing
  function mount(container, draw) {
    let lastW = 0, raf = 0;
    function go() {
      const w = Math.round(container.getBoundingClientRect().width);
      if (w < 10 || w === lastW) return;
      lastW = w; container.replaceChildren(); draw(w, container);
      // hand-written notes that would run past the chart are dropped instead of spilling out of it
      const cr = container.getBoundingClientRect();
      container.querySelectorAll(".pen-t").forEach((t) => { const r = t.getBoundingClientRect(); if (r.left < cr.left - 1 || r.right > cr.right + 1) t.remove(); });
    }
    new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(go); }).observe(container);
    go();
    return { redraw: () => { lastW = 0; go(); } };
  }
  /** run fn(width) now and whenever the element's width changes; unlike mount() it leaves the element's content alone */
  function onResize(el, fn) {
    let w0 = 0, raf = 0;
    new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => { const w = Math.round(el.getBoundingClientRect().width); if (w !== w0) { w0 = w; fn(w); } });
    }).observe(el);
    const w = Math.round(el.getBoundingClientRect().width); w0 = w; fn(w);
  }
  function seg(host, opts, cur, on) {
    host.replaceChildren();
    opts.forEach((o) => {
      const b = H("button", { type: "button", "aria-pressed": String(o.v === cur), text: o.t });
      b.addEventListener("click", () => {
        host.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true"); on(o.v);
      });
      host.appendChild(b);
    });
  }
  function legend(host, items) {
    host.replaceChildren();
    items.forEach((it) => {
      const sp = H("span");
      const svg = S("svg", { viewBox: "-8 -8 16 16" }, sp);
      if (it.kind === "mark") marker(svg, it.shape, 0, 0, 4.2, it.score, { noRing: true, hollow: it.hollow });
      else if (it.kind === "tick") S("path", { d: "M0 -7v14", style: "stroke:var(--ink);stroke-width:3.5" }, svg);
      else if (it.kind === "dash") S("path", { d: "M0 -7v14", style: "stroke:" + (it.color || "var(--ink)") + ";stroke-width:1.6;stroke-dasharray:3 2.5" }, svg);
      else if (it.kind === "line") S("path", { d: "M-8 0H8", style: "stroke:" + it.color + ";stroke-width:3;stroke-linecap:round" }, svg);
      sp.appendChild(document.createTextNode(it.text));
      host.appendChild(sp);
    });
  }
  function dataTable(anchor, key, head, rows, caption) {
    const old = anchor.parentElement.querySelector('details.data[data-key="' + key + '"]');
    if (old) old.remove();
    const det = H("details", { class: "data", "data-key": key });
    det.appendChild(H("summary", { text: caption || "Data table" }));
    const wrap = H("div", { class: "tablewrap" });
    const tb = H("table", { class: "tb" });
    const thead = H("thead"), trh = H("tr");
    head.forEach((h) => trh.appendChild(H("th", { scope: "col", text: h })));
    thead.appendChild(trh); tb.appendChild(thead);
    const tbody = H("tbody");
    rows.forEach((r) => { const tr = H("tr"); r.forEach((c, i) => tr.appendChild(H(i ? "td" : "th", i ? { text: c } : { scope: "row", text: c }))); tbody.appendChild(tr); });
    tb.appendChild(tbody); wrap.appendChild(tb); det.appendChild(wrap);
    anchor.insertAdjacentElement("afterend", det);
  }
  function axisBottom(g, x, lo, hi, y, ticks, fmt, opts) {
    opts = opts || {};
    ticks.forEach((t) => {
      S("path", { d: "M" + x(t) + " " + (opts.top || 0) + "V" + y, class: t === 0 && opts.zero ? "zero-l" : "grid-l" }, g);
      const tx = T(fmt ? fmt(t) : String(t), { x: x(t), y: y + 17, "text-anchor": "middle", class: "lbl" }, g);
    });
  }
  const A = { S: S, T: T, H: H, $: $, num: num, signed: signed, mount: mount, onResize: onResize, seg: seg, legend: legend, dataTable: dataTable, Tip: Tip, marker: marker, niceTicks: niceTicks, sc: sc,
    SC_LABEL: SC_LABEL, SC_VAR: SC_VAR, SC_SHAPE: SC_SHAPE, DS_NAME: DS_NAME, axisBottom: axisBottom };
  window.ChartKit = A;

  const scoreLegend = (ids) => ids.map((s) => ({ kind: "mark", shape: SC_SHAPE[s], score: s, text: SC_LONG[s] }));
  const penPath = (parent, d, extra) => S("path", Object.assign({ d: d, class: "pen-s", filter: "url(#wob)" }, extra || {}), parent);

  // ------------------------------------------------------------------------------------------ strip: eight scores, one dot per partition
  function initStrip() {
    const cont = $("#chart-strip"); if (!cont) return;
    let ds = "resisc45";
    legend($("#leg-strip"), [{ kind: "mark", shape: "circle", score: "random", text: "one class partition" }, { kind: "tick", text: "mean of the partitions" },
      { kind: "dash", text: "unpruned ResNet-18" }, { kind: "dash", color: "var(--c-random)", text: "random choice" }]);
    const m = mount(cont, (W) => {
      const d0 = D.strip[ds], rows = d0.rows, mobile = W < 640;
      const L = mobile ? 86 : 120, R = mobile ? 10 : 128, Tp = 70, rowH = mobile ? 34 : 40, B = 38;
      const H_ = Tp + rows.length * rowH + B;
      const vals = rows.flatMap((r) => r.vals).concat([d0.dense.m]);
      const lo = Math.floor(Math.min.apply(null, vals) - 1), hi = Math.ceil(Math.max.apply(null, vals) + 1);
      const x = sc(lo, hi, L, W - R);
      const svg = S("svg", { viewBox: "0 0 " + W + " " + H_, role: "presentation" }, cont);
      const g = S("g", { class: "ax" }, svg);
      axisBottom(g, x, lo, hi, Tp + rows.length * rowH, niceTicks(lo, hi, mobile ? 4 : 7), (t) => t + "%", { top: Tp - 6 });
      const rnd = rows.find((r) => r.id === "random");
      // random's mean and the unpruned line
      const lineTop = Tp - 8, lineBot = Tp + rows.length * rowH;
      S("path", { d: "M" + x(rnd.m) + " " + lineTop + "V" + lineBot, style: "stroke:var(--c-random);stroke-width:1.6;stroke-dasharray:4 3;fill:none" }, svg);
      S("path", { d: "M" + x(d0.dense.m) + " " + lineTop + "V" + lineBot, style: "stroke:var(--ink);stroke-width:1.6;stroke-dasharray:4 3;fill:none" }, svg);
      T("random", { x: x(rnd.m), y: Tp - 14, "text-anchor": "middle", class: "lbl" }, svg);
      T("unpruned " + num(d0.dense.m), { x: x(d0.dense.m), y: Tp - 14, "text-anchor": "middle", class: "lbl lbl-strong" }, svg);
      rows.forEach((r, i) => {
        const cy = Tp + i * rowH + rowH / 2;
        if (i) S("path", { d: "M" + 8 + " " + (cy - rowH / 2) + "H" + (W - 4), class: "grid-l", style: "opacity:.6" }, svg);
        S("circle", { cx: 12, cy: cy, r: 5.5, style: "fill:" + SC_VAR[r.id] + (["G", "V", "D"].includes(r.id) ? ";opacity:.55" : "") }, svg);
        T(SC_LABEL[r.id], { x: 26, y: cy + 5, class: "lbl-d" }, svg);
        const rg = Pen.rng(11 + i * 7);
        r.vals.forEach((v, k) => {
          const dy = (rg() - 0.5) * (rowH - 18);
          const dot = S("circle", { cx: x(v), cy: cy + dy, r: 4, style: "fill:" + SC_VAR[r.id] + ";opacity:" + (["G", "V", "D"].includes(r.id) ? 0.5 : 0.7) }, svg);
          S("circle", { cx: x(v), cy: cy + dy, r: 4, style: "fill:none;stroke:var(--surface);stroke-width:1.2;opacity:.9" }, svg);
          const hit = S("circle", { cx: x(v), cy: cy + dy, r: 12, class: "hit", tabindex: 0, role: "img", "aria-label": SC_LONG[r.id] + ", partition " + r.seeds[k] + ", " + num(v) + "%" }, svg);
          Tip.bind(hit, () => ({ head: num(v, 1) + "%", lines: [{ color: SC_VAR[r.id], text: SC_LONG[r.id] + " · partition " + r.seeds[k] }, { text: "mean " + num(r.m) + " · sd " + num(r.sd) }] }));
        });
        S("path", { d: "M" + x(r.m) + " " + (cy - 12) + "V" + (cy + 12), style: "stroke:var(--surface);stroke-width:7;stroke-linecap:round" }, svg);
        S("path", { d: "M" + x(r.m) + " " + (cy - 12) + "V" + (cy + 12), style: "stroke:" + SC_VAR[r.id] + ";stroke-width:4;stroke-linecap:round" }, svg);
        if (!mobile) T(num(r.m) + " ±" + num(r.sd), { x: W - R + 14, y: cy + 4, class: "lbl lbl-strong" }, svg);
      });
      if (!mobile) T("mean ± sd", { x: W - R + 14, y: Tp - 14, class: "lbl" }, svg);
      // the hand-drawn spread
      const means = rows.map((r) => r.m), mn = Math.min.apply(null, means), mx = Math.max.apply(null, means);
      const bx1 = x(mn), bx2 = x(mx), spreadTxt = num(mx - mn) + (mobile ? " points apart" : " points, best to worst"), half = spreadTxt.length * 3.9 + 4;
      const cxm = Math.min(Math.max((bx1 + bx2) / 2, half), W - half);
      penPath(svg, Pen.bracket(bx1, bx2, 34, 9, 4));
      T(spreadTxt, { x: cxm, y: 25, "text-anchor": "middle", class: "pen-t" }, svg);
    });
    const read = $("#read-strip");
    function words() {
      const d0 = D.strip[ds], means = d0.rows.map((r) => r.m);
      const dv = d0.dense.vals;
      read.replaceChildren(
        document.createTextNode("Best and worst of the eight are "), H("b", { text: num(Math.max.apply(null, means) - Math.min.apply(null, means)) + " points" }),
        document.createTextNode(" apart on " + DS_NAME[ds] + ". Single partitions of the unpruned network run from "), H("b", { text: num(Math.min.apply(null, dv)) + "%" }),
        document.createTextNode(" to "), H("b", { text: num(Math.max.apply(null, dv)) + "%" }), document.createTextNode("."));
      const d0r = D.strip[ds].rows;
      dataTable(cont, "strip", ["Score"].concat(d0r[0].seeds.map((s) => "p" + s), ["mean", "sd"]),
        d0r.map((r) => [SC_LONG[r.id]].concat(r.vals.map((v) => num(v)), [num(r.m), num(r.sd)])), "Data table: accuracy per partition");
    }
    seg($("#seg-strip"), Object.keys(DS_NAME).map((k) => ({ v: k, t: DS_NAME[k] })), ds, (v) => { ds = v; m.redraw(); words(); });
    words();
  }

  // ------------------------------------------------------------------------------------------ forest: gain over random
  function initForest() {
    const cont = $("#chart-forest"); if (!cont) return;
    let sid = D.forest[0].id;
    const order = ["l1", "bn_gamma", "taylor", "dacis"];
    legend($("#leg-forest"), scoreLegend(order).concat([{ kind: "line", color: "var(--ink)", text: "random choice = zero" }]));
    const m = mount(cont, (W) => {
      const f = D.forest.find((x) => x.id === sid), dss = ["resisc45", "eurosat13"], stacked = W < 760;
      let lo = 0, hi = 0;
      dss.forEach((d) => order.forEach((s) => { lo = Math.min(lo, f.ds[d][s].lo); hi = Math.max(hi, f.ds[d][s].hi); }));
      lo = Math.floor((lo - 0.8) / 2) * 2; hi = Math.ceil((hi + 1.2) / 2) * 2;
      const gap = 36, pw = stacked ? W : (W - gap) / 2, rowH = 54, Tp = 44, B = 36, ph = Tp + order.length * rowH + B;
      const H_ = stacked ? ph * 2 + 10 : ph;
      const svg = S("svg", { viewBox: "0 0 " + W + " " + H_, role: "presentation" }, cont);
      dss.forEach((d, pi) => {
        const ox = stacked ? 0 : pi * (pw + gap), oy = stacked ? pi * (ph + 10) : 0;
        const g = S("g", { transform: "translate(" + ox + " " + oy + ")" }, svg);
        const L = 96, R = 22, x = sc(lo, hi, L, pw - R);
        T(DS_NAME[d], { x: 0, y: 18, class: "lbl-d", style: "font-size:17px;font-weight:700" }, g);
        const ax = S("g", { class: "ax" }, g);
        axisBottom(ax, x, lo, hi, Tp + order.length * rowH, niceTicks(lo, hi, stacked ? 5 : 6), (t) => (t > 0 ? "+" : "") + t, { top: Tp - 4, zero: true });
        T(pw < 420 ? "points against random choice" : "points of 5-shot accuracy against random choice", { x: (L + pw - R) / 2, y: Tp + order.length * rowH + 33, "text-anchor": "middle", class: "lbl" }, g);
        order.forEach((s, i) => {
          const e = f.ds[d][s], cy = Tp + i * rowH + rowH / 2;
          T(SC_LABEL[s], { x: 0, y: cy + 5, class: "lbl-d" }, g);
          S("path", { d: "M" + x(e.lo) + " " + cy + "H" + x(e.hi), style: "stroke:" + SC_VAR[s] + ";stroke-width:4;stroke-linecap:round" }, g);
          marker(g, SC_SHAPE[s], x(e.m), cy, 6.5, s);
          T(signed(e.m), { x: x(e.m), y: cy - 14, "text-anchor": "middle", class: "lbl lbl-strong" }, g);
          const hit = S("rect", { x: x(Math.min(e.lo, e.m)) - 12, y: cy - 22, width: Math.max(24, x(e.hi) - x(e.lo) + 24), height: 44, class: "hit", tabindex: 0, role: "img",
            "aria-label": SC_LONG[s] + " minus random on " + DS_NAME[d] + ": " + signed(e.m) + " points, interval " + signed(e.lo) + " to " + signed(e.hi) }, g);
          Tip.bind(hit, () => ({ head: signed(e.m) + " points", lines: [{ color: SC_VAR[s], text: SC_LONG[s] + " minus random" }, { text: "95% interval " + signed(e.lo) + " to " + signed(e.hi) }, { text: "ahead in " + e.ahead + " of " + e.n + " partitions" }] }));
        });
        // pen: circle the Taylor row when it clears zero, and say so about DACIS when it does not
        if (pi === 0) {
          const tay = f.ds[d].taylor, ti = order.indexOf("taylor"), tcy = Tp + ti * rowH + rowH / 2;
          if (tay.lo > 0) {
            const cx = (x(tay.lo) + x(tay.hi)) / 2, rx = Math.max(34, (x(tay.hi) - x(tay.lo)) / 2 + 22);
            penPath(g, Pen.circle(cx, tcy - 2, rx, 26, 5));
            const room = x(0) - L;
            if (room > 205) T("only one clear of zero", { x: x(0) - 10, y: tcy + 7, "text-anchor": "end", class: "pen-t" }, g);
          }
          const dac = f.ds[d].dacis, di = order.indexOf("dacis"), dcy = Tp + di * rowH + rowH / 2;
          if (dac.lo <= 0 && dac.hi >= 0 && pw - R - x(dac.hi) > 205) T("can't tell it from dice", { x: x(dac.hi) + 14, y: dcy + 7, class: "pen-t" }, g);
        }
      });
    });
    seg($("#seg-forest"), D.forest.map((f) => ({ v: f.id, t: f.label.replace(" of the weights", "") })), sid, (v) => { sid = v; m.redraw(); table(); });
    function table() {
      const f = D.forest.find((x) => x.id === sid), rows = [];
      ["resisc45", "eurosat13"].forEach((d) => order.forEach((s) => { const e = f.ds[d][s]; rows.push([DS_NAME[d], SC_LONG[s], signed(e.m), signed(e.lo) + " to " + signed(e.hi), e.ahead + " of " + e.n]); }));
      dataTable(cont, "forest", ["Test set", "Score minus random", "gain (points)", "95% interval", "partitions ahead"], rows);
    }
    table();
  }

  // ------------------------------------------------------------------------------------------ lead over random as the cut gets harsher
  function initLead() {
    const cont = $("#chart-lead"); if (!cont) return;
    let ds = "resisc45";
    const sets = [{ k: "r18", title: "ResNet-18", scores: ["l1", "dacis", "taylor"] }, { k: "mbv2", title: "MobileNetV2", scores: ["l1", "bn_gamma", "dacis", "taylor"] }];
    legend($("#leg-lead"), scoreLegend(["l1", "bn_gamma", "taylor", "dacis"]).concat([{ kind: "line", color: "var(--ink)", text: "random choice = zero" }]));
    const m = mount(cont, (W) => {
      const stacked = W < 760, gap = 40, pw = stacked ? W : (W - gap) / 2, Tp = 36, B = 44, ph = stacked ? 300 : 330;
      const H_ = stacked ? 2 * ph + 10 : ph;
      let lo = 0, hi = 0;
      sets.forEach((s) => { const L = D.lead[ds][s.k]; s.scores.forEach((id) => L.scores[id].forEach((e) => { lo = Math.min(lo, e.lo); hi = Math.max(hi, e.hi); })); });
      lo = Math.floor((lo - 0.5) / 2) * 2; hi = Math.ceil((hi + 0.5) / 2) * 2;
      const svg = S("svg", { viewBox: "0 0 " + W + " " + H_, role: "presentation" }, cont);
      sets.forEach((s, pi) => {
        const L = D.lead[ds][s.k], ox = stacked ? 0 : pi * (pw + gap), oy = stacked ? pi * (ph + 10) : 0;
        const g = S("g", { transform: "translate(" + ox + " " + oy + ")" }, svg);
        const Lm = 40, Rm = 74, xx = sc(66, 6, Lm, pw - Rm), yy = sc(lo, hi, ph - B, Tp);
        T(s.title, { x: 0, y: 16, class: "lbl-d", style: "font-size:17px;font-weight:700" }, g);
        niceTicks(lo, hi, 6).forEach((t) => {
          S("path", { d: "M" + Lm + " " + yy(t) + "H" + (pw - Rm), class: t === 0 ? "zero-l" : "grid-l" }, g);
          T((t > 0 ? "+" : "") + t, { x: Lm - 8, y: yy(t) + 4, "text-anchor": "end", class: "lbl" }, g);
        });
        L.budgets.forEach((b) => { S("path", { d: "M" + xx(b) + " " + (ph - B) + "v5", class: "axis-l" }, g); T(b + "%", { x: xx(b), y: ph - B + 20, "text-anchor": "middle", class: "lbl" }, g); });
        T(pw < 420 ? "share kept → harsher cut" : "share of the parameters kept → harsher cut", { x: (Lm + pw - Rm) / 2, y: ph - 6, "text-anchor": "middle", class: "lbl" }, g);
        s.scores.forEach((id) => {
          const pts = L.scores[id];
          const dpath = pts.map((e, i) => (i ? "L" : "M") + xx(L.budgets[i]) + " " + yy(e.m)).join(" ");
          S("path", { d: dpath, style: "fill:none;stroke:" + SC_VAR[id] + ";stroke-width:2.4;stroke-linejoin:round" }, g);
        });
        const ends = [];
        s.scores.forEach((id) => {
          const pts = L.scores[id];
          pts.forEach((e, i) => {
            const cx = xx(L.budgets[i]), cy = yy(e.m);
            if (id === "taylor") S("path", { d: "M" + cx + " " + yy(e.lo) + "V" + yy(e.hi), style: "stroke:" + SC_VAR[id] + ";stroke-width:2;opacity:.45" }, g);
            marker(g, SC_SHAPE[id], cx, cy, 5, id);
            const hit = S("circle", { cx: cx, cy: cy, r: 14, class: "hit", tabindex: 0, role: "img", "aria-label": SC_LONG[id] + " at " + L.budgets[i] + "% kept: " + signed(e.m) + " points" }, g);
            Tip.bind(hit, () => ({ head: signed(e.m) + " points", lines: [{ color: SC_VAR[id], text: SC_LONG[id] + " minus random, " + L.budgets[i] + "% kept" }, { text: "95% interval " + signed(e.lo) + " to " + signed(e.hi) }, { text: e.n + " partitions" }] }));
          });
          ends.push({ id: id, y: yy(pts[pts.length - 1].m) });
        });
        // end labels: nudge apart where two lines finish close together
        ends.sort((a, b) => a.y - b.y);
        for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
        ends.forEach((e) => T(SC_LABEL[e.id], { x: xx(L.budgets[L.budgets.length - 1]) + 12, y: e.y + 4, class: "lbl-d", style: "font-size:13px" }, g));
        if (ds === "resisc45") {
          if (s.k === "mbv2") {
            const bn = L.scores.bn_gamma[2], tay = L.scores.taylor[2];
            const bx = xx(25), by = yy(bn.m);
            T(signed(bn.m) + " for BN scale", { x: xx(40) - 6, y: yy(bn.m) + 32, "text-anchor": "middle", class: "pen-t" }, g);
            const ar = Pen.arrow(xx(40) + 40, yy(bn.m) + 26, bx - 14, by + 8, { bend: -0.25, seed: 3, head: 9 });
            penPath(g, ar.d); penPath(g, ar.head);
          } else {
            const tay = L.scores.taylor[3];
            T("the lead grows", { x: xx(30), y: yy(tay.m) - 26, "text-anchor": "middle", class: "pen-t" }, g);
            const ar = Pen.arrow(xx(30) + 36, yy(tay.m) - 22, xx(10) - 10, yy(tay.m) - 8, { bend: 0.2, seed: 8, head: 9 });
            penPath(g, ar.d); penPath(g, ar.head);
          }
        }
      });
    });
    function table() {
      const rows = [];
      sets.forEach((s) => { const L = D.lead[ds][s.k]; s.scores.forEach((id) => L.scores[id].forEach((e, i) => rows.push([s.title, L.budgets[i] + "%", SC_LONG[id], signed(e.m), signed(e.lo) + " to " + signed(e.hi), String(e.n)]))); });
      dataTable(cont, "lead", ["Network", "kept", "Score minus random", "gain (points)", "95% interval", "partitions"], rows);
    }
    seg($("#seg-lead"), [{ v: "resisc45", t: "RESISC45" }, { v: "eurosat13", t: "EuroSAT-MS" }], ds, (v) => { ds = v; m.redraw(); table(); });
    table();
  }

  // ------------------------------------------------------------------------------------------ race: taylor against random, partition by partition
  function initRace() {
    const cont = $("#chart-race"); if (!cont) return;
    legend($("#leg-race"), [{ kind: "mark", shape: "circle", score: "random", hollow: true, text: "random choice" }, { kind: "mark", shape: "circle", score: "taylor", text: "Taylor" }]);
    const panels = [{ id: "r18_30", title: "ResNet-18, 30% kept" }, { id: "mb_40", title: "MobileNetV2, 40% kept" }, { id: "mb_25", title: "MobileNetV2, 25% kept" }];
    mount(cont, (W) => {
      const cols = W >= 900 ? 3 : W >= 600 ? 2 : 1, gap = 34, pw = (W - gap * (cols - 1)) / cols, rowH = cols === 1 ? 26 : 25, Tp = 62, B = 34, ph = Tp + 10 * rowH + B;
      const rowsN = Math.ceil(panels.length / cols), H_ = rowsN * ph + (rowsN - 1) * 16;
      const svg = S("svg", { viewBox: "0 0 " + W + " " + H_, role: "presentation" }, cont);
      panels.forEach((p, pi) => {
        const r = D.race[p.id], ox = (pi % cols) * (pw + gap), oy = Math.floor(pi / cols) * (ph + 16);
        const g = S("g", { transform: "translate(" + ox + " " + oy + ")" }, svg);
        const idx = r.seeds.map((s, i) => i).sort((a, b) => r.random[a] - r.random[b]);
        const all = r.taylor.concat(r.random), lo = Math.floor(Math.min.apply(null, all) - 1), hi = Math.ceil(Math.max.apply(null, all) + 1);
        const L = 28, Rm = 8, x = sc(lo, hi, L, pw - Rm);
        const gaps = r.taylor.map((t, i) => t - r.random[i]), mg = gaps.reduce((a, b) => a + b, 0) / gaps.length;
        T(p.title, { x: 0, y: 16, class: "lbl-d", style: "font-size:16px;font-weight:700" }, g);
        T("Taylor ahead in " + r.ahead + " of " + r.seeds.length + " · mean gap " + signed(mg), { x: 0, y: 36, class: "lbl" }, g);
        const ax = S("g", { class: "ax" }, g);
        axisBottom(ax, x, lo, hi, Tp + 10 * rowH, niceTicks(lo, hi, pw < 300 ? 3 : 4), (t) => t + "%", { top: Tp - 2 });
        idx.forEach((k, row) => {
          const cy = Tp + row * rowH + rowH / 2, a = r.random[k], b = r.taylor[k];
          T(String(r.seeds[k]), { x: L - 8, y: cy + 4, "text-anchor": "end", class: "lbl", style: "font-size:10.5px" }, g);
          S("path", { d: "M" + x(a) + " " + cy + "H" + x(b), style: "stroke:var(--c-taylor);stroke-width:3;stroke-linecap:round;opacity:.55" }, g);
          marker(g, "circle", x(a), cy, 5, "random", { hollow: true });
          marker(g, "circle", x(b), cy, 5, "taylor");
          const hit = S("rect", { x: Math.min(x(a), x(b)) - 10, y: cy - rowH / 2, width: Math.abs(x(b) - x(a)) + 20, height: rowH, class: "hit", tabindex: 0, role: "img",
            "aria-label": "Partition " + r.seeds[k] + ": random " + num(a) + "%, Taylor " + num(b) + "%" }, g);
          Tip.bind(hit, () => ({ head: signed(b - a) + " points", lines: [{ color: "var(--c-taylor)", text: "Taylor " + num(b) + "%" }, { color: "var(--c-random)", text: "Random " + num(a) + "%" }, { text: "partition " + r.seeds[k] }] }));
        });
      });
    });
    const rows = [];
    panels.forEach((p) => { const r = D.race[p.id]; r.seeds.forEach((s, i) => rows.push([p.title, "p" + s, num(r.random[i]), num(r.taylor[i]), signed(r.taylor[i] - r.random[i])])); });
    dataTable(cont, "race", ["Setting", "partition", "random (%)", "Taylor (%)", "gain (points)"], rows);
  }

  // ------------------------------------------------------------------------------------------ schedules against plain prune-and-recover
  function initSched() {
    const cont = $("#chart-sched"); if (!cont) return;
    legend($("#leg-sched"), [{ kind: "mark", shape: "tri", score: "dacis", text: "DACIS at 30% of the parameters" }, { kind: "line", color: "var(--ink)", text: "plain prune-and-recover (P) = zero" }]);
    const names = { PMP: "prune, meta-train, prune", MP: "meta-train, then prune", PGP: "prune, probe, prune", PM: "prune, then meta-train" };
    const subs = { PMP: "the PMP schedule", MP: "", PGP: "the probe updates no weights", PM: "" };
    mount(cont, (W) => {
      const mobile = W < 640, L = mobile ? 150 : 230, R = 20, rowH = 46, Tp = 34, grpGap = 40, B = 34;
      const dss = ["resisc45", "eurosat13"], rowsPer = 4;
      const H_ = Tp + dss.length * (rowsPer * rowH + 50) + B;
      let lo = 0, hi = 0;
      dss.forEach((d) => D.schedules[d].forEach((e) => { lo = Math.min(lo, e.lo); hi = Math.max(hi, e.hi); }));
      lo = Math.floor((lo - 0.3)); hi = Math.ceil(hi + 0.7);
      const x = sc(lo, hi, L, W - R), svg = S("svg", { viewBox: "0 0 " + W + " " + H_, role: "presentation" }, cont);
      const ax = S("g", { class: "ax" }, svg);
      axisBottom(ax, x, lo, hi, H_ - B, niceTicks(lo, hi, 6), (t) => (t > 0 ? "+" : "") + t, { top: Tp - 6, zero: true });
      T(mobile ? "points vs plain prune-and-recover" : "points of 5-shot accuracy against plain prune-and-recover", { x: (L + W - R) / 2, y: H_ - 4, "text-anchor": "middle", class: "lbl" }, svg);
      dss.forEach((d, di) => {
        const y0 = Tp + di * (rowsPer * rowH + 50);
        T(DS_NAME[d], { x: 0, y: y0 + 4, class: "lbl-d", style: "font-size:17px;font-weight:700" }, svg);
        D.schedules[d].forEach((e, i) => {
          const cy = y0 + 22 + i * rowH + rowH / 2;
          T(names[e.id], { x: 0, y: cy - (subs[e.id] && !mobile ? 2 : -4), class: "lbl-d", style: "font-size:" + (mobile ? 13 : 14.5) + "px" }, svg);
          if (subs[e.id] && !mobile) T(subs[e.id], { x: 0, y: cy + 14, class: "lbl", style: "font-size:11px;fill:var(--ink-3)" }, svg);
          S("path", { d: "M" + x(e.lo) + " " + cy + "H" + x(e.hi), style: "stroke:var(--c-dacis);stroke-width:4;stroke-linecap:round" }, svg);
          marker(svg, "tri", x(e.m), cy, 6.5, "dacis");
          T(signed(e.m), { x: x(e.m), y: cy - 14, "text-anchor": "middle", class: "lbl lbl-strong" }, svg);
          const hit = S("rect", { x: x(e.lo) - 12, y: cy - 20, width: x(e.hi) - x(e.lo) + 24, height: 40, class: "hit", tabindex: 0, role: "img", "aria-label": names[e.id] + " on " + DS_NAME[d] + ": " + signed(e.m) + " points" }, svg);
          Tip.bind(hit, () => ({ head: signed(e.m) + " points", lines: [{ color: "var(--c-dacis)", text: names[e.id] + " minus P" }, { text: "95% interval " + signed(e.lo) + " to " + signed(e.hi) }, { text: DS_NAME[d] + ", " + e.n + " partitions" }] }));
          if (d === "resisc45" && e.id === "PGP" && !mobile) {
            T("no weights touched,", { x: x(e.hi) + 18, y: cy - 2, class: "pen-t" }, svg);
            T("same gain", { x: x(e.hi) + 18, y: cy + 20, class: "pen-t" }, svg);
          }
        });
      });
    });
    const rows = [];
    ["resisc45", "eurosat13"].forEach((d) => D.schedules[d].forEach((e) => rows.push([DS_NAME[d], names[e.id], signed(e.m), signed(e.lo) + " to " + signed(e.hi), String(e.n)])));
    dataTable(cont, "sched", ["Test set", "Schedule minus P", "gain (points)", "95% interval", "partitions"], rows);
  }

  // ------------------------------------------------------------------------------------------ scatter: accuracy against multiply-accumulates
  function initScatter() {
    const cont = $("#chart-scatter"); if (!cont) return;
    let ds = "resisc45";
    legend($("#leg-scatter"), scoreLegend(["random", "l1", "bn_gamma", "taylor", "dacis"]).concat([
      { kind: "mark", shape: "circle", score: "random", hollow: true, text: "ResNet-18 pruned" }, { kind: "mark", shape: "square", score: "random", hollow: true, text: "MobileNetV2 pruned" }, { kind: "mark", shape: "diamond", score: "ink", text: "never pruned" }]));
    const DENSE_NAME = { r18: "ResNet-18", mbv2: "MobileNetV2", mbv3s: "MobileNetV3-Small", eff: "EfficientNet-B0" };
    const SHORT = { r18: "R18", mbv2: "V2", mbv3s: "V3-S", eff: "Eff-B0" };
    const legHost = $("#leg-scatter");
    const m = mount(cont, (W) => {
      const oldNote = legHost.querySelector(".short-note"); if (oldNote) oldNote.remove();
      if (W < 640) legHost.appendChild(H("span", { class: "short-note", text: "R18 ResNet-18 \u00b7 V2 MobileNetV2 \u00b7 V3-S MobileNetV3-Small \u00b7 Eff-B0 EfficientNet-B0" }));
      const pts = D.tradeoff[ds], pruned = pts.filter((p) => p.kind === "pruned"), dense = pts.filter((p) => p.kind === "dense");
      const mobile = W < 640, Lm = mobile ? 40 : 54, Rm = 16, Tp = 28, B = 52, H_ = mobile ? 400 : 480;
      const xs = pts.map((p) => p.macs), ys = pts.map((p) => p.m);
      const xlo = Math.log10(Math.min.apply(null, xs) * 0.75), xhi = Math.log10(Math.max.apply(null, xs) * 1.4);
      const ylo = Math.floor(Math.min.apply(null, ys) - 2), yhi = Math.ceil(Math.max.apply(null, ys) + 2);
      const X = (v) => sc(xlo, xhi, Lm, W - Rm)(Math.log10(v)), Y = sc(ylo, yhi, H_ - B, Tp);
      const svg = S("svg", { viewBox: "0 0 " + W + " " + H_, role: "presentation" }, cont);
      const ax = S("g", { class: "ax" }, svg);
      niceTicks(ylo, yhi, 6).forEach((t) => { S("path", { d: "M" + Lm + " " + Y(t) + "H" + (W - Rm), class: "grid-l" }, ax); T(t + "%", { x: Lm - 8, y: Y(t) + 4, "text-anchor": "end", class: "lbl" }, ax); });
      [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1].filter((t) => Math.log10(t) > xlo && Math.log10(t) < xhi).forEach((t) => {
        S("path", { d: "M" + X(t) + " " + Tp + "V" + (H_ - B), class: "grid-l" }, ax); T(String(t), { x: X(t), y: H_ - B + 18, "text-anchor": "middle", class: "lbl" }, ax);
      });
      T(mobile ? "multiply-accumulates (G), log scale" : "multiply-accumulates per image (G), log scale → more work", { x: (Lm + W - Rm) / 2, y: H_ - 10, "text-anchor": "middle", class: "lbl" }, svg);
      const best = dense.find((p) => p.family === "mbv3s");
      // the line nothing pruned reaches
      S("path", { d: "M" + Lm + " " + Y(best.m) + "H" + (W - Rm), style: "stroke:var(--ink);stroke-width:1.8;stroke-dasharray:7 5;fill:none" }, svg);
      // taylor trend per family
      [["r18", "PMP"], ["mbv2", "P"]].forEach(([fam, sched]) => {
        const tp = pruned.filter((p) => p.family === fam && p.score === "taylor" && p.sched === sched).sort((a, b) => a.macs - b.macs);
        if (tp.length > 1) S("path", { d: tp.map((p, i) => (i ? "L" : "M") + X(p.macs) + " " + Y(p.m)).join(" "), style: "fill:none;stroke:var(--c-taylor);stroke-width:1.8;opacity:.55" }, svg);
      });
      const order = ["random", "l1", "bn_gamma", "G", "V", "D", "dacis", "taylor"];
      const drawn = [];
      pruned.slice().sort((a, b) => order.indexOf(a.score) - order.indexOf(b.score)).forEach((p) => {
        const sh = p.family === "mbv2" ? "square" : "circle";
        marker(svg, sh, X(p.macs), Y(p.m), 4.2, p.score, { opacity: ["G", "V", "D"].includes(p.score) ? 0.55 : 0.95, noRing: false });
        drawn.push({ p: p, x: X(p.macs), y: Y(p.m) });
      });
      dense.forEach((p) => {
        const star = p.family === "mbv3s";
        const g = S("g", { transform: "translate(" + X(p.macs) + " " + Y(p.m) + ")" }, svg);
        const d = "M0 -9L9 0L0 9L-9 0z";
        S("path", { d: d, style: "fill:var(--surface);stroke:var(--surface);stroke-width:6" }, g);
        S("path", { d: d, style: star ? "fill:var(--hi);stroke:var(--ink);stroke-width:2" : "fill:var(--ink)" }, g);
        const left = X(p.macs) > W - 190;
        const nm = DENSE_NAME[p.family];
        T(mobile ? SHORT[p.family] : nm + " " + num(p.m) + "%", { x: mobile ? 0 : (left ? -14 : 14), y: mobile ? 22 : 4, "text-anchor": mobile ? "middle" : (left ? "end" : "start"), class: "lbl-d", style: "font-size:" + (mobile ? 11 : 14) + "px" }, g);
        drawn.push({ p: p, x: X(p.macs), y: Y(p.m) });
      });
      // pen note above the line: no pruned dot can be up there, so nothing is hidden
      if (!mobile) {
        const nx = Math.min(X(best.macs) + 26, W - 330), ny = Math.max(Tp + 26, Y(best.m) - 52);
        T("nothing we pruned gets above this line", { x: nx, y: ny, class: "pen-t" }, svg);
        const ar = Pen.arrow(nx + 250, ny + 8, nx + 292, Y(best.m) - 5, { bend: -0.3, seed: 2, head: 9 });
        penPath(svg, ar.d); penPath(svg, ar.head);
      }
      // nearest-point hover
      const hov = S("circle", { r: 9, style: "fill:none;stroke:var(--pen);stroke-width:2.2;opacity:0" }, svg);
      const ov = S("rect", { x: Lm, y: Tp, width: W - Lm - Rm, height: H_ - B - Tp, style: "fill:transparent", tabindex: 0, role: "img", "aria-label": "Scatter plot; the table below lists every network" }, svg);
      const near = (e) => {
        const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W, py = ((e.clientY - r.top) / r.height) * H_;
        let best2 = null, bd = 28;
        drawn.forEach((q) => { const dd = Math.hypot(q.x - px, q.y - py); if (dd < bd) { bd = dd; best2 = q; } });
        return best2;
      };
      const show = (e) => {
        const q = near(e);
        if (!q) { hov.style.opacity = 0; Tip.hide(); return; }
        hov.setAttribute("cx", q.x); hov.setAttribute("cy", q.y); hov.style.opacity = 1;
        const p = q.p;
        Tip.show({ head: num(p.m) + "%", lines: [{ color: p.score ? SC_VAR[p.score] : "var(--ink)", text: p.label }, { text: num(p.macs, 3) + " G multiply-accumulates" }, { text: "sd " + num(p.sd) + " over " + p.n + " partitions" }] }, e.clientX, e.clientY);
      };
      ov.addEventListener("pointermove", show); ov.addEventListener("pointerdown", show);
      ov.addEventListener("pointerleave", () => { hov.style.opacity = 0; Tip.hide(); });
    });
    function table() {
      const rows = D.tradeoff[ds].slice().sort((a, b) => b.m - a.m).map((p) => [p.label, num(p.macs, 3), num(p.m), num(p.sd), String(p.n)]);
      dataTable(cont, "scatter", ["Network", "G MACs", "5-shot (%)", "sd", "partitions"], rows, "Data table: every network in the plot");
    }
    seg($("#seg-scatter"), [{ v: "resisc45", t: "RESISC45" }, { v: "eurosat13", t: "EuroSAT-MS" }, { v: "plantvillage", t: "PlantVillage" }], ds, (v) => { ds = v; m.redraw(); table(); });
    table();
  }

  A.init = function () { initStrip(); initForest(); initLead(); initRace(); initSched(); initScatter(); };
})();
