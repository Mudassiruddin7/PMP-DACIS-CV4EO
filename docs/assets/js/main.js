/* Page behaviour: the pruning meter, section highlight, the small drawn icons, the ledger numbers. */
(function () {
  "use strict";
  const D = window.SITE_DATA, K = window.ChartKit, $ = (s, r) => (r || document).querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };

  // ---------------------------------------------------------------- ledger numbers come from the same data as the charts
  const L = D.lead.resisc45, f1 = (x) => K.signed(x);
  const vals = {
    "lead.r18.30": f1(L.r18.scores.taylor[1].m), "lead.r18.10": f1(L.r18.scores.taylor[3].m), "lead.mb.25": f1(L.mbv2.scores.taylor[2].m),
    "bn.mb.25": f1(L.mbv2.scores.bn_gamma[2].m), "dacis.r18.30": f1(L.r18.scores.dacis[1].m),
  };
  document.querySelectorAll("[data-n]").forEach((n) => { const v = vals[n.getAttribute("data-n")]; if (v) n.textContent = v; });

  // ---------------------------------------------------------------- small drawn icons for the score guide
  (function icons() {
    const cells = $("#l1cells");
    if (cells) {
      const op = [0.9, 0.15, 0.5, 0.1, 0.75, 0.3, 0.55, 0.2, 0.4];
      op.forEach((o, i) => { const x = 22 + (i % 3) * 27, y = 18 + Math.floor(i / 3) * 27; el("rect", { class: "st", x: x, y: y, width: 25, height: 25, style: "fill:var(--c-l1);fill-opacity:" + o }, cells); });
    }
    const jg = $("#jenga");
    if (jg) {
      for (let i = 0; i < 5; i++) {
        const pulled = i === 2, x = pulled ? 54 : 28, y = 16 + i * 17;
        el("rect", { class: "st", x: x, y: y, width: 70, height: 14, rx: 2, style: "fill:" + (pulled ? "var(--c-taylor)" : "currentColor") + ";fill-opacity:" + (pulled ? 0.75 : 0.12) }, jg);
      }
    }
    const mx = $("#mixer");
    if (mx) {
      [["G", 0.3, 36], ["V", 0.2, 75], ["D", 0.5, 114]].forEach(([n, w, x]) => {
        const y = 90 - w * 150;
        el("path", { class: "st", d: "M" + x + " 16V90" }, mx);
        el("rect", { class: "st", x: x - 12, y: y - 6, width: 24, height: 12, rx: 2, style: "fill:var(--c-dacis);fill-opacity:.85" }, mx);
        const t1 = el("text", { x: x, y: y - 12, "text-anchor": "middle" }, mx); t1.textContent = String(w).replace("0", "");
        const t2 = el("text", { x: x, y: 108, "text-anchor": "middle", style: "fill:currentColor" }, mx); t2.textContent = n;
      });
    }
  })();

  // ---------------------------------------------------------------- the meter: the page is slowly pruned to 30% as you read
  (function meter() {
    const svg = $("#meter-svg"), out = $("#meter-out"), N = 40;
    const rng = Pen.rng(21), order = Array.from({ length: N }, (_, i) => i).map((i) => [rng(), i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    const rank = new Array(N); order.forEach((idx, r) => (rank[idx] = r));
    const bars = [];
    for (let i = 0; i < N; i++) bars.push(el("rect", { x: i * 3 + 0.5, y: 2, width: 2, height: 14, class: "m-on" }, svg));
    let ticking = false;
    function upd() {
      ticking = false;
      const max = document.documentElement.scrollHeight - window.innerHeight, p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      const kept = 100 - 70 * p, cut = Math.round(N * (1 - kept / 100));
      bars.forEach((b, i) => b.setAttribute("class", rank[i] < cut ? "m-off" : "m-on"));
      out.textContent = "kept " + Math.round(kept) + "%";
    }
    window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(upd); } }, { passive: true });
    upd();
  })();

  // ---------------------------------------------------------------- section highlight in the nav
  (function spy() {
    const links = Array.from(document.querySelectorAll(".nav a")), map = new Map();
    links.forEach((a) => { const s = document.getElementById(a.getAttribute("href").slice(1)); if (s) map.set(s, a); });
    if (!("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) { links.forEach((a) => a.removeAttribute("aria-current")); map.get(e.target).setAttribute("aria-current", "true"); } });
    }, { rootMargin: "-45% 0px -50% 0px" });
    map.forEach((a, s) => io.observe(s));
  })();

  // ---------------------------------------------------------------- charts and figures
  K.init();
  K.initFigs();
})();
