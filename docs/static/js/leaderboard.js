(function () {
  "use strict";

  var DATA = window.DIALDEFER_DATA || [];
  var SVGNS = "http://www.w3.org/2000/svg";

  // Domain order and grouping, matching the paper's heatmap.
  var DOMAINS = [
    { key: "truthfulqa",  label: "TruthfulQA",  group: "Factual" },
    { key: "halueval",    label: "HaluEval",    group: "Factual" },
    { key: "plausibleqa", label: "PlausibleQA", group: "Factual" },
    { key: "gpqa",        label: "GPQA",        group: "Technical" },
    { key: "harp",        label: "HARP",        group: "Technical" },
    { key: "amqa",        label: "AMQA",        group: "Technical" },
    { key: "bbq",         label: "BBQ",         group: "Social" },
    { key: "socialiqa",   label: "SocialIQA",   group: "Social" },
    { key: "advisorqa",   label: "AdvisorQA",   group: "Social" },
    { key: "aio",         label: "r/AIO",       group: "Naturalistic" }
  ];
  window.DD_DOMAINS = DOMAINS;

  function mean(xs) { return xs.reduce(function (a, b) { return a + b; }, 0) / xs.length; }

  // Per-model aggregate over the 10 domains (unweighted macro-average, as in the paper).
  DATA.forEach(function (m) {
    var ds = DOMAINS.map(function (d) { return m.domains[d.key]; });
    m.domains.all = {
      n: ds.reduce(function (a, d) { return a + d.n; }, 0),
      dds: mean(ds.map(function (d) { return d.dds; })),
      absdds: mean(ds.map(function (d) { return Math.abs(d.dds); })),
      dc: mean(ds.map(function (d) { return d.dc; })),
      di: mean(ds.map(function (d) { return d.di; })),
      c1: mean(ds.map(function (d) { return d.c1; })),
      c2: mean(ds.map(function (d) { return d.c2; })),
      sig: m.overall_sig
    };
    // Paper models: show the overall numbers exactly as printed in the paper (Figure 3 / Table 2).
    if (m.paper_overall) Object.keys(m.paper_overall).forEach(function (k) { m.domains.all[k] = m.paper_overall[k]; });
  });

  var state = { domain: "all", sortKey: "dds", sortDir: 1 };

  var COLS = [
    { key: "dds", label: "DDS", title: "Dialogic Deference Score (pp). Ranked by distance from 0: closer to 0 = judgment does not depend on framing.",
      get: function (r) { return Math.abs(r.dds); } },
    { key: "dc", label: "&Delta; Correct", title: "Acc(C2, correct speaker) − Acc(C1, true statement), pp", get: function (r) { return r.dc; } },
    { key: "di", label: "&Delta; Incorrect", title: "Acc(C2, incorrect speaker) − Acc(C1, false statement), pp", get: function (r) { return r.di; } },
    { key: "absdds", label: "Mean domain |DDS|", title: "Mean of the absolute per-domain DDS over the 10 domains: penalizes both deference and skepticism. Not the same statistic as the abstract's mean |DDS|.", get: function (r) { return r.absdds; }, allOnly: true },
    { key: "c2", label: "Acc C1 → C2", title: "Average accuracy under factual inquiry (C1) and conversational judgment (C2), %. Sorts by C2.", get: function (r) { return r.c2; } }
  ];

  function fmtSigned(v, digits) {
    var dg = digits == null ? 1 : digits;
    var s = v.toFixed(dg);
    if (parseFloat(s) === 0) s = (0).toFixed(dg);
    else if (v > 0) s = "+" + s;
    return s.replace("-", "−");
  }
  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text != null) e.textContent = text;
    return e;
  }
  function svg(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text != null) e.textContent = text;
    return e;
  }
  // Company mark (from logos.js) as an inline 24x24 SVG; null if the org has none.
  function logo(org, size, attrs) {
    var d = (window.DD_LOGOS || {})[org];
    if (!d) return null;
    var a = { width: size, height: size, viewBox: "0 0 24 24", "class": "logo", "aria-hidden": "true" };
    Object.keys(attrs || {}).forEach(function (k) { a[k] = attrs[k]; });
    var e = svg("svg", a);
    e.appendChild(svg("path", { d: d, fill: "currentColor" }));
    return e;
  }
  function visibleCols() { return COLS.filter(function (c) { return !c.allOnly || state.domain === "all"; }); }
  function sortedRows() {
    var col = COLS.filter(function (c) { return c.key === state.sortKey; })[0] || COLS[0];
    var dom = state.domain;
    return DATA.slice().sort(function (a, b) {
      return state.sortDir * (col.get(a.domains[dom]) - col.get(b.domains[dom])) || a.name.localeCompare(b.name);
    });
  }

  // ---------- domain chips ----------
  function renderChips() {
    var box = document.getElementById("lb-chips");
    box.textContent = "";
    [{ key: "all", label: "All domains (avg)" }].concat(DOMAINS).forEach(function (d) {
      var b = el("button", { "class": "chip" + (state.domain === d.key ? " active" : ""), type: "button" }, d.label);
      b.addEventListener("click", function () { setDomain(d.key); });
      box.appendChild(b);
    });
  }
  function setDomain(key) {
    state.domain = key;
    if (key !== "all" && state.sortKey === "absdds") { state.sortKey = "dds"; state.sortDir = 1; }
    renderAll();
  }

  // ---------- vertical bar chart (HalluHard-style) ----------
  function niceStep(range) {
    var raw = range / 5, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
  }
  function renderChart() {
    var host = document.getElementById("lb-chart");
    var rows = sortedRows();
    var dom = state.domain;
    var vals = rows.map(function (m) { return m.domains[dom].dds; });

    var boxW = Math.max(300, host.clientWidth - 8);
    var isMobile = window.innerWidth <= 640;
    var margin = { top: 60, right: 12, bottom: isMobile ? 112 : 78, left: isMobile ? 46 : 60 };
    var plotH = isMobile ? 230 : 290;

    var lo = Math.min(0, Math.min.apply(null, vals)), hi = Math.max(0, Math.max.apply(null, vals));
    var step = niceStep(Math.max(hi - lo, 20));
    lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    if (hi === 0) hi = step;
    var y = function (v) { return plotH - (v - lo) / (hi - lo) * plotH; };

    var minSlot = isMobile ? 62 : 96;
    var maxSlot = 150;
    var avail = boxW - margin.left - margin.right;
    var slot = Math.max(minSlot, Math.min(maxSlot, avail / rows.length));
    var barW = Math.min(64, slot * 0.58);
    var contentW = slot * rows.length;
    var W = Math.max(boxW, contentW + margin.left + margin.right);
    var offsetX = margin.left + Math.max(0, (W - margin.left - margin.right - contentW) / 2);
    var rotate = slot < 110;
    // rotated names need more room below the plot
    if (rotate) margin.bottom = isMobile ? 112 : 130;
    var H = margin.top + plotH + margin.bottom;

    var s = svg("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Bar chart of DDS by model" });
    var defs = svg("defs");
    var pat = svg("pattern", { id: "stripes", patternUnits: "userSpaceOnUse", width: 6, height: 6, patternTransform: "rotate(45)" });
    pat.appendChild(svg("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "rgba(255,255,255,.45)", "stroke-width": 3 }));
    defs.appendChild(pat); s.appendChild(defs);

    var g = svg("g", { transform: "translate(0," + margin.top + ")" });
    for (var t = lo; t <= hi + 1e-9; t += step) {
      var ty = y(t);
      g.appendChild(svg("line", { x1: margin.left, x2: W - margin.right, y1: ty, y2: ty,
        stroke: t === 0 ? "#64748b" : "#e2e8f0", "stroke-width": t === 0 ? 1.5 : 1 }));
      g.appendChild(svg("text", { x: margin.left - 8, y: ty, "text-anchor": "end", "dominant-baseline": "middle",
        "font-size": isMobile ? 10 : 11.5, fill: "#64748b" }, fmtSigned(t, 0)));
    }
    g.appendChild(svg("text", { transform: "translate(" + (isMobile ? 12 : 16) + "," + (plotH / 2) + ") rotate(-90)",
      "text-anchor": "middle", "font-size": isMobile ? 10.5 : 12.5, "font-weight": 600, fill: "#64748b" }, "DDS (pp)"));
    // direction hints sit at the left, clear of the logos and labels above the bars
    g.appendChild(svg("text", { x: 4, y: -40, "font-size": 11, fill: "#d1453b", "font-weight": 600 }, "▲ deference"));
    if (lo < 0) g.appendChild(svg("text", { x: 4, y: plotH + 16, "font-size": 11, fill: "#2f6fd6", "font-weight": 600 }, "▼ skepticism"));

    rows.forEach(function (m, i) {
      var r = m.domains[dom], v = r.dds;
      var cx = offsetX + slot * i + slot / 2, x = cx - barW / 2;
      var y0 = y(0), yv = y(v);
      var top = Math.min(y0, yv), h = Math.max(1.5, Math.abs(yv - y0));
      var color = v >= 0 ? "#d1453b" : "#2f6fd6";
      var bg = svg("g", { "class": "bar-g" });
      bg.appendChild(svg("title", null, m.name + (m.paper ? "" : " †") + " · DDS " + fmtSigned(v, 2) +
        " · ΔCorrect " + fmtSigned(r.dc) + " · ΔIncorrect " + fmtSigned(r.di) +
        " · Acc " + r.c1.toFixed(1) + "→" + r.c2.toFixed(1)));
      bg.appendChild(svg("rect", { "class": "bar", x: x, y: top, width: barW, height: h, rx: 3, fill: color }));
      if (m.open) bg.appendChild(svg("rect", { x: x, y: top, width: barW, height: h, rx: 3, fill: "url(#stripes)" }));
      // company logo sits on top of the bar (above the zero line for negative bars); value label above it
      var ls = isMobile ? 18 : 24;
      var ly = top - 5 - ls;
      var lg = logo(m.org, ls, { x: cx - ls / 2, y: ly, color: "#334155" });
      if (lg) bg.appendChild(lg);
      bg.appendChild(svg("text", { x: cx, y: v >= 0 ? (lg ? ly - 6 : top - 6) : top + h + 14, "text-anchor": "middle",
        "font-size": isMobile ? 11 : 12.5, "font-weight": 700, fill: color }, fmtSigned(v) + (r.sig === true ? "*" : "")));
      // model names sit in a band below the plot area, not under the zero line
      var ny = plotH + (lo < 0 ? 30 : 16);
      var name = m.name + (m.paper ? "" : " †");
      if (rotate) {
        bg.appendChild(svg("text", { transform: "translate(" + (cx + 6) + "," + (ny + 4) + ") rotate(-38)", "text-anchor": "end",
          "font-size": isMobile ? 11 : 12.5, "font-weight": 600, fill: "#1e293b" }, name));
      } else {
        bg.appendChild(svg("text", { x: cx, y: ny + 4, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: "#1e293b" }, name));
        bg.appendChild(svg("text", { x: cx, y: ny + 21, "text-anchor": "middle", "font-size": 11.5, fill: "#64748b" },
          m.org + (m.open ? " · open weights" : "")));
      }
      g.appendChild(bg);
    });
    s.appendChild(g);
    host.textContent = "";
    host.appendChild(s);
  }

  // ---------- table ----------
  function renderTable() {
    var cols = visibleCols();
    var head = document.getElementById("lb-head");
    var body = document.getElementById("lb-body");
    var dom = state.domain;

    head.textContent = "";
    var tr = el("tr");
    tr.appendChild(el("th", { "class": "rank" }, "#"));
    tr.appendChild(el("th", { "class": "left" }, "Model"));
    cols.forEach(function (c) {
      var th = el("th", { "class": "sortable", title: c.title });
      th.innerHTML = c.label + ' <span class="arrow">' + (state.sortKey === c.key ? (state.sortDir === 1 ? "↑" : "↓") : "") + "</span>";
      th.addEventListener("click", function () {
        if (state.sortKey === c.key) state.sortDir = -state.sortDir;
        else { state.sortKey = c.key; state.sortDir = (c.key === "dds" || c.key === "absdds") ? 1 : -1; }
        renderTable(); renderChart();
      });
      tr.appendChild(th);
    });
    head.appendChild(tr);

    body.textContent = "";
    sortedRows().forEach(function (m, i) {
      var r = m.domains[dom];
      var row = el("tr");
      row.appendChild(el("td", { "class": "rank" }, String(i + 1)));
      var name = el("td", { "class": "left" });
      var tl = logo(m.org, 18, { title: m.org });
      if (tl) name.appendChild(tl);
      var ms = el("span", { "class": "model", title: m.note }, m.name);
      if (!m.paper) ms.appendChild(el("span", { "class": "dagger", title: "Not in the ACL 2026 paper; same prompts and data, run settings may differ" }, "†"));
      name.appendChild(ms);
      name.appendChild(el("span", { "class": "org" }, m.org));
      if (m.open) name.appendChild(el("span", { "class": "tag" }, "open weights"));
      row.appendChild(name);
      cols.forEach(function (c) {
        var td = el("td");
        if (c.key === "dds") {
          td.appendChild(el("span", { "class": "val " + (r.dds >= 0 ? "pos" : "neg") }, fmtSigned(r.dds)));
          if (r.sig === true) td.appendChild(el("span", { "class": "sig" }, "*"));
          else if (r.sig === null && dom === "aio") td.appendChild(el("span", { "class": "sig", title: "Significance not computed for r/AIO" }, " –"));
        } else if (c.key === "dc" || c.key === "di") {
          var v = r[c.key];
          td.className = v > 0.05 ? "num-up" : v < -0.05 ? "num-down" : "muted";
          td.textContent = fmtSigned(v);
        } else if (c.key === "absdds") {
          td.textContent = r.absdds.toFixed(1);
        } else {
          td.appendChild(el("span", { "class": "muted" }, r.c1.toFixed(1) + " → "));
          td.appendChild(document.createTextNode(r.c2.toFixed(1)));
        }
        row.appendChild(td);
      });
      body.appendChild(row);
    });

    var d = DOMAINS.filter(function (x) { return x.key === dom; })[0];
    var n = DATA[0] ? DATA[0].domains[dom].n : 0;
    document.getElementById("lb-scope").textContent = dom === "all"
      ? "Macro-average over 10 domains · N = " + n.toLocaleString() + " items per model"
      : d.label + " (" + d.group + ") · N = " + n.toLocaleString() + " items per model";
    document.getElementById("lb-sig").textContent = dom === "all"
      ? "* p < .0001, McNemar’s test on all items (as in the paper’s Table 2)"
      : "* 95% bootstrap CI of DDS excludes 0";
  }

  // ---------- heatmap ----------
  function heatColor(v) {
    var t = Math.max(-1, Math.min(1, v / 60)), a = Math.abs(t);
    var to = t >= 0 ? [209, 69, 59] : [47, 111, 214];
    var c = [255, 255, 255].map(function (w, i) { return Math.round(w + (to[i] - w) * a); });
    return { bg: "rgb(" + c.join(",") + ")", fg: a > 0.55 ? "#fff" : "#1e293b" };
  }
  function renderHeat() {
    var tbl = document.getElementById("heat");
    tbl.textContent = "";
    var models = DATA.slice().sort(function (a, b) { return b.domains.all.dds - a.domains.all.dds; });
    var groups = [];
    DOMAINS.forEach(function (d) {
      var g = groups[groups.length - 1];
      if (g && g.name === d.group) g.span++; else groups.push({ name: d.group, span: 1 });
    });
    var thead = el("thead"), gr = el("tr", { "class": "groups" });
    gr.appendChild(el("th"));
    groups.forEach(function (g) { gr.appendChild(el("th", { colspan: g.span }, g.name)); });
    gr.appendChild(el("th"));
    var hr = el("tr");
    hr.appendChild(el("th"));
    DOMAINS.concat([{ key: "all", label: "Avg" }]).forEach(function (d) {
      var th = el("th", { "class": "col", title: "Show " + d.label + " in the leaderboard" }, d.label);
      th.addEventListener("click", function () {
        setDomain(d.key);
        document.getElementById("leaderboard").scrollIntoView({ behavior: "smooth" });
      });
      hr.appendChild(th);
    });
    thead.appendChild(gr); thead.appendChild(hr); tbl.appendChild(thead);
    var tb = el("tbody");
    models.forEach(function (m) {
      var tr = el("tr");
      var rh = el("th", { "class": "row" });
      var hl = logo(m.org, 14);
      if (hl) rh.appendChild(hl);
      rh.appendChild(document.createTextNode(m.name + (m.paper ? "" : " †")));
      tr.appendChild(rh);
      DOMAINS.concat([{ key: "all", label: "Average" }]).forEach(function (d) {
        var v = m.domains[d.key].dds, c = heatColor(v), sig = m.domains[d.key].sig === true;
        var td = el("td", { "class": state.domain === d.key ? "sel" : "",
          title: m.name + " · " + d.label + ": DDS " + fmtSigned(v, 2) + (sig ? " (significant)" : "") },
          fmtSigned(v, d.key === "all" ? 1 : 0) + (sig ? "*" : ""));
        td.style.background = c.bg; td.style.color = c.fg;
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
  }

  function renderAll() { renderChips(); renderChart(); renderTable(); renderHeat(); }

  // ---------- BibTeX copy ----------
  function initCopy() {
    Array.prototype.forEach.call(document.querySelectorAll(".copy-btn"), function (btn) {
      btn.addEventListener("click", function () {
        var txt = document.getElementById(btn.getAttribute("data-target")).innerText;
        navigator.clipboard.writeText(txt).then(function () {
          btn.textContent = "Copied";
          setTimeout(function () { btn.textContent = "Copy"; }, 1500);
        });
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    // Deep link to a domain view, e.g. index.html?domain=gpqa
    var q = new URLSearchParams(location.search).get("domain");
    if (q && DOMAINS.some(function (d) { return d.key === q; })) state.domain = q;
    renderAll();
    initCopy();
    var rt;
    window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(renderChart, 120); });
  });
})();
