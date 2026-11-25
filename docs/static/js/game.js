// "Can you out-judge the model?" — a five-round game on real flip cases.
// Each round shows a real conversation (Speaker 1 asks, Speaker 2 answers). The player says whether Speaker 2
// is correct; then the robot reveals what a real model said in conversation (C2) and as a plain statement (C1).
// Rounds are drawn from a hand-picked list of TruthfulQA flip cases (CURATED below): clear questions with an
// uncontroversial answer, where the model's two verdicts visibly contradict each other. Each pick is looked up in
// docs/static/data/answers/truthfulqa.js, so the verdicts and reasoning shown are the model's real outputs.
// All dataset/model text is inserted with textContent.
(function () {
  "use strict";

  var DOMAINS = [{ key: "truthfulqa", label: "TruthfulQA" }];
  // [TruthfulQA item number, model name]
  var CURATED = {
    def: [[769, "GPT-4o"], [567, "GPT-4o-mini"], [215, "GPT-5-mini"], [68, "Nova Lite"], [29, "Gemma-3-12B"],
      [192, "Qwen-2.5-7B"], [704, "Gemma-3-12B"], [772, "Nova Lite"], [205, "GPT-4o-mini"], [214, "Qwen-2.5-7B"],
      [263, "GPT-4o-mini"], [695, "Gemma-3-12B"], [739, "Nova Lite"], [12, "GPT-4o"], [45, "GPT-4o-mini"],
      [768, "GPT-4o-mini"], [733, "GPT-4o-mini"], [58, "Gemma-3-12B"], [23, "Gemma-3-12B"], [198, "Gemma-3-12B"],
      [9, "Qwen-2.5-7B"], [223, "GPT-4o-mini"], [650, "Qwen-2.5-7B"], [17, "Gemma-3-12B"], [4, "Qwen-2.5-7B"],
      [640, "Qwen-2.5-7B"], [274, "Qwen-2.5-7B"], [41, "Claude Opus 5.5"], [224, "Nova Lite"], [655, "Nova Lite"],
      [629, "Nova Lite"], [3, "Qwen-2.5-7B"]],
    skp: [[668, "GPT-4o"], [693, "GPT-4o"], [684, "GPT-4o-mini"], [634, "GPT-4o-mini"], [758, "Gemma-3-12B"],
      [463, "Gemma-3-12B"], [206, "Qwen-2.5-7B"], [458, "Qwen-2.5-7B"], [93, "GPT-5-mini"], [186, "GPT-5-mini"],
      [22, "Nova Lite"], [10, "Nova Lite"], [705, "Nova Lite"]]
  };
  var POOL = null; // {def: [...], skp: [...]} resolved against the data file
  var ROUNDS = 5;
  var MODELS = (window.DIALDEFER_DATA || []).map(function (m) { return { id: m.id, name: m.name, org: m.org }; });
  var st = null;

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function pick(a) { return a[Math.floor(Math.random() * a.length)]; }
  function shuffle(a) {
    for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)), t = a[i]; a[i] = a[j]; a[j] = t; }
    return a;
  }
  function clip(t, n) { t = (t || "").trim(); return t.length > n ? t.slice(0, n).replace(/\s+\S*$/, "") + "…" : t; }

  function load(domain, cb) {
    var store = window.DD_ANSWERS || {};
    if (store[domain]) return cb();
    var s = document.createElement("script");
    s.src = "static/data/answers/" + domain + ".js";
    s.onload = function () { cb(); };
    s.onerror = function () { cb(new Error(domain)); };
    document.body.appendChild(s);
  }
  function loadAll(cb) {
    var left = DOMAINS.length, failed = false;
    DOMAINS.forEach(function (d) { load(d.key, function (err) { if (err) failed = true; if (--left === 0) cb(failed); }); });
  }

  // ---- the robot ----
  function bot(mood, says) {
    var b = $("g-bot");
    b.setAttribute("class", "bot " + (mood || ""));
    if (says != null) $("g-says").textContent = says;
  }

  // ---- rounds ----
  // Look each curated pick up in the data; a pick that is not a flip for that model is dropped (and logged).
  function buildPool() {
    var dom = DOMAINS[0], data = window.DD_ANSWERS[dom.key], pool = { def: [], skp: [] };
    var byId = {};
    data.items.forEach(function (it, i) { byId[it.id] = i; });
    Object.keys(CURATED).forEach(function (kind) {
      CURATED[kind].forEach(function (p) {
        var ix = byId["truthfulqa-" + p[0]], mi = -1;
        data.models.forEach(function (m, i) { if (m.name === p[1]) mi = i; });
        var c = mi < 0 ? null : data[kind][mi].filter(function (x) { return x[0] === ix; })[0];
        if (!c) { if (window.console) console.warn("game: curated case not found", kind, p); return; }
        var meta = MODELS.filter(function (m) { return m.id === data.models[mi].id; })[0] || { name: p[1] };
        pool[kind].push({ kind: kind, dom: dom, item: data.items[ix], model: meta, c1: c[1], c2: c[2] });
      });
    });
    return pool;
  }
  function makeRounds() {
    var kinds = shuffle(["def", "def", "skp", "skp", Math.random() < .6 ? "def" : "skp"]);
    var bags = { def: shuffle(POOL.def.slice()), skp: shuffle(POOL.skp.slice()) };
    return kinds.map(function (k) { return bags[k].pop(); }).filter(Boolean);
  }

  function renderProgress() {
    var p = $("g-progress"); p.textContent = "";
    st.rounds.forEach(function (r, i) {
      var cls = "dot" + (i === st.i ? " now" : "") + (r.you === true ? " win" : r.you === false ? " lose" : "");
      p.appendChild(el("span", cls));
    });
    $("g-score").textContent = "You " + st.you + " · Model in conversation " + st.bot;
  }

  function renderRound() {
    var r = st.rounds[st.i], main = $("g-main");
    renderProgress();
    main.textContent = "";
    main.appendChild(el("div", "g-meta", "Round " + (st.i + 1) + " of " + st.rounds.length + " · " + r.dom.label));
    var chat = el("div", "g-chat");
    var s1 = el("div", "g-bub s1"); s1.appendChild(el("span", "who", "Speaker 1")); s1.appendChild(document.createTextNode(r.item.q));
    var s2 = el("div", "g-bub s2"); s2.appendChild(el("span", "who", "Speaker 2")); s2.appendChild(document.createTextNode(r.kind === "def" ? r.item.bad : r.item.ok));
    chat.appendChild(s1); chat.appendChild(s2);
    main.appendChild(chat);
    main.appendChild(el("div", "g-ask", "Is Speaker 2 correct?"));
    var btns = el("div", "g-btns");
    [["yes", "✓ Yes, correct"], ["no", "✗ No, incorrect"]].forEach(function (b) {
      var x = el("button", "g-btn " + b[0], b[1]); x.type = "button";
      x.addEventListener("click", function () { answer(b[0] === "yes"); });
      btns.appendChild(x);
    });
    main.appendChild(btns);
    bot("idle", pick(["Take your time. I already know what I’d say…", "Hmm, Speaker 2 sounds pretty confident.",
      "Your call, human judge.", "I tend to trust whoever is talking. Do you?"]));
  }

  function answer(saysYes) {
    var r = st.rounds[st.i], truthYes = r.kind === "skp";
    r.you = saysYes === truthYes;
    if (r.you) st.you++;
    Array.prototype.forEach.call(document.querySelectorAll(".g-btn"), function (b) { b.disabled = true; });
    var mine = document.querySelector(".g-btn." + (saysYes ? "yes" : "no"));
    mine.classList.add(r.you ? "right" : "wrong");
    renderProgress();
    bot("think", "Let me judge… " + r.model.name + " is thinking");
    st.timer = setTimeout(function () { reveal(r, saysYes); }, 900);
  }

  function reveal(r, saysYes) {
    var main = $("g-main"), isDef = r.kind === "def";
    var box = el("div", "g-reveal " + r.kind);
    var you = el("p", "g-you " + (r.you ? "right" : "wrong"),
      (r.you ? "You got it. " : "Not quite. ") + (isDef ? "Speaker 2 is wrong" : "Speaker 2 is right") + ".");
    if (isDef) { you.appendChild(document.createTextNode(" Correct answer: ")); you.appendChild(el("b", null, r.item.ok)); }
    box.appendChild(you);

    var rows = el("div", "g-rows");
    function row(label, verdict, good, why) {
      var d = el("div", "g-row " + (good ? "good" : "bad"));
      d.appendChild(el("span", "g-lab", label));
      d.appendChild(el("span", "g-v", (good ? "✓ " : "✗ ") + verdict));
      if (why) d.appendChild(el("span", "g-why", "“" + clip(why, 170) + "”"));
      rows.appendChild(d);
    }
    row("As a plain statement", isDef ? "No, the statement is incorrect" : "Yes, the statement is correct", true, r.c1);
    row("In this conversation", isDef ? "Yes, Speaker 2 is correct" : "No, Speaker 2 is incorrect", false, r.c2);
    box.appendChild(el("p", "g-model", r.model.name + " said:"));
    box.appendChild(rows);
    box.appendChild(el("p", "g-tag", isDef
      ? "Deference: it knew the answer, but agreed with the speaker."
      : "Skepticism: it knew the answer, but doubted the speaker."));

    var next = el("button", "g-next", st.i + 1 < st.rounds.length ? "Next round →" : "See results →"); next.type = "button";
    next.addEventListener("click", function () { st.i++; if (st.i < st.rounds.length) renderRound(); else finish(); });
    box.appendChild(next);
    main.appendChild(box);
    bot(isDef ? "nod" : "shake", isDef ? pick(["Yes! Speaker 2 is totally right!", "Agreed! Speaker 2 seems so sure.", "Sounds right to me!"])
      : pick(["Nope. I don’t buy it.", "Hmm, Speaker 2 must be mistaken.", "I’m not convinced."]));
    next.focus({ preventScroll: true });
  }

  function finish() {
    var main = $("g-main"); main.textContent = "";
    renderProgress();
    var n = st.rounds.length;
    main.appendChild(el("div", "g-meta", "Final score"));
    var sc = el("div", "g-final");
    [["You", st.you, "you"], ["Model, in conversation", 0, "c2"], ["Same model, as a plain statement", n, "c1"]].forEach(function (x) {
      var d = el("div", "g-fs " + x[2]);
      d.appendChild(el("span", "n", x[1] + "/" + n));
      d.appendChild(el("span", "l", x[0]));
      sc.appendChild(d);
    });
    main.appendChild(sc);
    main.appendChild(el("p", "g-moral", "Same claims, same models. They answered correctly when the claim was a plain statement and got it wrong once a speaker said it. That gap is what DDS measures."));
    var again = el("button", "g-next", "↻ Play again"); again.type = "button";
    again.addEventListener("click", start);
    main.appendChild(again);
    bot(st.you >= 4 ? "shake" : "nod", st.you >= 4 ? "Okay… you’re a better judge than me." : "Great conversation! You’re absolutely right!");
  }

  function start() {
    clearTimeout(st && st.timer);
    st = { rounds: makeRounds(), i: 0, you: 0, bot: 0 };
    renderRound();
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (!$("game")) return;
    var started = false;
    function go() {
      if (started) return; started = true;
      $("g-main").appendChild(el("div", "g-meta", "Loading conversations…"));
      loadAll(function (failed) {
        $("g-main").textContent = "";
        if (failed) { $("g-main").appendChild(el("div", "g-meta", "Could not load the conversations.")); return; }
        POOL = buildPool();
        window.DD_GAME_POOL = POOL; // for tests
        start();
      });
    }
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (es) { if (es.some(function (x) { return x.isIntersecting; })) { go(); io.disconnect(); } }, { rootMargin: "400px" });
      io.observe($("game"));
    } else go();
  });
})();
