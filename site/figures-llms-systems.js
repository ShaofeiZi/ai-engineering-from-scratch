/* figures-llms-systems.js：阶段 10（从零构建 LLM）、阶段 12（多模态）和阶段 13
   （工具与协议）的交互式课程图示。在 lesson-figures.js 之后加载，并通过
   window.LF.register 注册。原生 ES5，无依赖，通过 CSS 变量应用主题。编写方式
   使用相同的围栏代码块：
       ```figure
       beam-search
       ``` */
(function () {
  'use strict';
  var LF = window.LF;
  if (!LF) { return; }
  var el = LF.el, svgEl = LF.svgEl, slider = LF.slider;

  // ── beam-search：每一步保留累积对数概率最高的 B 个序列 ─────────────────
  function beamSearch(host) {
    var state = { B: 3, steps: 4 };
    var W = 520, H = 240, PAD = 26;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 任意节点的候选子节点都使用确定性的逐步对数概率
    var STEP_LP = [-0.22, -0.51, -0.92, -1.39, -1.90];
    function px(s) { return PAD + s / state.steps * (W - 2 * PAD); }
    function py(rank, rows) { return PAD + (rank + 0.5) / rows * (H - 2 * PAD); }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var B = state.B, rows = B;
      // 每个存活束都有一个累积对数概率；根节点是概率为 0 的单束
      var beams = [{ lp: 0, y: py(0, 1), x: px(0) }];
      var s, kept = 1;
      for (s = 1; s <= state.steps; s++) {
        var cands = [];
        beams.forEach(function (b) {
          for (var c = 0; c < B; c++) { cands.push({ lp: b.lp + STEP_LP[c], from: b }); }
        });
        cands.sort(function (a, z) { return z.lp - a.lp; });
        var survivors = cands.slice(0, B);
        survivors.forEach(function (c, r) {
          c.x = px(s); c.y = py(r, B);
          svg.appendChild(svgEl('line', { x1: c.from.x, y1: c.from.y, x2: c.x, y2: c.y,
            stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '1.4', opacity: '0.55' }));
        });
        // 被丢弃的候选项淡出
        cands.slice(B).forEach(function (c, r) {
          var dy = py(B + r, B + cands.length - B);
          svg.appendChild(svgEl('line', { x1: c.from.x, y1: c.from.y, x2: px(s), y2: dy,
            stroke: 'var(--rule-soft,#ccc)', 'stroke-width': '1', 'stroke-dasharray': '3 3', opacity: '0.5' }));
        });
        beams = survivors; kept = survivors.length;
      }
      // 在上层绘制保留的节点
      beams.forEach(function (b) {
        svg.appendChild(svgEl('circle', { cx: b.x, cy: b.y, r: '4', fill: 'var(--blueprint,#3553ff)' }));
      });
      svg.appendChild(svgEl('circle', { cx: px(0), cy: py(0, 1), r: '5', fill: 'var(--ink,#1a1a1a)' }));
      var best = beams[0].lp;
      meta.textContent = (B === 1 ? 'B = 1 is greedy decoding' : 'B = ' + B + ' beams kept per step')
        + '  ·  best sequence log-prob ' + best.toFixed(2);
      formula.textContent = 'expand each beam into B children, score by Σ log p, keep the top ' + B + '  ·  ' + state.steps + ' steps';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'B', 'beam width B', 1, 5, 1),
      slider(state, 'steps', 'decode steps', 1, 5, 1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['BEAM SEARCH']), el('span', {}, ['drag the beam width'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Each step expands every surviving beam into B candidate continuations, scores them by cumulative log-probability, and keeps only the best B. Greedy decoding is the special case B equals 1: one path, no backtracking. Wider beams explore more but cost proportionally more compute.'])
    ]));
    state._render();
  }

  // ── speculative-decoding：草稿长度、接受率及由此产生的加速比 ───────────
  function speculativeDecoding(host) {
    var state = { gamma: 4, accept: 0.7 };
    var rows = el('div', {});
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    state._render = function () {
      var g = state.gamma, a = state.accept;
      // 首次拒绝前的预期接受前缀长度（验证器仍会发出一个纠正 token），
      // 上限为 g；如果全部 g 个 token 均通过，再加上奖励 token
      var expAcc = 0, prob = 1, i;
      for (i = 1; i <= g; i++) { expAcc += prob * a; prob *= a; }
      var allPass = Math.pow(a, g);
      // 每次验证生成的 token 数：连续接受数 + 1（纠正或奖励）
      var tokensPerPass = expAcc + 1;
      // 一次验证替代 tokensPerPass 个顺序执行的目标步骤
      var speedup = tokensPerPass;
      while (rows.firstChild) rows.removeChild(rows.firstChild);
      // 可视化一行 g 个草稿 token：前 floor(expAcc) 个被接受（蓝色），随后是一个验证 token
      var acceptedShown = Math.min(g, Math.round(expAcc));
      var strip = el('div', { class: 'lf-grid' });
      for (i = 0; i < g; i++) {
        var on = i < acceptedShown;
        var b = el('i'); b.style.width = '100%';
        if (!on) b.style.background = 'var(--rule-soft,#ccc)';
        var lab = el('label', {}, ['draft ' + (i + 1), el('b', {}, [on ? 'accept' : 'reject'])]);
        if (!on) lab.style.opacity = '0.45';
        strip.appendChild(el('div', { class: 'lf-ctrl' }, [lab, el('div', { class: 'lf-bar' }, [b])]));
      }
      rows.appendChild(strip);
      num.innerHTML = speedup.toFixed(2) + ' <small>x tokens / verify pass</small>';
      meta.textContent = 'expected ' + expAcc.toFixed(2) + ' of ' + g + ' drafts accepted  ·  all-' + g
        + '-pass chance ' + (allPass * 100).toFixed(0) + '%  ·  + 1 token from the target each pass';
      formula.textContent = 'draft γ = ' + g + ' tokens, verify in one target pass, accept rate α = ' + a.toFixed(2)
        + '  →  ~' + speedup.toFixed(2) + ' tokens per target call';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'gamma', 'draft length γ', 1, 8, 1),
      slider(state, 'accept', 'acceptance rate α', 0.1, 0.99, 0.01)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['SPECULATIVE DECODING']), el('span', {}, ['drag draft and accept rate'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [rows, num, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['A small draft model proposes γ tokens cheaply; the large target model verifies them all in a single parallel pass and accepts the longest correct prefix, then emits one more token itself. The higher the agreement rate, the longer the accepted run, and the more target calls each pass replaces.'])
    ]));
    state._render();
  }

  // ── moe-routing：将 token 路由至 top-k 专家，对比活跃参数与总参数并展示均衡性 ──
  function moeRouting(host) {
    var state = { experts: 8, topk: 2 };
    var W = 520, H = 200, PAD = 24;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var bar = el('i');
    var barWrap = el('div', { class: 'lf-bar' }, [bar]);
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    var TOKENS = 6;
    // 确定性路由：token t 优先选择从 (t*3) mod E 开始的专家
    function routeOf(t, E, k) {
      var picks = [], j;
      for (j = 0; j < k; j++) { picks.push((t * 3 + j) % E); }
      return picks;
    }
    state._render = function () {
      var E = state.experts, k = Math.min(state.topk, E);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var tokX = PAD, expX = W - PAD - 8;
      var load = [], e;
      for (e = 0; e < E; e++) { load.push(0); }
      var t;
      for (t = 0; t < TOKENS; t++) {
        var ty = PAD + (t + 0.5) / TOKENS * (H - 2 * PAD);
        svg.appendChild(svgEl('circle', { cx: tokX, cy: ty, r: '4', fill: 'var(--ink,#1a1a1a)' }));
        var picks = routeOf(t, E, k);
        picks.forEach(function (pe) {
          load[pe]++;
          var ey = PAD + (pe + 0.5) / E * (H - 2 * PAD);
          svg.appendChild(svgEl('line', { x1: tokX + 4, y1: ty, x2: expX, y2: ey,
            stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '1', opacity: '0.5' }));
        });
      }
      for (e = 0; e < E; e++) {
        var ey = PAD + (e + 0.5) / E * (H - 2 * PAD);
        var busy = load[e] > 0;
        svg.appendChild(svgEl('rect', { x: expX, y: ey - 5, width: '8', height: '10',
          fill: busy ? 'var(--blueprint,#3553ff)' : 'var(--rule-soft,#ccc)' }));
      }
      var activeFrac = k / E;
      num.innerHTML = (activeFrac * 100).toFixed(0) + ' <small>% of expert params active</small>';
      bar.style.width = (activeFrac * 100).toFixed(0) + '%';
      // 负载均衡：理想情况是每位专家处理 TOKENS*k/E；报告最大值/平均值不均衡率
      var avg = TOKENS * k / E;
      var mx = Math.max.apply(null, load);
      var imbal = avg > 0 ? mx / avg : 1;
      barWrap.classList.toggle('over', imbal > 1.6);
      meta.textContent = 'top-' + k + ' of ' + E + ' experts per token  ·  load imbalance (max/avg) ' + imbal.toFixed(2)
        + (imbal > 1.6 ? '  ·  needs balancing loss' : '  ·  reasonably balanced');
      formula.textContent = 'active fraction = k / E = ' + k + ' / ' + E + ' = ' + (activeFrac * 100).toFixed(0)
        + '%  ·  total params unchanged, compute scales with k';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'experts', 'experts E', 2, 12, 1),
      slider(state, 'topk', 'top-k routed', 1, 4, 1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['MIXTURE OF EXPERTS']), el('span', {}, ['drag experts and k'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, num, barWrap, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['A router sends each token to its top-k experts out of E. Only k of E expert blocks run per token, so the active compute is the fraction k over E even though every parameter still lives in memory. Uneven routing overloads a few experts, which is why MoE training adds a load-balancing loss.'])
    ]));
    state._render();
  }

  // ── context-window-slide：超出固定窗口的 token 会被丢弃 ─────────────────
  function contextWindowSlide(host) {
    var state = { seq: 14, window: 8 };
    var W = 520, H = 130, PAD = 20;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    var MAX = 24;
    state._render = function () {
      var n = state.seq, win = state.window;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var firstKept = Math.max(0, n - win);
      var cw = (W - 2 * PAD) / MAX;
      var bw = cw * 0.82, gap = cw * 0.18;
      var y = PAD + 18;
      var i;
      for (i = 0; i < n; i++) {
        var x = PAD + i * cw + gap / 2;
        var inWin = i >= firstKept;
        svg.appendChild(svgEl('rect', { x: x.toFixed(1), y: y, width: bw.toFixed(1), height: '28', rx: '2',
          fill: inWin ? 'var(--blueprint,#3553ff)' : 'var(--rule-soft,#ccc)',
          opacity: inWin ? '1' : '0.6' }));
      }
      // 窗口括号
      var wx0 = PAD + firstKept * cw, wx1 = PAD + n * cw;
      svg.appendChild(svgEl('rect', { x: wx0.toFixed(1), y: (y - 8).toFixed(1),
        width: (wx1 - wx0).toFixed(1), height: '44', fill: 'none',
        stroke: 'var(--warn,#b8870f)', 'stroke-width': '1.5' }));
      var dropped = Math.max(0, n - win);
      num.innerHTML = dropped + ' <small>tokens dropped</small>';
      meta.textContent = 'sequence ' + n + ' tokens · window ' + win + '  ·  '
        + (dropped > 0 ? 'oldest ' + dropped + ' fall outside the rolling context' : 'everything still fits');
      formula.textContent = 'attention sees only the last ' + win + ' positions; tokens before index '
        + firstKept + ' are no longer attended to';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'seq', 'sequence length', 1, MAX, 1),
      slider(state, 'window', 'context window', 1, 16, 1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['CONTEXT WINDOW']), el('span', {}, ['drag length past the window'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, num, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['A model can only attend over a fixed window. As the sequence grows past it, the oldest tokens slide out of the orange frame and are no longer visible to attention. This is the rolling context: recent tokens stay, early ones are forgotten unless they are summarized or retrieved back in.'])
    ]));
    state._render();
  }

  // ── perplexity-loss：困惑度 = e^loss，在 V 上随机猜测时困惑度为 V ───────
  function perplexityLoss(host) {
    var state = { loss: 2.0, logV: 4.7 };
    var W = 520, H = 200, PAD = 32, LMAX = 7;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function human(x) { var u = ['', 'K', 'M']; var i = 0; while (x >= 1000 && i < u.length - 1) { x /= 1000; i++; } return x.toFixed(x < 10 ? 1 : 0) + u[i]; }
    function px(l) { return PAD + l / LMAX * (W - 2 * PAD); }
    var PPMAX = Math.exp(LMAX);
    function py(pp) { return H - PAD - Math.log(pp) / Math.log(PPMAX) * (H - 2 * PAD); }
    state._render = function () {
      var loss = state.loss, V = Math.pow(10, state.logV);
      var pp = Math.exp(loss);
      var randomLoss = Math.log(V);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var d = '', i;
      for (i = 0; i <= 120; i++) { var l = LMAX * i / 120; d += (i ? 'L' : 'M') + px(l).toFixed(1) + ' ' + py(Math.exp(l)).toFixed(1) + ' '; }
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      // 随机基线：位于 loss = ln V 的竖线
      var rx = px(Math.min(LMAX, randomLoss));
      svg.appendChild(svgEl('line', { x1: rx, y1: PAD, x2: rx, y2: H - PAD,
        stroke: 'var(--warn,#b8870f)', 'stroke-width': '1.5', 'stroke-dasharray': '4 3' }));
      svg.appendChild(svgEl('circle', { cx: px(Math.min(LMAX, loss)), cy: py(pp), r: '5', fill: 'var(--blueprint,#3553ff)' }));
      num.innerHTML = (pp < 1000 ? pp.toFixed(pp < 10 ? 2 : 0) : human(pp)) + ' <small>perplexity</small>';
      meta.textContent = 'cross-entropy ' + loss.toFixed(2) + ' nats  ·  random over V = ' + human(V)
        + ' has loss ln V = ' + randomLoss.toFixed(2) + ', perplexity ' + human(V);
      formula.textContent = 'perplexity = e^loss   ·   a uniform guess over V tokens scores loss ln V and perplexity exactly V';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'loss', 'cross-entropy loss (nats)', 0.1, 7.0, 0.05),
      slider(state, 'logV', 'vocabulary V (10^x)', 2, 5.5, 0.1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['PERPLEXITY']), el('span', {}, ['drag the loss'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, num, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Perplexity is the exponential of cross-entropy loss, read as the effective number of equally likely choices the model is deciding between per token. A model that guessed uniformly over a vocabulary of size V would score perplexity exactly V (the orange line), so any useful model must land well below it.'])
    ]));
    state._render();
  }

  // ── continuous-batching：对比 GPU 槽位的静态填充与连续填充 ──────────────
  function continuousBatching(host) {
    var state = { mode: 'continuous', slots: 4 };
    var W = 520, H = 200, PAD = 26;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var bar = el('i');
    var barWrap = el('div', { class: 'lf-bar' }, [bar]);
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 到达槽位的请求长度（以步数计），取值确定
    var LENS = [3, 7, 2, 9, 4, 6, 5, 8, 3, 7];
    var STEPS = 12;
    state._render = function () {
      var S = state.slots;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var rowH = (H - 2 * PAD) / S, cw = (W - 2 * PAD) / STEPS;
      var busy = 0, total = S * STEPS;
      var queue = LENS.slice(S); // 前 S 个请求之后的剩余请求
      var qi = 0;
      var r;
      for (r = 0; r < S; r++) {
        var y = PAD + r * rowH + 2;
        var t = 0;
        var curLen = LENS[r];
        var start = 0;
        while (t < STEPS) {
          // 从起点开始运行当前请求 curLen 步
          var runEnd = Math.min(STEPS, start + curLen);
          var x = PAD + start * cw;
          svg.appendChild(svgEl('rect', { x: x.toFixed(1), y: y.toFixed(1),
            width: ((runEnd - start) * cw - 2).toFixed(1), height: (rowH - 4).toFixed(1), rx: '2',
            fill: 'var(--blueprint,#3553ff)', opacity: '0.85' }));
          busy += (runEnd - start);
          t = runEnd;
          if (state.mode === 'continuous' && qi < queue.length) {
            // 立即用队列中的下一个请求填充释放的槽位
            start = t; curLen = queue[qi++];
          } else {
            // 静态模式：槽位保持空闲，直至整个批次按最大长度运行完毕
            break;
          }
        }
      }
      var util;
      if (state.mode === 'static') {
        // 静态批处理持续运行，直到首批请求中最长的请求结束
        var maxLen = Math.max.apply(null, LENS.slice(0, S));
        var work = 0, k;
        for (k = 0; k < S; k++) { work += Math.min(STEPS, LENS[k]); }
        util = work / (S * Math.min(STEPS, maxLen));
        // 为静态模式绘制空闲尾段（灰色）
        for (r = 0; r < S; r++) {
          var ll = Math.min(STEPS, LENS[r]);
          var maxl = Math.min(STEPS, maxLen);
          if (ll < maxl) {
            var yy = PAD + r * rowH + 2;
            svg.appendChild(svgEl('rect', { x: (PAD + ll * cw).toFixed(1), y: yy.toFixed(1),
              width: ((maxl - ll) * cw - 2).toFixed(1), height: (rowH - 4).toFixed(1), rx: '2',
              fill: 'var(--rule-soft,#ccc)', opacity: '0.7' }));
          }
        }
      } else {
        util = busy / total;
      }
      var pct = Math.round(util * 100);
      num.innerHTML = pct + ' <small>% GPU utilization</small>';
      bar.style.width = pct + '%';
      barWrap.classList.toggle('over', pct < 60);
      meta.textContent = state.mode === 'continuous'
        ? 'finished slots are refilled from the queue immediately, so the batch stays full'
        : 'every slot waits for the longest request in the batch before any new one starts';
      formula.textContent = 'utilization = busy slot-steps / total slot-steps  ·  ' + S + ' slots over ' + STEPS + ' steps';
    };
    var sel = LF.select(state, 'mode', 'batching', [['continuous', 'continuous'], ['static', 'static']]);
    var grid = el('div', { class: 'lf-grid' }, [
      sel,
      slider(state, 'slots', 'GPU slots', 2, 6, 1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['CONTINUOUS BATCHING']), el('span', {}, ['toggle static vs continuous'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, num, barWrap, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Requests in a batch finish at different times because they generate different numbers of tokens. Static batching holds every slot until the longest request ends, leaving grey idle time. Continuous batching refills each freed slot from the queue the moment it opens, keeping the GPU full and lifting utilization.'])
    ]));
    state._render();
  }

  // ── image-patch-tokens：将图像拆分成 (size/patch)^2 个图块 token ─────────
  function imagePatchTokens(host) {
    var state = { size: 224, patch: 16 };
    var W = 520, H = 240, PAD = 16, BOX = 200;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    state._render = function () {
      var size = state.size, patch = state.patch;
      var perSide = Math.max(1, Math.ceil(size / patch));
      var n = perSide * perSide;
      var padded = perSide * patch;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var ox = PAD, oy = (H - BOX) / 2, cell = BOX / perSide;
      svg.appendChild(svgEl('rect', { x: ox, y: oy.toFixed(1), width: BOX, height: BOX,
        fill: 'var(--bg-surface,#eee)', stroke: 'var(--ink-soft,#555)', 'stroke-width': '1.5' }));
      var i;
      for (i = 1; i < perSide; i++) {
        var g = ox + i * cell;
        svg.appendChild(svgEl('line', { x1: g.toFixed(1), y1: oy.toFixed(1), x2: g.toFixed(1), y2: (oy + BOX).toFixed(1),
          stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '0.8', opacity: '0.7' }));
        var gy = oy + i * cell;
        svg.appendChild(svgEl('line', { x1: ox, y1: gy.toFixed(1), x2: (ox + BOX), y2: gy.toFixed(1),
          stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '0.8', opacity: '0.7' }));
      }
      // 右侧的一排 token 方块，为保证可读性而限制数量
      var tx = ox + BOX + 28, ty = oy, ts = 12, cols = 6;
      var shown = Math.min(n, 36);
      for (i = 0; i < shown; i++) {
        var cx = tx + (i % cols) * (ts + 3);
        var cy = ty + Math.floor(i / cols) * (ts + 3);
        svg.appendChild(svgEl('rect', { x: cx.toFixed(1), y: cy.toFixed(1), width: ts, height: ts, rx: '2',
          fill: 'var(--blueprint,#3553ff)', opacity: '0.8' }));
      }
      num.innerHTML = LF.fmtInt(n) + ' <small>patch tokens</small>';
      meta.textContent = perSide + ' x ' + perSide + ' grid  ·  each ' + patch + ' x ' + patch
        + ' px patch becomes one token' + (padded !== size ? ' · image padded to ' + padded + 'px' : '') + ' (plus a CLS token in ViT)';
      formula.textContent = 'tokens = ⌈size / patch⌉² = ⌈' + size + ' / ' + patch + '⌉² = ' + perSide + '² = ' + n;
    };
    var grid = el('div', { class: 'lf-grid' }, [
      LF.select(state, 'size', 'image size (px)', [['224', 224], ['256', 256], ['336', 336], ['384', 384], ['448', 448]]),
      LF.select(state, 'patch', 'patch size (px)', [['8', 8], ['14', 14], ['16', 16], ['32', 32]])
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['IMAGE PATCH TOKENS']), el('span', {}, ['pick image and patch size'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, num, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['A Vision Transformer cuts the image into a grid of fixed-size patches and treats each patch as one token, exactly like a word in text. The token count is the square of size over patch, so halving the patch size quadruples the sequence and the attention cost.'])
    ]));
    state._render();
  }

  // ── multimodal-fusion：两个编码器映射到共享空间，对比早期融合与晚期融合 ─
  function multimodalFusion(host) {
    var state = { mode: 'late' };
    var W = 520, H = 230, PAD = 20;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function box(x, y, w, h, label, fill) {
      var g = svgEl('g', {}, []);
      g.appendChild(svgEl('rect', { x: x, y: y, width: w, height: h, rx: '3',
        fill: fill || 'var(--bg-surface,#eee)', stroke: 'var(--ink-soft,#555)', 'stroke-width': '1.2' }));
      var t = svgEl('text', { x: (x + w / 2).toFixed(1), y: (y + h / 2 + 4).toFixed(1),
        'text-anchor': 'middle', 'font-family': 'monospace', 'font-size': '11', fill: 'var(--ink,#1a1a1a)' });
      t.appendChild(document.createTextNode(label));
      g.appendChild(t);
      return g;
    }
    function arrow(x1, y1, x2, y2) {
      return svgEl('line', { x1: x1, y1: y1, x2: x2, y2: y2,
        stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '1.6' });
    }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var imgY = 36, txtY = 150, colW = 92, colH = 36, h2 = 18;
      // 输入
      svg.appendChild(box(PAD, imgY, colW, colH, 'image', 'var(--bg,#fafaf5)'));
      svg.appendChild(box(PAD, txtY, colW, colH, 'text', 'var(--bg,#fafaf5)'));
      // 编码器
      var encX = PAD + colW + 40;
      svg.appendChild(box(encX, imgY, colW, colH, 'img enc'));
      svg.appendChild(box(encX, txtY, colW, colH, 'txt enc'));
      svg.appendChild(arrow(PAD + colW, imgY + h2, encX, imgY + h2));
      svg.appendChild(arrow(PAD + colW, txtY + h2, encX, txtY + h2));
      // 投影到共享空间
      var projX = encX + colW + 40;
      if (state.mode === 'late') {
        // 两者各自独立投影；融合发生在最后比较两个向量时
        svg.appendChild(box(projX, imgY, colW, colH, 'proj'));
        svg.appendChild(box(projX, txtY, colW, colH, 'proj'));
        svg.appendChild(arrow(encX + colW, imgY + h2, projX, imgY + h2));
        svg.appendChild(arrow(encX + colW, txtY + h2, projX, txtY + h2));
        var sx = projX + colW + 30, sy = (imgY + txtY) / 2;
        svg.appendChild(box(sx, sy, 70, colH, 'shared', 'var(--bg-surface,#eee)'));
        svg.appendChild(arrow(projX + colW, imgY + h2, sx, sy + 6));
        svg.appendChild(arrow(projX + colW, txtY + h2, sx, sy + colH - 6));
      } else {
        // 早期融合：将 token 拼接成单个序列并联合建模
        var fy = (imgY + txtY) / 2;
        svg.appendChild(box(projX, fy, 80, colH, 'concat', 'var(--bg-surface,#eee)'));
        svg.appendChild(arrow(encX + colW, imgY + h2, projX, fy + 8));
        svg.appendChild(arrow(encX + colW, txtY + h2, projX, fy + colH - 8));
        var jx = projX + 80 + 30;
        svg.appendChild(box(jx, fy, 78, colH, 'joint xfmr'));
        svg.appendChild(arrow(projX + 80, fy + h2, jx, fy + h2));
      }
      meta.textContent = state.mode === 'late'
        ? 'late fusion: encode each modality separately, project into one space, compare at the end (CLIP-style)'
        : 'early fusion: interleave image and text tokens into one sequence and model them jointly';
      formula.textContent = state.mode === 'late'
        ? 'sim = cos( proj(img enc(image)), proj(txt enc(text)) )'
        : 'joint = transformer( [ img tokens ; text tokens ] )';
    };
    var grid = el('div', {}, [LF.select(state, 'mode', 'fusion point', [['late fusion', 'late'], ['early fusion', 'early']])]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['MULTIMODAL FUSION']), el('span', {}, ['toggle early vs late'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['An image encoder and a text encoder each map their input into vectors. Late fusion projects both into a shared embedding space and compares them only at the end, which is how contrastive models like CLIP align images and captions. Early fusion concatenates the token streams and models them jointly from the start, letting the two modalities attend to each other throughout.'])
    ]));
    state._render();
  }

  // ── mcp-tool-call：客户端与服务器之间的 JSON-RPC 往返，结果写入上下文 ──
  function mcpToolCall(host) {
    var state = { step: 2 };
    var W = 520, H = 250, PAD = 18;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var code = el('div', { class: 'lf-formula' });
    var meta = el('div', { class: 'lf-meta' });
    var STEPS = [
      'tools/list: client asks the server which functions exist',
      'server returns the registry of available functions and their schemas',
      'tools/call: client invokes get_weather with arguments',
      'server runs the function and returns the result',
      'result is appended to the context and the model continues'
    ];
    var CODE = [
      '--> { "jsonrpc": "2.0", "id": 1, "method": "tools/list" }',
      '<-- { "result": { "tools": [ { "name": "get_weather", ... } ] } }',
      '--> { "jsonrpc": "2.0", "id": 2, "method": "tools/call",\n      "params": { "name": "get_weather", "arguments": { "city": "Pune" } } }',
      '<-- { "id": 2, "result": { "content": [ { "type": "text", "text": "31 C, clear" } ] } }',
      'context += tool result  ->  model writes the final answer'
    ];
    function box(x, y, w, h, label, active) {
      var g = svgEl('g', {});
      g.appendChild(svgEl('rect', { x: x, y: y, width: w, height: h, rx: '4',
        fill: active ? 'var(--blueprint,#3553ff)' : 'var(--bg-surface,#eee)',
        stroke: 'var(--ink-soft,#555)', 'stroke-width': '1.2' }));
      var t = svgEl('text', { x: (x + w / 2).toFixed(1), y: (y + h / 2 + 4).toFixed(1),
        'text-anchor': 'middle', 'font-family': 'monospace', 'font-size': '11',
        fill: active ? 'var(--bg,#fafaf5)' : 'var(--ink,#1a1a1a)' });
      t.appendChild(document.createTextNode(label));
      g.appendChild(t);
      return g;
    }
    state._render = function () {
      var s = state.step;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var bw = 120, bh = 38;
      var clientX = PAD, serverX = W - PAD - bw, midY = 30;
      var clientActive = (s === 0 || s === 2 || s === 4);
      var serverActive = (s === 1 || s === 3);
      svg.appendChild(box(clientX, midY, bw, bh, 'client / host', clientActive));
      svg.appendChild(box(serverX, midY, bw, bh, 'MCP server', serverActive));
      // 服务器下方的注册表
      svg.appendChild(box(serverX, midY + bh + 16, bw, 30, 'fn registry', s === 1));
      // 客户端下方的上下文
      svg.appendChild(box(clientX, midY + bh + 16, bw, 30, 'model context', s === 4));
      // 二者之间的消息箭头
      var ay = midY + bh + 92;
      var goingRight = (s === 0 || s === 2);
      var x1 = clientX + bw, x2 = serverX;
      if (goingRight) {
        svg.appendChild(svgEl('line', { x1: x1, y1: ay, x2: x2, y2: ay, stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
        svg.appendChild(svgEl('polygon', { points: (x2) + ',' + ay + ' ' + (x2 - 9) + ',' + (ay - 5) + ' ' + (x2 - 9) + ',' + (ay + 5), fill: 'var(--blueprint,#3553ff)' }));
      } else if (s === 1 || s === 3) {
        svg.appendChild(svgEl('line', { x1: x2, y1: ay, x2: x1, y2: ay, stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
        svg.appendChild(svgEl('polygon', { points: (x1) + ',' + ay + ' ' + (x1 + 9) + ',' + (ay - 5) + ' ' + (x1 + 9) + ',' + (ay + 5), fill: 'var(--blueprint,#3553ff)' }));
      }
      var dir = svgEl('text', { x: (W / 2).toFixed(1), y: (ay - 10).toFixed(1), 'text-anchor': 'middle',
        'font-family': 'monospace', 'font-size': '10', fill: 'var(--ink-mute,#777)' });
      dir.appendChild(document.createTextNode(goingRight ? 'request -->' : (s === 4 ? 'result feeds back' : '<-- response')));
      svg.appendChild(dir);
      // 步骤标签
      var lbl = svgEl('text', { x: (W / 2).toFixed(1), y: (H - 14).toFixed(1), 'text-anchor': 'middle',
        'font-family': 'monospace', 'font-size': '10.5', fill: 'var(--ink-soft,#555)' });
      lbl.appendChild(document.createTextNode((s + 1) + ' / ' + STEPS.length + '  ' + STEPS[s]));
      svg.appendChild(lbl);
      code.textContent = CODE[s];
      meta.textContent = 'JSON-RPC 2.0 over the transport  ·  the result becomes a context message the model reads next';
    };
    var grid = el('div', {}, [slider(state, 'step', 'round-trip step', 0, 4, 1)]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['MCP TOOL CALL']), el('span', {}, ['drag through the round trip'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, code, meta])]),
      el('div', { class: 'lf-cap' }, ['The Model Context Protocol speaks JSON-RPC between a client and a server. The client first lists the functions the server exposes, then calls one by name with arguments. The server runs it and returns a structured result, which the client appends to the model context so the next generation step can use it.'])
    ]));
    state._render();
  }

  LF.register({
    'beam-search': beamSearch,
    'speculative-decoding': speculativeDecoding,
    'moe-routing': moeRouting,
    'context-window-slide': contextWindowSlide,
    'perplexity-loss': perplexityLoss,
    'continuous-batching': continuousBatching,
    'image-patch-tokens': imagePatchTokens,
    'multimodal-fusion': multimodalFusion,
    'mcp-tool-call': mcpToolCall
  });
})();
