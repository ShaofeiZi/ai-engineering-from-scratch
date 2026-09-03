/* figures-dl.js — 第三阶段（深度学习核心）的交互式课程图示。
   在 lesson-figures.js 之后加载；通过 LF.register 注册九个 widget。
   无依赖，仅 ES5，主题通过 CSS 变量。 */
(function () {
  'use strict';
  var LF = window.LF;
  if (!LF) { return; }
  var el = LF.el, svgEl = LF.svgEl, slider = LF.slider, select = LF.select, clamp = LF.clamp;

  // ── perceptron-boundary：拖动权重，移动决策线 ──────────
  function perceptronBoundary(host) {
    // 两个线性可分的簇（确定性，数据空间 x,y 在 [-3,3]）。
    var pos = [[1.4, 1.2], [2.0, 0.6], [1.0, 2.1], [2.4, 1.7], [0.7, 1.0], [1.8, 2.4]];
    var neg = [[-1.3, -1.0], [-2.0, -0.5], [-0.8, -1.8], [-2.3, -1.6], [-0.6, -0.7], [-1.7, -2.2]];
    var state = { w1: 1, w2: 1, b: 0 };
    var W = 520, H = 230, PAD = 28, RNG = 3;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function px(x) { return PAD + (x + RNG) / (2 * RNG) * (W - 2 * PAD); }
    function py(y) { return H - PAD - (y + RNG) / (2 * RNG) * (H - 2 * PAD); }
    function score(p) { return state.w1 * p[0] + state.w2 * p[1] + state.b; }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      // 坐标轴
      svg.appendChild(svgEl('line', { x1: px(-RNG), y1: py(0), x2: px(RNG), y2: py(0), stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      svg.appendChild(svgEl('line', { x1: px(0), y1: py(-RNG), x2: px(0), y2: py(RNG), stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      // 决策线 w1 x + w2 y + b = 0 → y = -(w1 x + b)/w2（或垂直）
      if (Math.abs(state.w2) > 1e-6) {
        var xa = -RNG, xb = RNG;
        var ya = -(state.w1 * xa + state.b) / state.w2;
        var yb = -(state.w1 * xb + state.b) / state.w2;
        svg.appendChild(svgEl('line', { x1: px(xa), y1: py(ya), x2: px(xb), y2: py(yb), stroke: 'var(--warn,#b8870f)', 'stroke-width': '2' }));
      } else if (Math.abs(state.w1) > 1e-6) {
        var xv = -state.b / state.w1;
        svg.appendChild(svgEl('line', { x1: px(xv), y1: py(-RNG), x2: px(xv), y2: py(RNG), stroke: 'var(--warn,#b8870f)', 'stroke-width': '2' }));
      }
      var miss = 0;
      pos.forEach(function (p) {
        var ok = score(p) > 0;
        if (!ok) miss++;
        svg.appendChild(svgEl('circle', { cx: px(p[0]), cy: py(p[1]), r: '5', fill: ok ? 'var(--blueprint,#3553ff)' : 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      });
      neg.forEach(function (p) {
        var ok = score(p) < 0;
        if (!ok) miss++;
        svg.appendChild(svgEl('rect', { x: px(p[0]) - 4, y: py(p[1]) - 4, width: '8', height: '8', fill: ok ? 'var(--ink-mute,#999)' : 'none', stroke: 'var(--ink-mute,#999)', 'stroke-width': '2' }));
      });
      var total = pos.length + neg.length;
      status.innerHTML = miss + ' <small>of ' + total + ' misclassified</small>';
      meta.textContent = miss === 0 ? 'all points correct: this line separates the two classes' : 'filled = correct, hollow = wrong side of the line';
      formula.textContent = 'predict + when  ' + state.w1.toFixed(1) + '·x + ' + state.w2.toFixed(1) + '·y + (' + state.b.toFixed(1) + ') > 0';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'w1', 'weight w1', -3, 3, 0.1),
      slider(state, 'w2', 'weight w2', -3, 3, 0.1),
      slider(state, 'b', 'bias b', -3, 3, 0.1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['PERCEPTRON BOUNDARY']), el('span', {}, ['drag the weights'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [status]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['A perceptron predicts by the sign of w·x + b, so its decision surface is a straight line. Drag the weights and bias to rotate and shift that line until every blue circle sits on the positive side and every grey square on the negative one.'])
    ]));
    state._render();
  }

  // ── mlp-forward：拖动输入，观察 2-3-1 网络激活 ───────────────────
  function mlpForward(host) {
    // 固定权重：W1 为 3x2，b1 长度 3；w2 长度 3，b2 为标量。tanh 隐藏层 + 输出。
    var W1 = [[1.2, -0.8], [-0.5, 1.4], [0.9, 0.7]], b1 = [0.1, -0.2, 0.0];
    var w2 = [1.1, -1.3, 0.8], b2 = 0.2;
    var state = { x1: 0.6, x2: -0.4 };
    var W = 520, H = 230;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function tanh(z) { var e = Math.exp(2 * z); return (e - 1) / (e + 1); }
    function actFill(a) { // a 在 [-1,1] → blueprint 于 +1，bg 于 -1
      var t = (a + 1) / 2;
      return 'rgba(53,83,255,' + (0.12 + 0.78 * t).toFixed(3) + ')';
    }
    var inX = 90, hidX = 260, outX = 430;
    var inY = [80, 150], hidY = [55, 115, 175], outY = 115;
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var x = [state.x1, state.x2];
      var h = [0, 0, 0], j, i;
      for (j = 0; j < 3; j++) { var z = b1[j]; for (i = 0; i < 2; i++) z += W1[j][i] * x[i]; h[j] = tanh(z); }
      var zo = b2; for (j = 0; j < 3; j++) zo += w2[j] * h[j];
      var out = tanh(zo);
      // 输入→隐藏层的边
      for (j = 0; j < 3; j++) for (i = 0; i < 2; i++) {
        var wgt = W1[j][i];
        svg.appendChild(svgEl('line', { x1: inX, y1: inY[i], x2: hidX, y2: hidY[j], stroke: wgt >= 0 ? 'var(--blueprint,#3553ff)' : 'var(--warn,#b8870f)', 'stroke-width': (0.4 + Math.abs(wgt)).toFixed(2), opacity: '0.45' }));
      }
      // 隐藏层→输出层的边
      for (j = 0; j < 3; j++) {
        svg.appendChild(svgEl('line', { x1: hidX, y1: hidY[j], x2: outX, y2: outY, stroke: w2[j] >= 0 ? 'var(--blueprint,#3553ff)' : 'var(--warn,#b8870f)', 'stroke-width': (0.4 + Math.abs(w2[j])).toFixed(2), opacity: '0.45' }));
      }
      // 节点：输入（原始，缩放到 [-1,1] 作为填充提示）、隐藏、输出
      [0, 1].forEach(function (i2) {
        svg.appendChild(svgEl('circle', { cx: inX, cy: inY[i2], r: '15', fill: actFill(clamp(x[i2], -1, 1)), stroke: 'var(--ink-soft,#555)', 'stroke-width': '1.2' }));
      });
      h.forEach(function (hv, j2) {
        svg.appendChild(svgEl('circle', { cx: hidX, cy: hidY[j2], r: '15', fill: actFill(hv), stroke: 'var(--ink-soft,#555)', 'stroke-width': '1.2' }));
      });
      svg.appendChild(svgEl('circle', { cx: outX, cy: outY, r: '18', fill: actFill(out), stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      num.innerHTML = out.toFixed(3) + ' <small>output</small>';
      meta.textContent = 'hidden = [' + h.map(function (v) { return v.toFixed(2); }).join(', ') + ']  ·  darker node = stronger activation';
      formula.textContent = 'h = tanh(W₁x + b₁),  y = tanh(w₂·h + b₂)';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'x1', 'input x1', -2, 2, 0.05),
      slider(state, 'x2', 'input x2', -2, 2, 0.05)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['MLP FORWARD PASS']), el('span', {}, ['drag the two inputs'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [num]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Two inputs feed three hidden units through fixed weights, each squashed by tanh, then combine into one output. Blue edges are positive weights, gold are negative; node shading shows how strongly each unit fires for the inputs you set.'])
    ]));
    state._render();
  }

  // ── backprop-vanishing：激活导数沿深度的乘积 ─────
  function backpropVanishing(host) {
    var state = { act: 'sigmoid', depth: 10 };
    var W = 520, H = 220, PAD = 34;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 每层导数幅度的代表值（典型中间激活区）
    function dPerLayer() {
      if (state.act === 'sigmoid') return 0.25;   // sigmoid'(x) 最大值 = 0.25
      if (state.act === 'tanh') return 0.42;       // 远离 0 处的典型 |tanh'|
      return 1.0;                                  // relu 导数对激活单元 = 1
    }
    function px(layer) { return PAD + (state.depth <= 1 ? 0 : (layer / (state.depth)) * (W - 2 * PAD)); }
    function py(logmag) { // logmag 在 [-9, 0] → 底部..顶部
      var t = clamp((logmag + 9) / 9, 0, 1);
      return H - PAD - t * (H - 2 * PAD);
    }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(svgEl('line', { x1: PAD, y1: py(0), x2: W - PAD, y2: py(0), stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1', 'stroke-dasharray': '3 3' }));
      var per = dPerLayer(), mag = 1, d = '', l;
      var lastLog = 0;
      for (l = 0; l <= state.depth; l++) {
        var lg = l * Math.log(per) / Math.LN10; // l 层后的 mag 的 log10
        lastLog = lg;
        d += (l ? 'L' : 'M') + px(l).toFixed(1) + ' ' + py(lg).toFixed(1) + ' ';
      }
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      for (l = 0; l <= state.depth; l += Math.max(1, Math.round(state.depth / 10))) {
        svg.appendChild(svgEl('circle', { cx: px(l), cy: py(l * Math.log(per) / Math.LN10), r: '2.5', fill: 'var(--blueprint,#3553ff)' }));
      }
      mag = Math.pow(per, state.depth);
      status.innerHTML = mag < 1e-4 ? '≈ ' + mag.toExponential(1) + ' <small>gradient</small>' : mag.toFixed(4) + ' <small>gradient</small>';
      var verdict = state.act === 'relu' ? 'stable: derivative stays at 1, gradient survives depth'
        : (mag < 1e-3 ? 'vanished: gradient is too small to train early layers' : 'shrinking with depth');
      meta.textContent = 'per-layer factor ' + per.toFixed(2) + '  ·  after ' + state.depth + ' layers  ·  ' + verdict;
      formula.textContent = '∂L/∂early ∝ Π σ′(zₗ) ≈ (' + per.toFixed(2) + ')^depth   (log scale)';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      select(state, 'act', 'activation', [['sigmoid', 'sigmoid'], ['tanh', 'tanh'], ['relu', 'relu']]),
      slider(state, 'depth', 'depth (layers)', 2, 20, 1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['VANISHING GRADIENTS']), el('span', {}, ['pick activation, drag depth'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [status]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Backprop multiplies one activation derivative per layer. Sigmoid caps that derivative at 0.25 and tanh stays below one, so the product collapses toward zero in deep nets (note the log axis). ReLU keeps a derivative of one for active units, which is why it made deep training practical.'])
    ]));
    state._render();
  }

  // ── optimizer-trajectory：在病态条件碗上的 SGD vs Momentum vs Adam
  function optimizerTrajectory(host) {
    var state = { opt: 'momentum', lr: 0.08 };
    var W = 520, H = 230, PAD = 26, STEPS = 30;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // f(x,y) = 0.5*(a x^2 + b y^2)，沟谷：a 小，b 大 → 病态条件
    var A = 1.0, B = 20.0, X0 = -2.6, Y0 = 0.9;
    var RX = 3, RY = 1.2;
    function px(x) { return PAD + (x + RX) / (2 * RX) * (W - 2 * PAD); }
    function py(y) { return H / 2 - (y / RY) * (H / 2 - PAD); }
    function run() {
      var x = X0, y = Y0, pts = [[x, y]];
      var beta = 0.9, vx = 0, vy = 0;          // momentum / adam 一阶矩
      var b2 = 0.999, sx = 0, sy = 0, t = 0;   // adam 二阶矩
      var eps = 1e-8;
      for (var s = 0; s < STEPS; s++) {
        var gx = A * x, gy = B * y;
        if (state.opt === 'sgd') {
          x -= state.lr * gx; y -= state.lr * gy;
        } else if (state.opt === 'momentum') {
          vx = beta * vx + gx; vy = beta * vy + gy;
          x -= state.lr * vx; y -= state.lr * vy;
        } else { // adam
          t++;
          vx = beta * vx + (1 - beta) * gx; vy = beta * vy + (1 - beta) * gy;
          sx = b2 * sx + (1 - b2) * gx * gx; sy = b2 * sy + (1 - b2) * gy * gy;
          var mhx = vx / (1 - Math.pow(beta, t)), mhy = vy / (1 - Math.pow(beta, t));
          var shx = sx / (1 - Math.pow(b2, t)), shy = sy / (1 - Math.pow(b2, t));
          x -= state.lr * 8 * mhx / (Math.sqrt(shx) + eps);
          y -= state.lr * 8 * mhy / (Math.sqrt(shy) + eps);
        }
        if (!isFinite(x) || !isFinite(y) || Math.abs(x) > RX || Math.abs(y) > RY) { pts.push([clamp(x, -RX, RX), clamp(y, -RY, RY)]); break; }
        pts.push([x, y]);
      }
      return pts;
    }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      // 沟谷等高线（椭圆）
      [0.3, 0.7, 1.2].forEach(function (lvl) {
        svg.appendChild(svgEl('ellipse', { cx: px(0), cy: py(0), rx: (px(Math.sqrt(2 * lvl / A)) - px(0)).toFixed(1), ry: (py(0) - py(Math.sqrt(2 * lvl / B))).toFixed(1), fill: 'none', stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1' }));
      });
      svg.appendChild(svgEl('line', { x1: PAD, y1: py(0), x2: W - PAD, y2: py(0), stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1', 'stroke-dasharray': '3 3' }));
      var pts = run(), d = '';
      pts.forEach(function (p, i) { d += (i ? 'L' : 'M') + px(p[0]).toFixed(1) + ' ' + py(p[1]).toFixed(1) + ' '; });
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '1.6' }));
      pts.forEach(function (p, i) { if (i % 2 === 0 || i === pts.length - 1) svg.appendChild(svgEl('circle', { cx: px(p[0]), cy: py(p[1]), r: i === pts.length - 1 ? '5' : '2.4', fill: 'var(--blueprint,#3553ff)' })); });
      svg.appendChild(svgEl('circle', { cx: px(0), cy: py(0), r: '3', fill: 'var(--warn,#b8870f)' }));
      var last = pts[pts.length - 1];
      var dist = Math.sqrt(last[0] * last[0] + last[1] * last[1]);
      status.innerHTML = '‖θ − θ*‖ = ' + dist.toFixed(3);
      meta.textContent = 'gold dot is the minimum  ·  ' + (state.opt === 'sgd' ? 'plain SGD zig-zags across the steep ravine wall' : state.opt === 'momentum' ? 'momentum averages out the zig-zag and rolls down the valley' : 'Adam rescales each axis, so the steep and flat directions advance together');
      formula.textContent = 'f(x,y) = ½(x² + 20y²)   condition number 20  ·  ' + STEPS + ' steps';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      select(state, 'opt', 'optimizer', [['SGD', 'sgd'], ['Momentum', 'momentum'], ['Adam', 'adam']]),
      slider(state, 'lr', 'learning rate', 0.01, 0.18, 0.005)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['OPTIMIZER TRAJECTORY']), el('span', {}, ['pick optimizer, drag lr'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [status]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['The loss is a narrow ravine: gentle along x, twenty times steeper along y. Plain SGD bounces across the steep walls and crawls down the valley. Momentum smooths the bounce; Adam normalizes each direction so both axes converge at a similar rate.'])
    ]));
    state._render();
  }

  // ── weight-init-variance：三种方案的激活标准差沿深度变化 ────
  function weightInitVariance(host) {
    var state = { scheme: 'xavier', fanin: 256 };
    var L = 10;
    var W = 520, H = 220, PAD = 34;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 线性/tanh 堆叠的方差递推：var_out = n * w_var * var_in。
    // 增益 g = n * w_var。朴素：w_var = 1（g = n，爆炸）。xavier：w_var=1/n（g≈1）。
    // he：w_var=2/n 配 relu 折半 → 有效 g≈1。
    function gain() {
      var n = state.fanin;
      if (state.scheme === 'naive') return n * 1.0 / 50;        // 缩放以使其可见地增长
      if (state.scheme === 'xavier') return n * (1.0 / n);      // = 1
      return 0.5 * n * (2.0 / n);                               // he 配 relu 折半 = 1
    }
    function px(l) { return PAD + l / L * (W - 2 * PAD); }
    function py(logstd) { // log10(std) 在 [-4,4]
      var t = clamp((logstd + 4) / 8, 0, 1);
      return H - PAD - t * (H - 2 * PAD);
    }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(svgEl('line', { x1: PAD, y1: py(0), x2: W - PAD, y2: py(0), stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1', 'stroke-dasharray': '3 3' }));
      var g = gain(), varc = 1, d = '', l, lastStd = 1;
      for (l = 0; l <= L; l++) {
        var std = Math.sqrt(varc);
        lastStd = std;
        d += (l ? 'L' : 'M') + px(l).toFixed(1) + ' ' + py(Math.log(std) / Math.LN10).toFixed(1) + ' ';
        varc *= g;
      }
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      for (l = 0; l <= L; l++) { var v = Math.pow(g, l); svg.appendChild(svgEl('circle', { cx: px(l), cy: py(Math.log(Math.sqrt(v)) / Math.LN10), r: '2.6', fill: 'var(--blueprint,#3553ff)' })); }
      status.innerHTML = lastStd < 1e-3 ? '≈ ' + lastStd.toExponential(1) + ' <small>std @ L10</small>' : lastStd.toFixed(lastStd < 10 ? 2 : 0) + ' <small>std @ L10</small>';
      var verdict = state.scheme === 'naive' ? 'exploding: activations blow up layer by layer'
        : 'stable: variance held near one across all ten layers';
      meta.textContent = 'per-layer gain ' + g.toFixed(2) + '  ·  ' + verdict;
      formula.textContent = state.scheme === 'naive' ? 'Var = 1 (too large)  →  gain = n·Var grows with width'
        : state.scheme === 'xavier' ? 'Var(w) = 1/n  →  gain ≈ 1' : 'Var(w) = 2/n  →  gain ≈ 1 after ReLU';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      select(state, 'scheme', 'init scheme', [['naive (large)', 'naive'], ['Xavier / Glorot', 'xavier'], ['He / Kaiming', 'he']]),
      slider(state, 'fanin', 'fan-in n', 64, 1024, 64)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['WEIGHT INIT VARIANCE']), el('span', {}, ['pick a scheme'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [status]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Each layer multiplies the activation variance by a gain of n·Var(w). Naive large weights make that gain grow with width, so activations explode (log axis). Xavier sets Var(w)=1/n and He sets 2/n for ReLU, both holding the gain near one so signal magnitude stays flat across depth.'])
    ]));
    state._render();
  }

  // ── dropout-mask：拖动 p，丢弃确定性比例的单元 ───────────
  function dropoutMask(host) {
    var state = { p: 0.3 };
    var N = 24;
    var W = 520, H = 200, COLS = 8, PAD = 24;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var rows = Math.ceil(N / COLS);
      var cw = (W - 2 * PAD) / COLS, ch = (H - 2 * PAD) / rows;
      var r = Math.min(cw, ch) / 2 - 5;
      var dropped = 0, i;
      var nDrop = Math.round(state.p * N);
      var dropSet = {};
      for (i = 0; i < nDrop; i++) { dropSet[Math.floor((i + 0.5) * N / Math.max(1, nDrop))] = true; }
      for (i = 0; i < N; i++) {
        var col = i % COLS, row = Math.floor(i / COLS);
        var cx = PAD + col * cw + cw / 2, cy = PAD + row * ch + ch / 2;
        var off = !!dropSet[i];
        if (off) dropped++;
        svg.appendChild(svgEl('circle', { cx: cx.toFixed(1), cy: cy.toFixed(1), r: r.toFixed(1), fill: off ? 'var(--rule-soft,#ddd)' : 'var(--blueprint,#3553ff)', stroke: off ? 'var(--rule-soft,#ccc)' : 'var(--blueprint,#3553ff)', 'stroke-width': '1', opacity: off ? '0.45' : '1' }));
      }
      var scale = 1 / (1 - Math.min(0.95, state.p));
      status.innerHTML = dropped + ' <small>of ' + N + ' dropped</small>';
      meta.textContent = 'kept units scaled by 1/(1−p) = ' + scale.toFixed(2) + ' so the expected sum is unchanged';
      formula.textContent = 'drop each unit with prob p = ' + state.p.toFixed(2) + ',  then divide survivors by (1 − p)';
    };
    var grid = el('div', {}, [slider(state, 'p', 'dropout rate p', 0, 0.9, 0.05)]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['DROPOUT MASK']), el('span', {}, ['drag the rate'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [status]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Dropout zeroes a fraction p of units each step so the network cannot lean on any single one. Because only the survivors pass signal, they are scaled up by 1/(1−p) to keep the expected activation the same, and at test time the full layer runs with no scaling.'])
    ]));
    state._render();
  }

  // ── batchnorm-effect：移动输入，观察 BN 将其重新中心化 ───────────────
  function batchnormEffect(host) {
    var state = { shift: 1.4, scaleIn: 1.8 };
    var W = 520, H = 220, PAD = 30;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    var RNG = 6;
    function px(x) { return PAD + (x + RNG) / (2 * RNG) * (W - 2 * PAD); }
    function py(v, peak) { return H - PAD - (v / peak) * (H - 2 * PAD); }
    function gauss(x, mu, sd) { return Math.exp(-0.5 * Math.pow((x - mu) / sd, 2)); }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(svgEl('line', { x1: px(0), y1: PAD, x2: px(0), y2: H - PAD, stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1', 'stroke-dasharray': '3 3' }));
      var muIn = state.shift, sdIn = Math.max(0.2, state.scaleIn);
      var i, d1 = '', d2 = '';
      // 激活前分布（平移并缩放）
      for (i = 0; i <= 140; i++) { var x = -RNG + 2 * RNG * i / 140; d1 += (i ? 'L' : 'M') + px(x).toFixed(1) + ' ' + py(gauss(x, muIn, sdIn), 1).toFixed(1) + ' '; }
      svg.appendChild(svgEl('path', { d: d1, fill: 'none', stroke: 'var(--ink-mute,#999)', 'stroke-width': '2' }));
      // BN 后：零均值，单位方差
      for (i = 0; i <= 140; i++) { var x2 = -RNG + 2 * RNG * i / 140; d2 += (i ? 'L' : 'M') + px(x2).toFixed(1) + ' ' + py(gauss(x2, 0, 1), 1).toFixed(1) + ' '; }
      svg.appendChild(svgEl('path', { d: d2, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      status.innerHTML = 'μ ' + muIn.toFixed(2) + ' → 0 <small>· σ ' + sdIn.toFixed(2) + ' → 1</small>';
      meta.textContent = 'grey is the raw pre-activation, blue is after batch norm  ·  recentred and rescaled every batch';
      formula.textContent = 'x̂ = (x − μ_B) / √(σ²_B + ε),  then  y = γ·x̂ + β';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'shift', 'input mean shift', -3, 3, 0.1),
      slider(state, 'scaleIn', 'input spread σ', 0.3, 3, 0.1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['BATCH NORM']), el('span', {}, ['drag the input shift'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [status]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Whatever mean and spread the layer below sends up (grey), batch norm subtracts the batch mean and divides by the batch standard deviation, snapping the distribution to zero mean and unit variance (blue). Learnable γ and β then let the network re-stretch it if a different scale is useful.'])
    ]));
    state._render();
  }

  // ── learning-curves：容量 vs 训练/验证损失，标记早停 ───────
  function learningCurves(host) {
    var state = { cap: 6 };
    var W = 520, H = 230, PAD = 34, CMAX = 14;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 训练单调下降；验证呈 U 形（偏差项下降，方差项上升）
    function train(c) { return 0.3 + 4.5 / (c + 0.5); }
    function val(c) { return 4.5 / (c + 0.5) + 0.11 * c + 0.45; }
    var best = 1, bv = 1e9, c;
    for (c = 1; c <= CMAX; c++) { if (val(c) < bv) { bv = val(c); best = c; } }
    var YMAX = Math.max(val(1), train(1), val(CMAX)) + 0.4;
    function px(c2) { return PAD + (c2 - 1) / (CMAX - 1) * (W - 2 * PAD); }
    function py(y) { return H - PAD - (y / YMAX) * (H - 2 * PAD); }
    function curve(fn, stroke) { var d = '', i; for (i = 0; i <= 80; i++) { var x = 1 + (CMAX - 1) * i / 80; d += (i ? 'L' : 'M') + px(x).toFixed(1) + ' ' + py(fn(x)).toFixed(1) + ' '; } return svgEl('path', { d: d, fill: 'none', stroke: stroke, 'stroke-width': '2' }); }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(svgEl('line', { x1: px(best), y1: PAD, x2: px(best), y2: H - PAD, stroke: 'var(--warn,#b8870f)', 'stroke-width': '1.5', 'stroke-dasharray': '4 3' }));
      svg.appendChild(curve(train, 'var(--ink-mute,#999)'));
      svg.appendChild(curve(val, 'var(--blueprint,#3553ff)'));
      svg.appendChild(svgEl('circle', { cx: px(state.cap), cy: py(val(state.cap)), r: '5', fill: 'var(--blueprint,#3553ff)' }));
      svg.appendChild(svgEl('circle', { cx: px(state.cap), cy: py(train(state.cap)), r: '4', fill: 'var(--ink-mute,#999)' }));
      var gap = val(state.cap) - train(state.cap);
      status.innerHTML = 'gap ' + gap.toFixed(2) + ' <small>· ' + (state.cap < best ? 'underfit' : state.cap > best ? 'overfit' : 'best') + '</small>';
      meta.textContent = 'train ' + train(state.cap).toFixed(2) + '  ·  val ' + val(state.cap).toFixed(2) + '  ·  early stop at capacity ' + best + ' (gold line)';
      formula.textContent = 'train loss falls with capacity; val loss is U-shaped; stop where val bottoms out';
    };
    var grid = el('div', {}, [slider(state, 'cap', 'model capacity / epochs', 1, CMAX, 1)]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['LEARNING CURVES']), el('span', {}, ['drag capacity'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [status]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Grey is training loss, blue is validation loss. More capacity always lowers training loss, but validation loss bottoms out and then climbs as the model starts memorizing noise. The widening gap is the overfit signal; the gold line marks where early stopping would freeze the model.'])
    ]));
    state._render();
  }

  // ── gradient-clipping：通过限制范数驯服爆炸的更新 ────────
  function gradientClipping(host) {
    var state = { thresh: 1.0, norm: 4.0 };
    var W = 520, H = 200, PAD = 32, GMAX = 8;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var bar = el('i');
    var barWrap = el('div', { class: 'lf-bar' }, [bar]);
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function px(g) { return PAD + g / GMAX * (W - 2 * PAD); }
    function py(g) { return H - PAD - g / GMAX * (H - 2 * PAD); }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      // 恒等线 y = x（阈值前的裁剪输出）
      svg.appendChild(svgEl('line', { x1: px(0), y1: py(0), x2: px(GMAX), y2: py(GMAX), stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1', 'stroke-dasharray': '3 3' }));
      // 裁剪响应：out = min(g, thresh)
      var t = state.thresh;
      var d = 'M' + px(0) + ' ' + py(0) + ' L' + px(t) + ' ' + py(t) + ' L' + px(GMAX) + ' ' + py(t);
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      // 阈值标记
      svg.appendChild(svgEl('line', { x1: px(t), y1: PAD, x2: px(t), y2: H - PAD, stroke: 'var(--warn,#b8870f)', 'stroke-width': '1', 'stroke-dasharray': '2 3' }));
      var clipped = Math.min(state.norm, t);
      var raw = state.norm;
      // 当前点
      svg.appendChild(svgEl('circle', { cx: px(raw), cy: py(clipped), r: '5', fill: 'var(--blueprint,#3553ff)' }));
      var scale = raw > t ? t / raw : 1;
      status.innerHTML = clipped.toFixed(2) + ' <small>clipped norm</small>';
      bar.style.width = Math.min(100, clipped / GMAX * 100) + '%';
      barWrap.classList.toggle('over', raw > t);
      meta.textContent = raw > t ? 'exploding: raw norm ' + raw.toFixed(2) + ' scaled by ' + scale.toFixed(2) + ' down to the cap'
        : 'within budget: gradient passes through unchanged';
      formula.textContent = 'if ‖g‖ > τ:  g ← g · τ / ‖g‖   →   clipped = min(‖g‖, τ) = min(' + raw.toFixed(1) + ', ' + t.toFixed(1) + ')';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'thresh', 'clip threshold τ', 0.2, 6, 0.1),
      slider(state, 'norm', 'raw gradient norm', 0.2, 8, 0.1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['GRADIENT CLIPPING']), el('span', {}, ['drag threshold and norm'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [status]), barWrap, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['When a gradient norm spikes, a single step can throw the weights off a cliff. Clipping rescales any gradient whose norm exceeds the threshold τ back down to τ, keeping direction but capping magnitude. Below τ the gradient is untouched; above it the update is tamed to min(‖g‖, τ).'])
    ]));
    state._render();
  }

  LF.register({
    'perceptron-boundary': perceptronBoundary,
    'mlp-forward': mlpForward,
    'backprop-vanishing': backpropVanishing,
    'optimizer-trajectory': optimizerTrajectory,
    'weight-init-variance': weightInitVariance,
    'dropout-mask': dropoutMask,
    'batchnorm-effect': batchnormEffect,
    'learning-curves': learningCurves,
    'gradient-clipping': gradientClipping
  });
})();
