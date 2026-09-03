/* figures-math2.js - 交互式数学基础小部件（阶段 01）。
   在 lesson-figures.js 之后加载，并通过 window.LF 注册。原生 ES5，
   无依赖，主题通过 CSS 变量定义。每个小部件确定性渲染。 */
(function () {
  'use strict';
  var LF = window.LF;
  if (!LF) { return; }
  var el = LF.el, svgEl = LF.svgEl, slider = LF.slider, select = LF.select;
  var fmtInt = LF.fmtInt, clamp = LF.clamp;

  // ── svd-rank-reconstruction：保留 k 个奇异值，观察能量回归 ──
  function svdRank(host) {
    // 一个固定的 8x8 图案。其奇异值是预置的（递减），因此
    // 保留能量 = sum(top-k sigma^2) / sum(all sigma^2) 是精确的。
    var sigma = [9.0, 5.4, 3.1, 1.8, 1.0, 0.55, 0.28, 0.12];
    var n = sigma.length;
    var total = 0, i;
    for (i = 0; i < n; i++) { total += sigma[i] * sigma[i]; }
    var state = { k: 2 };
    var W = 520, H = 230, PAD = 30, CELL = 18, GX = 360;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var bar = el('i');
    var barWrap = el('div', { class: 'lf-bar' }, [bar]);
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function cell(r, c) {
      // 平滑的、利于低秩重建的目标强度，取值于 [0,1]
      return 0.5 + 0.5 * Math.cos((r + c) * Math.PI / (n - 1));
    }
    state._render = function () {
      while (svg.firstChild) { svg.removeChild(svg.firstChild); }
      var keep = state.k, energy = 0, j;
      for (j = 0; j < keep; j++) { energy += sigma[j] * sigma[j]; }
      var frac = energy / total;
      // 重建质量随保留能量提升：将单元格向灰色混合
      var r, c;
      for (r = 0; r < n; r++) {
        for (c = 0; c < n; c++) {
          var v = cell(r, c);
          var approx = 0.5 + (v - 0.5) * frac;
          var g = Math.round(clamp(approx, 0, 1) * 255);
          svg.appendChild(svgEl('rect', {
            x: PAD + c * CELL, y: PAD + r * CELL, width: CELL - 1, height: CELL - 1,
            fill: 'rgb(' + g + ',' + g + ',' + g + ')'
          }));
        }
      }
      // 右侧的奇异值谱条形图
      var maxS = sigma[0], bw = 14, sx = GX;
      for (j = 0; j < n; j++) {
        var bh = sigma[j] / maxS * 120;
        var on = j < keep;
        svg.appendChild(svgEl('rect', {
          x: sx + j * (bw + 4), y: PAD + 120 - bh, width: bw, height: bh,
          fill: on ? 'var(--blueprint,#3553ff)' : 'var(--rule-soft,#ddd)'
        }));
      }
      num.innerHTML = (frac * 100).toFixed(1) + ' <small>% energy</small>';
      bar.style.width = (frac * 100).toFixed(1) + '%';
      meta.textContent = 'rank ' + keep + ' of ' + n + '  ·  stores ' + (keep * (2 * n + 1)) +
        ' numbers vs ' + (n * n) + ' full  ·  blue bars are the kept singular values';
      formula.textContent = 'A_k = sum_{i<k} sigma_i u_i v_iT   ·   energy = sum top-k sigma^2 / sum all sigma^2';
    };
    var grid = el('div', {}, [slider(state, 'k', 'singular values kept (k)', 1, n, 1)]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['SVD LOW-RANK']), el('span', {}, ['drag the rank k'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [num]), barWrap, meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Keeping the k largest singular values gives the best rank-k approximation of a matrix. The first few capture most of the energy, so a low rank reconstructs the pattern almost exactly while storing far fewer numbers. That is the whole idea behind compression and low-rank adapters.'])
    ]));
    state._render();
  }

  // ── tensor-broadcast：两个形状的尾部维度是否对齐？ ──────────────────
  function tensorBroadcast(host) {
    var state = { a0: 8, a1: 1, a2: 3, b0: 1, b1: 4, b2: 3 };
    var rows = el('div', {});
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    state._render = function () {
      var a = [state.a0, state.a1, state.a2];
      var b = [state.b0, state.b1, state.b2];
      var out = [], ok = true, i, why = '';
      for (i = 0; i < 3; i++) {
        var x = a[i], y = b[i];
        if (x === y) { out.push(x); }
        else if (x === 1) { out.push(y); }
        else if (y === 1) { out.push(x); }
        else { ok = false; out.push('x'); if (!why) { why = 'dim ' + i + ': ' + x + ' vs ' + y + ' (neither is 1)'; } }
      }
      while (rows.firstChild) { rows.removeChild(rows.firstChild); }
      function shapeRow(label, vals, hi) {
        var cells = [];
        vals.forEach(function (v, idx) {
          var stretched = hi && (a[idx] === 1 || b[idx] === 1) && a[idx] !== b[idx] && (label !== 'result');
          cells.push(el('span', {
            class: 'lf-formula',
            style: 'display:inline-block;min-width:34px;text-align:center;padding:4px 6px;margin:2px;border:1px solid var(--rule-soft,#ddd);color:' +
              (v === 'x' ? 'var(--warn,#b8870f)' : 'var(--ink,#1a1a1a)')
          }, [String(v)]));
        });
        return el('div', { style: 'display:flex;align-items:center;gap:8px;margin:4px 0' },
          [el('span', { class: 'lf-meta', style: 'min-width:62px' }, [label])].concat(cells));
      }
      rows.appendChild(shapeRow('shape A', a, true));
      rows.appendChild(shapeRow('shape B', b, true));
      rows.appendChild(shapeRow('result', out, false));
      status.innerHTML = ok ? 'broadcasts' : 'mismatch';
      meta.textContent = ok ? 'result shape (' + out.join(', ') + ')  ·  a 1 stretches to match the other'
        : 'cannot broadcast  ·  ' + why;
      formula.textContent = 'align trailing dims; each pair must be equal or one of them 1';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'a0', 'A dim 0', 1, 8, 1), slider(state, 'b0', 'B dim 0', 1, 8, 1),
      slider(state, 'a1', 'A dim 1', 1, 8, 1), slider(state, 'b1', 'B dim 1', 1, 8, 1),
      slider(state, 'a2', 'A dim 2', 1, 8, 1), slider(state, 'b2', 'B dim 2', 1, 8, 1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['BROADCASTING']), el('span', {}, ['drag two shapes'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [rows, el('div', { style: 'margin-top:10px' }, [status]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Broadcasting lets arrays of different shapes combine without copying data. Line the shapes up from the right; each pair of dimensions must be equal, or one must be 1 and gets stretched. Any other clash is an error. This is why a bias vector adds cleanly to a whole batch.'])
    ]));
    state._render();
  }

  // ── logsumexp-stability：朴素 exp 会溢出，减去最大值后保持有限 ─
  function logsumexpStability(host) {
    var base = [1.0, 0.5, -0.3];
    var state = { big: 700 };
    var rows = el('div', {});
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    state._render = function () {
      var x = [state.big].concat(base);
      // 朴素：log(sum(exp(x)))  -- exp(710+) 在 float64 中溢出为 Infinity
      var naiveSum = 0, i;
      for (i = 0; i < x.length; i++) { naiveSum += Math.exp(x[i]); }
      var naive = Math.log(naiveSum);
      // 稳定：m + log(sum(exp(x - m)))
      var m = x[0];
      for (i = 1; i < x.length; i++) { if (x[i] > m) { m = x[i]; } }
      var s = 0;
      for (i = 0; i < x.length; i++) { s += Math.exp(x[i] - m); }
      var stable = m + Math.log(s);
      var overflow = !isFinite(naive);
      while (rows.firstChild) { rows.removeChild(rows.firstChild); }
      function line(label, val, warn) {
        return el('div', { style: 'display:flex;justify-content:space-between;margin:4px 0' }, [
          el('span', { class: 'lf-meta' }, [label]),
          el('span', { class: 'lf-formula', style: 'color:' + (warn ? 'var(--warn,#b8870f)' : 'var(--blueprint,#3553ff)') },
            [isFinite(val) ? val.toFixed(4) : 'Infinity (overflow)'])
        ]);
      }
      rows.appendChild(line('naive log(sum exp x)', naive, overflow));
      rows.appendChild(line('stable m + log(sum exp(x-m))', stable, false));
      status.innerHTML = overflow ? 'naive overflows' : 'both agree';
      meta.textContent = overflow
        ? 'exp(' + state.big + ') is beyond float64 range (~exp 709), so the naive sum is Infinity; the stable form returns ' + stable.toFixed(4)
        : 'the two forms are algebraically equal and both finite here (max = ' + m + ')';
      formula.textContent = 'logsumexp(x) = m + log( sum exp(x - m) ),  m = max(x)';
    };
    var grid = el('div', {}, [slider(state, 'big', 'largest logit value', 1, 1500, 1)]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['LOG-SUM-EXP']), el('span', {}, ['drag the logit'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [rows, el('div', { style: 'margin-top:10px' }, [status]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Computing log(sum(exp(x))) directly overflows once any logit passes about 709, where exp exceeds the float64 range. Subtracting the maximum first shifts the largest term to exp(0) = 1, so the sum stays finite. The result is identical because the subtracted maximum is added back outside the log.'])
    ]));
    state._render();
  }

  // ── norm-unit-balls：L1 菱形、L2 圆形、Linf 方形；读出点的范数 ─
  function normUnitBalls(host) {
    var state = { which: 'l2', px: 0.6, py: 0.5 };
    var W = 260, H = 230, CX = 130, CY = 115, R = 90;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    function toX(u) { return CX + u * R; }
    function toY(v) { return CY - v * R; }
    state._render = function () {
      while (svg.firstChild) { svg.removeChild(svg.firstChild); }
      // 坐标轴
      svg.appendChild(svgEl('line', { x1: toX(-1.3), y1: CY, x2: toX(1.3), y2: CY, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      svg.appendChild(svgEl('line', { x1: CX, y1: toY(-1.3), x2: CX, y2: toY(1.3), stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      // 单位球
      var shape;
      if (state.which === 'l1') {
        shape = svgEl('polygon', { points: [toX(1) + ',' + toY(0), toX(0) + ',' + toY(1), toX(-1) + ',' + toY(0), toX(0) + ',' + toY(-1)].join(' '), fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' });
      } else if (state.which === 'linf') {
        shape = svgEl('rect', { x: toX(-1), y: toY(1), width: 2 * R, height: 2 * R, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' });
      } else {
        shape = svgEl('circle', { cx: CX, cy: CY, r: R, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' });
      }
      svg.appendChild(shape);
      // 该点及其向量
      svg.appendChild(svgEl('line', { x1: CX, y1: CY, x2: toX(state.px), y2: toY(state.py), stroke: 'var(--ink-mute,#999)', 'stroke-width': '1.5' }));
      svg.appendChild(svgEl('circle', { cx: toX(state.px), cy: toY(state.py), r: '5', fill: 'var(--warn,#b8870f)' }));
      var ax = Math.abs(state.px), ay = Math.abs(state.py);
      var norm = state.which === 'l1' ? ax + ay : state.which === 'linf' ? Math.max(ax, ay) : Math.sqrt(ax * ax + ay * ay);
      var nm = state.which === 'l1' ? 'L1' : state.which === 'linf' ? 'Linf' : 'L2';
      num.innerHTML = norm.toFixed(3) + ' <small>' + nm + ' norm</small>';
      var formula = state.which === 'l1' ? '|x| + |y|' : state.which === 'linf' ? 'max(|x|, |y|)' : 'sqrt(x^2 + y^2)';
      meta.textContent = nm + ' of (' + state.px.toFixed(2) + ', ' + state.py.toFixed(2) + ') = ' + formula + '  ·  the outline is every point of norm 1';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      select(state, 'which', 'norm', [['L2 (Euclidean)', 'l2'], ['L1 (Manhattan)', 'l1'], ['Linf (max)', 'linf']]),
      slider(state, 'px', 'point x', -1.2, 1.2, 0.05),
      slider(state, 'py', 'point y', -1.2, 1.2, 0.05)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['NORM UNIT BALLS']), el('span', {}, ['pick a norm'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [num]), meta])]),
      el('div', { class: 'lf-cap' }, ['A norm measures length, and its unit ball is every vector of length one. L2 sums squares and gives a circle; L1 sums absolute values and gives a diamond; Linf takes the largest coordinate and gives a square. Which norm you choose changes what counts as close, which is why it shapes regularization and distance.'])
    ]));
    state._render();
  }

  // ── monte-carlo-pi：四分之一圆内的比例估计 pi ────────
  function monteCarloPi(host) {
    var state = { n: 200 };
    var W = 230, H = 230, PAD = 14, S = 200;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 确定性的低偏差点（以黄金比例共轭为加项的递增序列），
    // 使图示每次渲染结果一致。
    var g1 = 0.7548776662466927, g2 = 0.5698402909980532;
    state._render = function () {
      while (svg.firstChild) { svg.removeChild(svg.firstChild); }
      svg.appendChild(svgEl('rect', { x: PAD, y: PAD, width: S, height: S, fill: 'none', stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1' }));
      svg.appendChild(svgEl('path', { d: 'M ' + PAD + ' ' + PAD + ' A ' + S + ' ' + S + ' 0 0 1 ' + (PAD + S) + ' ' + (PAD + S), fill: 'none', stroke: 'var(--ink-soft,#555)', 'stroke-width': '1.5' }));
      var inside = 0, i;
      var px = 0.123, py = 0.456;
      for (i = 0; i < state.n; i++) {
        px = (px + g1) % 1; py = (py + g2) % 1;
        var hit = (px * px + py * py) <= 1;
        if (hit) { inside++; }
        if (state.n <= 1200) {
          svg.appendChild(svgEl('circle', {
            cx: PAD + px * S, cy: PAD + (1 - py) * S, r: '1.6',
            fill: hit ? 'var(--blueprint,#3553ff)' : 'var(--ink-mute,#bbb)'
          }));
        }
      }
      var est = 4 * inside / state.n;
      num.innerHTML = est.toFixed(4) + ' <small>~ pi</small>';
      meta.textContent = inside + ' of ' + fmtInt(state.n) + ' inside  ·  error ' + Math.abs(est - Math.PI).toFixed(4) + '  ·  shrinks like 1/sqrt(N)';
      formula.textContent = 'pi ~ 4 * (points inside quarter circle) / N   ·   true pi = 3.14159';
    };
    var grid = el('div', {}, [slider(state, 'n', 'samples N', 20, 5000, 20)]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['MONTE CARLO PI']), el('span', {}, ['drag the sample count'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [num]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Scatter points in the unit square and count how many fall inside the quarter circle. That fraction is the ratio of areas, pi/4, so four times it estimates pi. More samples tighten the estimate, but the error only falls like one over the square root of N, the defining cost of Monte Carlo.'])
    ]));
    state._render();
  }

  // ── linear-system-conditioning：两条直线趋于平行，条件数激增 ─
  function linearConditioning(host) {
    // 系统：line1 x + y = 2（固定）。line2 的斜率朝 line1 方向调整。
    var state = { tilt: 60 };
    var W = 260, H = 230, CX = 130, CY = 115, SC = 28;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function toX(x) { return CX + x * SC; }
    function toY(y) { return CY - y * SC; }
    state._render = function () {
      while (svg.firstChild) { svg.removeChild(svg.firstChild); }
      // 直线 1：a1 x + b1 y = c1  ->  x + y = 2
      var a1 = 1, b1 = 1, c1 = 2;
      // 当 tilt -> 100 时，Line 2 的角度趋近于 Line 1。line1 方向角为 135deg。
      var t = state.tilt / 100;
      var ang = (135 - 55 * t) * Math.PI / 180; // 80deg .. 135deg
      var a2 = Math.cos(ang), b2 = Math.sin(ang);
      var c2 = a2 * 1 + b2 * 1; // 强制两条直线都经过解 (1,1)
      var det = a1 * b2 - a2 * b1;
      // 通过奇异值计算 2x2 矩阵的条件数
      var M = [[a1, b1], [a2, b2]];
      var ata00 = M[0][0] * M[0][0] + M[1][0] * M[1][0];
      var ata01 = M[0][0] * M[0][1] + M[1][0] * M[1][1];
      var ata11 = M[0][1] * M[0][1] + M[1][1] * M[1][1];
      var tr = ata00 + ata11, dt = ata00 * ata11 - ata01 * ata01;
      var disc = Math.sqrt(Math.max(0, tr * tr / 4 - dt));
      var l1 = tr / 2 + disc, l2 = tr / 2 - disc;
      var cond = Math.sqrt(l1 / Math.max(l2, 1e-12));
      function drawLine(a, b, c, st) {
        // a x + b y = c，采样 x 的范围
        var pts = [], xx;
        for (xx = -4; xx <= 4.01; xx += 8) {
          if (Math.abs(b) > 1e-6) { pts.push([xx, (c - a * xx) / b]); }
        }
        if (pts.length === 2) {
          svg.appendChild(svgEl('line', { x1: toX(pts[0][0]), y1: toY(pts[0][1]), x2: toX(pts[1][0]), y2: toY(pts[1][1]), stroke: st, 'stroke-width': '2' }));
        }
      }
      svg.appendChild(svgEl('line', { x1: toX(-4), y1: CY, x2: toX(4), y2: CY, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      svg.appendChild(svgEl('line', { x1: CX, y1: toY(-4), x2: CX, y2: toY(4), stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      drawLine(a1, b1, c1, 'var(--ink-mute,#999)');
      drawLine(a2, b2, c2, 'var(--blueprint,#3553ff)');
      svg.appendChild(svgEl('circle', { cx: toX(1), cy: toY(1), r: '5', fill: 'var(--warn,#b8870f)' }));
      num.innerHTML = (cond < 1000 ? cond.toFixed(1) : cond.toExponential(1)) + ' <small>cond number</small>';
      meta.textContent = (cond > 50 ? 'ill-conditioned: ' : 'well-conditioned: ') +
        'det = ' + det.toFixed(3) + '  ·  near-parallel lines make the intersection hypersensitive to noise';
      formula.textContent = 'kappa = sigma_max / sigma_min   ·   small noise in b shifts the solution by up to kappa times';
    };
    var grid = el('div', {}, [slider(state, 'tilt', 'tilt line 2 toward line 1', 0, 98, 1)]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['CONDITIONING']), el('span', {}, ['drag toward parallel'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [num]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['A 2x2 system is the intersection of two lines. When the lines cross at a wide angle the solution is sharp and stable. As they tilt toward parallel the determinant shrinks, the condition number explodes, and a tiny change in the inputs swings the intersection far away. Ill-conditioned systems amplify noise.'])
    ]));
    state._render();
  }

  // ── random-walk-diffusion：一维随机游走的扩散以 sqrt(t) 量级增长 ──────────
  function randomWalkDiffusion(host) {
    var state = { t: 50 };
    var W = 520, H = 220, PAD = 30;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    var TMAX = 200;
    // 通过每个游走者固定的符号序列生成若干确定性样本路径
    var walkers = 7;
    function step(seed, k) {
      // 确定性的伪符号 ∈ {-1,+1}
      var v = Math.sin(seed * 12.9898 + k * 78.233) * 43758.5453;
      v = v - Math.floor(v);
      return v < 0.5 ? -1 : 1;
    }
    function px(s) { return PAD + s / TMAX * (W - 2 * PAD); }
    function py(v) { return H / 2 - v / Math.sqrt(TMAX) * (H / 2 - PAD) * 0.9; }
    state._render = function () {
      while (svg.firstChild) { svg.removeChild(svg.firstChild); }
      svg.appendChild(svgEl('line', { x1: PAD, y1: H / 2, x2: W - PAD, y2: H / 2, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      // 理论上的 +/- 一倍标准差包络：std = sqrt(t)
      var dUp = '', dDn = '', i;
      for (i = 0; i <= 120; i++) {
        var s = TMAX * i / 120;
        var sd = Math.sqrt(s);
        dUp += (i ? 'L' : 'M') + px(s).toFixed(1) + ' ' + py(sd).toFixed(1) + ' ';
        dDn += (i ? 'L' : 'M') + px(s).toFixed(1) + ' ' + py(-sd).toFixed(1) + ' ';
      }
      svg.appendChild(svgEl('path', { d: dUp, fill: 'none', stroke: 'var(--warn,#b8870f)', 'stroke-width': '1.5', 'stroke-dasharray': '4 3' }));
      svg.appendChild(svgEl('path', { d: dDn, fill: 'none', stroke: 'var(--warn,#b8870f)', 'stroke-width': '1.5', 'stroke-dasharray': '4 3' }));
      var w, ends = [];
      for (w = 0; w < walkers; w++) {
        var pos = 0, d = '';
        d += 'M' + px(0).toFixed(1) + ' ' + py(0).toFixed(1) + ' ';
        var k;
        for (k = 1; k <= state.t; k++) {
          pos += step(w + 1, k);
          d += 'L' + px(k).toFixed(1) + ' ' + py(pos).toFixed(1) + ' ';
        }
        ends.push(pos);
        svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '1.2', opacity: '0.7' }));
        svg.appendChild(svgEl('circle', { cx: px(state.t), cy: py(pos), r: '3', fill: 'var(--blueprint,#3553ff)' }));
      }
      var sdTheory = Math.sqrt(state.t);
      num.innerHTML = sdTheory.toFixed(2) + ' <small>std = sqrt(t)</small>';
      meta.textContent = 't = ' + state.t + ' steps  ·  endpoints spread out like sqrt(t), not t  ·  dashed orange is the +/- one std envelope';
      formula.textContent = 'each step +/-1 with equal odds  ·  Var(position) = t,  std = sqrt(t)';
    };
    var grid = el('div', {}, [slider(state, 't', 'steps t', 1, TMAX, 1)]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['RANDOM WALK']), el('span', {}, ['drag the step count'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [num]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['A one-dimensional walk takes a plus-or-minus-one step each tick. Steps are independent so variances add: after t steps the variance is t and the typical distance from the start is the square root of t. Diffusion spreads slowly, which is why the walk wanders but rarely runs straight away.'])
    ]));
    state._render();
  }

  // ── roots-of-unity：n 个复数的 n 次单位根均匀分布在单位圆上 ───
  function rootsOfUnity(host) {
    var state = { n: 5 };
    var W = 260, H = 240, CX = 130, CY = 120, R = 95;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    state._render = function () {
      while (svg.firstChild) { svg.removeChild(svg.firstChild); }
      svg.appendChild(svgEl('line', { x1: CX - R - 14, y1: CY, x2: CX + R + 14, y2: CY, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      svg.appendChild(svgEl('line', { x1: CX, y1: CY - R - 14, x2: CX, y2: CY + R + 14, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      svg.appendChild(svgEl('circle', { cx: CX, cy: CY, r: R, fill: 'none', stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1.5' }));
      var pts = '', k;
      var coords = [];
      for (k = 0; k < state.n; k++) {
        var ang = 2 * Math.PI * k / state.n;
        var x = CX + R * Math.cos(ang), y = CY - R * Math.sin(ang);
        coords.push([x, y]);
        pts += (k ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
      }
      pts += 'Z';
      svg.appendChild(svgEl('path', { d: pts, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '1', opacity: '0.45' }));
      coords.forEach(function (c, k2) {
        svg.appendChild(svgEl('line', { x1: CX, y1: CY, x2: c[0], y2: c[1], stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '0.8' }));
        svg.appendChild(svgEl('circle', { cx: c[0], cy: c[1], r: k2 === 0 ? '5' : '4', fill: k2 === 0 ? 'var(--warn,#b8870f)' : 'var(--blueprint,#3553ff)' }));
      });
      num.innerHTML = state.n + ' <small>roots</small>';
      meta.textContent = 'spaced ' + (360 / state.n).toFixed(1) + ' deg apart  ·  k = 0 (orange) is always 1  ·  they sum to 0 for n > 1';
      formula.textContent = 'z_k = exp(2*pi*i*k/n) = cos(2*pi*k/n) + i*sin(2*pi*k/n),  k = 0..n-1';
    };
    var grid = el('div', {}, [slider(state, 'n', 'n (number of roots)', 1, 16, 1)]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['ROOTS OF UNITY']), el('span', {}, ['drag n'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [num]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['The n complex nth-roots of unity are the solutions of z to the n equals one. They sit evenly around the unit circle at angles two pi k over n, one of them always at 1. These evenly spaced points are the sampling frequencies behind the discrete Fourier transform.'])
    ]));
    state._render();
  }

  // ── graph-degree-distribution：度数之和等于边数的两倍 ─────────
  function graphDegrees(host) {
    var state = { nodes: 6, edges: 7 };
    var W = 260, H = 240, CX = 130, CY = 110, R = 80;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    state._render = function () {
      while (svg.firstChild) { svg.removeChild(svg.firstChild); }
      var n = state.nodes;
      var maxEdges = n * (n - 1) / 2;
      var e = Math.min(state.edges, maxEdges);
      // 确定性的边表：按固定顺序枚举所有点对，取前 e 条
      var pairs = [], i, j;
      for (i = 0; i < n; i++) { for (j = i + 1; j < n; j++) { pairs.push([i, j]); } }
      // 交错排列，使较早的边在环上均匀分布而非聚集
      pairs.sort(function (a, b) { return ((a[1] - a[0]) - (b[1] - b[0])) || (a[0] - b[0]); });
      var deg = [];
      for (i = 0; i < n; i++) { deg.push(0); }
      var used = pairs.slice(0, e);
      var coords = [];
      for (i = 0; i < n; i++) {
        var ang = 2 * Math.PI * i / n - Math.PI / 2;
        coords.push([CX + R * Math.cos(ang), CY + R * Math.sin(ang)]);
      }
      used.forEach(function (p) {
        deg[p[0]]++; deg[p[1]]++;
        svg.appendChild(svgEl('line', { x1: coords[p[0]][0], y1: coords[p[0]][1], x2: coords[p[1]][0], y2: coords[p[1]][1], stroke: 'var(--rule-soft,#ccc)', 'stroke-width': '1.4' }));
      });
      coords.forEach(function (c, idx) {
        svg.appendChild(svgEl('circle', { cx: c[0], cy: c[1], r: '11', fill: 'var(--blueprint,#3553ff)' }));
        svg.appendChild(svgEl('text', { x: c[0], y: c[1] + 4, 'text-anchor': 'middle', 'font-size': '11', 'font-family': 'monospace', fill: 'var(--bg,#fafaf5)' }, []));
        svg.lastChild.appendChild(document.createTextNode(String(deg[idx])));
      });
      var sumDeg = 0;
      for (i = 0; i < n; i++) { sumDeg += deg[i]; }
      num.innerHTML = sumDeg + ' <small>= 2 * ' + used.length + ' edges</small>';
      meta.textContent = 'each node label is its degree  ·  average degree ' + (sumDeg / n).toFixed(2) +
        (e < state.edges ? '  ·  capped at ' + maxEdges + ' (complete graph)' : '');
      formula.textContent = 'handshake lemma: sum of degrees = 2 * (number of edges)';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'nodes', 'nodes', 3, 10, 1),
      slider(state, 'edges', 'edges', 0, 20, 1)
    ]);
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, ['GRAPH DEGREES']), el('span', {}, ['drag nodes and edges'])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, [svg, el('div', { style: 'margin-top:10px' }, [num]), meta, formula])]),
      el('div', { class: 'lf-cap' }, ['Every edge touches two nodes, so it adds one to each of their degrees. Add up the degrees of all nodes and you have counted every edge exactly twice. This handshake lemma holds for any graph and forces the number of odd-degree nodes to be even.'])
    ]));
    state._render();
  }

  LF.register({
    'svd-rank-reconstruction': svdRank,
    'tensor-broadcast': tensorBroadcast,
    'logsumexp-stability': logsumexpStability,
    'norm-unit-balls': normUnitBalls,
    'monte-carlo-pi': monteCarloPi,
    'linear-system-conditioning': linearConditioning,
    'random-walk-diffusion': randomWalkDiffusion,
    'roots-of-unity': rootsOfUnity,
    'graph-degree-distribution': graphDegrees
  });
})();
