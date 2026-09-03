/* figures-genai-rl.js — 阶段 8（生成式 AI）和阶段 9（强化学习）的交互式课程图示。
   在 lesson-figures.js 之后加载，并通过 window.LF 注册。无依赖，仅使用 ES5，
   通过 CSS 变量应用主题。 */
(function () {
  'use strict';
  var LF = window.LF;
  if (!LF) { return; }
  var el = LF.el, svgEl = LF.svgEl, slider = LF.slider, select = LF.select;
  var clamp = LF.clamp;

  function frame(host, label, hint, grid, outKids, caption) {
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, [label]), el('span', {}, [hint])]),
      el('div', { class: 'lf-body' }, [grid, el('div', { class: 'lf-out' }, outKids)]),
      el('div', { class: 'lf-cap' }, [caption])
    ]));
  }

  // ── diffusion-denoise：随着噪声被逐步去除，一维信号逐渐显现 ──────────────
  function diffusionDenoise(host) {
    var W = 520, H = 240, PAD = 30, N = 96, T = 50;
    var state = { t: 35 };
    // 固定干净信号 x0 和噪声样本，使渲染结果具有确定性
    var x0 = [], noise = [], i, seed = 12345;
    function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff * 2 - 1; }
    for (i = 0; i < N; i++) {
      var u = i / (N - 1);
      x0.push(0.6 * Math.sin(u * Math.PI * 2) + 0.25 * Math.sin(u * Math.PI * 6));
      noise.push(rnd());
    }
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function px(j) { return PAD + j / (N - 1) * (W - 2 * PAD); }
    function py(v) { return H / 2 - v / 1.6 * (H / 2 - PAD); }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(svgEl('line', { x1: PAD, y1: H / 2, x2: W - PAD, y2: H / 2, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      // alpha_bar 从 1（干净，t=0）逐渐降至约 0（纯噪声，t=T）
      var ab = Math.pow(Math.cos((state.t / T) * Math.PI / 2), 2);
      var sA = Math.sqrt(ab), sN = Math.sqrt(1 - ab), d = '', j;
      for (j = 0; j < N; j++) {
        var xt = sA * x0[j] + sN * noise[j];
        d += (j ? 'L' : 'M') + px(j).toFixed(1) + ' ' + py(xt).toFixed(1) + ' ';
      }
      // 淡显干净目标的轮廓
      var dc = '';
      for (j = 0; j < N; j++) { dc += (j ? 'L' : 'M') + px(j).toFixed(1) + ' ' + py(x0[j]).toFixed(1) + ' '; }
      svg.appendChild(svgEl('path', { d: dc, fill: 'none', stroke: 'var(--ink-mute,#999)', 'stroke-width': '1.5', 'stroke-dasharray': '4 3', opacity: '0.5' }));
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '1.8' }));
      meta.textContent = 't = ' + state.t + ' of ' + T + '  ·  signal ' + Math.round(sA * 100) + '%  ·  noise ' + Math.round(sN * 100) + '%  ·  ' + (state.t < 8 ? 'almost clean' : state.t > 42 ? 'near pure noise' : 'denoising');
      formula.textContent = 'x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) noise   ·   alpha_bar_t = cos^2((t/T)·pi/2)';
    };
    var grid = el('div', {}, [slider(state, 't', 'timestep t (0 = clean, T = noise)', 0, T, 1)]);
    frame(host, 'DIFFUSION DENOISE', 'drag the timestep',
      grid, [svg, meta, formula],
      'A diffusion model learns to reverse a noising process. At t = T the signal is pure noise; as t falls toward 0 the model removes noise step by step and the underlying signal (grey dashes) re-emerges. Each x_t is a fixed mix of the clean signal and the same noise, weighted by the schedule.');
    state._render();
  }

  // ── noise-schedule：对比扩散步骤中的线性与余弦 alpha_bar ────────────────
  function noiseSchedule(host) {
    var W = 520, H = 220, PAD = 32, T = 1000;
    var state = { sched: 'cosine', t: 500 };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function ab(t) { return state.sched === 'cosine' ? Math.pow(Math.cos((t / T) * Math.PI / 2), 2) : Math.pow(1 - t / T, 2.2); }
    function px(t) { return PAD + t / T * (W - 2 * PAD); }
    function py(v) { return H - PAD - v * (H - 2 * PAD); }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(svgEl('line', { x1: PAD, y1: H - PAD, x2: W - PAD, y2: H - PAD, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      var d = '', i, t;
      for (i = 0; i <= 160; i++) { t = T * i / 160; d += (i ? 'L' : 'M') + px(t).toFixed(1) + ' ' + py(ab(t)).toFixed(1) + ' '; }
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      var mx = px(state.t), v = ab(state.t);
      svg.appendChild(svgEl('line', { x1: mx, y1: PAD, x2: mx, y2: H - PAD, stroke: 'var(--ink-mute,#999)', 'stroke-width': '1', 'stroke-dasharray': '3 3' }));
      svg.appendChild(svgEl('circle', { cx: mx, cy: py(v), r: '5', fill: 'var(--blueprint,#3553ff)' }));
      var snr = v / Math.max(1e-6, 1 - v);
      meta.textContent = 'alpha_bar = ' + v.toFixed(3) + '  ·  SNR = ' + snr.toFixed(2) + '  ·  ' + (state.sched === 'cosine' ? 'cosine keeps signal longer in mid steps' : 'linear destroys signal early');
      formula.textContent = state.sched === 'cosine'
        ? 'alpha_bar_t = cos^2((t/T)·pi/2)   ·   SNR(t) = alpha_bar_t / (1 - alpha_bar_t)'
        : 'alpha_bar_t = (1 - t/T)^2.2   ·   SNR(t) = alpha_bar_t / (1 - alpha_bar_t)';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      select(state, 'sched', 'schedule', [['cosine', 'cosine'], ['linear', 'linear']]),
      slider(state, 't', 'diffusion step t', 0, T, 10)
    ]);
    frame(host, 'NOISE SCHEDULE', 'pick a schedule',
      grid, [svg, meta, formula],
      'alpha_bar is the fraction of signal surviving at step t, and its ratio to the remaining noise is the signal-to-noise ratio. A linear schedule wipes out the signal fast in the early steps; the cosine schedule decays more gently in the middle, leaving useful signal for longer and giving the model more informative intermediate targets.');
    state._render();
  }

  // ── vae-latent-grid：遍历二维潜在空间，观察解码形状的变化 ───────────────
  function vaeLatentGrid(host) {
    var W = 520, H = 240, CX = 380, CY = 120, R = 78;
    var state = { z1: 0, z2: 0 };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 左侧：带移动点的潜在平面；右侧：由 (z1,z2) 解码出的参数化形状
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var planeX = 40, planeY = 40, planeW = 160, planeH = 160;
      svg.appendChild(svgEl('rect', { x: planeX, y: planeY, width: planeW, height: planeH, fill: 'none', stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1' }));
      svg.appendChild(svgEl('line', { x1: planeX, y1: planeY + planeH / 2, x2: planeX + planeW, y2: planeY + planeH / 2, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      svg.appendChild(svgEl('line', { x1: planeX + planeW / 2, y1: planeY, x2: planeX + planeW / 2, y2: planeY + planeH, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      var dx = planeX + planeW / 2 + state.z1 / 3 * (planeW / 2);
      var dy = planeY + planeH / 2 - state.z2 / 3 * (planeH / 2);
      svg.appendChild(svgEl('circle', { cx: dx, cy: dy, r: '5', fill: 'var(--blueprint,#3553ff)' }));
      // 解码：z1 控制瓣数和尖锐程度，z2 控制圆润与星形之间的变化
      var pts = 80, k, dpath = '';
      var lobes = 3 + Math.round((state.z1 + 3) / 6 * 5); // 3..8
      var spike = (state.z2 + 3) / 6; // 0..1
      for (k = 0; k <= pts; k++) {
        var ang = k / pts * Math.PI * 2;
        var rad = R * (1 - spike * 0.55 * Math.abs(Math.cos(lobes * ang / 2)));
        var x = CX + rad * Math.cos(ang), y = CY + rad * Math.sin(ang);
        dpath += (k ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
      }
      dpath += 'Z';
      svg.appendChild(svgEl('path', { d: dpath, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      meta.textContent = 'latent (z1, z2) = (' + state.z1.toFixed(1) + ', ' + state.z2.toFixed(1) + ')  ·  decoded ' + lobes + ' lobes  ·  ' + (spike < 0.25 ? 'round' : spike > 0.7 ? 'spiky' : 'mixed');
      formula.textContent = 'x = decoder(z),  z ~ N(0, I)   ·   nearby z decode to similar shapes (smooth latent space)';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'z1', 'latent z1', -3, 3, 0.1),
      slider(state, 'z2', 'latent z2', -3, 3, 0.1)
    ]);
    frame(host, 'VAE LATENT GRID', 'drag z1 and z2',
      grid, [svg, meta, formula],
      'A VAE maps inputs to a smooth latent space and decodes points back to outputs. The square on the left is a slice of that space; the dot is your latent code. Moving it morphs the decoded shape on the right continuously, because the decoder is trained so that nearby codes produce nearby outputs.');
    state._render();
  }

  // ── gan-minimax：生成器与判别器的平衡及失效模式 ─────────────────────────
  function ganMinimax(host) {
    var W = 520, H = 220, PAD = 34;
    // bal：-1 表示生成器大幅领先，0 表示均衡，+1 表示判别器大幅领先
    var state = { bal: 0 };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function px(b) { return PAD + (b + 1) / 2 * (W - 2 * PAD); }
    function py(v) { return H - PAD - clamp(v, 0, 3) / 3 * (H - 2 * PAD); }
    // D 对伪样本的准确率随 bal 升高；当 D 确信无疑时，传给 G 的梯度消失
    function dLoss(b) { return 0.4 + 0.9 * (1 - Math.abs(b)); } // 在极端状态下，占优一方的损失最低
    function gLoss(b) { return 0.5 + 1.4 * (b + 1) / 2; } // D 越强，生成器承受的损失越大
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(svgEl('line', { x1: PAD, y1: H - PAD, x2: W - PAD, y2: H - PAD, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      // bal = 0 处的均衡标记
      var ex = px(0);
      svg.appendChild(svgEl('line', { x1: ex, y1: PAD, x2: ex, y2: H - PAD, stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1', 'stroke-dasharray': '3 3' }));
      function curve(fn, st) { var d = '', i, b; for (i = 0; i <= 100; i++) { b = -1 + 2 * i / 100; d += (i ? 'L' : 'M') + px(b).toFixed(1) + ' ' + py(fn(b)).toFixed(1) + ' '; } svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: st, 'stroke-width': '2' })); }
      curve(dLoss, 'var(--ink-mute,#999)');
      curve(gLoss, 'var(--blueprint,#3553ff)');
      var b = state.bal;
      svg.appendChild(svgEl('circle', { cx: px(b), cy: py(gLoss(b)), r: '5', fill: 'var(--blueprint,#3553ff)' }));
      svg.appendChild(svgEl('circle', { cx: px(b), cy: py(dLoss(b)), r: '4', fill: 'var(--ink-mute,#999)' }));
      var mode;
      if (b > 0.55) mode = 'discriminator too strong: vanishing gradient';
      else if (b < -0.55) mode = 'generator dominates: mode collapse risk';
      else mode = 'near equilibrium: useful gradients flow';
      status.innerHTML = mode;
      meta.textContent = 'generator loss ' + gLoss(b).toFixed(2) + '  ·  discriminator loss ' + dLoss(b).toFixed(2) + '  ·  gradient to G ' + ((1 - Math.abs(b)) * 100).toFixed(0) + '%';
      formula.textContent = 'min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]   ·   balance keeps the game informative';
    };
    var grid = el('div', {}, [slider(state, 'bal', 'balance (-1 = G ahead, +1 = D ahead)', -1, 1, 0.05)]);
    frame(host, 'GAN MINIMAX', 'drag the balance',
      grid, [svg, el('div', { style: 'margin-top:12px' }, [status]), meta, formula],
      'A GAN is a two-player game: the generator (blue) tries to fool the discriminator (grey), which tries to tell real from fake. Both must improve together. If the discriminator wins decisively its gradient to the generator vanishes; if the generator races ahead it can collapse onto a few outputs. Healthy training stays near the dashed equilibrium.');
    state._render();
  }

  // ── qlearning-gridworld：4x4 网格及不同训练回合的价值快照 ────────────────
  function qlearningGridworld(host) {
    var W = 520, H = 240, GRID = 4, CELL = 52, OX = 40, OY = 18;
    var GOAL = 3, PIT = 9; // 索引 = 行*4 + 列；目标位于 (0,3)，陷阱位于 (2,1)
    var state = { ep: 200 };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 在确定性 4x4 网格上通过价值迭代得到的收敛值（gamma 0.9，步长 -0.04）
    var GAMMA = 0.9, STEP = -0.04, Rgoal = 1, Rpit = -1;
    function neighbors(s) {
      var r = Math.floor(s / GRID), c = s % GRID, out = [];
      if (r > 0) out.push(s - GRID); if (r < GRID - 1) out.push(s + GRID);
      if (c > 0) out.push(s - 1); if (c < GRID - 1) out.push(s + 1);
      return out;
    }
    var Vstar = []; var i;
    for (i = 0; i < GRID * GRID; i++) Vstar.push(0);
    Vstar[GOAL] = Rgoal; Vstar[PIT] = Rpit;
    (function () { var it, s, nb, best, k; for (it = 0; it < 200; it++) { for (s = 0; s < GRID * GRID; s++) { if (s === GOAL || s === PIT) continue; nb = neighbors(s); best = -1e9; for (k = 0; k < nb.length; k++) best = Math.max(best, Vstar[nb[k]]); Vstar[s] = STEP + GAMMA * best; } } })();
    function valueAt(s, ep) { if (s === GOAL) return Rgoal; if (s === PIT) return Rpit; return Vstar[s] * clamp(ep / 300, 0, 1); }
    function shade(v) {
      // 将 [-1,1] 范围内的 v 映射为 blueprint（正值）或 warn（负值）的不透明度
      if (v >= 0) return { fill: 'var(--blueprint,#3553ff)', op: (0.08 + 0.6 * Math.min(1, v)).toFixed(2) };
      return { fill: 'var(--warn,#b8870f)', op: (0.08 + 0.6 * Math.min(1, -v)).toFixed(2) };
    }
    function bestDir(s, ep) {
      var nb = neighbors(s), best = -1e9, dir = null, k;
      for (k = 0; k < nb.length; k++) { var vv = valueAt(nb[k], ep); if (vv > best) { best = vv; dir = nb[k]; } }
      return dir;
    }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var s, r, c;
      for (s = 0; s < GRID * GRID; s++) {
        r = Math.floor(s / GRID); c = s % GRID;
        var x = OX + c * CELL, y = OY + r * CELL;
        var v = valueAt(s, state.ep), sh = shade(v);
        svg.appendChild(svgEl('rect', { x: x, y: y, width: CELL, height: CELL, fill: sh.fill, opacity: sh.op, stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1' }));
        var cx = x + CELL / 2, cy = y + CELL / 2;
        if (s === GOAL) { svg.appendChild(svgEl('text', { x: cx, y: cy + 4, 'text-anchor': 'middle', 'font-size': '12', fill: 'var(--ink,#1a1a1a)', 'font-family': 'monospace' }, [document.createTextNode('GOAL')])); }
        else if (s === PIT) { svg.appendChild(svgEl('text', { x: cx, y: cy + 4, 'text-anchor': 'middle', 'font-size': '13', fill: 'var(--ink,#1a1a1a)', 'font-family': 'monospace' }, [document.createTextNode('PIT')])); }
        else {
          svg.appendChild(svgEl('text', { x: cx, y: y + CELL - 6, 'text-anchor': 'middle', 'font-size': '9', fill: 'var(--ink-mute,#777)', 'font-family': 'monospace' }, [document.createTextNode(v.toFixed(2))]));
          if (state.ep > 20) {
            var dir = bestDir(s, state.ep);
            if (dir != null) {
              var dr = Math.floor(dir / GRID) - r, dc = (dir % GRID) - c;
              var ax = cx + dc * 14, ay = (cy - 4) + dr * 14;
              svg.appendChild(svgEl('line', { x1: cx, y1: cy - 4, x2: ax, y2: ay, stroke: 'var(--ink,#1a1a1a)', 'stroke-width': '1.6' }));
              svg.appendChild(svgEl('circle', { cx: ax, cy: ay, r: '2.4', fill: 'var(--ink,#1a1a1a)' }));
            }
          }
        }
      }
      meta.textContent = 'episode ' + state.ep + ' of 300  ·  ' + (state.ep < 30 ? 'values still near zero' : state.ep < 200 ? 'value spreading from the goal' : 'policy converged');
      formula.textContent = 'Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a\' Q(s\',a\') - Q(s,a) ]   ·   gamma = 0.9';
    };
    var grid = el('div', {}, [slider(state, 'ep', 'training episodes', 0, 300, 10)]);
    frame(host, 'Q-LEARNING GRIDWORLD', 'drag the episodes',
      grid, [svg, meta, formula],
      'The agent learns to reach the goal and avoid the pit. Cell shading is the learned state value (blue good, amber bad) and the arrows are the greedy policy. Early on values sit near zero; with training, value propagates outward from the goal and the arrows line up into a path that routes around the pit.');
    state._render();
  }

  // ── value-iteration-gamma：价值沿一维链向目标传播 ────────────────────────
  function valueIterationGamma(host) {
    var W = 520, H = 200, N = 10, CELL = 44, OX = 36, OY = 70;
    var state = { gamma: 0.9 };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 目标位于最右侧单元格，奖励为 1；每步无成本；V(s) = gamma^(dist)
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var g = state.gamma, i, vals = [];
      for (i = 0; i < N; i++) vals.push(Math.pow(g, (N - 1 - i)));
      for (i = 0; i < N; i++) {
        var x = OX + i * CELL, y = OY, v = vals[i];
        var op = (0.08 + 0.7 * v).toFixed(2);
        svg.appendChild(svgEl('rect', { x: x, y: y, width: CELL - 4, height: CELL - 4, fill: 'var(--blueprint,#3553ff)', opacity: op, stroke: 'var(--rule-soft,#ddd)', 'stroke-width': '1' }));
        var cx = x + (CELL - 4) / 2;
        svg.appendChild(svgEl('text', { x: cx, y: y + (CELL - 4) / 2 + 4, 'text-anchor': 'middle', 'font-size': '9', fill: 'var(--ink,#1a1a1a)', 'font-family': 'monospace' }, [document.createTextNode(v.toFixed(2))]));
        if (i === N - 1) svg.appendChild(svgEl('text', { x: cx, y: y - 8, 'text-anchor': 'middle', 'font-size': '10', fill: 'var(--ink-mute,#777)', 'font-family': 'monospace' }, [document.createTextNode('GOAL')]));
        if (i < N - 1) { svg.appendChild(svgEl('line', { x1: x + CELL - 6, y1: y + (CELL - 4) / 2, x2: x + CELL + 2, y2: y + (CELL - 4) / 2, stroke: 'var(--ink-mute,#999)', 'stroke-width': '1.2' })); }
      }
      var reach = vals[0];
      meta.textContent = 'gamma = ' + g.toFixed(2) + '  ·  value 9 steps from goal = ' + reach.toFixed(3) + '  ·  ' + (g < 0.6 ? 'short-sighted: distant reward fades' : g > 0.95 ? 'far-sighted: value carries across the chain' : 'moderate horizon');
      formula.textContent = 'V(s) = gamma^(distance to goal)   ·   higher gamma propagates value further from the goal';
    };
    var grid = el('div', {}, [slider(state, 'gamma', 'discount gamma', 0.1, 0.99, 0.01)]);
    frame(host, 'VALUE ITERATION', 'drag gamma',
      grid, [svg, meta, formula],
      'Value iteration backs reward up from the goal one step at a time. On this chain the only reward is at the rightmost cell, so each state is worth gamma raised to its distance from the goal. A small gamma makes distant reward nearly worthless (short-sighted); a gamma near 1 carries strong value all the way back along the chain.');
    state._render();
  }

  // ── epsilon-greedy：探索/利用比例与累积遗憾 ──────────────────────────────
  function epsilonGreedy(host) {
    var W = 520, H = 210, PAD = 32, N = 500;
    var state = { eps: 0.1, decay: 1 }; // decay 为 1 表示固定值，为 0 表示衰减调度
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var bar = el('i');
    var barWrap = el('div', { class: 'lf-bar' }, [bar]);
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function epsAt(t) { return state.decay > 0.5 ? state.eps : state.eps / (1 + t / 60); }
    function px(t) { return PAD + t / N * (W - 2 * PAD); }
    function py(v, vmax) { return H - PAD - clamp(v / vmax, 0, 1) * (H - 2 * PAD); }
    var GAP = 0.4; // 每次探索性拉动（次优臂）的遗憾成本
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(svgEl('line', { x1: PAD, y1: H - PAD, x2: W - PAD, y2: H - PAD, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      var t, regret = 0, pts = [], vmax = 1e-6;
      for (t = 0; t <= N; t++) { regret += epsAt(t) * GAP; pts.push(regret); }
      vmax = pts[N];
      var d = '';
      for (t = 0; t <= N; t += 5) { d += (t ? 'L' : 'M') + px(t).toFixed(1) + ' ' + py(pts[t], vmax).toFixed(1) + ' '; }
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '2' }));
      var e0 = epsAt(0), eEnd = epsAt(N);
      bar.style.width = (e0 * 100).toFixed(0) + '%';
      meta.textContent = 'start explore ' + Math.round(e0 * 100) + '% / exploit ' + Math.round((1 - e0) * 100) + '%  ·  end explore ' + Math.round(eEnd * 100) + '%  ·  total regret ' + vmax.toFixed(1);
      formula.textContent = state.decay > 0.5
        ? 'fixed epsilon: regret grows linearly forever  ·  P(explore) = ' + state.eps.toFixed(2)
        : 'decaying epsilon_t = epsilon_0 / (1 + t/60)  ·  regret levels off as exploration fades';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'eps', 'epsilon (explore rate)', 0, 0.5, 0.01),
      select(state, 'decay', 'schedule', [['fixed', '1'], ['decaying', '0']])
    ]);
    frame(host, 'EPSILON-GREEDY', 'drag epsilon',
      grid, [el('div', { class: 'lf-meta' }, ['explore share of actions']), barWrap, svg, meta, formula],
      'With probability epsilon the agent explores a random action; otherwise it exploits its current best estimate. The bar shows the explore/exploit split and the curve is cumulative regret, the reward given up by not always picking the best arm. A fixed epsilon piles up regret forever; a decaying schedule explores early then exploits, so regret flattens.');
    state._render();
  }

  // ── discount-horizon：有效时域与几何权重衰减 ─────────────────────────────
  function discountHorizon(host) {
    var W = 520, H = 210, PAD = 32, TMAX = 40;
    var state = { gamma: 0.9 };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var num = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    function px(t) { return PAD + t / TMAX * (W - 2 * PAD); }
    function py(w) { return H - PAD - w * (H - 2 * PAD); }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(svgEl('line', { x1: PAD, y1: H - PAD, x2: W - PAD, y2: H - PAD, stroke: 'var(--rule-soft,#eee)', 'stroke-width': '1' }));
      var g = state.gamma, t;
      for (t = 0; t <= TMAX; t++) {
        var w = Math.pow(g, t);
        svg.appendChild(svgEl('rect', { x: px(t) - 3, y: py(w), width: 6, height: (H - PAD) - py(w), fill: 'var(--blueprint,#3553ff)', opacity: '0.85' }));
      }
      var hor = 1 / (1 - g);
      // 标记有效时域
      var hx = px(Math.min(TMAX, hor));
      svg.appendChild(svgEl('line', { x1: hx, y1: PAD, x2: hx, y2: H - PAD, stroke: 'var(--warn,#b8870f)', 'stroke-width': '1.5', 'stroke-dasharray': '4 3' }));
      num.innerHTML = hor.toFixed(1) + ' <small>step horizon</small>';
      meta.textContent = 'gamma = ' + g.toFixed(2) + '  ·  weight at the horizon = ' + Math.pow(g, hor).toFixed(2) + ' (about 1/e)  ·  reward 20 steps out weighs ' + Math.pow(g, 20).toFixed(3);
      formula.textContent = 'return = sum_t gamma^t r_t   ·   effective horizon = 1/(1 - gamma)   ·   weights decay geometrically';
    };
    var grid = el('div', {}, [slider(state, 'gamma', 'discount gamma', 0.5, 0.99, 0.01)]);
    frame(host, 'DISCOUNT HORIZON', 'drag gamma',
      grid, [svg, el('div', { style: 'margin-top:10px' }, [num]), meta, formula],
      'Future reward t steps away is weighted by gamma^t, so the weights fall off geometrically (the bars). The dashed line marks the effective horizon 1/(1 - gamma), where the weight has dropped to about 1/e. Raising gamma stretches that horizon, making the agent care about reward much further into the future.');
    state._render();
  }

  // ── policy-gradient-landscape：通过梯度上升攀向奖励峰值 ──────────────────
  function policyGradientLandscape(host) {
    var W = 520, H = 230, PAD = 30;
    var state = { lr: 0.15, steps: 14, theta0: -2.4 };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var status = el('span', { class: 'lf-num' });
    var meta = el('div', { class: 'lf-meta' });
    var formula = el('div', { class: 'lf-formula' });
    // 奖励 J(theta)：平滑的峰形曲面，最大值位于 theta = 1.2 附近
    function J(t) { return 3 * Math.exp(-0.35 * (t - 1.2) * (t - 1.2)) + 0.4 * Math.exp(-0.8 * (t + 2) * (t + 2)); }
    function grad(t) { var h = 1e-3; return (J(t + h) - J(t - h)) / (2 * h); }
    function px(t) { return PAD + (t + 3.5) / 7 * (W - 2 * PAD); }
    function py(v) { return H - PAD - v / 3.4 * (H - 2 * PAD); }
    state._render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var d = '', i, x;
      for (i = 0; i <= 140; i++) { x = -3.5 + 7 * i / 140; d += (i ? 'L' : 'M') + px(x).toFixed(1) + ' ' + py(J(x)).toFixed(1) + ' '; }
      svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: 'var(--rule-soft,#ccc)', 'stroke-width': '2' }));
      var th = state.theta0, pts = [], t;
      for (t = 0; t <= state.steps; t++) { pts.push(th); th = th + state.lr * grad(th); th = clamp(th, -3.4, 3.4); }
      var pd = '';
      pts.forEach(function (p, idx) { pd += (idx ? 'L' : 'M') + px(p).toFixed(1) + ' ' + py(J(p)).toFixed(1) + ' '; });
      svg.appendChild(svgEl('path', { d: pd, fill: 'none', stroke: 'var(--blueprint,#3553ff)', 'stroke-width': '1.5', 'stroke-dasharray': '4 3' }));
      pts.forEach(function (p, idx) { svg.appendChild(svgEl('circle', { cx: px(p), cy: py(J(p)), r: idx === pts.length - 1 ? '5' : '3', fill: 'var(--blueprint,#3553ff)' })); });
      var last = pts[pts.length - 1];
      var atPeak = Math.abs(last - 1.2) < 0.2;
      status.innerHTML = atPeak ? 'reached the peak' : 'J(theta) = ' + J(last).toFixed(2);
      meta.textContent = 'theta = ' + last.toFixed(2) + '  ·  reward ' + J(last).toFixed(3) + ' of ' + J(1.2).toFixed(2) + ' max  ·  ' + (state.lr > 0.6 ? 'large lr: may overshoot the peak' : 'climbing');
      formula.textContent = 'theta <- theta + lr · grad_theta J(theta)   ·   ascent moves toward higher expected reward';
    };
    var grid = el('div', { class: 'lf-grid' }, [
      slider(state, 'lr', 'learning rate', 0.02, 1.0, 0.02),
      slider(state, 'steps', 'steps', 1, 40, 1),
      slider(state, 'theta0', 'start theta', -3.2, 3.2, 0.1)
    ]);
    frame(host, 'POLICY GRADIENT', 'drag the learning rate',
      grid, [svg, el('div', { style: 'margin-top:12px' }, [status]), meta, formula],
      'Policy gradient methods adjust the policy parameter theta in the direction that raises expected reward. Here the grey curve is the reward landscape and the dots are ascent steps climbing toward the peak. A small rate creeps up slowly; too large a rate overshoots; a local bump on the left can trap the climb if it starts there.');
    state._render();
  }

  LF.register({
    'diffusion-denoise': diffusionDenoise,
    'noise-schedule': noiseSchedule,
    'vae-latent-grid': vaeLatentGrid,
    'gan-minimax': ganMinimax,
    'qlearning-gridworld': qlearningGridworld,
    'value-iteration-gamma': valueIterationGamma,
    'epsilon-greedy': epsilonGreedy,
    'discount-horizon': discountHorizon,
    'policy-gradient-landscape': policyGradientLandscape
  });
})();
