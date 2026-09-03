(function () {
  'use strict';
  var LF = window.LF;
  if (!LF) { return; }

  var el = LF.el, svgEl = LF.svgEl;
  var INK = 'var(--ink,#1a1a1a)', SOFT = 'var(--ink-soft,#555)', MUTE = 'var(--ink-mute,#777)';
  var BP = 'var(--blueprint,#3553ff)', BG = 'var(--bg,#fafaf5)', SURF = 'var(--bg-surface,#eee)';
  var RULE = 'var(--rule-soft,#ddd)', WARN = 'var(--warn,#b8870f)';

  function anim(attr, vals, dur, extra) {
    var a = { attributeName: attr, values: vals, dur: dur, repeatCount: 'indefinite' };
    if (extra) for (var k in extra) a[k] = extra[k];
    return svgEl('animate', a);
  }
  function animT(type, vals, dur, extra) {
    var a = { attributeName: 'transform', type: type, values: vals, dur: dur, repeatCount: 'indefinite' };
    if (extra) for (var k in extra) a[k] = extra[k];
    return svgEl('animateTransform', a);
  }
  function card(host, label, hint, svg, caption) {
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, [label]), el('span', {}, [hint])]),
      el('div', { class: 'lf-body' }, [el('div', { class: 'lf-out' }, [svg])]),
      el('div', { class: 'lf-cap' }, [caption])
    ]));
  }
  function txt(x, y, s, fill, size, anchor) {
    return svgEl('text', {
      x: x, y: y, fill: fill || SOFT, 'font-size': size || 11,
      'font-family': 'var(--font-mono,monospace)', 'text-anchor': anchor || 'middle'
    }, [svgEl('tspan', {}, [document.createTextNode(s)])]);
  }

  // ── BPE标记器:相邻的字节对合并成一个符号,反复 ───
  function bpeMerge(host) {
    var svg = svgEl('svg', { viewBox: '0 0 520 230' });
    var cells = ['l', 'o', 'w', 'e', 's', 't'];
    var bw = 46, gap = 8, x0 = 90, y = 60;
    svg.appendChild(txt(60, y + 18, 'bytes', MUTE, 10, 'middle'));
    var i;
    for (i = 0; i < cells.length; i++) {
      var x = x0 + i * (bw + gap);
      svg.appendChild(svgEl('rect', { x: x, y: y, width: bw, height: 36, rx: 4, fill: BG, stroke: RULE, 'stroke-width': '1.5' }));
      svg.appendChild(txt(x + bw / 2, y + 24, cells[i], INK, 15));
    }
    // 突出框,在相邻的对之间滑动 ("最频繁的对"扫描)
    var hx0 = x0 - 3, pairW = bw * 2 + gap + 6;
    var slots = [];
    for (i = 0; i < cells.length - 1; i++) slots.push((hx0 + i * (bw + gap)) + ',' + (y - 3));
    var hl = svgEl('rect', { x: hx0, y: y - 3, width: pairW, height: 42, rx: 5, fill: 'none', stroke: BP, 'stroke-width': '2.5', opacity: '0.9' });
    hl.appendChild(animT('translate', '0 0;' + ((cells.length - 2) * (bw + gap)) + ' 0;0 0', '5s'));
    svg.appendChild(hl);
    // 合并符号在下面增长: "es" + "t" -> "est"
    var my = 150;
    svg.appendChild(txt(60, my + 22, 'merge', MUTE, 10, 'middle'));
    var down = svgEl('path', { d: 'M260 104 L260 142', stroke: BP, 'stroke-width': '1.6', fill: 'none', 'marker-end': '', 'stroke-dasharray': '5 4' });
    down.appendChild(anim('stroke-dashoffset', '18;0', '1.2s'));
    svg.appendChild(down);
    svg.appendChild(svgEl('polygon', { points: '256,142 264,142 260,150', fill: BP }));
    var merged = svgEl('g', {});
    merged.appendChild(svgEl('rect', { x: 224, y: my, width: 72, height: 38, rx: 5, fill: BP, opacity: '0.14', stroke: BP, 'stroke-width': '2' }));
    merged.appendChild(txt(260, my + 25, 'est', BP, 16));
    merged.appendChild(anim('opacity', '0.15;1;1;0.15', '5s'));
    svg.appendChild(merged);
    svg.appendChild(txt(420, my + 6, '→ new id', MUTE, 10, 'middle'));
    svg.appendChild(txt(420, my + 24, 'vocab + 1', SOFT, 12, 'middle'));
    card(host, 'BPE MERGE LOOP', 'scan · merge · repeat',
      svg,
      'Byte-Pair Encoding starts from raw bytes and repeatedly finds the most frequent adjacent pair, merges it into one new symbol, and grows the vocabulary by one id. The scan box sweeps every neighboring pair; the winner collapses into a single token.');
  }

  // ──滑动窗口:一个固定的窗口在ID流上滑动,每步T+1 ──
  function slidingWindow(host) {
    var svg = svgEl('svg', { viewBox: '0 0 520 220' });
    var n = 16, bw = 26, x0 = 26, y = 70;
    var ids = ['12', '7', '93', '4', '58', '2', '31', '9', '77', '6', '40', '15', '8', '61', '3', '22'];
    var i;
    svg.appendChild(txt(20, 50, 'token id stream', MUTE, 10, 'start'));
    for (i = 0; i < n; i++) {
      var x = x0 + i * (bw + 4);
      svg.appendChild(svgEl('rect', { x: x, y: y, width: bw, height: 30, rx: 3, fill: BG, stroke: RULE, 'stroke-width': '1' }));
      svg.appendChild(txt(x + bw / 2, y + 20, ids[i], SOFT, 11));
    }
    var step = bw + 4, win = 5;
    var maxShift = (n - win - 1) * step;
    // 输入窗口 (TID)
    var inW = win * step - 4;
    var wg = svgEl('g', {});
    wg.appendChild(svgEl('rect', { x: x0 - 3, y: y - 6, width: inW + 6, height: 42, rx: 5, fill: BP, opacity: '0.12', stroke: BP, 'stroke-width': '2' }));
    wg.appendChild(txt(x0 - 3 + (inW + 6) / 2, y - 12, 'input  (B, T)', BP, 10, 'middle'));
    // 目标窗口转移一个
    wg.appendChild(svgEl('rect', { x: x0 - 3 + step, y: y + 40, width: inW + 6, height: 24, rx: 5, fill: WARN, opacity: '0.14', stroke: WARN, 'stroke-width': '1.6' }));
    wg.appendChild(txt(x0 - 3 + step + (inW + 6) / 2, y + 80, 'target = input shifted by one', WARN, 10, 'middle'));
    wg.appendChild(animT('translate', '0 0;' + maxShift + ' 0', '6s', { calcMode: 'discrete' }));
    svg.appendChild(wg);
    card(host, 'SLIDING WINDOW', 'window slides · stride',
      svg,
      'The tokenizer flattens the corpus into one long id stream. A fixed-length window of T ids slides along it by a stride, and the target is the same window shifted left by one for next-token prediction. Smaller strides overlap more and inflate the example count.');
  }

  // ── 多头注意力：一次投影分流为 H 个并行注意力头 ──
  function multiHead(host) {
    var svg = svgEl('svg', { viewBox: '0 0 520 240' });
    var inX = 40, inY = 100;
    svg.appendChild(svgEl('rect', { x: inX, y: inY, width: 60, height: 40, rx: 5, fill: BG, stroke: BP, 'stroke-width': '2' }));
    svg.appendChild(txt(inX + 30, inY + 18, '(B,T,D)', INK, 10));
    svg.appendChild(txt(inX + 30, inY + 32, 'input', MUTE, 9));
    // 投影块
    var pjX = 150;
    svg.appendChild(svgEl('rect', { x: pjX, y: inY - 6, width: 64, height: 52, rx: 5, fill: BP, opacity: '0.12', stroke: BP, 'stroke-width': '2' }));
    svg.appendChild(txt(pjX + 32, inY + 14, 'Linear', BP, 11));
    svg.appendChild(txt(pjX + 32, inY + 30, 'D→3D', SOFT, 9));
    var flow = svgEl('line', { x1: inX + 60, y1: inY + 20, x2: pjX, y2: inY + 20, stroke: BP, 'stroke-width': '2', 'stroke-dasharray': '6 5' });
    flow.appendChild(anim('stroke-dashoffset', '22;0', '1s'));
    svg.appendChild(flow);
    // 发散的头
    var heads = 4, hX = 300, hY = [30, 90, 150, 210];
    var j;
    for (j = 0; j < heads; j++) {
      var hy = hY[j];
      var path = svgEl('path', { d: 'M' + (pjX + 64) + ' ' + (inY + 20) + ' C ' + (hX - 40) + ' ' + (inY + 20) + ', ' + (hX - 30) + ' ' + hy + ', ' + hX + ' ' + hy, fill: 'none', stroke: BP, 'stroke-width': '1.4', opacity: '0.55', 'stroke-dasharray': '5 4' });
      path.appendChild(anim('stroke-dashoffset', '18;0', '1.2s', { begin: (j * 0.18) + 's' }));
      svg.appendChild(path);
      var hg = svgEl('g', {});
      hg.appendChild(svgEl('rect', { x: hX, y: hy - 14, width: 86, height: 28, rx: 4, fill: BG, stroke: BP, 'stroke-width': '1.4' }));
      hg.appendChild(txt(hX + 43, hy + 4, 'head ' + (j + 1), SOFT, 10));
      hg.appendChild(anim('opacity', '0.4;1;0.4', '2.6s', { begin: (j * 0.4) + 's' }));
      svg.appendChild(hg);
      // 转向
      var c2 = svgEl('path', { d: 'M' + (hX + 86) + ' ' + hy + ' C ' + (470) + ' ' + hy + ', ' + 466 + ' ' + (inY + 20) + ', ' + 486 + ' ' + (inY + 20), fill: 'none', stroke: MUTE, 'stroke-width': '1.2', opacity: '0.5' });
      svg.appendChild(c2);
    }
    svg.appendChild(svgEl('rect', { x: 486, y: inY, width: 30, height: 40, rx: 5, fill: BP, opacity: '0.12', stroke: BP, 'stroke-width': '1.6' }));
    svg.appendChild(txt(501, inY + 24, '∥', BP, 16));
    svg.appendChild(txt(348, 12, 'scaled dot-product · causal mask · softmax', MUTE, 9, 'middle'));
    card(host, 'MULTI-HEAD ATTENTION', 'one projection · H heads',
      svg,
      'A single linear layer projects the input to Q, K, and V, then reshapes into H parallel heads of size D/H. Each head runs scaled dot-product attention under the causal mask independently, and the outputs concatenate back to D. The heads specialize during training.');
  }

  // 训练循环:向前,向后,步骤,然后下降损失
  function trainingLoop(host) {
    var svg = svgEl('svg', { viewBox: '0 0 520 240' });
    // 沿环路排列的训练阶段流水线
    var cx = 150, cy = 110, r = 78;
    var stages = ['batch', 'forward', 'loss', 'backward', 'AdamW'];
    var ring = svgEl('circle', { cx: cx, cy: cy, r: r, fill: 'none', stroke: RULE, 'stroke-width': '1.5', 'stroke-dasharray': '4 5' });
    svg.appendChild(ring);
    var k;
    for (k = 0; k < stages.length; k++) {
      var ang = -Math.PI / 2 + k * 2 * Math.PI / stages.length;
      var sx = cx + r * Math.cos(ang), sy = cy + r * Math.sin(ang);
      var g = svgEl('g', {});
      g.appendChild(svgEl('circle', { cx: sx, cy: sy, r: 7, fill: BP }));
      g.appendChild(txt(sx, sy - 12, stages[k], SOFT, 9));
      g.appendChild(anim('opacity', '0.35;1;0.35', '2.5s', { begin: (k * 0.5) + 's' }));
      svg.appendChild(g);
    }
    // 沿环路移动的 token
    var dot = svgEl('circle', { r: 5, fill: WARN });
    dot.appendChild(svgEl('animateMotion', { dur: '2.5s', repeatCount: 'indefinite', path: 'M' + cx + ' ' + (cy - r) + ' A ' + r + ' ' + r + ' 0 1 1 ' + (cx - 0.01) + ' ' + (cy - r) + ' Z' }));
    svg.appendChild(dot);
    svg.appendChild(txt(cx, cy + 4, 'every', MUTE, 9));
    svg.appendChild(txt(cx, cy + 16, 'step', MUTE, 9));
    // 损失曲线向右下降
    var bx = 280, by = 40, bw = 210, bh = 150;
    svg.appendChild(svgEl('line', { x1: bx, y1: by, x2: bx, y2: by + bh, stroke: RULE, 'stroke-width': '1' }));
    svg.appendChild(svgEl('line', { x1: bx, y1: by + bh, x2: bx + bw, y2: by + bh, stroke: RULE, 'stroke-width': '1' }));
    svg.appendChild(txt(bx - 6, by + 6, 'loss', MUTE, 9, 'end'));
    svg.appendChild(txt(bx + bw, by + bh + 14, 'steps', MUTE, 9, 'end'));
    var d = 'M' + bx + ' ' + (by + 8), i, pts = 60;
    for (i = 1; i <= pts; i++) {
      var t = i / pts;
      var lx = bx + t * bw;
      var ly = by + 8 + (bh - 20) * (1 - Math.exp(-3.2 * t)) - 4 * Math.sin(t * 22) * Math.exp(-2 * t);
      d += ' L' + lx.toFixed(1) + ' ' + ly.toFixed(1);
    }
    var curve = svgEl('path', { d: d, fill: 'none', stroke: BP, 'stroke-width': '2', 'stroke-dasharray': '600', 'stroke-dashoffset': '600' });
    curve.appendChild(anim('stroke-dashoffset', '600;0', '4s'));
    svg.appendChild(curve);
    card(host, 'TRAINING LOOP', 'forward · backward · step',
      svg,
      'Each step pulls a batch, runs the forward pass to logits, scores cross-entropy loss, backpropagates, and takes an AdamW step under the LR schedule. Repeated thousands of times, the loss curve falls and flattens. Periodic eval and sample probes catch divergence the scalar loss hides.');
  }

  // ── 分类头替换：冻结主干，移除旧头，接入新的二分类头 ──
  function headSwap(host) {
    var svg = svgEl('svg', { viewBox: '0 0 520 230' });
    var bx = 60, by = 40, bw = 150, bh = 150;
    // 冻结的模型主干
    svg.appendChild(svgEl('rect', { x: bx, y: by, width: bw, height: bh, rx: 6, fill: SURF, stroke: MUTE, 'stroke-width': '1.6', 'stroke-dasharray': '5 4' }));
    svg.appendChild(txt(bx + bw / 2, by + 22, 'transformer body', SOFT, 11));
    svg.appendChild(txt(bx + bw / 2, by + 38, '(frozen)', MUTE, 10));
    var ly;
    for (ly = 0; ly < 4; ly++) {
      var yy = by + 58 + ly * 24;
      var lr = svgEl('rect', { x: bx + 22, y: yy, width: bw - 44, height: 16, rx: 3, fill: BG, stroke: MUTE, 'stroke-width': '1' });
      svg.appendChild(lr);
      // 子的雪花脉冲,读取"结"
      svg.appendChild(txt(bx + bw / 2, yy + 12, 'block ' + (ly + 1), MUTE, 8));
    }
    // 射箭出身 (聚合式表示)
    var arr = svgEl('line', { x1: bx + bw, y1: by + bh / 2, x2: bx + bw + 50, y2: by + bh / 2, stroke: BP, 'stroke-width': '2', 'stroke-dasharray': '6 5' });
    arr.appendChild(anim('stroke-dashoffset', '22;0', '1s'));
    svg.appendChild(arr);
    svg.appendChild(txt(bx + bw + 25, by + bh / 2 - 8, 'pooled', MUTE, 9));
    // 旧头 (上)
    var oldH = svgEl('g', {});
    oldH.appendChild(svgEl('rect', { x: 290, y: 40, width: 130, height: 44, rx: 6, fill: 'none', stroke: MUTE, 'stroke-width': '1.6', 'stroke-dasharray': '4 4' }));
    oldH.appendChild(txt(355, 60, 'old LM head', MUTE, 11));
    oldH.appendChild(txt(355, 76, 'vocab projection', MUTE, 9));
    oldH.appendChild(animT('translate', '0 0;36 -18', '3s', { calcMode: 'spline', keySplines: '0.4 0 0.6 1', keyTimes: '0;1', additive: 'sum' }));
    oldH.appendChild(anim('opacity', '1;0.15', '3s'));
    svg.appendChild(oldH);
    // 新的头脑 (下面,蓝图)
    var newH = svgEl('g', {});
    newH.appendChild(svgEl('rect', { x: 290, y: 130, width: 130, height: 50, rx: 6, fill: BP, opacity: '0.14', stroke: BP, 'stroke-width': '2.2' }));
    newH.appendChild(txt(355, 152, 'new classifier head', BP, 10));
    newH.appendChild(txt(355, 168, 'linear → 2 logits', SOFT, 9));
    newH.appendChild(anim('opacity', '0.1;1', '1.4s', { begin: '0.8s', fill: 'freeze' }));
    svg.appendChild(newH);
    svg.appendChild(svgEl('line', { x1: 220, y1: by + bh / 2, x2: 290, y2: 155, stroke: BP, 'stroke-width': '1.6' }));
    card(host, 'CLASSIFIER HEAD SWAP', 'freeze body · swap head',
      svg,
      'A pretrained model is a frozen body plus a head. To classify, you keep the expensive body and lift off the vocabulary head, gluing on a small linear layer to two logits. Train head-only for speed or unfreeze the body for accuracy when the domain drifts.');
  }

  // ── DPO：相对冻结参考模型，提高选中回答概率并降低拒绝回答概率 ──
  function dpoPreference(host) {
    var svg = svgEl('svg', { viewBox: '0 0 520 230' });
    var bx = 70, by = 30, bw = 380, bh = 150;
    svg.appendChild(svgEl('line', { x1: bx, y1: by, x2: bx, y2: by + bh, stroke: RULE, 'stroke-width': '1' }));
    svg.appendChild(svgEl('line', { x1: bx, y1: by + bh, x2: bx + bw, y2: by + bh, stroke: RULE, 'stroke-width': '1' }));
    svg.appendChild(txt(bx - 8, by + 8, 'log p', MUTE, 9, 'end'));
    svg.appendChild(txt(bx + bw, by + bh + 14, 'training steps', MUTE, 9, 'end'));
    // 结结冰的参考线 (平面分段)
    var refY = by + bh / 2;
    svg.appendChild(svgEl('line', { x1: bx, y1: refY, x2: bx + bw, y2: refY, stroke: MUTE, 'stroke-width': '1.4', 'stroke-dasharray': '6 4' }));
    svg.appendChild(txt(bx + bw - 4, refY - 6, 'frozen reference π_ref', MUTE, 9, 'end'));
    // 选择的升级
    var i, pts = 50, dc = 'M' + bx + ' ' + refY, dr = 'M' + bx + ' ' + refY;
    for (i = 1; i <= pts; i++) {
      var t = i / pts, lx = bx + t * bw;
      var up = refY - (bh / 2 - 14) * (1 - Math.exp(-2.8 * t));
      var dn = refY + (bh / 2 - 14) * (1 - Math.exp(-2.4 * t));
      dc += ' L' + lx.toFixed(1) + ' ' + up.toFixed(1);
      dr += ' L' + lx.toFixed(1) + ' ' + dn.toFixed(1);
    }
    var cv = svgEl('path', { d: dc, fill: 'none', stroke: BP, 'stroke-width': '2.4', 'stroke-dasharray': '480', 'stroke-dashoffset': '480' });
    cv.appendChild(anim('stroke-dashoffset', '480;0', '3.6s'));
    svg.appendChild(cv);
    var rv = svgEl('path', { d: dr, fill: 'none', stroke: WARN, 'stroke-width': '2.4', 'stroke-dasharray': '480', 'stroke-dashoffset': '480' });
    rv.appendChild(anim('stroke-dashoffset', '480;0', '3.6s'));
    svg.appendChild(rv);
    svg.appendChild(txt(bx + bw - 4, by + 18, 'chosen  y_w  ↑', BP, 10, 'end'));
    svg.appendChild(txt(bx + bw - 4, by + bh - 8, 'rejected  y_l  ↓', WARN, 10, 'end'));
    // 状扩张差距标志
    var gap = svgEl('g', {});
    gap.appendChild(svgEl('line', { x1: bx + bw - 60, y1: refY - (bh / 2 - 16), x2: bx + bw - 60, y2: refY + (bh / 2 - 16), stroke: INK, 'stroke-width': '1', 'stroke-dasharray': '2 3' }));
    gap.appendChild(anim('opacity', '0;0;1', '3.6s', { fill: 'freeze' }));
    svg.appendChild(gap);
    card(host, 'DPO PREFERENCE', 'chosen ↑ · rejected ↓',
      svg,
      'DPO trains the policy directly on chosen-over-rejected pairs, with no separate reward model. The loss is a sigmoid over the log-probability gap relative to a frozen reference, so the chosen completion is pushed up and the rejected pulled down while the KL anchor keeps the policy honest.');
  }

  // ,再开始,再开始,再开始,再开始
  function corpusStream(host) {
    var svg = svgEl('svg', { viewBox: '0 0 520 230' });
    // 源云
    svg.appendChild(svgEl('rect', { x: 24, y: 90, width: 70, height: 44, rx: 22, fill: BG, stroke: BP, 'stroke-width': '1.8' }));
    svg.appendChild(txt(59, 110, 'remote', SOFT, 10));
    svg.appendChild(txt(59, 124, 'shards', SOFT, 10));
    // 管道,有流动的字节
    var pipeY = 112;
    svg.appendChild(svgEl('line', { x1: 94, y1: pipeY, x2: 250, y2: pipeY, stroke: RULE, 'stroke-width': '8' }));
    var path = 'M100 ' + pipeY + ' H244';
    var b;
    for (b = 0; b < 4; b++) {
      var dot = svgEl('circle', { r: 4, fill: BP });
      dot.appendChild(svgEl('animateMotion', { dur: '2.4s', repeatCount: 'indefinite', path: path, begin: (b * 0.6) + 's' }));
      svg.appendChild(dot);
    }
    // 简历标记:闪的字节对比标志 (网络下降 + 范围简历)
    var resume = svgEl('g', {});
    resume.appendChild(svgEl('line', { x1: 175, y1: pipeY - 22, x2: 175, y2: pipeY + 22, stroke: WARN, 'stroke-width': '2' }));
    resume.appendChild(txt(175, pipeY - 28, 'Range resume', WARN, 8));
    resume.appendChild(anim('opacity', '1;0.15;1', '2.4s'));
    svg.appendChild(resume);
    // 压缩区块
    svg.appendChild(svgEl('rect', { x: 250, y: 92, width: 60, height: 40, rx: 5, fill: BP, opacity: '0.12', stroke: BP, 'stroke-width': '1.8' }));
    svg.appendChild(txt(280, 110, 'zstd', BP, 10));
    svg.appendChild(txt(280, 124, 'stream', SOFT, 8));
    // 文档落入 LSH 桶，近重复项会被标记并丢弃
    var bx = 340, j;
    for (j = 0; j < 3; j++) {
      var bxj = bx + j * 56;
      svg.appendChild(svgEl('rect', { x: bxj, y: 150, width: 44, height: 34, rx: 4, fill: BG, stroke: RULE, 'stroke-width': '1.4' }));
      svg.appendChild(txt(bxj + 22, 196, 'LSH ' + (j + 1), MUTE, 8));
      var falling = svgEl('circle', { cx: bxj + 22, cy: 70, r: 4, fill: j === 1 ? WARN : BP });
      falling.appendChild(anim('cy', '70;167', '2.2s', { begin: (j * 0.5) + 's' }));
      falling.appendChild(anim('opacity', j === 1 ? '1;1;0.1' : '1;1;1', '2.2s', { begin: (j * 0.5) + 's' }));
      svg.appendChild(falling);
    }
    svg.appendChild(txt(bx + 56, 36, 'duplicate dropped', WARN, 9, 'middle'));
    // 报表
    svg.appendChild(svgEl('line', { x1: 310, y1: 112, x2: 340, y2: 112, stroke: BP, 'stroke-width': '2' }));
    card(host, 'CORPUS DOWNLOADER', 'stream · resume · dedup',
      svg,
      'Shards stream from the remote source and decompress through Zstandard without buffering the whole file. A verified byte offset lets a dropped connection resume with an HTTP Range request, and each document gets a MinHash signature bucketed by LSH so near-duplicates collide and drop before reaching the manifest.');
  }

  // 升:一个标志物在升坡上进入升的升.
  function cosineWarmup(host) {
    var svg = svgEl('svg', { viewBox: '0 0 520 230' });
    var bx = 50, by = 30, bw = 430, bh = 150, base = by + bh;
    svg.appendChild(svgEl('line', { x1: bx, y1: by, x2: bx, y2: base, stroke: RULE, 'stroke-width': '1' }));
    svg.appendChild(svgEl('line', { x1: bx, y1: base, x2: bx + bw, y2: base, stroke: RULE, 'stroke-width': '1' }));
    svg.appendChild(txt(bx - 6, by + 6, 'lr', MUTE, 9, 'end'));
    svg.appendChild(txt(bx + bw, base + 14, 'step', MUTE, 9, 'end'));
    var warmFrac = 0.18, peak = bh - 14, minY = bh * 0.06;
    function lrAt(t) {
      if (t < warmFrac) return peak * (t / warmFrac);
      var u = (t - warmFrac) / (1 - warmFrac);
      return minY + (peak - minY) * 0.5 * (1 + Math.cos(Math.PI * u));
    }
    function X(t) { return bx + t * bw; }
    function Y(v) { return base - v; }
    // 热化区域阴影标记
    svg.appendChild(svgEl('rect', { x: bx, y: by, width: bw * warmFrac, height: bh, fill: WARN, opacity: '0.07' }));
    svg.appendChild(txt(bx + bw * warmFrac / 2, by + 14, 'warmup', WARN, 9, 'middle'));
    svg.appendChild(txt(bx + bw * (warmFrac + (1 - warmFrac) / 2), by + 14, 'cosine decay', BP, 9, 'middle'));
    var i, pts = 120, d = 'M' + X(0) + ' ' + Y(lrAt(0));
    for (i = 1; i <= pts; i++) { var t = i / pts; d += ' L' + X(t).toFixed(1) + ' ' + Y(lrAt(t)).toFixed(1); }
    var curve = svgEl('path', { d: d, fill: 'none', stroke: BP, 'stroke-width': '2.4', 'stroke-dasharray': '700', 'stroke-dashoffset': '700' });
    curve.appendChild(anim('stroke-dashoffset', '700;0', '5s', { repeatCount: '1', fill: 'freeze' }));
    svg.appendChild(curve);
    // 标记点骑行时间表
    var sampleX = [], sampleY = [];
    for (i = 0; i <= 40; i++) { var tt = i / 40; sampleX.push(X(tt).toFixed(1)); sampleY.push(Y(lrAt(tt)).toFixed(1)); }
    var dot = svgEl('circle', { r: 6, fill: WARN });
    dot.appendChild(anim('cx', sampleX.join(';'), '5s'));
    dot.appendChild(anim('cy', sampleY.join(';'), '5s'));
    svg.appendChild(dot);
    // 顶点点线路指南
    svg.appendChild(svgEl('line', { x1: bx, y1: Y(peak), x2: X(warmFrac), y2: Y(peak), stroke: RULE, 'stroke-width': '1', 'stroke-dasharray': '3 3' }));
    svg.appendChild(txt(bx + 4, Y(peak) - 4, 'lr_max', SOFT, 8, 'start'));
    card(host, 'COSINE LR WARMUP', 'ramp up · decay down',
      svg,
      'The schedule ramps the learning rate linearly from zero to lr_max over the warmup steps, protecting the brittle first updates, then follows the upper half of a cosine curve back toward lr_min. The marker rides the exact value the optimizer uses at each step.');
  }

  LF.register({
    'cap-bpe-merge': bpeMerge,
    'cap-sliding-window': slidingWindow,
    'cap-multihead-attention': multiHead,
    'cap-training-loop': trainingLoop,
    'cap-classifier-head-swap': headSwap,
    'cap-dpo-preference': dpoPreference,
    'cap-corpus-downloader': corpusStream,
    'cap-cosine-warmup': cosineWarmup
  });
})();
