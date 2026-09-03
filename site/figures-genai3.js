/* figures-genai3.js：阶段 8（生成式 AI）和阶段 5（NLP）的动画课程图示。
   在 lesson-figures.js 之后加载，并通过 window.LF 注册。无依赖，仅使用 ES5，
   通过 CSS 变量应用主题。动画仅使用 SMIL（声明式），无 JS 循环和实际计算。
   每个图示都是单个静态 SVG 场景，由浏览器添加动画；在无头或减少动态效果模式下
   渲染为静止帧。 */
(function () {
  'use strict';
  var LF = window.LF;
  if (!LF) { return; }
  var el = LF.el, svgEl = LF.svgEl;

  var BP = 'var(--blueprint,#3553ff)';
  var MUTE = 'var(--ink-mute,#999)';
  var SOFT = 'var(--rule-soft,#ddd)';
  var WARN = 'var(--warn,#b8870f)';
  var INK = 'var(--ink,#1a1a1a)';

  function frame(host, label, hint, svg, caption) {
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, [label]), el('span', {}, [hint])]),
      el('div', { class: 'lf-body' }, [el('div', { class: 'lf-out' }, [svg])]),
      el('div', { class: 'lf-cap' }, [caption])
    ]));
  }
  function anim(attr, vals, dur, extra) {
    var a = { attributeName: attr, values: vals, dur: dur, repeatCount: 'indefinite' };
    if (extra) for (var k in extra) a[k] = extra[k];
    if (a.keySplines && !a.calcMode) a.calcMode = 'spline';
    return svgEl('animate', a);
  }
  function txt(x, y, s, size, fill, anchor) {
    return svgEl('text', { x: x, y: y, 'text-anchor': anchor || 'middle', 'font-size': size || 11, fill: fill || MUTE, 'font-family': 'monospace' }, [document.createTextNode(s)]);
  }

  // ── gx-var-next-scale：下一尺度预测，网格按 1→2→4→8 增长 ─────────────────
  function varNextScale(host) {
    var W = 520, H = 250, svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var scales = [1, 2, 4, 8];
    var bx = [40, 150, 280, 430], by = 70, size = 96;
    scales.forEach(function (n, si) {
      var x0 = bx[si], cell = size / n;
      // 按尺度顺序出现：阶段 si 在 si/4 时刻亮起
      var t0 = si / 4, t1 = (si + 0.5) / 4;
      var g = svgEl('g', {});
      // 边框
      g.appendChild(svgEl('rect', { x: x0, y: by, width: size, height: size, fill: 'none', stroke: SOFT, 'stroke-width': '1' }));
      var r, c;
      for (r = 0; r < n; r++) for (c = 0; c < n; c++) {
        var rect = svgEl('rect', { x: (x0 + c * cell).toFixed(1), y: (by + r * cell).toFixed(1), width: (cell - 1.2).toFixed(1), height: (cell - 1.2).toFixed(1), fill: BP, opacity: '0' });
        rect.appendChild(anim('opacity', '0;0;0.85;0.85', '8s', { keyTimes: '0;' + t0.toFixed(3) + ';' + t1.toFixed(3) + ';1', keySplines: '0 0 1 1;.4 0 .2 1;0 0 1 1' }));
        g.appendChild(rect);
      }
      g.appendChild(txt(x0 + size / 2, by + size + 16, n + 'x' + n, 11, MUTE));
      svg.appendChild(g);
      // 指向下一尺度的条件箭头
      if (si < scales.length - 1) {
        var ax = x0 + size + 4, axe = bx[si + 1] - 4;
        var ar = svgEl('line', { x1: ax, y1: by + size / 2, x2: axe, y2: by + size / 2, stroke: MUTE, 'stroke-width': '1.6', 'marker-end': 'none', opacity: '0.25' });
        ar.appendChild(anim('opacity', '0.25;0.25;1;0.25', '8s', { keyTimes: '0;' + t1.toFixed(3) + ';' + ((si + 1) / 4).toFixed(3) + ';1', keySplines: '0 0 1 1;.4 0 .2 1;0 0 1 1' }));
        svg.appendChild(ar);
        svg.appendChild(svgEl('polygon', { points: (axe - 6) + ',' + (by + size / 2 - 4) + ' ' + axe + ',' + (by + size / 2) + ' ' + (axe - 6) + ',' + (by + size / 2 + 4), fill: MUTE }));
      }
    });
    svg.appendChild(txt(W / 2, 30, 'predict each scale in one pass, conditioned on all coarser scales', 11, MUTE));
    frame(host, 'NEXT-SCALE PREDICTION', 'coarse to fine', svg,
      'VAR generates an image as a sequence of resolutions, not pixels in raster order. It predicts a 1x1 token summary, then a 2x2 grid, then 4x4, then 8x8, each scale produced in a single parallel pass and conditioned on every coarser scale before it. The generation-order problem of pixel-by-pixel models disappears.');
  }

  // ── gx-fid-distributions：用两个特征云之间的距离表示 FID ────────────────
  function fidDistributions(host) {
    var W = 520, H = 240, svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    svg.appendChild(svgEl('rect', { x: 30, y: 40, width: W - 60, height: H - 80, fill: 'none', stroke: SOFT, 'stroke-width': '1' }));
    svg.appendChild(txt(40, 30, 'Inception feature space (2 of 2048 dims)', 11, MUTE, 'start'));
    // 真实特征云：固定的近似高斯团块，中心偏左下；生成特征云从远处移动至重叠
    var realC = [180, 150], genStart = [400, 90], genEnd = [205, 138];
    function blob(cx, cy, st, pts) {
      var g = svgEl('g', {}), i;
      var off = [[0, 0], [22, -14], [-18, 16], [30, 18], [-26, -20], [12, 28], [-34, 4], [8, -30]];
      for (i = 0; i < pts; i++) {
        g.appendChild(svgEl('circle', { cx: (cx + off[i][0]).toFixed(1), cy: (cy + off[i][1]).toFixed(1), r: '4', fill: st, opacity: '0.8' }));
      }
      return g;
    }
    svg.appendChild(blob(realC[0], realC[1], BP, 8));
    // 生成团块通过 animateTransform 从初始偏移位置平移至重叠位置
    var genG = blob(genEnd[0], genEnd[1], WARN, 8);
    var dx0 = genStart[0] - genEnd[0], dy0 = genStart[1] - genEnd[1];
    genG.appendChild(svgEl('animateTransform', { attributeName: 'transform', type: 'translate', values: dx0 + ' ' + dy0 + ';0 0;0 0;' + dx0 + ' ' + dy0, keyTimes: '0;0.45;0.7;1', dur: '9s', repeatCount: 'indefinite', calcMode: 'spline', keySplines: '.4 0 .2 1;0 0 1 1;.4 0 .2 1' }));
    svg.appendChild(genG);
    // 均值之间的距离线
    var line = svgEl('line', { x1: realC[0], y1: realC[1], x2: genStart[0], y2: genStart[1], stroke: INK, 'stroke-width': '1.4', 'stroke-dasharray': '5 4' });
    line.appendChild(anim('x2', genStart[0] + ';' + genEnd[0] + ';' + genEnd[0] + ';' + genStart[0], '9s', { keyTimes: '0;0.45;0.7;1', keySplines: '.4 0 .2 1;0 0 1 1;.4 0 .2 1' }));
    line.appendChild(anim('y2', genStart[1] + ';' + genEnd[1] + ';' + genEnd[1] + ';' + genStart[1], '9s', { keyTimes: '0;0.45;0.7;1', keySplines: '.4 0 .2 1;0 0 1 1;.4 0 .2 1' }));
    svg.appendChild(line);
    svg.appendChild(svgEl('circle', { cx: realC[0], cy: realC[1], r: '3', fill: INK }));
    svg.appendChild(txt(110, 215, 'real (blue)   generated (amber)', 11, MUTE, 'start'));
    var fidLabel = txt(W - 40, 215, 'FID = high', 12, WARN, 'end');
    fidLabel.appendChild(svgEl('animate', { attributeName: 'fill', values: WARN + ';' + BP + ';' + BP + ';' + WARN, keyTimes: '0;0.45;0.7;1', dur: '9s', repeatCount: 'indefinite' }));
    svg.appendChild(fidLabel);
    frame(host, 'FRECHET INCEPTION DISTANCE', 'two distributions', svg,
      'FID does not score images one by one. It fits a Gaussian to the real images and another to the generated images in Inception feature space, then measures the distance between those two distributions through their means and covariances. As the generated cloud moves to overlap the real one, the distance, and the FID, falls toward zero.');
  }

  // ── gx-patchgan：判别器为重叠图块构成的网格评分 ─────────────────────────
  function patchgan(host) {
    var W = 520, H = 240, svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var ix = 60, iy = 50, isz = 150, n = 4;
    svg.appendChild(txt(ix + isz / 2, 36, 'generated image y  (conditioned on input x)', 11, MUTE));
    svg.appendChild(svgEl('rect', { x: ix, y: iy, width: isz, height: isz, fill: BP, opacity: '0.06', stroke: SOFT, 'stroke-width': '1' }));
    var cell = isz / n, r, c, k = 0;
    // 右侧为判定网格；感受野框扫过时，各图块依次亮起
    var gx = 330, gy = 50, gcell = 30;
    var rf = svgEl('rect', { x: ix, y: iy, width: cell + 8, height: cell + 8, fill: 'none', stroke: WARN, 'stroke-width': '1.8' });
    for (r = 0; r < n; r++) for (c = 0; c < n; c++) {
      var t = k / (n * n);
      var verdict = svgEl('rect', { x: gx + c * gcell, y: gy + r * gcell, width: gcell - 2, height: gcell - 2, fill: BP, opacity: '0.1', stroke: SOFT, 'stroke-width': '0.8' });
      verdict.appendChild(anim('opacity', '0.1;0.1;0.85;0.85', '8s', { keyTimes: '0;' + t.toFixed(3) + ';' + Math.min(1, t + 0.05).toFixed(3) + ';1', keySplines: '0 0 1 1;.3 0 .2 1;0 0 1 1' }));
      svg.appendChild(verdict);
      k++;
    }
    // 感受野标记逐单元格扫过图像
    var rx = [], ry = [], kt = [];
    for (r = 0; r < n; r++) for (c = 0; c < n; c++) { rx.push((ix + c * cell - 4).toFixed(1)); ry.push((iy + r * cell - 4).toFixed(1)); kt.push((rx.length - 1) / (n * n)); }
    kt.push(1); rx.push(rx[rx.length - 1]); ry.push(ry[ry.length - 1]);
    rf.appendChild(svgEl('animate', { attributeName: 'x', values: rx.join(';'), keyTimes: kt.join(';'), dur: '8s', repeatCount: 'indefinite', calcMode: 'discrete' }));
    rf.appendChild(svgEl('animate', { attributeName: 'y', values: ry.join(';'), keyTimes: kt.join(';'), dur: '8s', repeatCount: 'indefinite', calcMode: 'discrete' }));
    svg.appendChild(rf);
    svg.appendChild(txt(gx + (n * gcell) / 2 - 1, gy + n * gcell + 18, 'NxN real/fake grid', 11, MUTE));
    svg.appendChild(txt(ix + isz / 2, iy + isz + 18, '70x70 receptive field', 11, WARN));
    frame(host, 'PATCHGAN DISCRIMINATOR', 'local realism', svg,
      'A PatchGAN does not output one real/fake score for the whole image. It slides a fixed receptive field across the output and judges each local patch independently, producing an NxN grid of verdicts that is then averaged. Realism is treated as a local property, which makes the discriminator smaller, faster, and sharper on high-frequency texture.');
  }

  // ── gx-stylegan-mapping：z 相互纠缠，w 相互解耦，并按尺度注入 ────────────
  function styleganMapping(host) {
    var W = 520, H = 250, svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    // 左侧：纠缠的 z（盘绕路径）；MLP；解耦的 w（与坐标轴对齐）；注入刻度
    svg.appendChild(txt(70, 30, 'Z: entangled', 11, MUTE));
    svg.appendChild(txt(250, 30, 'f (8-layer MLP)', 11, BP));
    svg.appendChild(txt(440, 30, 'W: disentangled', 11, MUTE));
    // 纠缠团块：一条打结路径
    svg.appendChild(svgEl('rect', { x: 30, y: 50, width: 110, height: 110, fill: 'none', stroke: SOFT, 'stroke-width': '1' }));
    var knot = svgEl('path', { d: 'M50 70 C 120 60, 60 130, 110 120 S 50 150, 120 90 S 70 70, 100 140', fill: 'none', stroke: BP, 'stroke-width': '1.8', 'stroke-dasharray': '260', 'stroke-dashoffset': '260' });
    knot.appendChild(anim('stroke-dashoffset', '260;0;0;260', '8s', { keyTimes: '0;0.35;0.7;1', keySplines: '.4 0 .2 1;0 0 1 1;.4 0 .2 1' }));
    svg.appendChild(knot);
    // 映射箭头
    svg.appendChild(svgEl('line', { x1: 145, y1: 105, x2: 200, y2: 105, stroke: MUTE, 'stroke-width': '1.4' }));
    svg.appendChild(svgEl('rect', { x: 200, y: 78, width: 90, height: 54, fill: BP, opacity: '0.08', stroke: SOFT, 'stroke-width': '1' }));
    var li;
    for (li = 0; li < 4; li++) svg.appendChild(svgEl('line', { x1: 210, y1: 88 + li * 11, x2: 280, y2: 88 + li * 11, stroke: BP, 'stroke-width': '1', opacity: '0.5' }));
    svg.appendChild(svgEl('line', { x1: 290, y1: 105, x2: 345, y2: 105, stroke: MUTE, 'stroke-width': '1.4' }));
    // 解耦的 W：两条清晰的正交轴（姿态/光照）
    svg.appendChild(svgEl('rect', { x: 350, y: 50, width: 110, height: 110, fill: 'none', stroke: SOFT, 'stroke-width': '1' }));
    var axH = svgEl('line', { x1: 360, y1: 105, x2: 360, y2: 105, stroke: BP, 'stroke-width': '2' });
    axH.appendChild(anim('x2', '360;450;450;360', '8s', { keyTimes: '0;0.5;0.7;1', keySplines: '.4 0 .2 1;0 0 1 1;.4 0 .2 1' }));
    var axV = svgEl('line', { x1: 405, y1: 155, x2: 405, y2: 155, stroke: WARN, 'stroke-width': '2' });
    axV.appendChild(anim('y2', '155;60;60;155', '8s', { keyTimes: '0;0.5;0.7;1', keySplines: '.4 0 .2 1;0 0 1 1;.4 0 .2 1' }));
    svg.appendChild(axH); svg.appendChild(axV);
    svg.appendChild(txt(405, 178, 'pose axis', 9, BP));
    // 沿合成堆栈向下排列的注入刻度
    var sy;
    for (sy = 0; sy < 4; sy++) {
      var ty = 200 + 0; // 单行分辨率块
      var rect = svgEl('rect', { x: 60 + sy * 100, y: 195, width: 70, height: 32, fill: BP, opacity: '0.08', stroke: SOFT, 'stroke-width': '1' });
      svg.appendChild(rect);
      svg.appendChild(txt(95 + sy * 100, 215, ['4x4', '16x16', '64x64', '1024'][sy], 10, MUTE));
      var inj = svgEl('circle', { cx: 95 + sy * 100, cy: 195, r: '3', fill: WARN, opacity: '0.2' });
      inj.appendChild(anim('opacity', '0.2;0.2;1;0.2', '8s', { keyTimes: '0;' + (0.55 + sy * 0.08).toFixed(2) + ';' + (0.6 + sy * 0.08).toFixed(2) + ';1', keySplines: '0 0 1 1;.3 0 .2 1;0 0 1 1' }));
      svg.appendChild(inj);
    }
    svg.appendChild(txt(W / 2, 245, 'w injected via AdaIN at every resolution', 10, WARN));
    frame(host, 'STYLEGAN MAPPING', 'untangle then inject', svg,
      'A plain generator feeds the noise vector z straight into the network, so every factor of variation stays tangled together. StyleGAN first maps z through an 8-layer MLP to an intermediate space W whose axes line up with meaningful factors like pose and lighting, then injects that w through AdaIN at every resolution. Disentangling first is what made style mixing and editing possible.');
  }

  // ── gx-hybrid-retrieval：通过 RRF 融合稀疏与稠密列表，然后重新排序 ───────
  function hybridRetrieval(host) {
    var W = 520, H = 250, svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    function col(x, title, items, color) {
      svg.appendChild(txt(x + 30, 38, title, 11, color));
      items.forEach(function (lab, i) {
        svg.appendChild(svgEl('rect', { x: x, y: 50 + i * 26, width: 60, height: 20, fill: color, opacity: (0.5 - i * 0.08).toFixed(2), stroke: SOFT, 'stroke-width': '0.6' }));
        svg.appendChild(txt(x + 30, 64 + i * 26, lab, 9, INK));
      });
    }
    col(30, 'BM25 (sparse)', ['D7', 'D2', 'D9', 'D4'], BP);
    col(130, 'dense', ['D2', 'D5', 'D7', 'D1'], MUTE);
    // RRF 融合列（中间）淡入
    var fg = svgEl('g', { opacity: '0' });
    fg.appendChild(anim('opacity', '0;0;1;1', '8s', { keyTimes: '0;0.35;0.5;1', keySplines: '0 0 1 1;.3 0 .2 1;0 0 1 1' }));
    fg.appendChild(txt(290, 38, 'RRF fusion', 11, WARN));
    ['D2', 'D7', 'D5', 'D9'].forEach(function (lab, i) {
      fg.appendChild(svgEl('rect', { x: 260, y: 50 + i * 26, width: 60, height: 20, fill: WARN, opacity: (0.5 - i * 0.08).toFixed(2), stroke: SOFT, 'stroke-width': '0.6' }));
      fg.appendChild(txt(290, 64 + i * 26, lab, 9, INK));
    });
    svg.appendChild(fg);
    // 从两个列表指向融合结果的合并箭头
    [70, 170].forEach(function (sx) {
      var ar = svgEl('line', { x1: sx + 20, y1: 100, x2: 255, y2: 90, stroke: SOFT, 'stroke-width': '1' });
      ar.appendChild(anim('opacity', '0;0;1;1', '8s', { keyTimes: '0;0.3;0.45;1' }));
      svg.appendChild(ar);
    });
    // 交叉编码器重排列最后淡入，并突出显示重排后的首位结果
    var rg = svgEl('g', { opacity: '0' });
    rg.appendChild(anim('opacity', '0;0;0;1;1', '8s', { keyTimes: '0;0.55;0.65;0.78;1', keySplines: '0 0 1 1;0 0 1 1;.3 0 .2 1;0 0 1 1' }));
    rg.appendChild(txt(430, 38, 'rerank top-5', 11, BP));
    ['D5', 'D2', 'D7'].forEach(function (lab, i) {
      rg.appendChild(svgEl('rect', { x: 400, y: 50 + i * 26, width: 60, height: 20, fill: i === 0 ? BP : SOFT, opacity: i === 0 ? '0.7' : '0.3', stroke: SOFT, 'stroke-width': '0.6' }));
      rg.appendChild(txt(430, 64 + i * 26, lab, 9, i === 0 ? 'var(--bg,#fff)' : INK));
    });
    svg.appendChild(rg);
    var ar2 = svgEl('line', { x1: 322, y1: 90, x2: 395, y2: 75, stroke: SOFT, 'stroke-width': '1' });
    ar2.appendChild(anim('opacity', '0;0;0;1', '8s', { keyTimes: '0;0.55;0.65;1' }));
    svg.appendChild(ar2);
    svg.appendChild(txt(W / 2, 232, 'each layer catches the failures of the one before', 11, MUTE));
    frame(host, 'HYBRID RETRIEVAL', 'sparse + dense + rerank', svg,
      'Production retrieval is a chain, not a single method. Sparse BM25 nails exact keywords; dense vectors catch paraphrase; Reciprocal Rank Fusion merges the two ranked lists using only positions, so their incompatible score scales do not matter; a cross-encoder then reranks the survivors by reading each query and document together. The final top result can differ from either input list.');
  }

  // ── gx-matryoshka：截断嵌入维度，向量随之缩短 ───────────────────────────
  function matryoshka(host) {
    var W = 520, H = 230, svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var x0 = 40, y0 = 70, full = 440, h = 34, dims = 64;
    // 由单元格组成的全宽向量；截断线向左扫动，越过的单元格变暗
    svg.appendChild(txt(W / 2, 40, 'one Matryoshka vector, truncated to a smaller prefix', 11, MUTE));
    var i, cw = full / dims;
    for (i = 0; i < dims; i++) {
      var cell = svgEl('rect', { x: (x0 + i * cw).toFixed(1), y: y0, width: (cw - 0.6).toFixed(1), height: h, fill: BP });
      // 每个单元格的“保留”比例：dims 从完整值 (1.0) 扫至 1/8 (0.125)，再返回
      var keep = (i + 1) / dims; // 包含当前单元格的向量比例
      // 截断线在 t 属于 [0,0.4] 时从 1 扫至 .125，停留后在 [0.6,1] 时返回；
      // 截断线经过时每个单元格变暗，返回途中再次亮起
      var lo = '0.1', hiOp = '0.85';
      if (keep <= 0.125) {
        cell.setAttribute('opacity', hiOp);
      } else {
        var fk = (keep - 0.125) / 0.875;
        var tDim = 0.4 * (1 - fk);
        var tRise = Math.min(0.6 + 0.4 * fk, 0.96);
        var op = hiOp + ';' + hiOp + ';' + lo + ';' + lo + ';' + hiOp + ';' + hiOp;
        var kt = '0;' + tDim.toFixed(3) + ';' + (tDim + 0.03).toFixed(3) + ';' + tRise.toFixed(3) + ';' + (tRise + 0.03).toFixed(3) + ';1';
        cell.appendChild(anim('opacity', op, '7s', { keyTimes: kt }));
      }
      svg.appendChild(cell);
    }
    svg.appendChild(svgEl('rect', { x: x0, y: y0, width: full, height: h, fill: 'none', stroke: SOFT, 'stroke-width': '1' }));
    // 扫动的截断线
    var cut = svgEl('line', { x1: x0 + full, y1: y0 - 8, x2: x0 + full, y2: y0 + h + 8, stroke: WARN, 'stroke-width': '2' });
    cut.appendChild(anim('x1', (x0 + full) + ';' + (x0 + full / 8) + ';' + (x0 + full / 8) + ';' + (x0 + full), '7s', { keyTimes: '0;0.4;0.6;1', keySplines: '.4 0 .2 1;0 0 1 1;.4 0 .2 1' }));
    cut.appendChild(anim('x2', (x0 + full) + ';' + (x0 + full / 8) + ';' + (x0 + full / 8) + ';' + (x0 + full), '7s', { keyTimes: '0;0.4;0.6;1', keySplines: '.4 0 .2 1;0 0 1 1;.4 0 .2 1' }));
    svg.appendChild(cut);
    var lbl = txt(x0 + full + 4, y0 + h + 28, '3072 dims', 11, WARN, 'middle');
    lbl.appendChild(anim('x', (x0 + full) + ';' + (x0 + full / 8) + ';' + (x0 + full / 8) + ';' + (x0 + full), '7s', { keyTimes: '0;0.4;0.6;1', keySplines: '.4 0 .2 1;0 0 1 1;.4 0 .2 1' }));
    svg.appendChild(lbl);
    svg.appendChild(txt(x0, y0 + h + 28, 'prefix kept', 11, MUTE, 'start'));
    svg.appendChild(txt(W / 2, 200, 'storage scales with kept dimensions; quality degrades gracefully', 11, MUTE));
    frame(host, 'MATRYOSHKA TRUNCATION', 'shed dimensions', svg,
      'A Matryoshka-trained embedding packs the most important information into its earliest dimensions. You can keep only a prefix of the vector, slicing 3072 floats down to a few hundred, and still retrieve well. Truncation cuts index storage several-fold while quality falls off gradually rather than collapsing, so you tune the dimension budget to your cost target.');
  }

  // ── gx-entity-linking：提及项 → 候选知识库条目 → 上下文消歧
  function entityLinking(host) {
    var W = 520, H = 250, svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    svg.appendChild(txt(70, 36, 'mention', 11, MUTE));
    svg.appendChild(svgEl('rect', { x: 30, y: 48, width: 90, height: 40, fill: BP, opacity: '0.12', stroke: SOFT, 'stroke-width': '1' }));
    svg.appendChild(txt(75, 73, '"Jordan"', 12, INK));
    svg.appendChild(txt(75, 120, 'context: beat the press', 9, MUTE));
    var cands = ['Q41421  M. Jordan (NBA)', 'Q44437  M.B. Jordan (actor)', 'Q810  Jordan (country)', 'Q3308285  M.I. Jordan (ML)'];
    var cy = [40, 90, 140, 190], cx = 250, correct = 0;
    cands.forEach(function (lab, i) {
      // 候选生成：全部显示；随后消歧保留 #0，并将其余候选变暗
      var box = svgEl('rect', { x: cx, y: cy[i], width: 230, height: 34, fill: BP, opacity: '0.1', stroke: SOFT, 'stroke-width': '1' });
      var t0 = 0.2 + i * 0.06;
      box.appendChild(anim('opacity', '0;0;0.18;0.18;' + (i === correct ? '0.7' : '0.06') + ';' + (i === correct ? '0.7' : '0.06'),
        '8s', { keyTimes: '0;' + t0.toFixed(2) + ';' + (t0 + 0.05).toFixed(2) + ';0.6;0.72;1', keySplines: '0 0 1 1;.3 0 .2 1;0 0 1 1;.3 0 .2 1;0 0 1 1' }));
      svg.appendChild(box);
      var t = txt(cx + 12, cy[i] + 22, lab, 10, INK, 'start');
      svg.appendChild(t);
      // 从提及项连接到候选项的边
      var e = svgEl('line', { x1: 122, y1: 68, x2: cx, y2: cy[i] + 17, stroke: SOFT, 'stroke-width': '1', opacity: '0' });
      e.appendChild(anim('opacity', '0;0;0.6;0.6', '8s', { keyTimes: '0;' + t0.toFixed(2) + ';' + (t0 + 0.05).toFixed(2) + ';1' }));
      svg.appendChild(e);
    });
    // 获胜项上的勾选标记
    var ok = svgEl('circle', { cx: cx + 245, cy: cy[correct] + 17, r: '7', fill: BP, opacity: '0' });
    ok.appendChild(anim('opacity', '0;0;0;1;1', '8s', { keyTimes: '0;0.6;0.7;0.78;1' }));
    svg.appendChild(ok);
    var okt = txt(cx + 245, cy[correct] + 21, '✓', 11, 'var(--bg,#fff)', 'middle');
    okt.appendChild(anim('opacity', '0;0;0;1;1', '8s', { keyTimes: '0;0.6;0.7;0.78;1' }));
    svg.appendChild(okt);
    svg.appendChild(txt(135, 235, 'generate candidates, then disambiguate by context', 11, MUTE, 'start'));
    frame(host, 'ENTITY LINKING', 'candidates then pick', svg,
      'Entity linking runs in two stages. Candidate generation pulls every knowledge-base entry a surface form like "Jordan" could refer to. Disambiguation then scores each candidate against the surrounding context and keeps one. Here the sports context resolves the mention to the basketball entry while the actor, country, and ML-professor candidates fade out.');
  }

  // ── gx-niah-decay：大海捞针准确率随深度和上下文长度增加而下降 ───────────
  function niahDecay(host) {
    var W = 520, H = 240, svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var hx = 40, hy = 50, hw = 300, hh = 150;
    svg.appendChild(txt(hx + hw / 2, 36, 'haystack: context grows left to right', 11, MUTE));
    // 草堆主体
    svg.appendChild(svgEl('rect', { x: hx, y: hy, width: hw, height: hh, fill: BP, opacity: '0.06', stroke: SOFT, 'stroke-width': '1' }));
    // 淡色文本行
    var ln;
    for (ln = 0; ln < 9; ln++) svg.appendChild(svgEl('line', { x1: hx + 10, y1: hy + 14 + ln * 16, x2: hx + hw - 10, y2: hy + 14 + ln * 16, stroke: MUTE, 'stroke-width': '1', opacity: '0.22' }));
    // 针：随时间向更深处（向右并向下）移动的标记
    var needle = svgEl('rect', { x: hx + 30, y: hy + 20, width: 46, height: 14, fill: WARN });
    needle.appendChild(anim('x', (hx + 20) + ';' + (hx + hw - 70) + ';' + (hx + 20), '10s', { keyTimes: '0;0.5;1', keySplines: '.4 0 .2 1;.4 0 .2 1' }));
    needle.appendChild(anim('y', (hy + 18) + ';' + (hy + hh - 34) + ';' + (hy + 18), '10s', { keyTimes: '0;0.5;1', keySplines: '.4 0 .2 1;.4 0 .2 1' }));
    svg.appendChild(needle);
    svg.appendChild(txt(hx + 53, hy + 30, 'needle', 8, 'var(--bg,#fff)'));
    // 右侧的准确率仪表：针越深入，条形高度越低
    var gx = 400, gtop = 50, gh = 150, gw = 50;
    svg.appendChild(svgEl('rect', { x: gx, y: gtop, width: gw, height: gh, fill: 'none', stroke: SOFT, 'stroke-width': '1' }));
    var fill = svgEl('rect', { x: gx, y: gtop, width: gw, height: gh, fill: BP, opacity: '0.7' });
    // y 与高度同步变化：高准确率（满）→ 低准确率（短）→ 高准确率
    fill.appendChild(anim('y', gtop + ';' + (gtop + gh * 0.72) + ';' + gtop, '10s', { keyTimes: '0;0.5;1', keySplines: '.4 0 .2 1;.4 0 .2 1' }));
    fill.appendChild(anim('height', gh + ';' + (gh * 0.28) + ';' + gh, '10s', { keyTimes: '0;0.5;1', keySplines: '.4 0 .2 1;.4 0 .2 1' }));
    fill.appendChild(svgEl('animate', { attributeName: 'fill', values: BP + ';' + WARN + ';' + BP, keyTimes: '0;0.5;1', dur: '10s', repeatCount: 'indefinite' }));
    svg.appendChild(fill);
    svg.appendChild(txt(gx + gw / 2, gtop + gh + 18, 'recall accuracy', 10, MUTE));
    svg.appendChild(txt(hx + hw / 2, hy + hh + 24, 'advertised context is not all usable', 11, MUTE));
    frame(host, 'NEEDLE IN A HAYSTACK', 'depth vs recall', svg,
      'A needle test plants a single fact at a controlled depth in a long context and asks the model to retrieve it. Recall is near-perfect for shallow needles in short contexts, but as the fact sits deeper inside a longer haystack the accuracy gauge falls. The advertised context window is rarely the usable one, which is why depth-by-length sweeps matter.');
  }

  LF.register({
    'gx-var-next-scale': varNextScale,
    'gx-fid-distributions': fidDistributions,
    'gx-patchgan': patchgan,
    'gx-stylegan-mapping': styleganMapping,
    'gx-hybrid-retrieval': hybridRetrieval,
    'gx-matryoshka': matryoshka,
    'gx-entity-linking': entityLinking,
    'gx-niah-decay': niahDecay
  });
})();
