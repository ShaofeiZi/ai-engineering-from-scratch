/* figures-systems3.js——覆盖 Phase 6（语音/音频）、Phase 8（生成式 AI）、
   Phase 11（LLM 工程）、Phase 12（多模态）和 Phase 13（工具与协议）的动画课程图示。
   在 lesson-figures.js 之后加载，使用共享 LF 工具包，并通过 LF.register 注册。
   这些 SVG 使用 SMIL 动画：由 <animate>/<animateTransform>/<animateMotion>/
   stroke-dashoffset 驱动，不使用 JS 渲染循环。无依赖，仅使用 ES5，主题由 CSS
   变量控制。编写方式与 docs/en.md 中的围栏 ```figure 代码块相同。 */
(function () {
  'use strict';
  var LF = window.LF;
  if (!LF) { return; }
  var el = LF.el, svgEl = LF.svgEl;

  // 带 SVG 和说明文字的卡片外壳。H 是 SVG viewBox 的高度。
  function shell(host, label, hint, svg, cap) {
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, [label]), el('span', {}, [hint])]),
      el('div', { class: 'lf-body' }, [el('div', { class: 'lf-out' }, [svg])]),
      el('div', { class: 'lf-cap' }, [cap])
    ]));
  }
  function newSvg(H) { return svgEl('svg', { viewBox: '0 0 520 ' + H }); }
  var BP = 'var(--blueprint,#3553ff)', MUTE = 'var(--ink-mute,#999)', SOFT = 'var(--rule-soft,#ddd)', WARN = 'var(--warn,#b8870f)', INK = 'var(--ink-soft,#555)';
  function anim(attr, vals, dur, extra) {
    var a = { attributeName: attr, values: vals, dur: dur, repeatCount: 'indefinite' };
    if (extra) for (var k in extra) a[k] = extra[k];
    return svgEl('animate', a);
  }
  function aTransform(type, vals, dur, extra) {
    var a = { attributeName: 'transform', type: type, values: vals, dur: dur, repeatCount: 'indefinite' };
    if (extra) for (var k in extra) a[k] = extra[k];
    return svgEl('animateTransform', a);
  }
  function txt(x, y, s, size, fill, anchor) {
    return svgEl('text', { x: x, y: y, 'font-size': size || 10, 'font-family': 'monospace', fill: fill || INK, 'text-anchor': anchor || 'start' }, [document.createTextNode(s)]);
  }

  // ── masked-diffusion-unmask（Show-o）：遮罩网格分步填充 ─────────────
  function maskedDiffusion(host) {
    var svg = newSvg(240);
    var n = 6, cell = 26, ox = 150, oy = 26;
    // 确定性的去遮罩顺序（类似置信度螺旋），36 个单元分 9 波完成
    var order = [14, 15, 20, 21, 13, 16, 19, 22, 8, 9, 10, 11, 26, 27, 28, 29, 7, 12, 25, 30, 2, 3, 4, 5, 1, 6, 24, 31, 0, 17, 18, 23, 32, 33, 34, 35];
    var WAVES = 6, total = n * n, per = total / WAVES, cyc = 6; // 秒
    for (var i = 0; i < total; i++) {
      var r = Math.floor(i / n), c = i % n;
      var wave = Math.floor(order.indexOf(i) / per);
      var begin = (wave / WAVES * cyc).toFixed(2) + 's';
      var g = svgEl('g', {});
      // 遮罩块（灰色）淡出，内容块（蓝色）淡入
      g.appendChild(svgEl('rect', { x: ox + c * cell, y: oy + r * cell, width: cell - 2, height: cell - 2, fill: 'var(--bg-surface,#eee)', stroke: SOFT, 'stroke-width': '0.5' }, [
        anim('fill-opacity', '1;1;0;0', cyc + 's', { keyTimes: '0;' + (wave / WAVES).toFixed(3) + ';' + ((wave + 0.7) / WAVES).toFixed(3) + ';1' })
      ]));
      g.appendChild(svgEl('rect', { x: ox + c * cell, y: oy + r * cell, width: cell - 2, height: cell - 2, fill: BP, 'fill-opacity': '0', stroke: SOFT, 'stroke-width': '0.5' }, [
        anim('fill-opacity', '0;0;' + (0.3 + 0.5 * ((r + c) % 2)).toFixed(2) + ';' + (0.3 + 0.5 * ((r + c) % 2)).toFixed(2), cyc + 's', { keyTimes: '0;' + (wave / WAVES).toFixed(3) + ';' + ((wave + 0.7) / WAVES).toFixed(3) + ';1' })
      ]));
      svg.appendChild(g);
    }
    svg.appendChild(txt(20, 40, 'text:', 11, INK));
    // 左侧为从左到右生成的文本 token 流（因果）
    for (var t = 0; t < 5; t++) {
      svg.appendChild(svgEl('rect', { x: 20, y: 56 + t * 26, width: 100, height: 18, rx: 2, fill: BP, 'fill-opacity': '0' }, [
        anim('fill-opacity', '0;0.7;0.7', '6s', { keyTimes: '0;' + (t / 8 + 0.02).toFixed(3) + ';1' })
      ]));
    }
    svg.appendChild(txt(ox, oy - 8, 'image: parallel masked unmasking', 10, MUTE));
    svg.appendChild(txt(20, 200, 'causal NTP', 10, MUTE));
    shell(host, 'SHOW-O UNIFIED', 'text left-to-right · image unmasks in parallel', svg,
      'Text tokens are generated one at a time, left to right, by next-token prediction. Image tokens start fully masked and are recovered in parallel: each step predicts every masked cell at once and keeps the most confident, re-masking the rest. After a handful of waves the whole image is filled in — far fewer steps than autoregressive image decoding.');
  }

  // ── any-to-any-stream（MIO）：四种模态 token 合并为一个流 ───────────
  function anyToAny(host) {
    var svg = newSvg(240);
    var mods = [
      { y: 30, label: 'text', col: BP },
      { y: 78, label: 'image', col: WARN },
      { y: 126, label: 'speech', col: BP },
      { y: 174, label: 'music', col: MUTE }
    ];
    var laneX = 24, mergeX = 250, outX = 470;
    // 中间是一个 transformer 块
    svg.appendChild(svgEl('rect', { x: mergeX, y: 60, width: 90, height: 110, rx: 4, fill: 'var(--bg-surface,#eee)', stroke: SOFT, 'stroke-width': '1' }));
    svg.appendChild(txt(mergeX + 45, 122, 'one', 11, INK, 'middle'));
    svg.appendChild(txt(mergeX + 45, 138, 'transformer', 11, INK, 'middle'));
    mods.forEach(function (m, mi) {
      svg.appendChild(txt(laneX, m.y + 4, m.label, 11, m.col));
      // 四个 token 方块沿路径流入 transformer，再从中流出
      var pIn = 'M ' + (laneX + 56) + ' ' + m.y + ' L ' + (mergeX - 6) + ' ' + (m.y < 100 ? 90 : 140);
      for (var k = 0; k < 3; k++) {
        var sq = svgEl('rect', { x: -5, y: -5, width: 10, height: 10, rx: 2, fill: m.col, 'fill-opacity': '0.8' });
        var mo = svgEl('animateMotion', { dur: '3s', repeatCount: 'indefinite', path: pIn, begin: (mi * 0.2 + k * 1).toFixed(2) + 's' });
        sq.appendChild(mo);
        sq.appendChild(anim('fill-opacity', '0;0.85;0', '3s', { begin: (mi * 0.2 + k * 1).toFixed(2) + 's' }));
        svg.appendChild(sq);
      }
    });
    // 输出流：不同模态的 token 交替沿单一通道离开
    var outCols = [BP, WARN, BP, MUTE, BP];
    var pOut = 'M ' + (mergeX + 90) + ' 115 L ' + outX + ' 115';
    for (var o = 0; o < 5; o++) {
      var os = svgEl('rect', { x: -6, y: -6, width: 12, height: 12, rx: 2, fill: outCols[o], 'fill-opacity': '0.85' });
      os.appendChild(svgEl('animateMotion', { dur: '2.5s', repeatCount: 'indefinite', path: pOut, begin: (o * 0.5).toFixed(2) + 's' }));
      os.appendChild(anim('fill-opacity', '0;0.9;0', '2.5s', { begin: (o * 0.5).toFixed(2) + 's' }));
      svg.appendChild(os);
    }
    svg.appendChild(txt(outX - 30, 100, 'any output', 10, MUTE, 'middle'));
    shell(host, 'ANY-TO-ANY STREAM', 'four modalities → one shared vocabulary', svg,
      'Text, image, speech, and music are each tokenized into one shared vocabulary, then interleaved into a single sequence that one causal transformer consumes. Because every modality is just tokens, the model can emit any modality as output — the decode stream alternates token types and streams out fast enough for conversation.');
  }

  // ── video-diffusion-denoise（Sora 风格）：含噪帧序列逐渐清晰 ────────
  function videoDenoise(host) {
    var svg = newSvg(220);
    var fw = 90, fh = 64, gap = 10, oy = 40, ox = 16, frames = 5;
    svg.appendChild(txt(ox, 28, 'spatiotemporal patches denoise over T steps', 10, MUTE));
    for (var f = 0; f < frames; f++) {
      var fx = ox + f * (fw + gap);
      var g = svgEl('g', {});
      // 帧边框
      g.appendChild(svgEl('rect', { x: fx, y: oy, width: fw, height: fh, fill: 'none', stroke: SOFT, 'stroke-width': '1' }));
      // 噪点层淡出
      var noise = svgEl('g', {});
      for (var s = 0; s < 10; s++) {
        var nx = fx + 6 + (s * 17 % (fw - 12));
        var ny = oy + 6 + ((s * 23) % (fh - 12));
        noise.appendChild(svgEl('rect', { x: nx, y: ny, width: 6, height: 6, fill: MUTE, 'fill-opacity': '0.7' }));
      }
      noise.appendChild(anim('opacity', '1;0', '5s', { begin: (f * 0.4).toFixed(2) + 's' }));
      g.appendChild(noise);
      // 清晰形状淡入并跨帧移动（移动的小球表示时间一致性）
      var cy = oy + fh / 2 + (f - 2) * 4;
      var ball = svgEl('circle', { cx: fx + 20 + f * 12, cy: cy, r: 12, fill: BP, 'fill-opacity': '0' });
      ball.appendChild(anim('fill-opacity', '0;0.8', '5s', { begin: (f * 0.4).toFixed(2) + 's' }));
      g.appendChild(ball);
      svg.appendChild(g);
      if (f < frames - 1) {
        svg.appendChild(svgEl('line', { x1: fx + fw, y1: oy + fh / 2, x2: fx + fw + gap, y2: oy + fh / 2, stroke: SOFT, 'stroke-width': '1', 'stroke-dasharray': '2 2' }));
      }
    }
    svg.appendChild(txt(ox, oy + fh + 24, 'frame 1', 9, MUTE));
    svg.appendChild(txt(ox + (frames - 1) * (fw + gap), oy + fh + 24, 'frame ' + frames, 9, MUTE));
    shell(host, 'VIDEO DIFFUSION', 'noise → coherent motion', svg,
      'A 3-D VAE compresses the clip into spatiotemporal patches, and a diffusion transformer denoises them. The grey speckle is the noise the model removes step by step; the blue shape resolving and shifting across frames is the temporal coherence the network has to model — the same object, lighting, and motion held consistent through the strip.');
  }

  // ── inpaint-mask-reinject：重新生成遮罩区域，同时保持上下文不变 ─────
  function inpaint(host) {
    var svg = newSvg(230);
    var ix = 140, iy = 30, iw = 240, ih = 168;
    // 外部图像（保留的上下文）——柔和填充
    svg.appendChild(svgEl('rect', { x: ix, y: iy, width: iw, height: ih, fill: BP, 'fill-opacity': '0.14', stroke: SOFT, 'stroke-width': '1' }));
    svg.appendChild(txt(ix + 8, iy + 18, 'kept context (re-injected each step)', 10, INK));
    // 遮罩区域
    var mx = ix + 70, my = iy + 56, mw = 100, mh = 84;
    var maskRect = svgEl('rect', { x: mx, y: my, width: mw, height: mh, fill: 'var(--bg-surface,#eee)', stroke: WARN, 'stroke-width': '1.5', 'stroke-dasharray': '5 3' });
    svg.appendChild(maskRect);
    // 虚线沿遮罩边界移动
    svg.appendChild(svgEl('rect', { x: mx, y: my, width: mw, height: mh, fill: 'none', stroke: WARN, 'stroke-width': '1.5', 'stroke-dasharray': '5 3' }, [
      anim('stroke-dashoffset', '0;-16', '1s')
    ]));
    // 遮罩内重新生成的内容：噪声淡出为清晰的蓝色填充
    var noise = svgEl('g', {});
    for (var s = 0; s < 12; s++) {
      noise.appendChild(svgEl('rect', { x: mx + 6 + (s * 13 % (mw - 12)), y: my + 6 + ((s * 19) % (mh - 12)), width: 7, height: 7, fill: MUTE, 'fill-opacity': '0.7' }));
    }
    noise.appendChild(anim('opacity', '1;1;0;0', '4s', { keyTimes: '0;0.15;0.75;1' }));
    svg.appendChild(noise);
    var fill = svgEl('rect', { x: mx + 4, y: my + 4, width: mw - 8, height: mh - 8, fill: BP, 'fill-opacity': '0' }, [
      anim('fill-opacity', '0;0;0.7;0.7', '4s', { keyTimes: '0;0.15;0.75;1' })
    ]);
    svg.appendChild(fill);
    svg.appendChild(txt(mx + mw / 2, my + mh + 18, 'regenerated only here', 10, WARN, 'middle'));
    // 侧边标签
    svg.appendChild(txt(18, 110, 'denoise', 11, INK));
    svg.appendChild(txt(18, 126, 'inside', 11, INK));
    svg.appendChild(txt(18, 142, 'mask', 11, INK));
    shell(host, 'INPAINTING', 'denoise inside the mask, hold the rest', svg,
      'Inpainting denoises only the masked region while re-injecting the known pixels at every step so the boundary stays consistent. The marching dashes are the mask edge; inside it the noise resolves into new content, while the surrounding context is held fixed and pixel-identical.');
  }

  // ── agentic-rag-loop：检索 → 推理 → 行动循环 ───────────────────────
  function agenticRag(host) {
    var svg = newSvg(240);
    var cx = 260, cy = 128, R = 78;
    var nodes = [
      { a: -90, label: 'retrieve', col: BP },
      { a: 30, label: 'reason', col: WARN },
      { a: 150, label: 'act / refine', col: BP }
    ];
    // 环形箭头轨迹
    svg.appendChild(svgEl('circle', { cx: cx, cy: cy, r: R, fill: 'none', stroke: SOFT, 'stroke-width': '1.5', 'stroke-dasharray': '6 6' }, [
      anim('stroke-dashoffset', '0;-48', '2s')
    ]));
    var pos = [];
    nodes.forEach(function (nd) {
      var rad = nd.a * Math.PI / 180;
      var x = cx + R * Math.cos(rad), y = cy + R * Math.sin(rad);
      pos.push([x, y]);
      svg.appendChild(svgEl('circle', { cx: x, cy: y, r: 30, fill: nd.col, 'fill-opacity': '0.18', stroke: nd.col, 'stroke-width': '1.5' }, [
        anim('stroke-opacity', '0.3;1;0.3', '6s', { begin: (nodes.indexOf(nd) * 2).toFixed(1) + 's' })
      ]));
      svg.appendChild(txt(x, y + 4, nd.label, 10, INK, 'middle'));
    });
    // 查询 token 沿循环轨道运行
    var orbit = 'M ' + pos[0][0] + ' ' + pos[0][1] + ' A ' + R + ' ' + R + ' 0 0 1 ' + pos[1][0] + ' ' + pos[1][1] +
      ' A ' + R + ' ' + R + ' 0 0 1 ' + pos[2][0] + ' ' + pos[2][1] +
      ' A ' + R + ' ' + R + ' 0 0 1 ' + pos[0][0] + ' ' + pos[0][1];
    var tok = svgEl('circle', { r: 7, fill: WARN });
    tok.appendChild(svgEl('animateMotion', { dur: '6s', repeatCount: 'indefinite', path: orbit, rotate: 'auto' }));
    svg.appendChild(tok);
    // 左侧语料库向检索环节提供数据
    svg.appendChild(txt(30, 60, 'corpus', 10, MUTE));
    for (var d = 0; d < 4; d++) {
      svg.appendChild(svgEl('rect', { x: 30, y: 70 + d * 18, width: 60, height: 12, rx: 1, fill: BP, 'fill-opacity': (0.25 + d * 0.12).toFixed(2) }));
    }
    svg.appendChild(txt(cx, 22, 'agentic RAG: loop until the answer is grounded', 10, MUTE, 'middle'));
    shell(host, 'AGENTIC RAG', 'retrieve → reason → act → repeat', svg,
      'Basic RAG retrieves once and answers. Agentic RAG loops: retrieve candidates, reason over whether they actually answer the query, then act — rewrite the query, rerank, or retrieve again. The orbiting token is one query cycling the loop until the context is good enough to answer, which is what multi-hop questions need.');
  }

  // ── mcp-nxm-collapse：N 个 host × M 个 server → 一个协议枢纽 ───────
  function mcpMatrix(host) {
    var svg = newSvg(250);
    var hosts = ['Claude', 'ChatGPT', 'Cursor'];
    var servers = ['db', 'calendar', 'files'];
    var hubX = 260, hubY = 125;
    // 枢纽
    svg.appendChild(svgEl('circle', { cx: hubX, cy: hubY, r: 26, fill: BP, 'fill-opacity': '0.18', stroke: BP, 'stroke-width': '1.5' }));
    svg.appendChild(txt(hubX, hubY + 4, 'MCP', 11, BP, 'middle'));
    var hy = [50, 125, 200], sy = [50, 125, 200];
    hosts.forEach(function (h, i) {
      var hx = 40;
      svg.appendChild(svgEl('rect', { x: hx, y: hy[i] - 14, width: 76, height: 28, rx: 3, fill: 'var(--bg-surface,#eee)', stroke: SOFT, 'stroke-width': '1' }));
      svg.appendChild(txt(hx + 38, hy[i] + 4, h, 10, INK, 'middle'));
      svg.appendChild(svgEl('line', { x1: hx + 76, y1: hy[i], x2: hubX - 26, y2: hubY, stroke: SOFT, 'stroke-width': '1', 'stroke-dasharray': '5 4' }, [
        anim('stroke-dashoffset', '0;-18', '1.2s', { begin: (i * 0.3).toFixed(1) + 's' })
      ]));
      // 请求包从 host 流向枢纽
      var pkt = svgEl('circle', { r: 5, fill: BP });
      pkt.appendChild(svgEl('animateMotion', { dur: '2.4s', repeatCount: 'indefinite', path: 'M ' + (hx + 76) + ' ' + hy[i] + ' L ' + (hubX - 26) + ' ' + hubY, begin: (i * 0.4).toFixed(1) + 's' }));
      pkt.appendChild(anim('opacity', '0;1;0', '2.4s', { begin: (i * 0.4).toFixed(1) + 's' }));
      svg.appendChild(pkt);
    });
    servers.forEach(function (sv, i) {
      var sx = 404;
      svg.appendChild(svgEl('rect', { x: sx, y: sy[i] - 14, width: 76, height: 28, rx: 3, fill: 'var(--bg-surface,#eee)', stroke: SOFT, 'stroke-width': '1' }));
      svg.appendChild(txt(sx + 38, sy[i] + 4, sv, 10, INK, 'middle'));
      svg.appendChild(svgEl('line', { x1: hubX + 26, y1: hubY, x2: sx, y2: sy[i], stroke: SOFT, 'stroke-width': '1', 'stroke-dasharray': '5 4' }, [
        anim('stroke-dashoffset', '0;-18', '1.2s', { begin: (i * 0.3 + 0.6).toFixed(1) + 's' })
      ]));
      var pkt2 = svgEl('circle', { r: 5, fill: WARN });
      pkt2.appendChild(svgEl('animateMotion', { dur: '2.4s', repeatCount: 'indefinite', path: 'M ' + (hubX + 26) + ' ' + hubY + ' L ' + sx + ' ' + sy[i], begin: (i * 0.4 + 1).toFixed(1) + 's' }));
      pkt2.appendChild(anim('opacity', '0;1;0', '2.4s', { begin: (i * 0.4 + 1).toFixed(1) + 's' }));
      svg.appendChild(pkt2);
    });
    svg.appendChild(txt(40, 232, 'N hosts', 10, MUTE));
    svg.appendChild(txt(404, 232, 'M servers', 10, MUTE));
    shell(host, 'MCP COLLAPSES N×M', 'one protocol, every host ↔ every server', svg,
      'Before MCP every host and every server spoke a bespoke protocol, an N×M integration matrix. MCP is one JSON-RPC spec in the middle: write one server and any compliant host discovers and calls its tools, resources, and prompts. Requests flow in, results flow back, through a single wire format.');
  }

  // ── a2a-task-lifecycle：agent 发送 Task，状态推进，artifact 返回 ─────
  function a2aLifecycle(host) {
    var svg = newSvg(240);
    var states = ['submitted', 'working', 'input-required', 'completed'];
    var sx = 150, dx = 92, sy = 70;
    // 客户端与远端 agent 方框
    svg.appendChild(svgEl('rect', { x: 20, y: 30, width: 90, height: 36, rx: 4, fill: BP, 'fill-opacity': '0.16', stroke: BP, 'stroke-width': '1.5' }));
    svg.appendChild(txt(65, 52, 'client agent', 10, INK, 'middle'));
    svg.appendChild(svgEl('rect', { x: 410, y: 30, width: 90, height: 36, rx: 4, fill: WARN, 'fill-opacity': '0.16', stroke: WARN, 'stroke-width': '1.5' }));
    svg.appendChild(txt(455, 52, 'remote agent', 10, INK, 'middle'));
    // Task 消息从客户端飞向远端
    var task = svgEl('rect', { x: -16, y: -8, width: 32, height: 16, rx: 3, fill: BP });
    task.appendChild(svgEl('animateMotion', { dur: '8s', repeatCount: 'indefinite', path: 'M 110 48 L 410 48', keyTimes: '0;0.12;1', keyPoints: '0;1;1', calcMode: 'linear' }));
    task.appendChild(anim('opacity', '0;1;1;0;0', '8s', { keyTimes: '0;0.02;0.1;0.14;1' }));
    svg.appendChild(task);
    svg.appendChild(svgEl('text', { x: 0, y: 4, 'font-size': 8, 'font-family': 'monospace', fill: 'var(--bg,#fff)', 'text-anchor': 'middle' }, [document.createTextNode('Task'),
      svgEl('animateMotion', { dur: '8s', repeatCount: 'indefinite', path: 'M 110 48 L 410 48', keyTimes: '0;0.12;1', keyPoints: '0;1;1', calcMode: 'linear' }),
      anim('opacity', '0;1;1;0;0', '8s', { keyTimes: '0;0.02;0.1;0.14;1' })]));
    // 状态标签依次点亮
    states.forEach(function (st, i) {
      var x = sx + i * dx;
      svg.appendChild(svgEl('rect', { x: x - 42, y: 120, width: 84, height: 26, rx: 13, fill: 'var(--bg-surface,#eee)', stroke: SOFT, 'stroke-width': '1' }));
      svg.appendChild(svgEl('rect', { x: x - 42, y: 120, width: 84, height: 26, rx: 13, fill: BP, 'fill-opacity': '0' }, [
        anim('fill-opacity', '0;0.85;0.85;0', '8s', { keyTimes: '0;' + (0.15 + i * 0.2).toFixed(2) + ';' + (0.32 + i * 0.2).toFixed(2) + ';1' })
      ]));
      svg.appendChild(txt(x, 137, st, 9, INK, 'middle'));
      if (i < states.length - 1) {
        svg.appendChild(svgEl('line', { x1: x + 42, y1: 133, x2: x + dx - 42, y2: 133, stroke: SOFT, 'stroke-width': '1' }));
      }
    });
    // 最后 artifact 从远端返回客户端
    var art = svgEl('rect', { x: -18, y: -9, width: 36, height: 18, rx: 3, fill: WARN });
    art.appendChild(svgEl('animateMotion', { dur: '8s', repeatCount: 'indefinite', path: 'M 410 190 L 110 190', keyTimes: '0;0.86;0.98;1', keyPoints: '0;0;1;1', calcMode: 'linear' }));
    art.appendChild(anim('opacity', '0;0;1;0', '8s', { keyTimes: '0;0.86;0.94;1' }));
    svg.appendChild(art);
    svg.appendChild(txt(260, 215, 'artifact', 9, WARN, 'middle'));
    svg.appendChild(txt(260, 100, 'task lifecycle (state stays opaque to caller)', 10, MUTE, 'middle'));
    shell(host, 'A2A TASK LIFECYCLE', 'submitted → working → completed', svg,
      'One agent sends a Task to another and watches only the state transitions: submitted, working, sometimes input-required, then completed. The remote agent\'s internal reasoning stays opaque — the caller sees state changes and, at the end, an artifact returned as the output.');
  }

  // ── rvq-codec-cascade：残差向量量化，语义与声学信息对比 ───────────
  function rvqCodec(host) {
    var svg = newSvg(230);
    // 左侧波形输入编码器
    var wd = 'M 24 120', wx;
    for (wx = 0; wx <= 80; wx++) {
      var xx = 24 + wx, yy = 120 + 26 * Math.sin(wx / 4) * Math.exp(-wx / 120);
      wd += ' L ' + xx + ' ' + yy.toFixed(1);
    }
    svg.appendChild(svgEl('path', { d: wd, fill: 'none', stroke: MUTE, 'stroke-width': '1.5' }));
    svg.appendChild(txt(24, 60, 'waveform', 10, MUTE));
    // 编码器块
    svg.appendChild(svgEl('rect', { x: 116, y: 96, width: 30, height: 48, rx: 3, fill: 'var(--bg-surface,#eee)', stroke: SOFT, 'stroke-width': '1' }));
    svg.appendChild(txt(131, 124, 'enc', 9, INK, 'middle'));
    // 级联码本；残差沿堆栈逐层缩小
    var books = [
      { y: 30, label: 'CB0  semantic', col: BP, amp: 1.0 },
      { y: 78, label: 'CB1  acoustic', col: WARN, amp: 0.55 },
      { y: 126, label: 'CB2  acoustic', col: WARN, amp: 0.32 },
      { y: 174, label: 'CB3  acoustic', col: WARN, amp: 0.18 }
    ];
    books.forEach(function (b, i) {
      var bx = 200;
      svg.appendChild(svgEl('rect', { x: bx, y: b.y, width: 150, height: 30, rx: 3, fill: b.col, 'fill-opacity': '0.12', stroke: b.col, 'stroke-width': '1' }));
      svg.appendChild(txt(bx + 8, b.y + 19, b.label, 10, INK));
      // 残差条逐渐缩短——用宽度脉冲动画展示“还有多少待量化”
      svg.appendChild(svgEl('rect', { x: bx + 100, y: b.y + 8, width: 40 * b.amp, height: 14, rx: 2, fill: b.col, 'fill-opacity': '0.7' }, [
        anim('fill-opacity', '0.3;0.8;0.3', '3s', { begin: (i * 0.4).toFixed(1) + 's' })
      ]));
      // 残差数据包从一个码本流向下一个码本
      if (i < books.length - 1) {
        var pk = svgEl('circle', { r: 4, fill: b.col });
        pk.appendChild(svgEl('animateMotion', { dur: '3s', repeatCount: 'indefinite', path: 'M ' + (bx + 75) + ' ' + (b.y + 30) + ' L ' + (bx + 75) + ' ' + books[i + 1].y, begin: (i * 0.5).toFixed(1) + 's' }));
        pk.appendChild(anim('opacity', '0;1;0', '3s', { begin: (i * 0.5).toFixed(1) + 's' }));
        svg.appendChild(pk);
      }
      // 编码器 → 第一个码本的链路
      if (i === 0) {
        svg.appendChild(svgEl('line', { x1: 146, y1: 120, x2: bx, y2: b.y + 15, stroke: SOFT, 'stroke-width': '1', 'stroke-dasharray': '4 3' }, [
          anim('stroke-dashoffset', '0;-14', '1s')
        ]));
      }
    });
    svg.appendChild(txt(200, 220, 'each codebook quantizes the residual of the last', 10, MUTE));
    shell(host, 'RVQ AUDIO CODEC', 'semantic codebook 0, acoustic 1..N', svg,
      'Instead of one giant codebook, neural audio codecs cascade small ones: the first quantizes the encoder output, each next one quantizes the residual left over. The shrinking bars are that residual getting smaller down the stack. Forcing codebook 0 to carry linguistic content (semantic) and the rest acoustic detail is what makes token-based speech models work.');
  }

  LF.register({
    'masked-diffusion-unmask': maskedDiffusion,
    'any-to-any-stream': anyToAny,
    'video-diffusion-denoise': videoDenoise,
    'inpaint-mask-reinject': inpaint,
    'agentic-rag-loop': agenticRag,
    'mcp-nxm-collapse': mcpMatrix,
    'a2a-task-lifecycle': a2aLifecycle,
    'rvq-codec-cascade': rvqCodec
  });
})();
