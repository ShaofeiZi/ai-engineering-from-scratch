/* figures-swarms3.js - 第 16 阶段（多智能体与集群）第三模块的动画课程图示，
   支持主题适配。在 lesson-figures.js 之后加载，并通过 window.LF 注册。
   无依赖，仅使用 ES5 和 SMIL 动画，通过 CSS 变量适配主题。编写时使用
   ```figure 代码块指定下列某个组件。 */
(function () {
  'use strict';
  var LF = window.LF;
  if (!LF) { return; }
  var el = LF.el, svgEl = LF.svgEl;

  function shell(host, label, hint, svg, cap) {
    host.appendChild(el('div', { class: 'lf' }, [
      el('div', { class: 'lf-head' }, [el('span', { class: 'lf-label' }, [label]), el('span', {}, [hint])]),
      el('div', { class: 'lf-body' }, [el('div', { class: 'lf-out' }, [svg])]),
      el('div', { class: 'lf-cap' }, [cap])
    ]));
  }
  function txt(x, y, s, size, fill, anchor) {
    var t = svgEl('text', { x: x, y: y, 'text-anchor': anchor || 'middle', 'font-family': 'var(--font-mono,monospace)', 'font-size': size || '10', fill: fill || 'var(--ink-mute,#777)' });
    t.appendChild(document.createTextNode(s));
    return t;
  }
  function anim(attr, vals, kt, dur, opts) {
    var a = { attributeName: attr, values: vals, keyTimes: kt, dur: dur + 's', repeatCount: 'indefinite' };
    if (opts) for (var k in opts) a[k] = opts[k];
    return svgEl('animate', a);
  }
  function motion(path, kt, kp, dur, begin) {
    return svgEl('animateMotion', { path: path, keyTimes: kt, keyPoints: kp, dur: dur + 's', begin: (begin || 0) + 's', repeatCount: 'indefinite', calcMode: 'linear' });
  }

  var BP = 'var(--blueprint,#3553ff)';
  var WARN = 'var(--warn,#b8870f)';
  var SOFT = 'var(--rule-soft,#ddd)';
  var SURF = 'var(--bg-surface,#eee)';
  var BG = 'var(--bg,#fafaf5)';
  var MUTE = 'var(--ink-mute,#777)';

  // ── sw-contract-net：管理者发布任务，三个竞标者闪回出价，最低出价者中标
  //    （FIPA 合同网）──────────────────────────────────────────────────────
  function contractNet(host) {
    var W = 520, H = 250, mx = 70, my = 125, period = 8;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var bx = 410, bys = [60, 125, 190], win = 2; // 获胜（最低价）竞标者的索引
    var i;
    for (i = 0; i < 3; i++) {
      svg.appendChild(svgEl('line', { x1: mx, y1: my, x2: bx, y2: bys[i], stroke: SOFT, 'stroke-width': '1.2' }));
    }
    // 公告脉冲向外传播，随后各出价错开返回
    for (i = 0; i < 3; i++) {
      var fwd = 'M' + mx + ',' + my + ' L' + bx + ',' + bys[i];
      var pkt = svgEl('circle', { r: '4', fill: BP });
      pkt.appendChild(anim('opacity', '0;1;1;0;0', '0;0.02;0.2;0.24;1', period));
      pkt.appendChild(motion(fwd, '0;0.2;1', '0;1;1', period));
      svg.appendChild(pkt);
      var back = 'M' + bx + ',' + bys[i] + ' L' + mx + ',' + my;
      var bid = svgEl('circle', { r: '4', fill: (i === win ? WARN : MUTE) });
      bid.appendChild(anim('opacity', '0;0;1;1;0;0', '0;0.3;0.34;0.55;0.6;1', period));
      bid.appendChild(motion(back, '0;0.34;0.6;1', '0;0;1;1', period));
      svg.appendChild(bid);
    }
    var mgr = svgEl('circle', { cx: mx, cy: my, r: '20', stroke: BP, 'stroke-width': '2', fill: SURF });
    svg.appendChild(mgr);
    svg.appendChild(txt(mx, my + 4, 'mgr', '10', BP));
    var labels = ['$9', '$7', '$4'];
    for (i = 0; i < 3; i++) {
      var c = svgEl('circle', { cx: bx, cy: bys[i], r: '16', stroke: (i === win ? WARN : MUTE), 'stroke-width': '2', fill: SURF });
      if (i === win) c.appendChild(anim('fill', SURF + ';' + SURF + ';' + WARN + ';' + WARN + ';' + SURF, '0;0.6;0.66;0.9;1', period));
      svg.appendChild(c);
      svg.appendChild(txt(bx, bys[i] + 4, labels[i], '10', (i === win ? WARN : MUTE)));
    }
    svg.appendChild(txt(W / 2, H - 14, 'announce  ->  bids return  ->  cheapest bid wins the contract', '10', MUTE));
    shell(host, 'CONTRACT NET', 'announce, bid, award', svg,
      'The FIPA contract-net protocol turns task allocation into a sealed auction. A manager broadcasts a call for proposals, idle agents reply with bids, and the manager awards the contract to the best bid. MCP tools/call and modern task markets are JSON-native restatements of this 1980 mechanism.');
  }

  // ── sw-work-stealing：任务落入共享队列；三个 worker 异步拉取任务，无需中央
  //    调度器 ───────────────────────────────────────────────────────────────
  function workStealing(host) {
    var W = 520, H = 250, qx = 60, qy = 50, qw = 400, period = 6;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    svg.appendChild(svgEl('rect', { x: qx, y: qy, width: qw, height: 30, fill: SURF, stroke: SOFT, 'stroke-width': '1.2', rx: '3' }));
    svg.appendChild(txt(qx + qw + 6, qy + 20, 'queue', '9', MUTE, 'start'));
    var wx = [120, 260, 400], wy = 190, i, j;
    for (i = 0; i < 3; i++) {
      svg.appendChild(svgEl('rect', { x: wx[i] - 26, y: wy - 22, width: 52, height: 44, fill: SURF, stroke: BP, 'stroke-width': '2', rx: '3' }));
      svg.appendChild(txt(wx[i], wy + 4, 'w' + i, '11', BP));
    }
    // 六个任务 token 落入队列，随后被 worker 向下拉取
    var slots = [90, 150, 210, 270, 330, 390];
    for (j = 0; j < 6; j++) {
      var tgt = j % 3;
      var begin = (j * (period / 6)).toFixed(2);
      var path = 'M' + slots[j] + ',' + (qy + 15) + ' L' + slots[j] + ',' + (qy + 15) + ' L' + wx[tgt] + ',' + (wy - 30);
      var g = svgEl('g', {});
      var sq = svgEl('rect', { x: -5, y: -5, width: 10, height: 10, fill: BP, rx: '2' });
      g.appendChild(sq);
      g.appendChild(anim('opacity', '0;1;1;1;0;0', '0;0.05;0.4;0.7;0.78;1', period, { begin: begin + 's' }));
      g.appendChild(motion(path, '0;0.35;1', '0;0;1', period, begin));
      svg.appendChild(g);
    }
    svg.appendChild(txt(W / 2, H - 14, 'workers pull tasks off the shared queue  ·  no orchestrator decides who does what', '10', MUTE));
    shell(host, 'WORK STEALING', 'pull, not push', svg,
      'A swarm has no central dispatcher. Tasks land on a shared queue and idle workers pull the next unit off it themselves. Coordination lives in the queue semantics, so the system scales until the queue does. The price is determinism: you trade a single coherent plan for throughput.');
  }

  // ── sw-handoff-routing：对话 token 在智能体之间逐个移交，每次移交都是返回
  //    下一个智能体的工具调用（OpenAI Swarm）───────────────────────────────
  function handoffRouting(host) {
    var W = 520, H = 240, period = 9;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var ax = [90, 260, 430], ay = [120, 70, 160], names = ['triage', 'billing', 'refund'];
    // 链路：分诊 -> 计费 -> 退款 -> 分诊
    var order = [0, 1, 2, 0];
    var i;
    for (i = 0; i < 3; i++) {
      var nxt = order[i + 1];
      svg.appendChild(svgEl('line', { x1: ax[order[i]], y1: ay[order[i]], x2: ax[nxt], y2: ay[nxt], stroke: SOFT, 'stroke-width': '1.2', 'stroke-dasharray': '4 3' }));
    }
    // 移动的对话 token 沿链路前进
    var tok = svgEl('circle', { r: '7', fill: WARN });
    var mpath = 'M' + ax[0] + ',' + ay[0] + ' L' + ax[1] + ',' + ay[1] + ' L' + ax[2] + ',' + ay[2] + ' L' + ax[0] + ',' + ay[0];
    tok.appendChild(motion(mpath, '0;0.33;0.66;1', '0;0.249;0.519;1', period));
    for (i = 0; i < 3; i++) {
      var lit = i === 0 ? '0;0.05;0.28;0.33' : (i === 1 ? '0.33;0.38;0.61;0.66' : '0.66;0.71;0.94;1');
      var kt = i === 0 ? '0;0.05;0.28;0.33;1' : (i === 1 ? '0;0.33;0.38;0.61;0.66;1' : '0;0.66;0.71;0.94;1');
      var vals = i === 0 ? (SURF + ';' + BP + ';' + BP + ';' + SURF + ';' + SURF)
        : i === 1 ? (SURF + ';' + SURF + ';' + BP + ';' + BP + ';' + SURF + ';' + SURF)
          : (SURF + ';' + SURF + ';' + BP + ';' + BP + ';' + SURF);
      var c = svgEl('circle', { cx: ax[i], cy: ay[i], r: '24', stroke: BP, 'stroke-width': '2', fill: SURF });
      c.appendChild(anim('fill', vals, kt, period));
      svg.appendChild(c);
      svg.appendChild(txt(ax[i], ay[i] + 4, names[i], '9', BP));
    }
    svg.appendChild(tok);
    svg.appendChild(txt(W / 2, H - 14, 'handoff = a tool call returning the next agent  ·  whoever holds the token is the orchestrator', '10', MUTE));
    shell(host, 'HANDOFF ROUTING', 'pass the conversation', svg,
      'OpenAI Swarm reduced orchestration to two primitives: routines (prompt plus tools) and handoffs (a tool that returns the next agent). There is no state machine. The model routes by calling the right handoff, and whichever agent currently holds the conversation is the one in charge.');
  }

  // ── sw-agent-card-discovery：客户端读取 Agent Card，然后推动任务经历其生命
  //    周期状态（A2A）──────────────────────────────────────────────────────
  function agentCard(host) {
    var W = 520, H = 250, period = 8;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var cx = 80, cy = 120;
    svg.appendChild(svgEl('rect', { x: cx - 36, y: cy - 26, width: 72, height: 52, fill: SURF, stroke: BP, 'stroke-width': '2', rx: '4' }));
    svg.appendChild(txt(cx, cy - 4, 'client', '10', BP));
    svg.appendChild(txt(cx, cy + 12, 'reads card', '7', MUTE));
    // 带有卡片的远程智能体
    var rx = 250, ry = 60;
    svg.appendChild(svgEl('rect', { x: rx - 40, y: ry - 22, width: 80, height: 44, fill: SURF, stroke: MUTE, 'stroke-width': '2', rx: '4' }));
    svg.appendChild(txt(rx, ry - 2, 'agent card', '9', MUTE));
    svg.appendChild(txt(rx, ry + 12, '/.well-known', '7', MUTE));
    // 发现数据包：客户端 -> 卡片
    var disc = svgEl('circle', { r: '4', fill: BP });
    disc.appendChild(anim('opacity', '0;1;1;0;0', '0;0.04;0.16;0.2;1', period));
    disc.appendChild(motion('M' + cx + ',' + (cy - 20) + ' L' + rx + ',' + (ry + 14), '0;0.18;1', '0;1;1', period));
    svg.appendChild(disc);
    // 任务生命周期：已提交 -> 工作中 -> 已完成
    var states = ['submitted', 'working', 'completed'];
    var sx = [330, 410, 480], sy = 175;
    for (var i = 0; i < 3; i++) {
      var on = i === 0 ? '0;0.2;0.45;0.5' : i === 1 ? '0.5;0.55;0.78;0.8' : '0.8;0.85;0.98;1';
      var kt = i === 0 ? '0;0.2;0.45;0.5;1' : i === 1 ? '0;0.5;0.55;0.78;0.8;1' : '0;0.8;0.85;0.98;1';
      var vals = i === 0 ? (SURF + ';' + WARN + ';' + WARN + ';' + SURF + ';' + SURF)
        : i === 1 ? (SURF + ';' + SURF + ';' + WARN + ';' + WARN + ';' + SURF + ';' + SURF)
          : (SURF + ';' + SURF + ';' + WARN + ';' + WARN + ';' + WARN);
      if (i < 2) svg.appendChild(svgEl('line', { x1: sx[i] + 14, y1: sy, x2: sx[i + 1] - 14, y2: sy, stroke: SOFT, 'stroke-width': '1.2' }));
      var c = svgEl('circle', { cx: sx[i], cy: sy, r: '13', stroke: WARN, 'stroke-width': '1.8', fill: SURF });
      c.appendChild(anim('fill', vals, kt, period));
      svg.appendChild(c);
      svg.appendChild(txt(sx[i], sy + 27, states[i], '7', MUTE));
    }
    svg.appendChild(txt(W / 2, H - 12, 'discover via Agent Card  ->  submit task  ->  opaque lifecycle returns artifacts', '9', MUTE));
    shell(host, 'A2A DISCOVERY', 'card then task', svg,
      'A2A is the horizontal wire protocol between agents. A client first fetches an Agent Card from a well-known URL to learn what a remote agent can do, then submits a task that moves through an opaque lifecycle (submitted, working, completed) and returns artifacts. It is HTTP plus REST, reframed with agents as first-class peers.');
  }

  // ── sw-debate-topology：相同的五个智能体依次重连为星形、链式、树形和图形
  //    拓扑；各阶段的边淡入淡出 ─────────────────────────────────────────────
  function debateTopology(host) {
    var W = 520, H = 250, period = 12;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var nx = [260, 130, 390, 175, 345], ny = [70, 140, 140, 210, 210];
    // 四种拓扑，每种都是五个节点之间的 [a,b] 边列表
    var topos = [
      [[0, 1], [0, 2], [0, 3], [0, 4]],            // 星形
      [[0, 1], [1, 3], [3, 4], [4, 2]],            // 链式
      [[0, 1], [0, 2], [1, 3], [1, 4]],            // 树形
      [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4], [0, 4]] // 图形
    ];
    var labels = ['STAR', 'CHAIN', 'TREE', 'GRAPH'];
    // 每个阶段占据周期的四分之一；为各阶段构建边组
    var p, e;
    for (p = 0; p < 4; p++) {
      var lo = (p / 4), hi = ((p + 1) / 4);
      var edges = topos[p];
      for (e = 0; e < edges.length; e++) {
        var a = edges[e][0], b = edges[e][1];
        var ln = svgEl('line', { x1: nx[a], y1: ny[a], x2: nx[b], y2: ny[b], stroke: BP, 'stroke-width': '1.6', opacity: '0' });
        var kt = '0;' + lo.toFixed(3) + ';' + (lo + 0.03).toFixed(3) + ';' + (hi - 0.03).toFixed(3) + ';' + hi.toFixed(3) + ';1';
        ln.appendChild(anim('opacity', '0;0;1;1;0;0', kt, period));
        svg.appendChild(ln);
      }
      // 阶段标签
      var lt = txt(W / 2, H - 30, labels[p], '11', BP);
      lt.setAttribute('opacity', '0');
      var ktl = '0;' + lo.toFixed(3) + ';' + (lo + 0.02).toFixed(3) + ';' + (hi - 0.02).toFixed(3) + ';' + hi.toFixed(3) + ';1';
      lt.appendChild(anim('opacity', '0;0;1;1;0;0', ktl, period));
      svg.appendChild(lt);
    }
    var i;
    for (i = 0; i < 5; i++) {
      svg.appendChild(svgEl('circle', { cx: nx[i], cy: ny[i], r: '15', stroke: BP, 'stroke-width': '2', fill: SURF }));
      svg.appendChild(txt(nx[i], ny[i] + 4, String(i), '11', BP));
    }
    svg.appendChild(txt(W / 2, H - 12, 'same agents, different wiring  ·  graph wins for research, coordination tax rises past ~4', '9', MUTE));
    shell(host, 'DEBATE TOPOLOGY', 'who talks to whom', svg,
      'Aggregating N agents is not just majority vote; the wiring matters. Star routes everything through a hub, chain passes a baton, tree branches, graph lets everyone argue. MultiAgentBench found graph best for research tasks, with a coordination tax that climbs once you pass about four agents.');
  }

  // ── sw-theory-of-mind：嵌套的信念气泡——智能体 A 对 B 关于 C 的信念建模，
  //    通过脉冲展示递归深度 ─────────────────────────────────────────────────
  function theoryOfMind(host) {
    var W = 520, H = 250, period = 9;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var ax = 130, ay = 130;
    // 智能体 A
    svg.appendChild(svgEl('circle', { cx: ax, cy: ay, r: '26', stroke: BP, 'stroke-width': '2.5', fill: SURF }));
    svg.appendChild(txt(ax, ay + 4, 'A', '13', BP));
    // 右侧的嵌套思考气泡：A 对 B 的建模，B 对 C 的建模
    var b1x = 300, b1y = 95, b2x = 430, b2y = 130, b3x = 380, b3y = 195;
    // 从 A 头部引出的小气泡
    svg.appendChild(svgEl('circle', { cx: ax + 30, cy: ay - 22, r: '3', fill: SOFT }));
    svg.appendChild(svgEl('circle', { cx: ax + 46, cy: ay - 36, r: '4', fill: SOFT }));
    // 第 1 层：A 相信 B 相信……
    var l1 = svgEl('ellipse', { cx: b1x, cy: b1y, rx: '56', ry: '34', stroke: BP, 'stroke-width': '1.8', fill: 'none' });
    l1.appendChild(anim('opacity', '0.25;1;1;0.25;0.25', '0;0.15;0.55;0.7;1', period));
    svg.appendChild(l1);
    svg.appendChild(txt(b1x, b1y - 14, "A thinks", '8', MUTE));
    svg.appendChild(txt(b1x, b1y + 4, 'B', '12', BP));
    // 嵌套第 2 层：B 相信 C
    var l2 = svgEl('ellipse', { cx: b2x, cy: b2y, rx: '40', ry: '26', stroke: WARN, 'stroke-width': '1.8', fill: 'none' });
    l2.appendChild(anim('opacity', '0.15;0.15;1;1;0.15;0.15', '0;0.3;0.45;0.62;0.72;1', period));
    svg.appendChild(l2);
    svg.appendChild(txt(b2x, b2y - 10, 'B thinks', '8', MUTE));
    svg.appendChild(txt(b2x, b2y + 8, 'C', '12', WARN));
    // 最深的第 3 层：C 的目标
    var l3 = svgEl('circle', { cx: b3x, cy: b3y, r: '18', stroke: MUTE, 'stroke-width': '1.6', fill: 'none' });
    l3.appendChild(anim('opacity', '0.1;0.1;0.1;1;1;0.1', '0;0.45;0.55;0.62;0.78;1', period));
    svg.appendChild(l3);
    svg.appendChild(txt(b3x, b3y + 4, 'goal', '8', MUTE));
    svg.appendChild(txt(W / 2, H - 12, 'A reasons about what B believes about C  ·  higher-order ToM is prompt-conditional', '9', MUTE));
    shell(host, 'THEORY OF MIND', 'beliefs about beliefs', svg,
      'Real coordination needs agents that model each other. Higher-order theory of mind is reasoning about what one agent believes about a third agent. Riedl 2025 found this only produces genuine, goal-directed differentiation under a theory-of-mind prompt; strip the prompt and the apparent coordination does not survive statistical controls.');
  }

  // ── sw-ctde：训练期间，集中式 critic 能看到所有智能体；随后连接断开，
  //    分散式 actor 基于局部视图运行（MARL CTDE）───────────────────────────
  function ctde(host) {
    var W = 520, H = 250, period = 10;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    var ax = [110, 260, 410], ay = 175;
    var crx = 260, cry = 60;
    // critic 框（仅在训练阶段的前半程出现/亮起）
    var critic = svgEl('rect', { x: crx - 50, y: cry - 22, width: 100, height: 44, fill: SURF, stroke: WARN, 'stroke-width': '2', rx: '4' });
    critic.appendChild(anim('opacity', '1;1;0.15;0.15;1', '0;0.45;0.52;0.95;1', period));
    svg.appendChild(critic);
    var ctxt = txt(crx, cry - 2, 'central critic', '9', WARN);
    ctxt.appendChild(anim('opacity', '1;1;0.2;0.2;1', '0;0.45;0.52;0.95;1', period));
    svg.appendChild(ctxt);
    var ptxt = txt(crx, cry + 13, 'sees all', '7', MUTE);
    ptxt.appendChild(anim('opacity', '1;1;0;0;1', '0;0.45;0.52;0.95;1', period));
    svg.appendChild(ptxt);
    var i;
    for (i = 0; i < 3; i++) {
      // critic 到 actor 的连接，仅在训练阶段的前半程可见
      var ln = svgEl('line', { x1: crx, y1: cry + 22, x2: ax[i], y2: ay - 22, stroke: WARN, 'stroke-width': '1.4', 'stroke-dasharray': '4 3' });
      ln.appendChild(anim('opacity', '1;1;0;0;1', '0;0.45;0.52;0.95;1', period));
      svg.appendChild(ln);
      // 执行者（actor）
      svg.appendChild(svgEl('rect', { x: ax[i] - 28, y: ay - 22, width: 56, height: 44, fill: SURF, stroke: BP, 'stroke-width': '2', rx: '3' }));
      svg.appendChild(txt(ax[i], ay + 2, 'actor ' + i, '8', BP));
      svg.appendChild(txt(ax[i], ay + 15, 'local obs', '7', MUTE));
    }
    // 阶段标签在训练/执行之间切换
    var tl = txt(W / 2, H - 30, 'TRAIN: critic sees everything', '10', WARN);
    tl.appendChild(anim('opacity', '1;1;0;0;1', '0;0.45;0.5;0.95;1', period));
    svg.appendChild(tl);
    var el2 = txt(W / 2, H - 30, 'EXECUTE: actors run alone', '10', BP);
    el2.setAttribute('opacity', '0');
    el2.appendChild(anim('opacity', '0;0;1;1;0', '0;0.5;0.55;0.92;1', period));
    svg.appendChild(el2);
    svg.appendChild(txt(W / 2, H - 12, 'centralized training, decentralized execution  ·  global info at train, local policies at test', '9', MUTE));
    shell(host, 'CTDE', 'train wide, run local', svg,
      'Centralized Training, Decentralized Execution is the spine of cooperative MARL. During training a critic sees every agent state and action, which fixes the non-stationarity that plagues independent learners. At test time the critic is gone and each actor runs on its own local observation. MADDPG, QMIX, and MAPPO are three takes on the same split.');
  }

  // ── sw-checkpoint-replay：一个 worker 推进任务后崩溃，租约被释放，新的
  //    worker 从上一个检查点恢复 ────────────────────────────────────────────
  function checkpointReplay(host) {
    var W = 520, H = 250, period = 10;
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H });
    // 检查点日志：一条线上的四个步骤
    var stx = [110, 210, 310, 410], sty = 80;
    var i;
    svg.appendChild(svgEl('line', { x1: 80, y1: sty, x2: 440, y2: sty, stroke: SOFT, 'stroke-width': '1.4' }));
    for (i = 0; i < 4; i++) {
      var lit = i === 0 ? '0;0.05;1' : i === 1 ? '0;0.2;0.25;1' : i === 2 ? '0;0.35;0.4;1' : '0;0.78;0.83;1';
      var vals = i === 0 ? (SURF + ';' + BP + ';' + BP) : i === 1 ? (SURF + ';' + SURF + ';' + BP + ';' + BP) : i === 2 ? (SURF + ';' + SURF + ';' + BP + ';' + BP) : (SURF + ';' + SURF + ';' + BP + ';' + BP);
      var c = svgEl('rect', { x: stx[i] - 11, y: sty - 11, width: 22, height: 22, rx: '3', stroke: BP, 'stroke-width': '1.8', fill: SURF });
      c.appendChild(anim('fill', vals, lit, period));
      svg.appendChild(c);
      svg.appendChild(txt(stx[i], sty + 26, 'ckpt ' + i, '7', MUTE));
    }
    // worker A：运行到 ckpt2 后崩溃（变为警告色并淡出）
    var wa = svgEl('g', {});
    wa.appendChild(svgEl('rect', { x: -26, y: -20, width: 52, height: 40, rx: '3', stroke: BP, 'stroke-width': '2', fill: SURF }));
    wa.appendChild(txt(0, 5, 'worker A', '8', BP));
    var waPath = 'M' + stx[0] + ',150 L' + stx[2] + ',150 L' + stx[2] + ',150';
    wa.appendChild(svgEl('animateMotion', { path: waPath, keyTimes: '0;0.4;1', keyPoints: '0;1;1', dur: period + 's', repeatCount: 'indefinite', calcMode: 'linear' }));
    wa.appendChild(anim('opacity', '1;1;1;0.15;0.15', '0;0.4;0.45;0.5;1', period));
    svg.appendChild(wa);
    // ckpt2 处的崩溃标记
    var crash = txt(stx[2], 135, 'crash', '9', WARN);
    crash.appendChild(anim('opacity', '0;0;1;1;0;0', '0;0.42;0.46;0.55;0.6;1', period));
    svg.appendChild(crash);
    // worker B：出现后从 ckpt2 恢复并运行到 ckpt3
    var wb = svgEl('g', {});
    wb.appendChild(svgEl('rect', { x: -26, y: -20, width: 52, height: 40, rx: '3', stroke: WARN, 'stroke-width': '2', fill: SURF }));
    wb.appendChild(txt(0, 5, 'worker B', '8', WARN));
    var wbPath = 'M' + stx[2] + ',200 L' + stx[2] + ',200 L' + stx[3] + ',200';
    wb.appendChild(svgEl('animateMotion', { path: wbPath, keyTimes: '0;0.55;1', keyPoints: '0;0;1', dur: period + 's', repeatCount: 'indefinite', calcMode: 'linear' }));
    wb.appendChild(anim('opacity', '0;0;1;1;1', '0;0.5;0.55;0.95;1', period));
    svg.appendChild(wb);
    svg.appendChild(txt(W / 2, H - 12, 'crash releases the lease  ·  worker B resumes from the last durable checkpoint', '9', MUTE));
    shell(host, 'CHECKPOINT REPLAY', 'crash then resume', svg,
      'Durable execution is what lets multi-agent systems scale past one laptop. The runtime writes a checkpoint after each step keyed by a thread id. When a worker crashes mid-run its lease is released, and another worker picks the task up and resumes from the last committed checkpoint instead of starting over.');
  }

  LF.register({
    'sw-contract-net': contractNet,
    'sw-work-stealing': workStealing,
    'sw-handoff-routing': handoffRouting,
    'sw-agent-card-discovery': agentCard,
    'sw-debate-topology': debateTopology,
    'sw-theory-of-mind': theoryOfMind,
    'sw-ctde': ctde,
    'sw-checkpoint-replay': checkpointReplay
  });
})();
