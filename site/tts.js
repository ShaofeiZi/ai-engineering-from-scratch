/**
 * 基于浏览器内置 SpeechSynthesis API 的朗读支持。
 *
 * 在包含可读文章内容的页面 header 中注入朗读按钮（位于语言选择器与主题切换
 * 按钮之间），并在播放期间提供可控制暂停、停止和速度的浮动控制栏。
 *
 * 朗读范围为文章正文：标题、段落、列表、表格、课程格言与 meta 标签、测验文本和
 * 图注。代码块与渲染后的图示会被跳过，因为朗读它们需要单独的解析层。
 *
 * 不发起网络请求且无依赖：全部使用原生 Web Speech API。
 */
(function () {
  'use strict';

  if (typeof window === 'undefined') return;
  var VERSION = '20260829a';
  if (window.__AIFS_TTS_VERSION === VERSION && window.AIFS_TTS) return;
  window.__AIFS_TTS_VERSION = VERSION;

  function tr(value, params) {
    if (window.AIFS_I18n && typeof window.AIFS_I18n.t === 'function') {
      return window.AIFS_I18n.t(value, params);
    }
    return String(value == null ? '' : value).replace(/\{([A-Za-z0-9_]+)\}/g, function (token, name) {
      return params && Object.prototype.hasOwnProperty.call(params, name) ? params[name] : token;
    });
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function localSilentMode() {
    if (location.hostname !== '127.0.0.1' && location.hostname !== 'localhost') return false;
    try { return new URLSearchParams(location.search).get('ttsTest') === 'silent'; } catch (e) { return false; }
  }

  function SilentUtterance(text) {
    this.text = text;
    this.rate = 1;
    this.lang = '';
    this.voice = null;
    this.onend = null;
    this.onerror = null;
  }

  function silentSynthesizer() {
    var current = null;
    return {
      speaking: false,
      pending: false,
      paused: false,
      onvoiceschanged: null,
      getVoices: function () {
        var locale = document.documentElement.getAttribute('lang') || 'en-US';
        return [{ name: 'Silent QA voice', lang: locale, voiceURI: 'aifs-silent', localService: true, default: true }];
      },
      speak: function (utterance) {
        current = utterance;
        this.speaking = true;
        this.pending = false;
      },
      cancel: function () {
        current = null;
        this.speaking = false;
        this.pending = false;
        this.paused = false;
      },
      pause: function () { this.paused = true; },
      resume: function () { this.paused = false; this.speaking = !!current; },
      addEventListener: function () {},
    };
  }

  var silentMode = localSilentMode();
  // 可选 override 与本地静音模式是轻量测试 seam，使浏览器 QA 可操作所有控件
  // 而不会真正发声。
  var synth = window.__AIFS_TTS_SYNTH__ || window.speechSynthesis;
  var Utterance = window.__AIFS_TTS_UTTERANCE__ || window.SpeechSynthesisUtterance;
  if (silentMode) {
    synth = silentSynthesizer();
    Utterance = SilentUtterance;
  }
  var supported = !!(synth && typeof Utterance === 'function');

  var RATE_KEY = 'tts:rate';
  var LEGACY_VOICE_KEY = 'tts:voice';
  var VOICE_KEY_PREFIX = 'tts:voice:';
  var MAX_CHUNK = 160;

  // 这些区域属于页面 chrome 而非内容，其中任何文本都不会被朗读。
  var HARD_SKIP = [
    'script',
    'style',
    'svg',
    'canvas',
    'noscript',
    'nav',
    'textarea',
    'input',
    'select',
    '.katex',
    '.lesson-sidebar',
    '.toc-sidebar',
    '.site-header',
    '.site-footer',
    '.tts-bar',
    '.copy-btn',
    '[aria-hidden="true"]',
    '[data-tts-skip]',
  ].join(',');

  // 默认跳过交互元素（复制按钮、tab、控件），但以下承载真实内容的元素除外。
  var ALLOW_SELECTOR = '.quiz-option,.quiz-explanation,[data-tts-read]';

  var INTERACTIVE_SKIP = 'button,code,[role="button"]';

  var BLOCK_SELECTOR = [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'li', 'blockquote', 'dd', 'dt', 'figcaption', 'summary', 'tr',
    // 课程正文和 panel 使用普通 div 组织文本。
    '.motto',
    '.lesson-meta-tag',
    '.ai-panel-title',
    '.ai-panel-subtitle',
    '.quiz-question-num',
    '.quiz-question-text',
    '.quiz-option',
    '.quiz-explanation',
    '.quiz-score-number',
    '.quiz-score-label',
    '.quiz-deeper',
    // 交互课程图：由 title 和 caption 承载讲解。
    '.lf-label',
    '.lf-cap',
    '[data-tts-read]',
  ].join(',');

  // 包含以下任一元素的 block 是 wrapper：只读它自身的文本，使包含代码块的
  // 列表项仍能朗读其句子，同时不朗读内部代码。
  var NESTED_PROBE = BLOCK_SELECTOR + ',pre';

  // 浏览器阻止 Storage 时会抛错，而不是返回 null。
  // （如关闭 cookie 的 Safari、沙箱 iframe），因此所有读取都经过这些函数；
  // lsGet() 位于收集流程的热路径中，绝不能抛错。
  function lsGet(key) {
    try {
      return localStorage.getItem(key);
    } catch (e) {
      return null;
    }
  }

  function lsSet(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch (e) {
      // Storage 被禁用；偏好设置将无法持久化。
    }
  }

  // 只有显式续读课程时才允许跨页面延续播放。保存目标 route，避免无关页面继承音频。
  var RESUME_KEY = 'tts:resume';

  function routeKey(url) {
    try {
      var parsed = new URL(url, location.href);
      if (parsed.origin !== location.origin) return '';
      var pathname = parsed.pathname.replace(/\/lesson\.html$/, '/lesson');
      var entries = Array.from(parsed.searchParams.entries()).sort(function (a, b) {
        return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0;
      });
      var normalized = new URLSearchParams();
      entries.forEach(function (entry) { normalized.append(entry[0], entry[1]); });
      var search = normalized.toString();
      return pathname + (search ? '?' + search : '');
    } catch (e) {
      return '';
    }
  }

  function setResumeTarget(url) {
    var target = routeKey(url);
    if (!target) return clearResumeTarget();
    try {
      sessionStorage.setItem(RESUME_KEY, JSON.stringify({ target: target, createdAt: Date.now() }));
    } catch (e) {
      // sessionStorage 可能被禁用；播放状态将无法跨页面延续。
    }
  }

  function clearResumeTarget() {
    try {
      sessionStorage.removeItem(RESUME_KEY);
    } catch (e) {
      // sessionStorage 可能被禁用。
    }
  }

  function takeResumeTarget() {
    var raw = null;
    try {
      raw = sessionStorage.getItem(RESUME_KEY);
      sessionStorage.removeItem(RESUME_KEY);
    } catch (e) {
      return false;
    }
    if (!raw) return false;
    try {
      var intent = JSON.parse(raw);
      return !!(
        intent &&
        intent.target === routeKey(location.href) &&
        typeof intent.createdAt === 'number' &&
        Date.now() - intent.createdAt < 60000
      );
    } catch (e) {
      return false;
    }
  }

  var reducedMotion =
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  var reducedMotionListener = null;

  function prefersReducedMotion() {
    return !!(reducedMotion && reducedMotion.matches);
  }

  function bindReducedMotionPreference() {
    if (!reducedMotion || reducedMotionListener) return;
    reducedMotionListener = function (event) {
      if (event.matches) commitDragInertiaForReducedMotion();
    };
    if (typeof reducedMotion.addEventListener === 'function') {
      reducedMotion.addEventListener('change', reducedMotionListener);
    } else if (typeof reducedMotion.addListener === 'function') {
      reducedMotion.addListener(reducedMotionListener);
    }
  }

  function disposeReducedMotionPreference() {
    if (!reducedMotion || !reducedMotionListener) return;
    if (typeof reducedMotion.removeEventListener === 'function') {
      reducedMotion.removeEventListener('change', reducedMotionListener);
    } else if (typeof reducedMotion.removeListener === 'function') {
      reducedMotion.removeListener(reducedMotionListener);
    }
    reducedMotionListener = null;
  }

  var state = {
    chunks: [],
    index: 0,
    mode: 'idle',
    message: '',
    scope: null,
    // 控制条折叠为单个圆钮，以及区分拖动与点击的 guard。
    collapsed: false,
    dragged: false,
    highlighted: null,
    utterance: null,
    // 播放健康状态：识别过期 callback 的序列 token、防止 GC 的强引用、
    // 卡顿计数器，以及卡顿时回退到的离线 voice。
    seq: 0,
    spoken: [],
    stalls: 0,
    idleTicks: 0,
    forcedLocal: null,
    watchdog: null,
    observer: null,
    refreshTimer: null,
    navigationTarget: '',
  };

  var els = {};

  /* ---------------------------------------------------------------- 文本 */

  function contentRoot(scope) {
    if (scope && scope.nodeType === 1 && document.contains(scope)) return scope;
    var candidates = [
      '[data-tts-root]',
      '.lesson-article',
      '#lessonContent',
      'main#main',
      'main',
      '.container',
    ];
    for (var i = 0; i < candidates.length; i++) {
      var el = document.querySelector(candidates[i]);
      if (el && el.textContent.trim().length > 40) return el;
    }
    return null;
  }

  function isSkipped(el) {
    if (!el.closest) return true;
    var explicit = el.closest('[data-tts-read]');
    if (explicit && !explicit.closest('[data-tts-skip],.site-header,.site-footer,.tts-bar,[aria-hidden="true"]')) return false;
    if (el.closest(HARD_SKIP)) return true;
    if (el.closest(ALLOW_SELECTOR)) return false;
    // 本阅读器不会朗读代码块和已渲染的图示。
    if (el.closest('pre')) return true;
    return !!el.closest(INTERACTIVE_SKIP);
  }

  function isVisible(el) {
    if (el.hidden) return false;
    // display:none 时 offsetParent 为 null（position:fixed 时也如此，但可读块均未使用）。
    return el.offsetParent !== null || el.getClientRects().length > 0;
  }

  function clean(text) {
    var isChinese = languageBase(pageLocale()) === 'zh';
    var value = String(text || '')
      .replace(/\s+/g, ' ')
      .replace(/[`*_#~|]+/g, ' ')
      .replace(/\s+([,.;:!?])/g, '$1')
      .trim();
    var replacements = [
      [/\bCI\s*\/\s*CD\b/gi, 'C I slash C D'],
      [/\bLLMs?\b/g, function (match) { return match === 'LLMs' ? 'L L M s' : 'L L M'; }],
      [/\bAPIs?\b/g, function (match) { return match === 'APIs' ? 'A P I s' : 'A P I'; }],
      [/\bMCP\b/g, 'M C P'],
      [/\bSLOs?\b/g, function (match) { return match === 'SLOs' ? 'S L O s' : 'S L O'; }],
      [/\bADRs?\b/g, function (match) { return match === 'ADRs' ? 'A D R s' : 'A D R'; }],
      [/\bJSON\b/g, 'J S O N'],
      [/\bHTTP\b/g, 'H T T P'],
      [/\bSDKs?\b/g, function (match) { return match === 'SDKs' ? 'S D K s' : 'S D K'; }],
      [/\s*[→⇒]\s*/g, isChinese ? ' 指向 ' : ' leads to '],
      [/\s*≤\s*/g, isChinese ? ' 小于或等于 ' : ' less than or equal to '],
      [/\s*≥\s*/g, isChinese ? ' 大于或等于 ' : ' greater than or equal to '],
    ];
    for (var i = 0; i < replacements.length; i++) value = value.replace(replacements[i][0], replacements[i][1]);
    return value.replace(/\s+/g, ' ').trim();
  }

  /** 将视觉换行保留为朗读时的单词边界。 */
  function readableText(el) {
    var out = '';
    for (var i = 0; i < el.childNodes.length; i++) {
      var node = el.childNodes[i];
      if (node.nodeType === 3) {
        out += node.nodeValue;
      } else if (node.nodeType === 1) {
        if (node.tagName === 'BR') out += ' ';
        else if (!isSkipped(node)) out += readableText(node);
      }
    }
    return out;
  }

  /** 在句子边界将长 block 拆为可朗读片段。 */
  function split(text) {
    if (text.length <= MAX_CHUNK) return [text];
    var sentences = text.match(/[^.!?]+[.!?]*\s*/g) || [text];
    var out = [];
    var buf = '';
    for (var i = 0; i < sentences.length; i++) {
      var s = sentences[i];
      while (s.length > MAX_CHUNK) {
        // 单个超长句：在范围内最后一个空格处切分。
        var cut = s.lastIndexOf(' ', MAX_CHUNK);
        if (cut <= 0) cut = MAX_CHUNK;
        if (buf) {
          out.push(buf.trim());
          buf = '';
        }
        out.push(s.slice(0, cut).trim());
        s = s.slice(cut);
      }
      if ((buf + s).length > MAX_CHUNK) {
        out.push(buf.trim());
        buf = s;
      } else {
        buf += s;
      }
    }
    if (buf.trim()) out.push(buf.trim());
    return out.filter(Boolean);
  }

  /** 属于此元素、但不属于任何嵌套可读 block 的文本。 */
  function ownText(el) {
    var out = '';
    for (var i = 0; i < el.childNodes.length; i++) {
      var n = el.childNodes[i];
      if (n.nodeType === 3) {
        out += n.nodeValue;
      } else if (n.nodeType === 1 && !n.matches(BLOCK_SELECTOR) && !isSkipped(n)) {
        if (n.tagName === 'BR') {
          out += ' ';
          continue;
        }
        // 进入普通 wrapper，避免重复读取嵌套 block。
        out += n.querySelector(BLOCK_SELECTOR) ? ownText(n) : readableText(n);
      }
    }
    return out;
  }

  function tableRowText(row) {
    var cells = Array.prototype.slice.call(row.querySelectorAll(':scope > th, :scope > td'));
    if (!cells.length) return '';
    var table = row.closest('table');
    var headerCells = table ? Array.prototype.slice.call(table.querySelectorAll('thead tr:first-child > th, thead tr:first-child > td')) : [];
    var inHead = !!row.closest('thead');
    if (inHead || cells.every(function (cell) { return cell.tagName === 'TH'; })) {
      return tr('Table columns.') + ' ' + cells.map(function (cell) { return clean(cell.textContent); }).filter(Boolean).join('. ');
    }
    return cells.map(function (cell, index) {
      var value = clean(cell.textContent);
      var label = headerCells[index] ? clean(headerCells[index].textContent) : '';
      return label && value ? label + ': ' + value : value;
    }).filter(Boolean).join('. ');
  }

  function sectionName(el) {
    if (!el || !el.closest) return '';
    if (/^H[1-6]$/.test(el.tagName)) return clean(readableText(el));
    var namedSection = el.closest('[data-tts-section]');
    var namedLabel = namedSection && namedSection.getAttribute('data-tts-section');
    if (namedLabel) return clean(tr(namedLabel));
    var section = el.closest('[data-tts-section],article,section');
    var heading = section && section.querySelector('h1,h2,h3,h4,h5,h6');
    if (heading) return clean(readableText(heading));
    var cursor = el.previousElementSibling;
    while (cursor) {
      if (/^H[1-6]$/.test(cursor.tagName)) return clean(readableText(cursor));
      cursor = cursor.previousElementSibling;
    }
    return '';
  }

  /** 按文档顺序构建播放队列：[{ text, el }]。 */
  function collect(scope) {
    var root = contentRoot(scope);
    if (!root) return [];
    var blocks = root.querySelectorAll(BLOCK_SELECTOR);
    var chunks = [];
    var seen = 0;
    for (var i = 0; i < blocks.length; i++) {
      var el = blocks[i];
      if (isSkipped(el) || !isVisible(el)) continue;
      if (el.closest('tr') && !el.matches('tr')) continue;
      var text;
      if (el.hasAttribute('data-tts-label')) {
        text = clean(tr(el.getAttribute('data-tts-label')));
      } else if (el.matches('tr')) {
        text = clean(tableRowText(el));
      } else if (el.querySelector(NESTED_PROBE)) {
        // wrapper（如包含代码块的列表项、包含标题的 panel）。
        // 仅朗读它自身的文本；嵌套 block 会单独轮到。
        text = clean(ownText(el));
      } else {
        text = clean(readableText(el));
        if (el.matches('.quiz-option')) {
          // 标记为 <span>A</span><span>answer</span>，两者之间没有空白，
          // 因此将字母作为单独节拍朗读。
          var letter = el.querySelector('.opt-letter');
          var label = letter ? clean(letter.textContent || '') : '';
          var rest = label ? clean(text.slice(label.length)) : text;
          text = tr('Option') + ' ' + (label ? label + '. ' : '') + rest;
        } else if (el.matches('.quiz-explanation')) {
          text = tr('Explanation.') + ' ' + text;
        } else if (el.matches('.lf-label')) {
          text = tr('Interactive figure:') + ' ' + text + '.';
        }
      }
      if (text.length < 2) continue;
      var parts = split(text);
      for (var j = 0; j < parts.length; j++) {
        chunks.push({
          text: parts[j],
          el: el,
          key: chunkKey(el),
          part: j,
          section: sectionName(el),
          words: parts[j].split(/\s+/).filter(Boolean).length,
        });
      }
      seen++;
      if (seen > 4000) break;
    }
    return chunks;
  }

  function chunkKey(el) {
    if (!el || !el.closest) return '';
    var keyed = el.closest('[data-tts-key]');
    if (keyed) return 'key:' + keyed.getAttribute('data-tts-key');
    if (el.id) return 'id:' + el.id;
    return '';
  }

  /* --------------------------------------------------------------- voice */

  /**
   * 各平台的 voice 质量差异很大，浏览器默认项往往是最差选择
   * （Windows 默认提供机械感很强的 SAPI5 voice）。为每个 voice 评分，
   * 让 "Auto" 选择现有的最佳神经网络或云端 voice。
   */

  // 优选名称，质量由高到低；与 voice.name 宽松匹配。
  var PREFERRED = [
    // Edge / Windows 11 神经网络 voice。
    'microsoft aria', 'microsoft jenny', 'microsoft guy', 'microsoft ava',
    'microsoft andrew', 'microsoft emma', 'microsoft brian', 'microsoft libby',
    'microsoft ryan', 'microsoft sonia',
    // Chrome 云端 voice。
    'google us english', 'google uk english female', 'google uk english male',
    // Apple 高质量 voice。
    'samantha', 'ava', 'allison', 'tom', 'evan', 'zoe', 'nathan', 'joelle',
    'serena', 'daniel', 'alex',
  ];

  // macOS 趣味 voice：偏喜剧效果，不适合朗读正文。
  var NOVELTY = /^(albert|bad news|bahh|bells|boing|bubbles|cellos|deranged|good news|jester|organ|superstar|trinoids|whisper|wobble|zarvox|junior|ralph|fred|kathy|bruce|princess|grandma|grandpa|rocko|shelley|sandy|eddy|flo|reed|grandpa|bells)\b/i;

  function pageLocale() {
    var content = state && contentRoot(state.scope);
    var value = content && content.getAttribute && content.getAttribute('lang');
    if (!value && content && content.closest) {
      var localized = content.closest('[lang]');
      value = localized && localized.getAttribute('lang');
    }
    if (!value) value = document.documentElement.getAttribute('lang') || navigator.language || 'en';
    return String(value).replace('_', '-').toLowerCase();
  }

  function languageBase(value) {
    return String(value || '').toLowerCase().split(/[-_]/)[0];
  }

  function elementLocale(element) {
    var localized = element && element.closest ? element.closest('[lang]') : null;
    return localized ? String(localized.getAttribute('lang') || pageLocale()).toLowerCase() : pageLocale();
  }

  function sameLanguage(voice, locale) {
    return languageBase(voice && voice.lang) === languageBase(locale);
  }

  function voiceKey(locale) {
    return VOICE_KEY_PREFIX + (locale || pageLocale());
  }

  function score(v, locale) {
    var name = (v.name || '').toLowerCase();
    var lang = (v.lang || '').toLowerCase();
    var s = 0;
    var wanted = locale || pageLocale();

    if (NOVELTY.test(v.name || '')) return -100;

    // voice 名称中的显式质量标记。
    if (/natural|neural/.test(name)) s += 60;
    if (/premium|enhanced/.test(name)) s += 50;
    if (/\bonline\b/.test(name)) s += 40;
    if (/^google/.test(name)) s += 35;
    // SAPI5 桌面 voice 是机械感较强的旧版集合。
    if (/desktop/.test(name)) s -= 30;
    if (v.localService === false) s += 15;

    for (var i = 0; i < PREFERRED.length; i++) {
      if (name.indexOf(PREFERRED[i]) !== -1) {
        s += 100 - i; // earlier in the list wins ties
        break;
      }
    }

    // 首先匹配内容语言。再优美的英文 voice，也不适合作为西班牙语、印地语、
    // 日语或其他 locale 的默认项。
    if (sameLanguage(v, wanted)) s += 260;
    else s -= 250;
    if (lang === wanted) s += 35;
    if (v.default) s += 2;

    return s;
  }

  function voices(locale) {
    var wanted = locale || pageLocale();
    var all = (synth.getVoices() || []).slice();
    var ranked = all.map(function (v, i) {
      return { v: v, s: score(v, wanted), i: i };
    });
    ranked.sort(function (a, b) {
      return b.s - a.s || a.i - b.i;
    });
    return ranked
      .filter(function (r) {
        return r.s > -100;
      })
      .map(function (r) {
        return r.v;
      });
  }

  function bestVoice(locale) {
    locale = locale || pageLocale();
    var list = voices(locale);
    for (var i = 0; i < list.length; i++) {
      if (sameLanguage(list[i], locale)) return list[i];
    }
    // 没有匹配的已安装 voice 时，让浏览器根据 utterance.lang 选择；
    // 强制使用无关的英文 voice 更糟。
    return null;
  }

  function selectedVoice(locale) {
    locale = locale || pageLocale();
    // 本次会话中，持续中断的 voice 已被替换。
    if (state.forcedLocal && sameLanguage(state.forcedLocal, locale)) return state.forcedLocal;
    state.forcedLocal = null;
    var wanted = lsGet(voiceKey(locale));
    if (!wanted && languageBase(locale) === 'en') wanted = lsGet(LEGACY_VOICE_KEY);
    var all = synth.getVoices() || [];
    if (wanted) {
      for (var i = 0; i < all.length; i++) {
        if (all[i].voiceURI === wanted && sameLanguage(all[i], locale)) return all[i];
      }
    }
    // 没有已保存选择（或系统升级后消失）：自动选择最佳项。
    return bestVoice(locale);
  }

  function fillVoices() {
    if (!els.voice) return;
    var locale = pageLocale();
    var list = voices(locale);
    if (!list.length) return;
    var current = lsGet(voiceKey(locale)) || '';
    var best = bestVoice(locale);
    els.voice.innerHTML = '';
    var def = document.createElement('option');
    def.value = '';
    def.textContent = tr('Auto — {name}', { name: best ? best.name : locale.toUpperCase() });
    els.voice.appendChild(def);
    for (var i = 0; i < list.length; i++) {
      var o = document.createElement('option');
      o.value = list[i].voiceURI;
      o.textContent =
        (sameLanguage(list[i], locale) ? '★ ' : '') + list[i].name + ' (' + list[i].lang + ')';
      els.voice.appendChild(o);
    }
    var selected = '';
    for (var voiceIndex = 0; voiceIndex < list.length; voiceIndex++) {
      if (list[voiceIndex].voiceURI === current && sameLanguage(list[voiceIndex], locale)) selected = current;
    }
    els.voice.value = selected;
    // 已保存但不再存在的 voice 回退到 Auto。
    if (els.voice.value !== selected) els.voice.value = '';
  }

  function rate() {
    var stored = parseFloat(lsGet(RATE_KEY));
    return stored >= 0.5 && stored <= 3 ? stored : 1;
  }

  /* ------------------------------------------------------------- 播放 */

  function isActive() {
    return state.mode !== 'idle';
  }

  function isPlaying() {
    return state.mode === 'playing';
  }

  function isPaused() {
    return state.mode === 'paused';
  }

  function isWaiting() {
    return state.mode === 'waiting';
  }

  function remainingMinutes() {
    var words = 0;
    for (var i = state.index; i < state.chunks.length; i++) words += state.chunks[i].words || 0;
    return words ? Math.max(1, Math.ceil(words / (180 * rate()))) : 0;
  }

  function highlight(el) {
    if (state.highlighted === el) return;
    if (state.highlighted) state.highlighted.classList.remove('tts-reading');
    state.highlighted = el || null;
    if (!el) return;
    el.classList.add('tts-reading');
    var box = el.getBoundingClientRect();
    if (box.top < 80 || box.bottom > window.innerHeight - 80) {
      // 每个 chunk 边界自动滚动是该功能中动态效果最强的部分，
      // 因此遵循与 CSS 相同的偏好设置。
      el.scrollIntoView({ block: 'center', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
    }
  }

  /**
   * 课程页面在首次绘制后仍会继续构建（panel、diagram、figure）。如果当前 block
   * 已被替换，则根据实时 DOM 重建队列，并通过文本保持当前位置。
   */
  function refreshQueue(restartIfMissing) {
    var current = state.chunks[state.index];
    var oldIndex = state.index;
    var fresh = collect(state.scope);
    if (!fresh.length) return false;
    var at = -1;
    for (var i = 0; i < fresh.length; i++) {
      if (current && fresh[i].el === current.el && fresh[i].text === current.text) {
        at = i;
        break;
      }
    }
    if (at < 0) {
      var keyedChunks = [];
      for (var keyIndex = 0; keyIndex < fresh.length; keyIndex++) {
        if (current && current.key && fresh[keyIndex].key === current.key) keyedChunks.push(keyIndex);
      }
      if (keyedChunks.length) {
        var partOffset = Math.min(current.part || 0, keyedChunks.length - 1);
        at = keyedChunks[partOffset];
      }
    }
    if (at < 0) {
      for (var j = 0; j < fresh.length; j++) {
        if (current && fresh[j].text === current.text) {
          at = j;
          break;
        }
      }
    }
    state.chunks = fresh;
    state.index = at >= 0 ? at : Math.min(oldIndex, fresh.length - 1);
    render();
    if (restartIfMissing && at < 0 && isPlaying()) {
      cancelSpeech();
      deferSpeak();
    }
    return true;
  }

  function scheduleRefresh() {
    if (!isActive()) return;
    clearTimeout(state.refreshTimer);
    state.refreshTimer = setTimeout(function () {
      state.refreshTimer = null;
      refreshQueue(true);
    }, 90);
  }

  function nonNarrationClasses(value) {
    return String(value || '')
      .split(/\s+/)
      .filter(function (name) {
        return name && name !== 'tts-reading' && name !== 'tts-active';
      })
      .sort()
      .join(' ');
  }

  function isNarrationMutation(mutation, target) {
    if (target.closest('.tts-bar,.tts-from-here,.tts-toggle')) return true;
    if (mutation.type !== 'attributes' || mutation.attributeName !== 'class') return false;

    // 高亮移动会先从上一个 block 移除 tts-reading，再添加到下一个 block。
    // Mutation callback 在两次操作后运行，因此只与 state.highlighted 比较会漏掉
    // 移除动作，并导致朗读器重建自身队列。仅在非朗读 class 未变化时忽略 class
    // mutation；真实的应用 class 变化仍需刷新可读 DOM。
    return nonNarrationClasses(mutation.oldValue) ===
      nonNarrationClasses(target.getAttribute('class'));
  }

  function observeQueue() {
    if (state.observer || typeof MutationObserver !== 'function' || !document.body) return;
    state.observer = new MutationObserver(function (mutations) {
      var meaningful = mutations.some(function (mutation) {
        var target = mutation.target && mutation.target.nodeType === 1 ? mutation.target : mutation.target.parentElement;
        if (!target) return false;
        if (isNarrationMutation(mutation, target)) return false;
        return true;
      });
      if (meaningful) scheduleRefresh();
    });
    state.observer.observe(document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeOldValue: true,
      attributeFilter: ['hidden', 'aria-hidden', 'open', 'class', 'style'],
    });
  }

  function disconnectQueueObserver() {
    clearTimeout(state.refreshTimer);
    state.refreshTimer = null;
    if (state.observer) state.observer.disconnect();
    state.observer = null;
  }

  function speakCurrent() {
    if (state.index >= state.chunks.length) {
      stop();
      return;
    }
    var stale = state.chunks[state.index].el;
    if (stale && (!document.contains(stale) || !isVisible(stale))) refreshQueue(false);
    var chunk = state.chunks[state.index];
    if (!chunk) {
      stop();
      return;
    }
    var u = new Utterance(chunk.text);
    u.rate = rate();
    var chunkLocale = elementLocale(chunk.el);
    u.lang = chunkLocale;
    var v = selectedVoice(chunkLocale);
    if (v) {
      u.voice = v;
      u.lang = v.lang;
    }
    // 已跳过 utterance 的 callback 不得推进队列：watchdog 可能重新朗读一个 chunk，
    // 而原 utterance 仍停留在引擎内部。
    var token = ++state.seq;

    u.onend = function () {
      if (token !== state.seq || !isPlaying()) return;
      state.index++;
      state.stalls = 0;
      render();
      // 某些 Chromium 版本会因在 onend 内同步调用 speak() 而卡住队列；
      // 先让出执行权更可靠。
      deferSpeak();
    };
    u.onerror = function (e) {
      // "interrupted"/"canceled" 是 stop()/next() 的正常结果。
      if (e && (e.error === 'interrupted' || e.error === 'canceled')) return;
      if (token !== state.seq || !isPlaying()) return;
      state.index++;
      state.stalls = 0;
      if (state.index < state.chunks.length) deferSpeak();
      else stop();
    };

    state.utterance = u;
    // Chromium 可能回收正在播放的 utterance 并将其截断，因此要强引用近期项。
    state.spoken.push(u);
    if (state.spoken.length > 8) state.spoken.shift();

    highlight(chunk.el);
    synth.speak(u);
  }

  /* ------------------------------------------------------- 从此处朗读 */

  /**
   * 任意节点所在的可读 block。当节点位于不可读内容（如代码块）中时，返回节点
   * 自身，以便调用方从其后续内容开始。
   */
  function blockOf(node) {
    var el = node && node.nodeType === 3 ? node.parentNode : node;
    var first = el;
    var root = contentRoot();
    while (el && el.nodeType === 1) {
      if (el.matches(BLOCK_SELECTOR) && !isSkipped(el)) return el;
      if (root && el === root) break;
      el = el.parentNode;
    }
    return first && first.nodeType === 1 ? first : null;
  }

  /** block 在队列中的位置：它自身，或紧随其后的下一项。 */
  function indexOfBlock(el) {
    if (!el) return 0;
    for (var i = 0; i < state.chunks.length; i++) {
      var c = state.chunks[i].el;
      if (c === el || (c && (c.contains(el) || el.contains(c)))) return i;
    }
    // 未入队（被跳过的 block）：继续查找文档中的下一项。
    for (var j = 0; j < state.chunks.length; j++) {
      var pos = el.compareDocumentPosition(state.chunks[j].el);
      if (pos & Node.DOCUMENT_POSITION_FOLLOWING) return j;
    }
    return 0;
  }

  /** 当前文本选区起点所在的 block（如有）。 */
  function selectedBlock() {
    var sel = window.getSelection && window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    if (!clean(sel.toString())) return null;
    var root = contentRoot();
    var node = sel.getRangeAt(0).startContainer;
    if (root && !root.contains(node.nodeType === 3 ? node.parentNode : node)) return null;
    return blockOf(node);
  }

  function readFromSelection() {
    var block = selectedBlock();
    if (!block) return false;
    hideSelectionButton();
    var sel = window.getSelection && window.getSelection();
    if (sel && sel.removeAllRanges) sel.removeAllRanges();
    return start(false, block);
  }

  function start(silentIfEmpty, fromEl, scopeEl) {
    if (!supported) {
      if (!silentIfEmpty) flash('Read aloud is unavailable in this browser');
      return false;
    }
    state.scope = scopeEl && document.contains(scopeEl) ? scopeEl : null;
    state.chunks = collect(state.scope);
    if (!state.chunks.length) {
      if (!silentIfEmpty) flash('Nothing to read on this page');
      state.scope = null;
      return false;
    }
    cancelSpeech();
    state.index = fromEl ? indexOfBlock(fromEl) : 0;
    state.mode = 'playing';
    state.message = '';
    state.stalls = 0;
    clearResumeTarget();
    observeQueue();
    startWatchdog();
    render();
    speakCurrent();
    return true;
  }

  function pause() {
    if (!isPlaying()) return;
    state.mode = 'paused';
    clearResumeTarget();
    synth.pause();
    render();
  }

  function resume() {
    if (!isPaused()) return;
    state.mode = 'playing';
    render();
    synth.resume();
    if (!state.utterance) speakCurrent();
  }

  function stop() {
    state.mode = 'idle';
    state.message = '';
    state.utterance = null;
    state.chunks = [];
    state.index = 0;
    state.scope = null;
    state.navigationTarget = '';
    clearResumeTarget();
    stopWatchdog();
    disconnectQueueObserver();
    cancelSpeech();
    highlight(null);
    hideSelectionButton();
    render();
  }

  function fail(message) {
    state.mode = 'error';
    state.message = message;
    clearResumeTarget();
    stopWatchdog();
    disconnectQueueObserver();
    cancelSpeech();
    highlight(null);
    render();
  }

  /**
   * 在课程导航之间延续播放。课程正文在 load 后获取，因此开始前需轮询，
   * 直到出现可读内容。
   */
  function autoResume() {
    if (!takeResumeTarget()) return;
    state.mode = 'waiting';
    render();

    var tries = 0;
    var lastSize = -1;
    var timer = setInterval(function () {
      if (!isWaiting()) {
        clearInterval(timer);
        return;
      }
      tries++;
      // 等待文章停止增长，否则会把页面即将替换的段落加入队列，
      // 高亮也会落在已脱离 DOM 的节点上。
      var root = contentRoot();
      var size = root ? root.textContent.trim().length : 0;
      if (!size || size !== lastSize) {
        lastSize = size;
        if (tries <= 60) return;
      }
      if (start(true)) {
        clearInterval(timer);
        armGestureFallback();
        return;
      }
      if (tries > 60) {
        // 约 15 秒：页面仍无内容可读，因此放弃交接。
        state.mode = 'idle';
        clearResumeTarget();
        clearInterval(timer);
        render();
      }
    }, 250);
  }

  function isLessonContinuationLink(link) {
    if (!link || !link.matches('.lesson-nav-btn,.continue-link')) return false;
    try {
      var url = new URL(link.href, location.href);
      return url.origin === location.origin && /\/lesson(?:\.html)?$/.test(url.pathname) && !!url.searchParams.get('path');
    } catch (e) {
      return false;
    }
  }

  function bindNavigationResume() {
    document.addEventListener('click', function (event) {
      var link = event.target.closest && event.target.closest('a[href]');
      if (!link) return;
      state.navigationTarget = '';
      clearResumeTarget();
      if (!isPlaying() || !isLessonContinuationLink(link)) return;
      if (silentMode) {
        var silentUrl = new URL(link.href, location.href);
        silentUrl.searchParams.set('ttsTest', 'silent');
        link.href = silentUrl.toString();
      }
      state.navigationTarget = routeKey(link.href);
      setResumeTarget(link.href);
    }, true);
  }

  /**
   * 某些浏览器拒绝在用户尚未交互的页面朗读。若发生此情况，首次点击或按键时启动。
   */
  function armGestureFallback() {
    if (synth.speaking) return;
    var retry = function () {
      document.removeEventListener('pointerdown', retry, true);
      document.removeEventListener('keydown', retry, true);
      if (isPlaying() && !synth.speaking) speakCurrent();
    };
    document.addEventListener('pointerdown', retry, true);
    document.addEventListener('keydown', retry, true);
    setTimeout(function () {
      if (isPlaying() && !synth.speaking) {
        flash('Press play or click the page to continue reading');
      }
    }, 1200);
  }

  function jump(delta) {
    if (!isPlaying() && !isPaused()) return;
    var next = state.index + delta;
    if (next < 0) next = 0;
    if (next >= state.chunks.length) {
      stop();
      return;
    }
    state.index = next;
    state.mode = 'playing';
    cancelSpeech();
    render();
    speakCurrent();
  }

  /**
   * cancel() 不会屏蔽被终止 utterance 的 callback：WebKit 仍会为已取消的
   * utterance 触发 onend，未提供可识别 `error` 字符串的引擎也会进入同一路径。
   * 因此先递增序列号，使之后到达的事件变为过期事件，不能推进队列或启动第二个阅读器。
   */
  function cancelSpeech() {
    state.seq++;
    if (supported) synth.cancel();
  }

  /**
   * 在新 task 中把控制权交给下一个 chunk。每次延迟都会记住调度时的序列号，
   * 因此若期间发生卡顿重试、跳转或停止等接管操作，该任务会静默退出，
   * 不会启动第二个阅读器。
   */
  function deferSpeak() {
    var expected = state.seq;
    setTimeout(function () {
      if (!isPlaying()) return;
      if (state.seq !== expected) return;
      speakCurrent();
    }, 0);
  }

  /**
   * 网络 voice（Google 以及 Edge 的 "Online (Natural)" 集合）可能丢弃
   * utterance，却不触发 onend 或 onerror。此时队列会静默卡住，控制条仍显示正在
   * 朗读，听起来就像“声音几秒后停止”。
   *
   * API 不会报告这种情况，因此需要轮询：若我们认为仍在播放，但引擎既未 speaking
   * 也未 pending，说明 utterance 已被丢弃。
   */
  function startWatchdog() {
    stopWatchdog();
    state.idleTicks = 0;
    state.watchdog = setInterval(function () {
      if (!isPlaying()) return;
      if (synth.speaking || synth.pending) {
        state.idleTicks = 0;
        return;
      }
      // 等待两个 tick，避免把 utterance 之间的正常间隔误判为卡顿。
      if (++state.idleTicks < 2) return;
      state.idleTicks = 0;
      recoverFromStall();
    }, 400);
  }

  function stopWatchdog() {
    if (state.watchdog) clearInterval(state.watchdog);
    state.watchdog = null;
  }

  function recoverFromStall() {
    state.stalls++;
    // 第四次无响应后放弃；第五次只会再跳过一个 chunk。
    if (state.stalls >= 4) {
      fail('Speech engine stopped responding');
      return;
    }
    var local = state.stalls >= 2 && !state.forcedLocal ? localVoice() : null;
    if (local) {
      // 持续中断的云端 voice 不会自行恢复；改用效果普通但不会中断的离线 voice。
      state.forcedLocal = local;
      flash(tr('Switched to {name} — the previous voice kept cutting out', { name: local.name }));
    } else if (state.stalls >= 3) {
      // 回退后仍卡顿：跳过该 chunk，而不是无限重试，以便继续朗读文章其余部分。
      state.index++;
      if (state.index >= state.chunks.length) {
        stop();
        return;
      }
      render();
    }
    cancelSpeech();
    deferSpeak();
  }

  /** 最佳离线 voice，在网络 voice 持续中断时使用。 */
  function localVoice() {
    var locale = pageLocale();
    var all = voices(locale);
    for (var i = 0; i < all.length; i++) {
      if (all[i].localService && sameLanguage(all[i], locale)) return all[i];
    }
    return null;
  }

  /* ------------------------------------------------------------------ ui */

  function flash(msg) {
    if (!els.bar) return;
    els.bar.hidden = false;
    els.bar.classList.add('is-visible');
    els.status.textContent = tr(msg);
    setTimeout(function () {
      if (!isActive()) {
        els.bar.classList.remove('is-visible');
        els.bar.hidden = true;
      }
    }, 2200);
  }

  function updateBarReserve(active) {
    if (!document.body) return;
    document.body.classList.toggle('tts-active', active);
    if (!active) {
      document.documentElement.style.removeProperty('--tts-bar-height');
      return;
    }
    requestAnimationFrame(function () {
      if (els.bar && !els.bar.hidden) {
        document.documentElement.style.setProperty('--tts-bar-height', Math.ceil(els.bar.getBoundingClientRect().height) + 'px');
      }
    });
  }

  function render() {
    var active = isActive();
    if (els.toggle) {
      els.toggle.classList.toggle('is-active', isPlaying() || isWaiting());
      els.toggle.setAttribute('aria-pressed', active && state.mode !== 'error' ? 'true' : 'false');
      els.toggle.setAttribute(
        'aria-label',
        tr(isPaused()
          ? 'Resume reading aloud'
          : state.mode === 'error'
            ? 'Dismiss read aloud error'
            : active
              ? 'Stop reading aloud'
              : 'Read this page aloud')
      );
      els.toggle.title = els.toggle.getAttribute('aria-label');
    }
    if (!els.bar) return;
    updateBarReserve(active);
    var wasHidden = els.bar.hidden;
    els.bar.hidden = !active;
    els.bar.classList.toggle('is-visible', active);
    if (active && wasHidden && els.bar.classList.contains('is-placed')) schedulePlacementBoundsRefresh();
    // 折叠后，圆钮上的扬声器图标是唯一剩余的播放反馈。
    els.bar.classList.toggle('is-reading', isPlaying() || isWaiting());
    if (!active) return;
    els.playPause.textContent = isPaused() ? '▶' : '⏸';
    els.playPause.setAttribute('aria-label', tr(isPaused() ? 'Resume' : 'Pause'));
    els.playPause.disabled = isWaiting() || state.mode === 'error';
    if (isWaiting()) {
      els.status.textContent = tr('Loading the next lesson…');
      els.progress.removeAttribute('value');
      return;
    }
    if (state.mode === 'error') {
      els.status.textContent = tr(state.message || 'Read aloud stopped');
      els.progress.value = 0;
      return;
    }
    var current = state.chunks[state.index] || {};
    var section = current.section ? current.section.slice(0, 52) : tr('Page');
    var minutes = remainingMinutes();
    var statusParams = {
      state: tr(isPaused() ? 'Paused' : 'Reading'),
      section: section,
      current: Math.min(state.index + 1, state.chunks.length),
      total: state.chunks.length,
      minutes: minutes
    };
    var status = tr(minutes
      ? '{state} · {section} · {current}/{total} · {minutes} min left'
      : '{state} · {section} · {current}/{total}', statusParams);
    els.status.textContent = status;
    els.progress.max = Math.max(1, state.chunks.length);
    els.progress.value = Math.min(state.index + 1, state.chunks.length);
  }

  function icon() {
    return (
      '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
      'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<polygon points="4 9 8 9 13 5 13 19 8 15 4 15"></polygon>' +
      '<path class="tts-wave-1" d="M16.5 8.5a5 5 0 0 1 0 7"></path>' +
      '<path class="tts-wave-2" d="M19.5 5.5a9 9 0 0 1 0 13"></path>' +
      '</svg>'
    );
  }

  function hashStartElement() {
    var hash = location.hash ? decodeURIComponent(location.hash.slice(1)) : '';
    if (!hash) return null;
    var target = document.getElementById(hash);
    var phaseMatch = hash.match(/^phase-(\d{1,2})$/);
    if (!target && phaseMatch) target = document.querySelector('.roadmap-node[data-phase="' + parseInt(phaseMatch[1], 10) + '"]');
    var root = contentRoot();
    if (!target || !root || !root.contains(target)) return null;
    return blockOf(target) || target;
  }

  function closeCompactNavigation() {
    var toggle = document.querySelector('.header-menu-toggle[aria-expanded="true"]');
    if (toggle) toggle.click();
  }

  function placeToggle(btn) {
    var header = btn.closest('.site-header') || document.querySelector('.site-header');
    var inner = header && header.querySelector('.header-inner');
    var themeToggle = header && header.querySelector('.theme-toggle:not(.tts-toggle)');
    if (!inner || !themeToggle) return;
    var compact = window.matchMedia && window.matchMedia('(max-width: 1100px)').matches;
    var menuToggle = inner.querySelector('.header-menu-toggle');
    if (compact && menuToggle) inner.insertBefore(btn, menuToggle);
    else themeToggle.parentNode.insertBefore(btn, themeToggle);
  }

  function buildButton() {
    var themeToggle = document.querySelector('.theme-toggle');
    if (!themeToggle || !themeToggle.parentNode) return null;
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'theme-toggle tts-toggle';
    btn.id = 'ttsToggle';
    btn.setAttribute('data-header-persistent', 'true');
    btn.innerHTML = icon();
    btn.setAttribute('aria-label', tr('Read this page aloud'));
    btn.title = tr('Read this page aloud');
    btn.setAttribute('aria-pressed', 'false');
    placeToggle(btn);
    var compact = window.matchMedia && window.matchMedia('(max-width: 1100px)');
    if (compact) {
      var reposition = function () { placeToggle(btn); };
      if (typeof compact.addEventListener === 'function') compact.addEventListener('change', reposition);
      else if (typeof compact.addListener === 'function') compact.addListener(reposition);
    }
    if (!supported) {
      btn.disabled = true;
      btn.setAttribute('aria-label', tr('Read aloud unavailable in this browser'));
      btn.title = tr('Read aloud unavailable in this browser');
      return btn;
    }
    btn.addEventListener('click', function () {
      closeCompactNavigation();
      if (isPaused()) resume();
      else if (isActive()) stop();
      else if (!readFromSelection()) start(false, hashStartElement());
    });
    return btn;
  }

  /* ------------------------------------------------- 折叠与拖动 */

  var COLLAPSED_KEY = 'tts:collapsed';
  var POS_KEY = 'tts:pos';
  var DRAG_SLOP = 4;
  var dragInertiaFrame = 0;
  var placementBoundsFrame = 0;
  var placementTransitionFrame = 0;
  var placementBounds = null;
  var placedPosition = null;
  var playerResizeObserver = null;

  function stopDragInertia() {
    if (dragInertiaFrame) window.cancelAnimationFrame(dragInertiaFrame);
    dragInertiaFrame = 0;
    if (els.bar) {
      els.bar.classList.remove('is-gliding');
      els.bar.style.removeProperty('transition');
    }
  }

  function commitDragInertiaForReducedMotion() {
    if (!els.bar || (!dragInertiaFrame && !els.bar.classList.contains('is-gliding'))) return;
    stopDragInertia();
    if (!els.bar.classList.contains('is-placed') || !placedPosition) return;
    els.bar.style.transition = 'none';
    place(placedPosition.x, placedPosition.y, true, placementBounds || refreshPlacementBounds());
    restorePlacementTransition();
  }

  /** 折叠后控制条仅显示扬声器圆钮，点击即可展开。 */
  function setCollapsed(on, quiet) {
    state.collapsed = !!on;
    if (!quiet) lsSet(COLLAPSED_KEY, on ? '1' : '0');
    if (!els.bar) return;
    els.bar.classList.toggle('is-collapsed', state.collapsed);
    if (els.collapse) {
      els.collapse.innerHTML = state.collapsed ? icon() : '▾';
      els.collapse.setAttribute('aria-expanded', state.collapsed ? 'false' : 'true');
      var label = tr(state.collapsed ? 'Expand read aloud controls' : 'Collapse controls');
      els.collapse.setAttribute('aria-label', label);
      els.collapse.title = label + tr(' (drag to move)');
    }
    schedulePlacementBoundsRefresh();
  }

  function savedPosition() {
    try {
      var raw = lsGet(POS_KEY);
      if (!raw) return null;
      var p = JSON.parse(raw);
      if (typeof p.x !== 'number' || typeof p.y !== 'number') return null;
      return p;
    } catch (e) {
      return null;
    }
  }

  /** 将控制条固定在 viewport 坐标处，替换默认锚定方式。 */
  function enterPlacedMode() {
    if (!els.bar || els.bar.classList.contains('is-placed')) return;
    els.bar.classList.add('is-placed');
    els.bar.style.left = '0px';
    els.bar.style.top = '0px';
  }

  function place(x, y, persist, limits) {
    if (!els.bar) return;
    limits = limits || placementBounds || refreshPlacementBounds();
    var cx = Math.min(Math.max(limits.minX, x), limits.maxX);
    var cy = Math.min(Math.max(limits.minY, y), limits.maxY);
    placedPosition = { x: cx, y: cy };
    enterPlacedMode();
    els.bar.style.transform = 'translate3d(' + cx + 'px,' + cy + 'px,0)';
    if (els.resetPosition) els.resetPosition.hidden = false;
    if (persist) lsSet(POS_KEY, JSON.stringify({ x: cx, y: cy }));
    return placedPosition;
  }

  function resetPosition() {
    stopDragInertia();
    lsSet(POS_KEY, '');
    placedPosition = null;
    if (!els.bar) return;
    els.bar.classList.remove('is-placed', 'is-gliding');
    els.bar.style.removeProperty('left');
    els.bar.style.removeProperty('top');
    els.bar.style.removeProperty('transform');
    els.bar.style.removeProperty('transition');
    if (els.resetPosition) els.resetPosition.hidden = true;
    updateBarReserve(isActive());
  }

  function refreshPlacementBounds(rect) {
    var measured = rect || (els.bar ? els.bar.getBoundingClientRect() : null);
    var width = measured && measured.width ? measured.width : placementBounds ? placementBounds.width : 0;
    var height = measured && measured.height ? measured.height : placementBounds ? placementBounds.height : 0;
    placementBounds = {
      minX: 8,
      minY: 8,
      maxX: Math.max(8, document.documentElement.clientWidth - width - 8),
      maxY: Math.max(8, window.innerHeight - height - 8),
      width: width,
      height: height,
    };
    return placementBounds;
  }

  function schedulePlacementBoundsRefresh() {
    if (placementBoundsFrame) return;
    placementBoundsFrame = window.requestAnimationFrame(function () {
      placementBoundsFrame = 0;
      if (!els.bar) return;
      if (els.bar.classList.contains('is-placed')) clampToViewport();
      else refreshPlacementBounds();
    });
  }

  function restorePlacementTransition() {
    if (placementTransitionFrame) window.cancelAnimationFrame(placementTransitionFrame);
    placementTransitionFrame = window.requestAnimationFrame(function () {
      placementTransitionFrame = 0;
      if (!els.bar || els.bar.classList.contains('is-dragging') || els.bar.classList.contains('is-gliding')) return;
      els.bar.style.removeProperty('transition');
    });
  }

  function resistEdge(value, min, max) {
    if (value < min) {
      var before = min - value;
      return min - (before * 0.3) / (1 + before / 96);
    }
    if (value > max) {
      var after = value - max;
      return max + (after * 0.3) / (1 + after / 96);
    }
    return value;
  }

  function placeDuringDrag(x, y, limits) {
    if (!els.bar) return;
    var resistedX = resistEdge(x, limits.minX, limits.maxX);
    var resistedY = resistEdge(y, limits.minY, limits.maxY);
    enterPlacedMode();
    els.bar.style.transform = 'translate3d(' + resistedX + 'px,' + resistedY + 'px,0)';
    placedPosition = { x: resistedX, y: resistedY };
    return placedPosition;
  }

  function clampToViewport() {
    if (!els.bar || !els.bar.classList.contains('is-placed')) return;
    stopDragInertia();
    var rect = els.bar.getBoundingClientRect();
    var limits = refreshPlacementBounds(rect);
    var current = placedPosition || { x: rect.left, y: rect.top };
    place(current.x, current.y, false, limits);
  }

  /**
   * 可将控制条拖到文章上的任意位置。只要指针没有实际移动，button 和 select 就保留
   * 自身行为，因此折叠后的圆钮既可点击，也可拖动。
   */
  function bindDrag(bar) {
    var active = false;
    var moved = false;
    var startX = 0;
    var startY = 0;
    var originX = 0;
    var originY = 0;
    var lastX = 0;
    var lastY = 0;
    var lastTime = 0;
    var velocityX = 0;
    var velocityY = 0;
    var currentX = 0;
    var currentY = 0;
    var dragLimits = null;

    function beginInertia(initialVelocityX, initialVelocityY, initialX, initialY, limits) {
      if (prefersReducedMotion()) {
        bar.style.transition = 'none';
        place(initialX, initialY, true, limits);
        restorePlacementTransition();
        return;
      }

      var x = initialX;
      var y = initialY;
      var vx = initialVelocityX;
      var vy = initialVelocityY;
      var previous = performance.now();
      bar.classList.add('is-gliding');
      bar.style.transition = 'none';

      function settle() {
        dragInertiaFrame = 0;
        bar.classList.remove('is-gliding');
        place(x, y, true, limits);
        restorePlacementTransition();
      }

      function glide(now) {
        var elapsed = Math.min(32, Math.max(1, now - previous));
        previous = now;

        x += vx * elapsed;
        y += vy * elapsed;

        if (x < limits.minX || x > limits.maxX) {
          x = Math.min(Math.max(limits.minX, x), limits.maxX);
          vx *= -0.24;
        }
        if (y < limits.minY || y > limits.maxY) {
          y = Math.min(Math.max(limits.minY, y), limits.maxY);
          vy *= -0.24;
        }

        var damping = Math.pow(0.9, elapsed / (1000 / 60));
        vx *= damping;
        vy *= damping;
        place(x, y, false, limits);

        if (Math.abs(vx) + Math.abs(vy) < 0.018) {
          settle();
          return;
        }
        dragInertiaFrame = window.requestAnimationFrame(glide);
      }

      dragInertiaFrame = window.requestAnimationFrame(glide);
    }

    bar.addEventListener('pointerdown', function (e) {
      if (e.button != null && e.button !== 0) return;
      if (window.matchMedia && window.matchMedia('(max-width: 720px)').matches) return;
      // 控制条展开时不要干扰真实控件；圆钮整体都是 button，因此也必须可拖动。
      if (!state.collapsed && e.target.closest('select,input,option')) return;
      stopDragInertia();
      var rect = bar.getBoundingClientRect();
      dragLimits = refreshPlacementBounds(rect);
      active = true;
      moved = false;
      startX = e.clientX;
      startY = e.clientY;
      originX = rect.left;
      originY = rect.top;
      currentX = originX;
      currentY = originY;
      placedPosition = { x: originX, y: originY };
      lastX = e.clientX;
      lastY = e.clientY;
      lastTime = e.timeStamp || performance.now();
      velocityX = 0;
      velocityY = 0;
    });

    bar.addEventListener('pointermove', function (e) {
      if (!active) return;
      var dx = e.clientX - startX;
      var dy = e.clientY - startY;
      if (!moved && Math.abs(dx) < DRAG_SLOP && Math.abs(dy) < DRAG_SLOP) return;
      if (!moved) {
        moved = true;
        bar.classList.add('is-dragging');
        if (bar.setPointerCapture) bar.setPointerCapture(e.pointerId);
      }
      e.preventDefault();
      var now = e.timeStamp || performance.now();
      var elapsed = Math.max(1, now - lastTime);
      var sampleX = (e.clientX - lastX) / elapsed;
      var sampleY = (e.clientY - lastY) / elapsed;
      velocityX = velocityX * 0.6 + sampleX * 0.4;
      velocityY = velocityY * 0.6 + sampleY * 0.4;
      lastX = e.clientX;
      lastY = e.clientY;
      lastTime = now;
      var placement = placeDuringDrag(originX + dx, originY + dy, dragLimits);
      currentX = placement.x;
      currentY = placement.y;
    });

    var end = function (e) {
      if (!active) return;
      active = false;
      if (!moved) return;
      bar.classList.remove('is-dragging');
      if (bar.releasePointerCapture && e.pointerId != null) {
        try {
          bar.releasePointerCapture(e.pointerId);
        } catch (err) {
          // pointer capture 可能已经释放。
        }
      }
      if (e.type === 'pointerup' && Math.abs(velocityX) + Math.abs(velocityY) >= 0.06) {
        beginInertia(velocityX, velocityY, currentX, currentY, dragLimits);
      } else {
        bar.style.transition = 'none';
        place(currentX, currentY, true, dragLimits);
        restorePlacementTransition();
      }
      // 吞掉完成拖动后即将产生的 click。取消的 gesture 不会触发 click，
      // 若此时启用 guard，反而会吞掉下一次真实点击。
      state.dragged = e.type === 'pointerup';
    };

    bar.addEventListener('pointerup', end);
    bar.addEventListener('pointercancel', end);
    window.addEventListener('resize', schedulePlacementBoundsRefresh);
    window.addEventListener('orientationchange', schedulePlacementBoundsRefresh);
    if (typeof ResizeObserver === 'function') {
      playerResizeObserver = new ResizeObserver(schedulePlacementBoundsRefresh);
      playerResizeObserver.observe(bar);
    }
  }

  function buildBar() {
    var bar = document.createElement('div');
    bar.className = 'tts-bar';
    bar.id = 'ttsBar';
    bar.hidden = true;
    bar.setAttribute('role', 'region');
    bar.setAttribute('aria-label', tr('Read aloud controls'));
    bar.innerHTML =
      '<button type="button" class="tts-btn" data-tts="prev" aria-label="' + escapeHtml(tr('Previous passage')) + '">⏪</button>' +
      '<button type="button" class="tts-btn tts-btn-main" data-tts="playpause" aria-label="' + escapeHtml(tr('Pause')) + '">⏸</button>' +
      '<button type="button" class="tts-btn" data-tts="next" aria-label="' + escapeHtml(tr('Next passage')) + '">⏩</button>' +
      '<span class="tts-status" id="ttsStatus" aria-live="polite">' + escapeHtml(tr('Reading')) + '</span>' +
      '<progress class="tts-progress" id="ttsProgress" max="1" value="0" aria-label="' + escapeHtml(tr('Narration progress')) + '"></progress>' +
      '<label class="tts-field"><span>' + escapeHtml(tr('Speed')) + '</span>' +
      '<select class="tts-select" id="ttsRate" aria-label="' + escapeHtml(tr('Reading speed')) + '">' +
      '<option value="0.75">0.75x</option><option value="1">1x</option>' +
      '<option value="1.25">1.25x</option><option value="1.5">1.5x</option>' +
      '<option value="1.75">1.75x</option><option value="2">2x</option></select></label>' +
      '<label class="tts-field tts-field-voice"><span>' + escapeHtml(tr('Voice')) + '</span>' +
      '<select class="tts-select" id="ttsVoice" aria-label="' + escapeHtml(tr('Voice')) + '"></select></label>' +
      '<button type="button" class="tts-btn tts-btn-reset" data-tts="reset" aria-label="' + escapeHtml(tr('Reset player position')) + '" hidden>' + escapeHtml(tr('Dock')) + '</button>' +
      '<button type="button" class="tts-btn tts-btn-stop" data-tts="stop" aria-label="' + escapeHtml(tr('Stop reading')) + '">' + escapeHtml(tr('Stop')) + '</button>' +
      '<button type="button" class="tts-btn tts-btn-collapse" data-tts="collapse" ' +
      'aria-label="' + escapeHtml(tr('Collapse controls')) + '" aria-expanded="true" title="' + escapeHtml(tr('Collapse (drag to move)')) + '">▾</button>';
    document.body.appendChild(bar);

    els.bar = bar;
    els.status = bar.querySelector('#ttsStatus');
    els.progress = bar.querySelector('#ttsProgress');
    els.playPause = bar.querySelector('[data-tts="playpause"]');
    els.rate = bar.querySelector('#ttsRate');
    els.voice = bar.querySelector('#ttsVoice');

    els.collapse = bar.querySelector('[data-tts="collapse"]');
    els.resetPosition = bar.querySelector('[data-tts="reset"]');

    bar.addEventListener('click', function (e) {
      // 结束拖动的 click 不应同时按下其下方的按钮。
      if (state.dragged) {
        state.dragged = false;
        return;
      }
      var target = e.target.closest('[data-tts]');
      if (!target) return;
      var action = target.getAttribute('data-tts');
      if (action === 'collapse') setCollapsed(!state.collapsed);
      else if (action === 'playpause') isPaused() ? resume() : pause();
      else if (action === 'stop') stop();
      else if (action === 'next') jump(1);
      else if (action === 'prev') jump(-1);
      else if (action === 'reset') resetPosition();
    });

    els.rate.value = String(rate());
    els.rate.addEventListener('change', function () {
      lsSet(RATE_KEY, els.rate.value);
      if (isPlaying()) {
        // rate 只对新 utterance 生效，因此需重启当前 chunk。
        cancelSpeech();
        speakCurrent();
      }
      render();
    });

    els.voice.addEventListener('change', function () {
      lsSet(voiceKey(), els.voice.value);
      // 显式选择会覆盖自动离线回退。
      state.forcedLocal = null;
      if (isPlaying()) {
        cancelSpeech();
        speakCurrent();
      }
    });

    bindDrag(bar);
    setCollapsed(lsGet(COLLAPSED_KEY) === '1', true);
    var pos = savedPosition();
    if (pos && !(window.matchMedia && window.matchMedia('(max-width: 720px)').matches)) {
      place(pos.x, pos.y, false);
    }

    fillVoices();
    if (typeof synth.onvoiceschanged !== 'undefined') {
      synth.addEventListener('voiceschanged', fillVoices);
    }
    return bar;
  }

  /**
   * 跟随文章内文本选区的“从此处朗读”chip。
   */
  function buildSelectionButton() {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'tts-from-here';
    btn.id = 'ttsFromHere';
    btn.hidden = true;
    btn.innerHTML = '<span aria-hidden="true">▶</span> ' + escapeHtml(tr('Read from here'));
    btn.title = tr('Read from here (Alt+R)');
    // mousedown 会在 click 到达前清除选区。
    btn.addEventListener('mousedown', function (e) {
      e.preventDefault();
    });
    btn.addEventListener('click', readFromSelection);
    document.body.appendChild(btn);
    els.fromHere = btn;
    return btn;
  }

  function refreshLanguage() {
    var wasPaused = isPaused();
    state.forcedLocal = null;
    state.stalls = 0;
    state.idleTicks = 0;
    if (!supported) {
      if (els.toggle) {
        els.toggle.setAttribute('aria-label', tr('Read aloud unavailable in this browser'));
        els.toggle.title = tr('Read aloud unavailable in this browser');
      }
      return;
    }
    if (isActive()) {
      refreshQueue(false);
      cancelSpeech();
      state.utterance = null;
      if (wasPaused) synth.resume();
      if (isPlaying()) {
        deferSpeak();
      }
    }
    if (els.bar) {
      var rateLabel = els.rate && els.rate.closest('label');
      var voiceLabel = els.voice && els.voice.closest('label');
      if (rateLabel && rateLabel.querySelector('span')) rateLabel.querySelector('span').textContent = tr('Speed');
      if (voiceLabel && voiceLabel.querySelector('span')) voiceLabel.querySelector('span').textContent = tr('Voice');
      els.bar.setAttribute('aria-label', tr('Read aloud controls'));
      var previous = els.bar.querySelector('[data-tts="prev"]');
      var next = els.bar.querySelector('[data-tts="next"]');
      var stopButton = els.bar.querySelector('[data-tts="stop"]');
      if (previous) previous.setAttribute('aria-label', tr('Previous passage'));
      if (next) next.setAttribute('aria-label', tr('Next passage'));
      if (els.progress) els.progress.setAttribute('aria-label', tr('Narration progress'));
      if (els.rate) els.rate.setAttribute('aria-label', tr('Reading speed'));
      if (els.voice) els.voice.setAttribute('aria-label', tr('Voice'));
      if (els.resetPosition) {
        els.resetPosition.setAttribute('aria-label', tr('Reset player position'));
        els.resetPosition.textContent = tr('Dock');
      }
      if (stopButton) {
        stopButton.setAttribute('aria-label', tr('Stop reading'));
        stopButton.textContent = tr('Stop');
      }
      fillVoices();
      setCollapsed(state.collapsed, true);
    }
    if (els.fromHere) {
      els.fromHere.innerHTML = '<span aria-hidden="true">▶</span> ' + escapeHtml(tr('Read from here'));
      els.fromHere.title = tr('Read from here (Alt+R)');
    }
    render();
  }

  function refreshContentLanguage() {
    state.forcedLocal = null;
    state.stalls = 0;
    state.idleTicks = 0;
    fillVoices();
    if (!isActive()) return render();
    var wasPaused = isPaused();
    refreshQueue(false);
    cancelSpeech();
    state.utterance = null;
    if (wasPaused) synth.resume();
    if (isPlaying()) deferSpeak();
    render();
  }

  function hideSelectionButton() {
    if (els.fromHere) els.fromHere.hidden = true;
  }

  function showSelectionButton() {
    if (!els.fromHere) return;
    // 仅在朗读运行时提供；控制条关闭时，应通过扬声器按钮进入。
    if (!isPlaying() && !isPaused()) {
      hideSelectionButton();
      return;
    }
    var sel = window.getSelection && window.getSelection();
    if (!selectedBlock()) {
      hideSelectionButton();
      return;
    }
    var rect = sel.getRangeAt(0).getBoundingClientRect();
    if (!rect || (!rect.width && !rect.height)) {
      hideSelectionButton();
      return;
    }
    els.fromHere.hidden = false;
    var top = rect.top + window.pageYOffset - els.fromHere.offsetHeight - 8;
    // 选区上方空间不足时翻转到其下方。
    if (rect.top < 60) top = rect.bottom + window.pageYOffset + 8;
    var left = rect.left + window.pageXOffset + rect.width / 2 - els.fromHere.offsetWidth / 2;
    var max = document.documentElement.clientWidth - els.fromHere.offsetWidth - 8;
    els.fromHere.style.top = Math.max(8, top) + 'px';
    els.fromHere.style.left = Math.min(Math.max(8, left), Math.max(8, max)) + 'px';
  }

  function bindSelection() {
    buildSelectionButton();
    var pending = null;
    var refresh = function () {
      clearTimeout(pending);
      pending = setTimeout(showSelectionButton, 10);
    };
    document.addEventListener('mouseup', refresh);
    document.addEventListener('keyup', function (e) {
      if (e.shiftKey || e.key === 'Shift' || /^Arrow/.test(e.key)) refresh();
    });
    document.addEventListener('selectionchange', function () {
      var sel = window.getSelection && window.getSelection();
      if (!sel || sel.isCollapsed) hideSelectionButton();
    });
    window.addEventListener('scroll', hideSelectionButton, { passive: true });
    window.addEventListener('resize', hideSelectionButton);
  }

  function resolveElement(target) {
    if (!target) return null;
    if (target.nodeType === 1) return target;
    if (typeof target !== 'string') return null;
    try {
      return document.querySelector(target);
    } catch (e) {
      return document.getElementById(target.replace(/^#/, ''));
    }
  }

  function startAt(target, options) {
    var element = resolveElement(target);
    if (!element) return false;
    var scope = options && options.scope ? resolveElement(options.scope) : null;
    if (!scope && options && options.section) scope = element.closest('[data-tts-section],article,section');
    closeCompactNavigation();
    return start(false, blockOf(element) || element, scope);
  }

  function bindSectionStarts() {
    document.addEventListener('click', function (event) {
      var control = event.target.closest && event.target.closest('[data-tts-start]');
      if (!control) return;
      event.preventDefault();
      var selector = control.getAttribute('data-tts-start');
      var target = selector ? resolveElement(selector) : control.closest('[data-tts-section],article,section');
      var scope = control.closest('[data-tts-section],article,section');
      if (target) start(false, blockOf(target) || target, scope);
    });
  }

  function stateSnapshot() {
    var current = state.chunks[state.index] || {};
    return {
      version: VERSION,
      supported: supported,
      silentMode: silentMode,
      mode: state.mode,
      index: state.index,
      total: state.chunks.length,
      section: current.section || '',
      locale: pageLocale(),
      remainingMinutes: remainingMinutes(),
      scoped: !!state.scope,
    };
  }

  window.AIFS_TTS = {
    version: VERSION,
    supported: supported,
    start: function () { return start(false, hashStartElement()); },
    startAt: startAt,
    pause: pause,
    resume: resume,
    stop: stop,
    refresh: function () { return refreshQueue(false); },
    refreshLanguage: refreshLanguage,
    getState: stateSnapshot,
  };

  function init() {
    if (document.getElementById('ttsToggle')) return;
    var btn = buildButton();
    if (!btn) return;
    els.toggle = btn;
    window.addEventListener('aifs:language-change', refreshLanguage);
    document.addEventListener('aifs:content-language-change', refreshContentLanguage);
    if (!supported) {
      document.dispatchEvent(new CustomEvent('aifs:tts-ready', { detail: stateSnapshot() }));
      return;
    }
    buildBar();
    bindReducedMotionPreference();
    bindSelection();
    bindSectionStarts();
    bindNavigationResume();
    render();

    // 遗留 utterance 会覆盖下一页继续朗读；跨导航延续的是 resume 标记，
    // 而不是音频本身。
    window.addEventListener('pagehide', function (event) {
      if (!state.navigationTarget) clearResumeTarget();
      cancelSpeech();
      if (!event.persisted) disposeReducedMotionPreference();
    });
    window.addEventListener('pageshow', function () {
      bindReducedMotionPreference();
      if (prefersReducedMotion()) commitDragInertiaForReducedMotion();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && !e.defaultPrevented && isActive()) stop();
      // chip 位于 tab 顺序末尾，因此为键盘用户提供快捷键：Alt+R 从选区起点朗读。
      if (e.altKey && !e.ctrlKey && !e.metaKey && (e.key === 'r' || e.key === 'R')) {
        if (selectedBlock() && readFromSelection()) e.preventDefault();
      }
    });

    autoResume();
    document.dispatchEvent(new CustomEvent('aifs:tts-ready', { detail: stateSnapshot() }));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
