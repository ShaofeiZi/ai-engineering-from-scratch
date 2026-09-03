/**
 * 命令面板：由 Cmd/Ctrl+K 或搜索按钮触发的全局搜索。
 *
 * 可搜索 focused path、lesson 标题、摘要、phase 名称、语言、类型和 glossary
 * 术语。中文搜索覆盖层会在面板第一次以中文打开时按需抓取一次。无外部依赖。
 *
 * API（挂载到 window.CmdPalette）：
 *   CmdPalette.open()   — 打开面板
 *   CmdPalette.close()  — 关闭面板
 *
 * 触发按钮：任意带有 [data-cmd-palette] 属性的元素。
 */
(function () {
  'use strict';

  // ── 常量 ─────────────────────────────────────────────────────────────
  var PALETTE_ID  = 'cmdPalette';
  var MAX_RESULTS = 12;
  var BODY_ATTR   = 'data-palette-open';
  var LANGUAGE_EVENT = 'aifs:language-change';
  var LANG_STORAGE_KEY = 'lang';

  // ── 模块状态 ─────────────────────────────────────────────────────────
  var _index      = null;   // 延迟构建的可搜索条目平铺数组
  var _indexLang  = '';
  var _activeIdx  = -1;
  var _isOpen     = false;
  var _prevFocus  = null;
  var _zhSearchAsset = null;
  var _zhSearchAssetPromise = null;

  function isCertificationSurface() {
    var pathname = typeof window !== 'undefined' && window.location
      ? String(window.location.pathname || '')
      : '';
    if (/(?:^|\/)(?:assessment|certification|certifications)(?:\.html)?\/?$/.test(pathname)) return true;
    if (!/(?:^|\/)lesson(?:\.html)?\/?$/.test(pathname) || typeof URLSearchParams === 'undefined') return false;
    try {
      return new URLSearchParams(window.location.search || '').get('path')
        .indexOf('certifications/claude/lessons/') === 0;
    } catch (_) {
      return false;
    }
  }

  function currentLang() {
    var api = (typeof window !== 'undefined') ? window.AIFS_I18n : null;
    var params;
    var value = '';
    if (isCertificationSurface()) return 'en';
    if (typeof window !== 'undefined' && window.AIFS_CHINESE_ONLY) return 'zh';
    if (api && typeof api.current === 'string' && api.current) return api.current;
    if (typeof window !== 'undefined' && typeof window.AIFS_currentLang === 'function') {
      value = window.AIFS_currentLang();
      if (value) return value;
    }
    if (typeof window !== 'undefined' && window.location && typeof URLSearchParams !== 'undefined') {
      try {
        params = new URLSearchParams(window.location.search || '');
        value = params.get('lang') || '';
      } catch (_) {
        value = '';
      }
    }
    if (!value && typeof window !== 'undefined' && window.localStorage) {
      try { value = window.localStorage.getItem(LANG_STORAGE_KEY) || ''; } catch (_) {}
    }
    return value || 'en';
  }

  function i18nApi() {
    return (typeof window !== 'undefined' && window.AIFS_I18n) ? window.AIFS_I18n : null;
  }

  function t(text, params) {
    var api = i18nApi();
    if (api && typeof api.t === 'function') {
      return api.t(text == null ? '' : String(text), params || null);
    }
    return interpolate(text == null ? '' : String(text), params || null);
  }

  function interpolate(text, params) {
    if (!params || typeof params !== 'object') return text;
    return String(text).replace(/\{([A-Za-z0-9_]+)\}/g, function (_, name) {
      return Object.prototype.hasOwnProperty.call(params, name) ? params[name] : _;
    });
  }

  function stableArtifactId(artifact, index) {
    if (artifact && artifact.id) return String(artifact.id);
    if (artifact && artifact.file) return String(artifact.file);
    if (artifact && artifact.lessonPath && artifact.name) {
      return artifact.lessonPath + '::' + artifact.name;
    }
    return 'artifact:' + String(index);
  }

  function stableGlossaryId(entry, index) {
    if (entry && entry.slug) return String(entry.slug);
    if (entry && entry.term) return String(entry.term);
    return 'glossary:' + String(index);
  }

  function appendUnique(parts, value) {
    var i;
    if (!value) return;
    value = String(value);
    for (i = 0; i < parts.length; i++) {
      if (parts[i] === value) return;
    }
    parts.push(value);
  }

  function combinedText() {
    var parts = [];
    var i;
    for (i = 0; i < arguments.length; i++) appendUnique(parts, arguments[i]);
    return parts.join(' ');
  }

  function entryString(entry, keys) {
    var i;
    if (!entry || typeof entry !== 'object') return '';
    for (i = 0; i < keys.length; i++) {
      if (typeof entry[keys[i]] === 'string' && entry[keys[i]]) return entry[keys[i]];
    }
    return '';
  }

  function normalizeSearchPayload(payload) {
    var normalized = { lessons: {}, artifacts: {}, glossary: {} };
    var key;
    if (!payload || typeof payload !== 'object') return normalized;
    if (payload.payload && typeof payload.payload === 'object') payload = payload.payload;

    if (payload.lessons && typeof payload.lessons === 'object') {
      for (key in payload.lessons) {
        if (Object.prototype.hasOwnProperty.call(payload.lessons, key)) {
          normalized.lessons[key] = payload.lessons[key];
        }
      }
    }
    if (payload.artifacts && typeof payload.artifacts === 'object') {
      for (key in payload.artifacts) {
        if (Object.prototype.hasOwnProperty.call(payload.artifacts, key)) {
          normalized.artifacts[key] = payload.artifacts[key];
        }
      }
    }
    if (payload.glossary && typeof payload.glossary === 'object') {
      for (key in payload.glossary) {
        if (Object.prototype.hasOwnProperty.call(payload.glossary, key)) {
          normalized.glossary[key] = payload.glossary[key];
        }
      }
    }
    return normalized;
  }

  function mergeSearchPayload(target, source) {
    var key;
    if (!source || typeof source !== 'object') return target;
    source = normalizeSearchPayload(source);
    for (key in source.lessons) {
      if (Object.prototype.hasOwnProperty.call(source.lessons, key)) target.lessons[key] = source.lessons[key];
    }
    for (key in source.artifacts) {
      if (Object.prototype.hasOwnProperty.call(source.artifacts, key)) target.artifacts[key] = source.artifacts[key];
    }
    for (key in source.glossary) {
      if (Object.prototype.hasOwnProperty.call(source.glossary, key)) target.glossary[key] = source.glossary[key];
    }
    return target;
  }

  function readJsonAsset(url) {
    if (typeof fetch !== 'function' || !url) {
      return Promise.reject(new Error('fetch unavailable'));
    }
    return fetch(url, { credentials: 'same-origin' }).then(function (response) {
      if (!response.ok) throw new Error('search asset request failed');
      return response.json();
    });
  }

  function resolveSearchAssetDescriptor(descriptor) {
    if (!descriptor) return Promise.reject(new Error('missing search asset descriptor'));
    if (typeof descriptor === 'function') {
      try {
        return Promise.resolve(descriptor()).then(resolveSearchAssetDescriptor);
      } catch (error) {
        return Promise.reject(error);
      }
    }
    if (typeof descriptor.then === 'function') {
      return Promise.resolve(descriptor).then(resolveSearchAssetDescriptor);
    }
    if (typeof descriptor === 'string') {
      return readJsonAsset(descriptor);
    }
    if (descriptor.payload || descriptor.data || descriptor.lessons || descriptor.artifacts || descriptor.glossary) {
      return Promise.resolve(descriptor.payload || descriptor.data || descriptor);
    }
    if (descriptor.src || descriptor.url || descriptor.href) {
      return readJsonAsset(descriptor.src || descriptor.url || descriptor.href);
    }
    if (Array.isArray(descriptor.files) || Array.isArray(descriptor.urls) || Array.isArray(descriptor.chunks)) {
      var files = descriptor.files || descriptor.urls || descriptor.chunks || [];
      return Promise.all(files.map(function (file) {
        if (typeof file === 'string') return readJsonAsset(file);
        return resolveSearchAssetDescriptor(file);
      })).then(function (parts) {
        var merged = { lessons: {}, artifacts: {}, glossary: {} };
        for (var i = 0; i < parts.length; i++) mergeSearchPayload(merged, parts[i]);
        return merged;
      });
    }
    return Promise.reject(new Error('unsupported search asset descriptor'));
  }

  function searchAssetDescriptor() {
    var source = (typeof window !== 'undefined') ? window.AIFS_I18N : null;
    if (!source || !source.zh || !source.zh.assets) return null;
    return source.zh.assets.search || null;
  }

  function ensureZhSearchAsset() {
    if (currentLang() !== 'zh') return Promise.resolve(null);
    if (_zhSearchAsset) return Promise.resolve(_zhSearchAsset);
    if (_zhSearchAssetPromise) return _zhSearchAssetPromise;

    _zhSearchAssetPromise = resolveSearchAssetDescriptor(searchAssetDescriptor())
      .then(function (payload) {
        _zhSearchAsset = normalizeSearchPayload(payload);
        return _zhSearchAsset;
      })
      .catch(function () {
        return null;
      })
      .then(function (payload) {
        _zhSearchAssetPromise = null;
        if (payload && currentLang() === 'zh') {
          rebuildIndex();
          refreshOpenPalette();
        }
        return payload;
      });

    return _zhSearchAssetPromise;
  }

  function translationForLesson(path) {
    if (!_zhSearchAsset || !_zhSearchAsset.lessons) return null;
    return _zhSearchAsset.lessons[path] || null;
  }

  function translationForArtifact(id) {
    if (!_zhSearchAsset || !_zhSearchAsset.artifacts) return null;
    return _zhSearchAsset.artifacts[id] || null;
  }

  function translationForGlossary(id) {
    if (!_zhSearchAsset || !_zhSearchAsset.glossary) return null;
    return _zhSearchAsset.glossary[id] || null;
  }

  function lessonLocalizedSummary(path, fallback) {
    var entry = translationForLesson(path);
    return entryString(entry, ['translation', 'summary', 'description', 'text']) || fallback || '';
  }

  function artifactLocalizedName(entry, fallback) {
    return entryString(entry, ['name', 'title', 'translation', 'label']) || fallback || '';
  }

  function artifactLocalizedSummary(entry, fallback) {
    return entryString(entry, ['summary', 'description', 'translation', 'text']) || fallback || '';
  }

  function glossaryLocalizedName(entry, fallback) {
    return entryString(entry, ['term', 'name', 'title', 'translation', 'label']) || fallback || '';
  }

  function glossaryLocalizedSummary(entry) {
    return entryString(entry, ['summary', 'description', 'translation', 'text']);
  }

  function makeDisplayItem(item, localized) {
    item.displayName = localized.name || item.name || '';
    item.displaySummary = localized.summary || item.summary || '';
    item.displayKeywords = localized.keywords || item.keywords || '';
    item.displayPhaseName = localized.phaseName || item.phaseName || '';
    item.displayType = localized.type || item.type || '';
    item.displayLang = localized.lang || item.lang || '';
    item.displayLevel = localized.level || item.level || '';
    item.displayArtKind = localized.artKind || item.artKind || '';
    item.displayExamCode = localized.examCode || item.examCode || '';
    item.searchName = combinedText(item.name, localized.name);
    item.searchSummary = combinedText(item.summary, localized.summary);
    item.searchKeywords = combinedText(item.keywords, localized.keywords);
    item.searchPhase = combinedText(item.phaseName, localized.phaseName);
    item.searchLang = combinedText(item.lang, localized.lang);
    item.searchType = combinedText(item.type, localized.type);
    item.searchSays = combinedText(item.says, localized.says);
    return item;
  }

  function learningPathEntryPath(entry) {
    return typeof entry === 'string' ? entry : entry && entry.path ? entry.path : '';
  }

  function learningPathDestination(lessonPath, learningPathId) {
    if (!lessonPath || !learningPathId) return '';
    return 'lesson?path=' + encodeURIComponent(lessonPath) +
      '&learningPath=' + encodeURIComponent(learningPathId);
  }

  function resultIndexForEnter(activeIndex, resultCount) {
    if (activeIndex >= 0 && activeIndex < resultCount) return activeIndex;
    return resultCount > 0 ? 0 : -1;
  }

  function navigationDestination(href, routeLinks) {
    var routes = routeLinks || (typeof window !== 'undefined' ? window.AIFSRouteLinks : null);
    return routes && typeof routes.adaptHref === 'function' ? routes.adaptHref(href) : href;
  }

  // ── 搜索索引 ─────────────────────────────────────────────────────────
  function certificationData() {
    var data = null;
    if (typeof CLAUDE_CERTIFICATION_DATA !== 'undefined' && CLAUDE_CERTIFICATION_DATA) {
      data = CLAUDE_CERTIFICATION_DATA;
    } else if (typeof CERTIFICATIONS !== 'undefined' && CERTIFICATIONS) {
      data = CERTIFICATIONS;
    }

    var tracks = null;
    if (typeof CERTIFICATION_TRACKS !== 'undefined' && CERTIFICATION_TRACKS) {
      tracks = Array.isArray(CERTIFICATION_TRACKS)
        ? CERTIFICATION_TRACKS
        : CERTIFICATION_TRACKS.tracks;
    }

    if (!data && tracks) data = { tracks: tracks };
    else if (data && !Array.isArray(data.tracks) && tracks) {
      data = Object.assign({}, data, { tracks: tracks });
    }
    return data;
  }

  /**
   * 基于 window.PHASES 和 window.GLOSSARY 构建一次平铺搜索索引。
   * 该过程是幂等的：后续调用直接返回缓存数组。
   */
  function buildIndex() {
    var lang = currentLang();
    var isZh = lang === 'zh';
    if (_index !== null && _indexLang === lang) return _index;
    _index = [];
    _indexLang = lang;

    if (typeof LEARNING_PATHS !== 'undefined' && Array.isArray(LEARNING_PATHS)) {
      for (var lp = 0; lp < LEARNING_PATHS.length; lp++) {
        var learningPath = LEARNING_PATHS[lp] || {};
        var route = Array.isArray(learningPath.lessons) ? learningPath.lessons : [];
        var firstLessonPath = route.length ? learningPathEntryPath(route[0]) : '';
        var learningPathId = learningPath.id || String(lp);
        if (!firstLessonPath) continue;
        var checkpointKeywords = Array.isArray(learningPath.checkpoints)
          ? learningPath.checkpoints.map(function (checkpoint) {
              return typeof checkpoint === 'string'
                ? checkpoint
                : checkpoint && (checkpoint.title || checkpoint.name || checkpoint.goal) || '';
            }).join(' ')
          : '';
        _index.push({
          kind:        'learning-path',
          id:          'lp:' + learningPathId,
          name:        learningPath.title || learningPathId,
          summary:     learningPath.summary || '',
          keywords:    [learningPath.keywords || '', checkpointKeywords, 'focused course route'].filter(Boolean).join(' '),
          lessonCount: route.length,
          minutes:     Number(learningPath.estimatedMinutes || 0),
          url:         learningPathDestination(firstLessonPath, learningPathId),
        });
        if (isZh) {
          makeDisplayItem(_index[_index.length - 1], {
            name: t(learningPath.title || learningPathId),
            summary: t(learningPath.summary || ''),
            keywords: t([learningPath.keywords || '', checkpointKeywords].filter(Boolean).join(' ')),
          });
        } else {
          makeDisplayItem(_index[_index.length - 1], {});
        }
      }
    }

    if (typeof PHASES !== 'undefined' && Array.isArray(PHASES)) {
      for (var i = 0; i < PHASES.length; i++) {
        var phase = PHASES[i];
        for (var j = 0; j < phase.lessons.length; j++) {
          var lesson = phase.lessons[j];

          // 提取供 lesson?path= 使用的 phases/…/… 路径
          var lessonPath = '';
          if (lesson.url) {
            var m = lesson.url.match(/(phases\/[^/?#]+\/[^/?#]+)/);
            if (m) lessonPath = m[1];
          }

          _index.push({
            kind:       'lesson',
            id:         'l:' + i + ':' + j,
            phaseId:    phase.id,
            phaseName:  phase.name,
            name:       lesson.name     || '',
            summary:    lesson.summary  || '',
            keywords:   lesson.keywords || '',
            type:       lesson.type     || '',
            lang:       lesson.lang     || '',
            status:     lesson.status   || '',
            lessonPath: lessonPath,
            url:        lesson.url      || '',
          });
          if (isZh) {
            makeDisplayItem(_index[_index.length - 1], {
              name: t(lesson.name || ''),
              summary: lessonLocalizedSummary(lessonPath, t(lesson.summary || '')),
              phaseName: t(phase.name || ''),
              type: t(lesson.type || ''),
              lang: t(lesson.lang || ''),
              keywords: t(lesson.keywords || ''),
            });
          } else {
            makeDisplayItem(_index[_index.length - 1], {});
          }
        }
      }
    }

    if (typeof GLOSSARY !== 'undefined' && Array.isArray(GLOSSARY)) {
      for (var k = 0; k < GLOSSARY.length; k++) {
        var g = GLOSSARY[k];
        _index.push({
          kind:    'glossary',
          id:      'g:' + k,
          name:    g.term  || '',
          summary: g.means || '',
          says:    g.says  || '',
          slug:    g.slug  || '',
          keywords: [
            g.category,
            g.whyItMatters,
            g.example,
            g.confusion,
            g.whyCalled,
            Array.isArray(g.aliases) ? g.aliases.join(' ') : '',
            Array.isArray(g.related) ? g.related.join(' ') : '',
          ].filter(Boolean).join(' '),
        });
        if (isZh) {
          var glossaryEntry = translationForGlossary(stableGlossaryId(g, k));
          makeDisplayItem(_index[_index.length - 1], {
            name: glossaryLocalizedName(glossaryEntry, t(g.term || '')),
            summary: glossaryLocalizedSummary(glossaryEntry) || g.means || '',
            says: entryString(glossaryEntry, ['says', 'aliases', 'keywords']),
            keywords: combinedText(
              t(g.category || ''),
              t(g.whyItMatters || ''),
              t(g.example || ''),
              t(g.confusion || ''),
              t(g.whyCalled || ''),
              entryString(glossaryEntry, ['keywords'])
            )
          });
        } else {
          makeDisplayItem(_index[_index.length - 1], {});
        }
      }
    }

    if (typeof ARTIFACTS !== 'undefined' && Array.isArray(ARTIFACTS)) {
      for (var a = 0; a < ARTIFACTS.length; a++) {
        var art = ARTIFACTS[a];
        var artifactId = stableArtifactId(art, a);
        _index.push({
          kind:       'artifact',
          id:         'a:' + artifactId,
          artKind:    art.kind || 'artifact',
          name:       art.name || '',
          summary:    art.description || '',
          keywords:   Array.isArray(art.tags) ? art.tags.join(' ') : '',
          phaseId:    art.phase,
          lesson:     art.lesson,
          lessonPath: art.lessonPath || '',
          file:       art.file || '',
        });
        if (isZh) {
          var artifactEntry = translationForArtifact(artifactId);
          makeDisplayItem(_index[_index.length - 1], {
            name: t(art.name || ''),
            summary: artifactLocalizedSummary(artifactEntry, t(art.description || '')),
            keywords: combinedText(t(Array.isArray(art.tags) ? art.tags.join(' ') : ''), entryString(artifactEntry, ['keywords'])),
            artKind: t(art.kind || 'artifact'),
          });
        } else {
          makeDisplayItem(_index[_index.length - 1], {});
        }
      }
    }

    // Certification 数据是可选的。只有当前页面已经加载了受支持的全局变量时
    // 才把它编入索引，不要仅为了搜索而额外抓取那份较大的数据包。
    var certs = certificationData();
    if (certs) {
      var tracks = Array.isArray(certs.tracks) ? certs.tracks : [];
      for (var trackIndex = 0; trackIndex < tracks.length; trackIndex++) {
        var track = tracks[trackIndex] || {};
        var trackId = track.id || track.slug || track.examCode || String(trackIndex);
        var domainNames = Array.isArray(track.domains)
          ? track.domains.map(function (domain) { return domain.name || domain.id || ''; }).join(' ')
          : '';
        _index.push({
          kind:     'certification-track',
          id:       'ct:' + trackId,
          name:     track.credential || track.name || track.shortName || track.examCode || 'Certification track',
          summary:  track.summary || track.audience || '',
          keywords: [track.shortName, track.examCode, track.level, track.audience, domainNames].filter(Boolean).join(' '),
          examCode: track.examCode || '',
          level:    track.level || '',
          url:      'certification?id=' + encodeURIComponent(trackId),
        });
        makeDisplayItem(_index[_index.length - 1], {});
      }

      var lessonMap = certs.lessonsByPath || {};
      var lessonList = Array.isArray(certs.lessons) ? certs.lessons : [];
      var certLessons = Object.keys(lessonMap).map(function (path) {
        var lesson = lessonMap[path] || {};
        return Object.assign({ path: path }, lesson);
      }).concat(lessonList);
      var seenCertLessons = {};

      for (var c = 0; c < certLessons.length; c++) {
        var certLesson = certLessons[c] || {};
        var certPath = certLesson.path || certLesson.lessonPath || '';
        if (!certPath || seenCertLessons[certPath]) continue;
        seenCertLessons[certPath] = true;
        _index.push({
          kind:       'certification-lesson',
          id:         'cl:' + certPath,
          name:       certLesson.name || certLesson.title || certLesson.slug || 'Certification lesson',
          summary:    certLesson.summary || '',
          keywords:   certLesson.keywords || '',
          type:       certLesson.type || '',
          lang:       certLesson.languages || certLesson.lang || '',
          lessonPath: certPath,
        });
        makeDisplayItem(_index[_index.length - 1], {});
      }
    }

    return _index;
  }

  function rebuildIndex() {
    _index = null;
    return buildIndex();
  }

  function refreshOpenPalette() {
    if (!_isOpen) return;
    var input = _inputEl();
    var query = input ? input.value.trim() : '';
    renderResults(query ? search(query) : []);
  }

  // ── 评分 ─────────────────────────────────────────────────────────────
  function scoreItem(item, q) {
    // q 在调用方里已经完成小写化和 trim
    var name     = (item.searchName || item.name || '').toLowerCase();
    var summary  = (item.searchSummary  || item.summary  || '').toLowerCase();
    var keywords = (item.searchKeywords || item.keywords || '').toLowerCase();
    var phase    = (item.searchPhase || item.phaseName || '').toLowerCase();
    var lang     = (item.searchLang  || item.lang  || '').toLowerCase();
    var type     = (item.searchType  || item.type  || '').toLowerCase();
    var says     = (item.searchSays  || item.says  || '').toLowerCase();

    var s = 0;

    // 全名精确匹配：最高优先级
    if (name === q) return 200;

    // 名称中的子串匹配（最重要的信号）
    if (name.startsWith(q))          s += 100;
    else if (name.indexOf(q) !== -1) s +=  70;
    if (item.kind === 'learning-path' && name.startsWith(q)) s += 100;

    // 多词查询：每个词都必须出现在名称中
    var words = q.split(/\s+/).filter(Boolean);
    if (words.length > 1) {
      var allInName = words.every(function (w) { return name.indexOf(w) !== -1; });
      if (allInName) {
        s += (s === 0 ? 65 : 20);
      } else {
        // 较弱信号：每个词分别散落在 name、summary、keywords、phase 中
        var blob = name + ' ' + summary + ' ' + keywords + ' ' + phase;
        var allInBlob = words.every(function (w) { return blob.indexOf(w) !== -1; });
        if (allInBlob) s += 15;
      }
    }

    // 辅助字段，按预期相关度排序
    if (summary.indexOf(q)  !== -1) s += 25;
    if (keywords.indexOf(q) !== -1) s += 22; // H3 headings: dense vocabulary
    if (says.indexOf(q)     !== -1) s += 22; // glossary "what people say"
    if (phase.indexOf(q)    !== -1) s += 18;
    if (lang.indexOf(q)     !== -1) s += 14;
    if (type.indexOf(q)     !== -1) s += 10;

    // 单词查询的回退策略：在名称 token 上做词边界前缀匹配
    if (s === 0 && words.length === 1) {
      var nameParts = name.split(/[\s\-–—:,]+/).filter(Boolean);
      for (var i = 0; i < nameParts.length; i++) {
        if (nameParts[i].startsWith(q)) { s += 30; break; }
      }
      // 最后兜底：单词只要出现在 keywords 或 summary 任意位置即可
      if (s === 0 && keywords.indexOf(q) !== -1) s += 18;
      if (s === 0 && summary.indexOf(q)  !== -1) s += 12;
    }

    return s;
  }

  function search(query) {
    var q = query.trim().toLowerCase();
    if (!q) return [];

    var items   = buildIndex();
    var results = [];

    for (var i = 0; i < items.length; i++) {
      var s = scoreItem(items[i], q);
      if (s > 0) results.push({ item: items[i], s: s });
    }

    results.sort(function (a, b) { return b.s - a.s; });
    return results.slice(0, MAX_RESULTS).map(function (r) { return r.item; });
  }

  // ── 工具函数 ─────────────────────────────────────────────────────────
  function escHtml(str) {
    var d = document.createElement('div');
    d.textContent = (str == null) ? '' : String(str);
    return d.innerHTML;
  }

  /**
   * 在 `text` 中高亮 `query` 的首次出现位置（或第一个匹配到的查询词）。
   * 返回的是 HTML 安全字符串，并在命中处包裹 <mark>。
   */
  function highlight(text, query) {
    if (!text) return '';
    if (!query) return escHtml(text);

    var lower = text.toLowerCase();
    var q     = query.trim().toLowerCase();
    var idx   = lower.indexOf(q);
    var matchLen = q.length;

    if (idx === -1) {
      // 逐个尝试每个单词
      var words = q.split(/\s+/).filter(Boolean);
      for (var i = 0; i < words.length; i++) {
        idx = lower.indexOf(words[i]);
        if (idx !== -1) { matchLen = words[i].length; break; }
      }
    }

    if (idx === -1) return escHtml(text);

    return (
      escHtml(text.slice(0, idx)) +
      '<mark>' + escHtml(text.slice(idx, idx + matchLen)) + '</mark>' +
      escHtml(text.slice(idx + matchLen))
    );
  }

  function truncate(str, max) {
    if (!str || str.length <= max) return str || '';
    var cut = str.slice(0, max).replace(/\s+\S*$/, '');
    return (cut.length > max * 0.6 ? cut : str.slice(0, max)) + '…';
  }

  // ── 面板 DOM（首次打开时延迟创建） ───────────────────────────────────
  function createPaletteDOM() {
    if (document.getElementById(PALETTE_ID)) return;

    // 识别平台，用于页脚快捷键提示
    var isMac = /Mac|iPhone|iPod|iPad/.test(
      (navigator.userAgentData && navigator.userAgentData.platform) ||
      navigator.platform || ''
    );
    var shortcutLabel = isMac ? '⌘K' : 'Ctrl+K';
    var paletteLabel = currentLang() === 'zh' ? '搜索学习路径、课程与术语表' : 'Search learning paths, lessons, and glossary';
    var searchPlaceholder = currentLang() === 'zh' ? '搜索学习路径、课程与术语表…' : 'Search paths, lessons, and glossary…';
    var searchLabel = currentLang() === 'zh' ? '搜索' : 'Search';
    var closeSearchLabel = currentLang() === 'zh' ? '关闭搜索' : 'Close search';
    var resultsLabel = currentLang() === 'zh' ? '搜索结果' : 'Search results';
    var navigateLabel = currentLang() === 'zh' ? '切换' : 'navigate';
    var openLabel = currentLang() === 'zh' ? '打开' : 'open';
    var closeLabel = currentLang() === 'zh' ? '关闭' : 'close';

    var el = document.createElement('div');
    el.id = PALETTE_ID;
    el.setAttribute('role', 'dialog');
    el.setAttribute('aria-modal', 'true');
    el.setAttribute('aria-label', paletteLabel);
    el.setAttribute('aria-hidden', 'true');
    el.inert = true;

    el.innerHTML =
      '<div class="cp-backdrop" id="cpBackdrop"></div>' +
      '<div class="cp-panel">' +
        '<div class="cp-search-row">' +
          '<svg class="cp-search-icon" width="16" height="16" viewBox="0 0 24 24"' +
          ' fill="none" stroke="currentColor" stroke-width="2.5"' +
          ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
            '<circle cx="11" cy="11" r="8"/>' +
            '<line x1="21" y1="21" x2="16.65" y2="16.65"/>' +
          '</svg>' +
          '<input class="cp-input" id="cpInput" type="search"' +
          ' placeholder="' + escHtml(searchPlaceholder) + '"' +
          ' autocomplete="off" autocorrect="off"' +
          ' autocapitalize="off" spellcheck="false"' +
          ' role="combobox" aria-label="' + escHtml(searchLabel) + '" aria-autocomplete="list"' +
          ' aria-haspopup="listbox" aria-expanded="false"' +
          ' aria-controls="cpResults">' +
          '<button class="cp-kbd-esc" id="cpClose" type="button"' +
          ' aria-label="' + escHtml(closeSearchLabel) + '">Esc</button>' +
        '</div>' +
        '<ul class="cp-results" id="cpResults"' +
        ' role="listbox" aria-label="' + escHtml(resultsLabel) + '"></ul>' +
        '<div class="cp-footer">' +
          '<span class="cp-footer-group">' +
            '<kbd>↑</kbd><kbd>↓</kbd>' +
            '<span class="cp-footer-label">' + escHtml(navigateLabel) + '</span>' +
          '</span>' +
          '<span class="cp-footer-group">' +
            '<kbd>↵</kbd>' +
            '<span class="cp-footer-label">' + escHtml(openLabel) + '</span>' +
          '</span>' +
          '<span class="cp-footer-group">' +
            '<kbd>Esc</kbd>' +
            '<span class="cp-footer-label">' + escHtml(closeLabel) + '</span>' +
          '</span>' +
          '<span class="cp-footer-shortcut">' + shortcutLabel + '</span>' +
        '</div>' +
      '</div>';

    document.body.appendChild(el);

    // 绑定内部交互
    document.getElementById('cpBackdrop').addEventListener('click', close);
    document.getElementById('cpClose').addEventListener('click', close);
    el.addEventListener('keydown', _onDialogKeyDown);

    var inp = document.getElementById('cpInput');
    inp.addEventListener('input', _onInput);
    inp.addEventListener('keydown', _onKeyDown);
  }

  function _palEl()   { return document.getElementById(PALETTE_ID); }
  function _inputEl() { return document.getElementById('cpInput'); }
  function _listEl()  { return document.getElementById('cpResults'); }

  function _clearActiveDescendant() {
    var input = _inputEl();
    if (input) input.removeAttribute('aria-activedescendant');
  }

  // ── 打开 / 关闭 ──────────────────────────────────────────────────────
  function open() {
    if (_isOpen) {
      // 已经打开时，只需确保输入框获得焦点
      var inp = _inputEl();
      if (inp) inp.focus();
      return;
    }

    _prevFocus = document.activeElement || null;
    _isOpen    = true;
    _activeIdx = -1;
    if (currentLang() === 'zh') ensureZhSearchAsset();

    createPaletteDOM();
    document.body.setAttribute(BODY_ATTR, '');

    var pal = _palEl();
    if (pal) {
      pal.inert = false;
      pal.setAttribute('aria-hidden', 'false');
      pal.classList.add('cp-open');
    }

    var input = _inputEl();
    if (input) {
      input.setAttribute('aria-expanded', 'true');
      _clearActiveDescendant();
      input.focus();
      var q = input.value.trim();
      renderResults(q ? search(q) : []);
    }
  }

  function close() {
    if (!_isOpen) return;
    _isOpen    = false;
    _activeIdx = -1;

    var pal = _palEl();
    if (pal) {
      pal.classList.remove('cp-open');
      pal.setAttribute('aria-hidden', 'true');
      pal.inert = true;
    }
    var input = _inputEl();
    if (input) input.setAttribute('aria-expanded', 'false');
    _clearActiveDescendant();
    document.body.removeAttribute(BODY_ATTR);

    // 焦点还给用户打开面板前所在的位置
    try {
      if (_prevFocus && typeof _prevFocus.focus === 'function') {
        _prevFocus.focus();
      }
    } catch (_) { /* 元素可能已经从 DOM 中移除 */ }
    _prevFocus = null;
  }

  // ── 渲染结果 ─────────────────────────────────────────────────────────
  function renderResults(results) {
    var list = _listEl();
    if (!list) return;

    var query = (_inputEl() ? _inputEl().value : '').trim();

    if (!query) {
      var inventory = buildIndex();
      var lessonCount = inventory.filter(function (item) { return item.kind === 'lesson'; }).length;
      var certificationLessonCount = inventory.filter(function (item) { return item.kind === 'certification-lesson'; }).length;
      var learningPathCount = inventory.filter(function (item) { return item.kind === 'learning-path'; }).length;
      var artifactCount = inventory.filter(function (item) { return item.kind === 'artifact'; }).length;
      var glossaryCount = inventory.filter(function (item) { return item.kind === 'glossary'; }).length;
      var inventoryParts = [lessonCount + ' lessons'];
      if (learningPathCount) {
        inventoryParts.push(learningPathCount + ' focused learning ' + (learningPathCount === 1 ? 'path' : 'paths'));
      }
      if (certificationLessonCount) {
        inventoryParts.push(certificationLessonCount + ' certification lessons');
      }
      inventoryParts.push(artifactCount + ' outputs');
      inventoryParts.push(glossaryCount + ' glossary terms');
      var localizedInventory = inventoryParts.map(function (part) { return t(part); });
      var inventoryMessage = t('Search {items}, and {last}', {
        items: localizedInventory.slice(0, -1).join('、'),
        last: localizedInventory[localizedInventory.length - 1]
      });
      list.innerHTML =
        '<li class="cp-empty" role="option" aria-disabled="true">' +
        escHtml(inventoryMessage) +
        '</li>';
      _activeIdx = -1;
      _clearActiveDescendant();
      return;
    }

    if (results.length === 0) {
      var noResultsHtml = escHtml(t('No results for')) + ' <em>' + escHtml(query) + '</em>';
      list.innerHTML =
        '<li class="cp-empty" role="option" aria-disabled="true">' +
        noResultsHtml +
        '</li>';
      _activeIdx = -1;
      _clearActiveDescendant();
      return;
    }

    var html = '';
    for (var i = 0; i < results.length; i++) {
      var r    = results[i];
      var dest = '';
      var chip = '';
      var chipClass = 'cp-item-chip';

      if (r.kind === 'learning-path') {
        dest = r.url;
        chip = t('Learning path');
        chipClass += ' cp-item-chip--alt';
      } else if (r.kind === 'lesson') {
        // 优先跳到站内阅读器；否则退回到 GitHub URL
        dest = r.lessonPath
          ? 'lesson?path=' + encodeURIComponent(r.lessonPath)
          : r.url;
        chip = t('Phase ' + String(r.phaseId).padStart(2, '0'));
      } else if (r.kind === 'certification-lesson') {
        dest = 'lesson?path=' + encodeURIComponent(r.lessonPath);
        chip = 'Certification';
        chipClass += ' cp-item-chip--alt';
      } else if (r.kind === 'certification-track') {
        dest = r.url;
        chip = r.examCode || 'Certification';
        chipClass += ' cp-item-chip--alt';
      } else if (r.kind === 'artifact') {
        // 跳到产出该 artifact 的 lesson
        dest = r.lessonPath
          ? 'lesson?path=' + encodeURIComponent(r.lessonPath)
          : ('https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/' + r.file);
        var ak = (r.artKind || 'artifact');
        chip = t(ak.charAt(0).toUpperCase() + ak.slice(1));
        chipClass += ' cp-item-chip--alt';
      } else {
        // 优先使用标准术语锚点。旧版生成数据则回退到 exact-name 查询，
        // 直到下一次 site build 为止。
        dest      = r.slug
          ? 'glossary.html#' + encodeURIComponent(r.slug)
          : 'glossary.html?q=' + encodeURIComponent(r.displayName || r.name);
        chip      = t('Glossary');
        chipClass += ' cp-item-chip--alt';
      }

      var displayName = r.displayName || r.name || '';
      var displaySummary = r.displaySummary || r.summary || '';
      var snippet = displaySummary ? truncate(displaySummary, 110) : '';
      var metaParts = [];
      if (r.kind === 'learning-path') {
        if (r.lessonCount) metaParts.push(t('{count} lessons', { count: String(r.lessonCount) }));
        if (r.minutes) {
          var hours = Math.floor(r.minutes / 60);
          var minutes = r.minutes % 60;
          metaParts.push(t(((hours ? hours + 'h' : '') + (minutes ? ' ' + minutes + 'm' : '')).trim()));
        }
      } else if (r.kind === 'lesson' || r.kind === 'certification-lesson') {
        if (r.displayType && r.displayType !== '—') metaParts.push(r.displayType);
        if (r.displayLang && r.displayLang !== '—') metaParts.push(r.displayLang);
      } else if (r.kind === 'certification-track') {
        if (r.displayLevel) metaParts.push(r.displayLevel);
      } else if (r.kind === 'artifact') {
        if (r.phaseId !== undefined && r.phaseId !== null) {
          metaParts.push(t('Phase ' + String(r.phaseId).padStart(2, '0')));
        }
      }
      var meta = metaParts.join(' · '); // ·

      html +=
        '<li class="cp-item" id="cpOption-' + i + '" role="option" aria-selected="false"' +
        ' data-idx="' + i + '"' +
        ' data-href="' + escHtml(dest) + '">' +
          '<div class="cp-item-body">' +
            '<span class="' + chipClass + '">' + escHtml(chip) + '</span>' +
            '<span class="cp-item-name">'    + highlight(displayName, query) + '</span>' +
            (snippet ? '<span class="cp-item-summary">' + highlight(snippet, query) + '</span>' : '') +
            (meta    ? '<span class="cp-item-meta">'    + escHtml(meta)             + '</span>' : '') +
          '</div>' +
          '<svg class="cp-item-arrow" width="12" height="12" viewBox="0 0 24 24"' +
          ' fill="none" stroke="currentColor" stroke-width="2"' +
          ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
            '<polyline points="9 18 15 12 9 6"/>' +
          '</svg>' +
        '</li>';
    }

    list.innerHTML = html;
    _activeIdx = -1;
    _clearActiveDescendant();

    // 绑定交互处理器
    var items = list.querySelectorAll('.cp-item');
    for (var j = 0; j < items.length; j++) {
      items[j].addEventListener('click',     _onItemClick);
      items[j].addEventListener('mousemove', _onItemMouseMove);
    }
  }

  // ── 事件处理 ─────────────────────────────────────────────────────────
  function _onInput(e) {
    var query = e.target.value;
    renderResults(search(query));
    _activeIdx = -1;
  }

  function _onKeyDown(e) {
    var list  = _listEl();
    var items = list ? list.querySelectorAll('.cp-item') : [];
    var count = items.length;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!count) return;
        _activeIdx = (_activeIdx + 1) % count;
        _updateActive(items);
        break;

      case 'ArrowUp':
        e.preventDefault();
        if (!count) return;
        _activeIdx = (_activeIdx - 1 + count) % count;
        _updateActive(items);
        break;

      case 'Enter': {
        e.preventDefault();
        var targetIndex = resultIndexForEnter(_activeIdx, count);
        var target = targetIndex >= 0 ? items[targetIndex] : null;
        if (target) _navigate(target);
        break;
      }

    }
  }

  function _onDialogKeyDown(e) {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      close();
      return;
    }

    if (e.key !== 'Tab') return;
    var input = _inputEl();
    var closeButton = document.getElementById('cpClose');
    if (!input || !closeButton) return;

    if (e.shiftKey && document.activeElement === input) {
      e.preventDefault();
      closeButton.focus();
    } else if (!e.shiftKey && document.activeElement === closeButton) {
      e.preventDefault();
      input.focus();
    }
  }

  function _updateActive(items) {
    var input = _inputEl();
    var activeId = '';
    for (var i = 0; i < items.length; i++) {
      var active = (i === _activeIdx);
      items[i].classList.toggle('cp-item--active', active);
      items[i].setAttribute('aria-selected', active ? 'true' : 'false');
      if (active) {
        activeId = items[i].id;
        items[i].scrollIntoView({ block: 'nearest', behavior: 'instant' });
      }
    }
    if (input && activeId) input.setAttribute('aria-activedescendant', activeId);
    else _clearActiveDescendant();
  }

  function _onItemClick(e) {
    _navigate(e.currentTarget);
  }

  function _onItemMouseMove(e) {
    var list = _listEl();
    if (!list) return;
    var idx = parseInt(e.currentTarget.getAttribute('data-idx'), 10);
    if (idx !== _activeIdx) {
      _activeIdx = idx;
      _updateActive(list.querySelectorAll('.cp-item'));
    }
  }

  function _navigate(item) {
    var href = item.getAttribute('data-href');
    if (!href) return;
    close();
    window.location.href = navigationDestination(href);
  }

  // ── 全局键盘快捷键（Cmd/Ctrl+K） ────────────────────────────────────
  if (typeof document !== 'undefined') {
    document.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (_isOpen) {
          // 面板已经打开时，只重新聚焦输入框
          var inp = _inputEl();
          if (inp) inp.focus();
        } else {
          open();
        }
      }
    });
  }

  // ── 初始化：绑定触发按钮，并预先构建索引 ──────────────────────────────
  function _init() {
    // 任意带有 [data-cmd-palette] 的元素在点击时打开面板
    var triggers = document.querySelectorAll('[data-cmd-palette]');
    for (var i = 0; i < triggers.length; i++) {
      triggers[i].addEventListener('click', function (e) {
        e.preventDefault();
        open();
      });
    }

    // 现在就构建核心索引，确保用户第一次输入时立即响应。中文覆盖层仍保持
    // 延迟加载，直到用户真正打开面板。在 lesson 页面中，certification-data.js
    // 是按需加载的，可能仍在请求中；等它稳定后重新构建索引，避免过早生成的
    // 纯核心缓存永久隐藏 certification tracks 与 lessons。
    buildIndex();

    var certificationReady = window.__AIFS_CERTIFICATION_DATA_READY;
    if (certificationReady && typeof certificationReady.then === 'function') {
      certificationReady.then(function () {
        rebuildIndex();
        refreshOpenPalette();
      }).catch(function () {
        // 如果可选的 certification bundle 加载失败，也保留已经构建好的
        // 核心索引可用。
      });
    }

    window.addEventListener(LANGUAGE_EVENT, function () {
      if (currentLang() === 'zh' && _isOpen) {
        ensureZhSearchAsset().then(function () {
          rebuildIndex();
          refreshOpenPalette();
        });
      } else {
        rebuildIndex();
        refreshOpenPalette();
      }
    });
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', _init);
    } else {
      _init();
    }
  }

  // ── 对外 API ────────────────────────────────────────────────────────
  if (typeof window !== 'undefined') {
    window.CmdPalette = { open: open, close: close };
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      rebuildIndex: rebuildIndex,
      search: search,
      learningPathDestination: learningPathDestination,
      navigationDestination: navigationDestination,
      resultIndexForEnter: resultIndexForEnter,
    };
  }

}());
