const fs = require('fs');
const path = require('path');
const express = require('express');
const MarkdownIt = require('markdown-it');
const { Marp } = require('@marp-team/marp-core');

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
});

function parseArgs(argv) {
  const args = { port: 8000, dir: '.' };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--port' && argv[i + 1]) {
      args.port = Number(argv[i + 1]);
      i += 1;
    } else if (arg === '--dir' && argv[i + 1]) {
      args.dir = argv[i + 1];
      i += 1;
    }
  }
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function listTopics(baseDir) {
  return fs.readdirSync(baseDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const topicPath = path.join(baseDir, entry.name);
      const metadataPath = path.join(topicPath, 'metadata.json');
      const paperListPath = path.join(topicPath, 'paper_list.jsonl');
      if (!fs.existsSync(metadataPath) || !fs.existsSync(paperListPath)) return null;
      try {
        const metadata = readJson(metadataPath);
        return {
          id: entry.name,
          title: typeof metadata.title === 'string' && metadata.title.trim() ? metadata.title.trim() : entry.name,
          keyword: Array.isArray(metadata.keyword) ? metadata.keyword.map((item) => String(item)) : [],
          path: topicPath,
        };
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .sort((a, b) => a.title.localeCompare(b.title));
}

function getTopicMap(baseDir) {
  const map = new Map();
  for (const topic of listTopics(baseDir)) {
    map.set(topic.id, topic);
  }
  return map;
}

function loadPapers(topicPath) {
  const filePath = path.join(topicPath, 'paper_list.jsonl');
  const lines = fs.readFileSync(filePath, 'utf8').split(/\r?\n/).filter(Boolean);
  return lines.map((line, index) => {
    const obj = JSON.parse(line);
    return {
      index: index + 1,
      title: String(obj.title || '').trim(),
      author: String(obj.author || '').trim(),
      year: String(obj.year || '').trim(),
      url: String(obj.url || '').trim(),
      summary: String(obj.summary || '').trim(),
      slide: String(obj.slide || '').trim(),
    };
  });
}

function normalizePdfUrl(url) {
  if (!url) return '';
  if (url.includes('/pdf/')) return url;
  if (url.includes('arxiv.org/abs/')) return `${url.replace('/abs/', '/pdf/')}.pdf`;
  return url;
}

function ensureInside(baseDir, topicId, fileName) {
  const topicMap = getTopicMap(baseDir);
  const topic = topicMap.get(topicId);
  if (!topic) return null;
  const safeName = path.basename(fileName);
  if (safeName !== fileName) return null;
  const resolved = path.resolve(topic.path, safeName);
  const root = path.resolve(topic.path);
  if (!resolved.startsWith(root + path.sep) && resolved !== root) return null;
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) return null;
  return { topic, filePath: resolved };
}

const HTML_PAGE = `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Viewer</title>
  <link rel="icon" href="/favicon.ico" sizes="32x32">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;500;700&display=swap" rel="stylesheet">
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$']],
        displayMath: [['$$', '$$']]
      },
      svg: {
        fontCache: 'global'
      }
    };
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {
      --bg: #f3f1ea;
      --panel: #fffdf8;
      --line: #d6d0c4;
      --text: #1f1d18;
      --muted: #706b63;
      --accent: dodgerblue;
      --col1: 260px;
      --col2: 380px;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; }
    body {
      font-family: "Noto Serif KR", Georgia, "Times New Roman", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, #f7efe0, transparent 32%),
        radial-gradient(circle at bottom right, #e5efe8, transparent 28%),
        var(--bg);
    }
    .app {
      display: grid;
      grid-template-columns: minmax(180px, var(--col1)) 6px minmax(240px, var(--col2)) 6px minmax(320px, 1fr);
      height: 100vh;
    }
    .panel {
      min-width: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      background: color-mix(in srgb, var(--panel) 92%, white 8%);
      border-right: 1px solid var(--line);
    }
    .panel-header {
      padding: 16px 18px 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.62);
      backdrop-filter: blur(10px);
    }
    .eyebrow {
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .panel-title {
      font-size: 22px;
      margin: 0;
      line-height: 1.15;
    }
    .panel-subtitle {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .list {
      margin: 0;
      padding: 10px;
      list-style: none;
      overflow: auto;
      flex: 1;
    }
    .item {
      border: 1px solid transparent;
      border-radius: 14px;
      padding: 12px 14px;
      cursor: pointer;
      transition: background 0.18s ease, border-color 0.18s ease, transform 0.12s ease;
      user-select: none;
    }
    .item + .item { margin-top: 8px; }
    .item:hover {
      background: rgba(30, 110, 255, 0.05);
      border-color: rgba(30, 110, 255, 0.16);
      transform: translateY(-1px);
    }
    .item.active {
      background: rgba(30, 110, 255, 0.08);
      border-color: rgba(30, 110, 255, 0.28);
      box-shadow: 0 8px 24px rgba(25, 42, 70, 0.08);
    }
    .topic-keywords {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .paper-title {
      font-size: 15px;
      line-height: 1.4;
      margin: 0 0 8px;
      font-weight: 700;
    }
    .paper-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .paper-item { border-left: 4px solid transparent; }
    .paper-item.gray { color: #8b8b8b; border-left-color: #c2c2c2; }
    .paper-item.black { color: #171717; border-left-color: #303030; }
    .paper-item.blue { color: var(--accent); border-left-color: var(--accent); }
    .splitter {
      background: linear-gradient(to right, transparent, rgba(70, 65, 58, 0.14), transparent);
      cursor: col-resize;
      position: relative;
    }
    .splitter::after {
      content: "";
      position: absolute;
      inset: 0;
      margin: auto;
      width: 2px;
      height: 72px;
      border-radius: 999px;
      background: rgba(70, 65, 58, 0.18);
    }
    .viewer {
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
    }
    .viewer-header {
      padding: 16px 18px 8px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.68);
      backdrop-filter: blur(10px);
    }
    .viewer-paper-title {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
    }
    .viewer-paper-meta {
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    .tabbar {
      display: flex;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    .tabbar button, .action-link {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 8px 14px;
      font: inherit;
      cursor: pointer;
      color: var(--text);
      text-decoration: none;
    }
    .tabbar button.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    .action-link[aria-disabled="true"] {
      opacity: 0.45;
      pointer-events: none;
    }
    .viewer-body {
      flex: 1;
      min-height: 0;
      position: relative;
    }
    .tab-panel {
      display: none;
      height: 100%;
      overflow: auto;
      background: rgba(255, 255, 255, 0.7);
    }
    .tab-panel.active { display: block; }
    iframe, embed {
      border: 0;
      width: 100%;
      height: 100%;
      background: white;
    }
    .empty-state {
      padding: 28px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 15px;
    }
    .markdown-body {
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 34px 64px;
      line-height: 1.7;
      font-size: 17px;
    }
    .markdown-body h1, .markdown-body h2, .markdown-body h3 {
      line-height: 1.25;
      margin-top: 1.6em;
    }
    .markdown-body pre {
      overflow: auto;
      padding: 14px;
      border-radius: 12px;
      background: #f7f4ec;
      border: 1px solid #e2dccc;
    }
    .markdown-body code {
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 0.92em;
    }
    .slide-html {
      --slide-scale: 1;
      --slide-base-width: 1280px;
      width: 100%;
      height: 100%;
      overflow: auto;
      background: #fff;
    }
    .slide-html .marp-root {
      min-height: 100%;
      padding: 28px 0 72px;
      display: flex;
      flex-direction: column;
      gap: 42px;
      align-items: center;
      background:
        linear-gradient(180deg, rgba(247, 243, 232, 0.8), rgba(255, 255, 255, 0.95));
    }
    .slide-html svg[data-marpit-svg] {
      display: block;
      width: calc(var(--slide-base-width) * var(--slide-scale));
      height: auto;
      max-width: none;
      margin-block: 22px;
      box-shadow: 0 18px 48px rgba(40, 40, 40, 0.12);
      border-radius: 8px;
      background: #fff;
    }
    .slide-html mjx-container {
      font-size: 82% !important;
    }
    .slide-html svg mjx-container {
      font-size: 80% !important;
    }
    @media (max-width: 960px) {
      .app {
        grid-template-columns: 1fr;
        grid-template-rows: 28vh 30vh 1fr;
      }
      .splitter { display: none; }
      .panel { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="app" id="app">
    <section class="panel">
      <div class="panel-header">
        <div class="eyebrow">Research Topics</div>
        <h1 class="panel-title">Fields</h1>
        <div class="panel-subtitle" id="topic-count"></div>
      </div>
      <ul class="list" id="topic-list"></ul>
    </section>
    <div class="splitter" data-resize="left"></div>
    <section class="panel">
      <div class="panel-header">
        <div class="eyebrow">Paper Inventory</div>
        <h2 class="panel-title" id="paper-panel-title">Papers</h2>
        <div class="panel-subtitle" id="paper-count">Select a topic.</div>
      </div>
      <ul class="list" id="paper-list"></ul>
    </section>
    <div class="splitter" data-resize="middle"></div>
    <section class="viewer">
      <div class="viewer-header">
        <div class="eyebrow">Reader</div>
        <h2 class="viewer-paper-title" id="viewer-title">Select a paper</h2>
        <div class="viewer-paper-meta" id="viewer-meta">Choose a topic on the left, then a paper in the middle column.</div>
        <div class="tabbar">
          <button type="button" class="tab-button active" data-tab="summary">Summary</button>
          <button type="button" class="tab-button" data-tab="slide">Slide</button>
          <button type="button" class="tab-button" data-tab="pdf">PDF</button>
          <a id="open-pdf" class="action-link" target="_blank" rel="noreferrer" aria-disabled="true">Open PDF</a>
        </div>
      </div>
      <div class="viewer-body">
        <div class="tab-panel" data-panel="pdf">
          <iframe id="pdf-frame" title="PDF viewer"></iframe>
          <div id="pdf-empty" class="empty-state" hidden>No PDF is available for this paper.</div>
        </div>
        <div class="tab-panel active" data-panel="summary">
          <div id="summary-content" class="markdown-body empty-state">Select a paper with a summary.</div>
        </div>
        <div class="tab-panel" data-panel="slide">
          <div id="slide-content" class="slide-html empty-state">Select a paper with a slide deck.</div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const state = {
      topics: [],
      currentTopic: null,
      papers: [],
      currentPaper: null,
    };

    let slideFitScheduled = false;

    const els = {
      topicList: document.getElementById('topic-list'),
      topicCount: document.getElementById('topic-count'),
      paperList: document.getElementById('paper-list'),
      paperCount: document.getElementById('paper-count'),
      paperPanelTitle: document.getElementById('paper-panel-title'),
      viewerTitle: document.getElementById('viewer-title'),
      viewerMeta: document.getElementById('viewer-meta'),
      pdfFrame: document.getElementById('pdf-frame'),
      pdfEmpty: document.getElementById('pdf-empty'),
      summaryContent: document.getElementById('summary-content'),
      slideContent: document.getElementById('slide-content'),
      openPdf: document.getElementById('open-pdf'),
    };

    function setActionLink(link, href) {
      if (href) {
        link.href = href;
        link.setAttribute('aria-disabled', 'false');
      } else {
        link.removeAttribute('href');
        link.setAttribute('aria-disabled', 'true');
      }
    }

    function paperClass(paper) {
      if (paper.slide) return 'blue';
      if (paper.summary) return 'black';
      return 'gray';
    }

    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(\`Request failed: \${res.status}\`);
      return res.json();
    }

    async function renderMath(container) {
      if (!window.MathJax || !window.MathJax.typesetPromise) return;
      await window.MathJax.typesetPromise([container]);
    }

    function fitSlidesToView() {
      const firstSlide = els.slideContent.querySelector('svg[data-marpit-svg]');
      if (!firstSlide || !els.slideContent.clientWidth || !els.slideContent.clientHeight) return;

      const viewBox = firstSlide.getAttribute('viewBox');
      let slideWidth = 0;
      let slideHeight = 0;

      if (viewBox) {
        const values = viewBox.split(/[\s,]+/).map(Number);
        if (values.length === 4 && values.every(Number.isFinite)) {
          slideWidth = values[2];
          slideHeight = values[3];
        }
      }

      if (!slideWidth || !slideHeight) {
        const widthAttr = Number(firstSlide.getAttribute('width'));
        const heightAttr = Number(firstSlide.getAttribute('height'));
        if (Number.isFinite(widthAttr) && Number.isFinite(heightAttr) && widthAttr > 0 && heightAttr > 0) {
          slideWidth = widthAttr;
          slideHeight = heightAttr;
        }
      }

      if (!slideWidth || !slideHeight) return;

      const availableWidth = Math.max(0, els.slideContent.clientWidth - 64);
      const availableHeight = Math.max(0, els.slideContent.clientHeight - 64);
      const scale = Math.max(0.1, Math.min(availableWidth / slideWidth, availableHeight / slideHeight));

      els.slideContent.style.setProperty('--slide-base-width', \`\${slideWidth}px\`);
      els.slideContent.style.setProperty('--slide-scale', String(scale));
    }

    function scheduleSlideFit() {
      if (slideFitScheduled) return;
      slideFitScheduled = true;
      window.requestAnimationFrame(() => {
        slideFitScheduled = false;
        fitSlidesToView();
      });
    }

    function renderTopics() {
      els.topicCount.textContent = \`\${state.topics.length} topics\`;
      els.topicList.innerHTML = '';
      for (const topic of state.topics) {
        const li = document.createElement('li');
        li.className = 'item' + (state.currentTopic && state.currentTopic.id === topic.id ? ' active' : '');
        li.innerHTML = \`
          <div class="paper-title">\${topic.title}</div>
          <div class="topic-keywords">\${topic.keyword.join(', ')}</div>
        \`;
        li.addEventListener('click', () => selectTopic(topic.id));
        els.topicList.appendChild(li);
      }
    }

    function renderPapers() {
      els.paperList.innerHTML = '';
      if (!state.currentTopic) {
        els.paperPanelTitle.textContent = 'Papers';
        els.paperCount.textContent = 'Select a topic.';
        return;
      }
      els.paperPanelTitle.textContent = state.currentTopic.title;
      els.paperCount.textContent = \`\${state.papers.length} papers\`;
      for (const paper of state.papers) {
        const isActive = state.currentPaper && state.currentPaper.index === paper.index;
        const li = document.createElement('li');
        li.className = \`item paper-item \${paperClass(paper)}\${isActive ? ' active' : ''}\`;
        li.innerHTML = \`
          <div class="paper-title">\${paper.title}</div>
          <div class="paper-meta">\${paper.author}<br>\${paper.year}</div>
        \`;
        li.addEventListener('click', () => selectPaper(paper.index));
        els.paperList.appendChild(li);
      }
    }

    async function loadTopics() {
      const data = await fetchJson('/api/topics');
      state.topics = data.topics;
      state.currentTopic = data.topics[0] || null;
      renderTopics();
      if (state.currentTopic) await loadPapers(state.currentTopic.id);
    }

    async function loadPapers(topicId) {
      const data = await fetchJson(\`/api/papers?topic=\${encodeURIComponent(topicId)}\`);
      state.papers = data.papers;
      state.currentPaper = state.papers[0] || null;
      renderPapers();
      await renderCurrentPaper();
    }

    async function selectTopic(topicId) {
      if (state.currentTopic && state.currentTopic.id === topicId) return;
      state.currentTopic = state.topics.find((topic) => topic.id === topicId) || null;
      renderTopics();
      await loadPapers(topicId);
    }

    async function selectPaper(index) {
      state.currentPaper = state.papers.find((paper) => paper.index === index) || null;
      renderPapers();
      await renderCurrentPaper();
    }

    async function renderCurrentPaper() {
      const paper = state.currentPaper;
      if (!paper) {
        els.viewerTitle.textContent = 'Select a paper';
        els.viewerMeta.textContent = 'Choose a topic on the left, then a paper in the middle column.';
        els.pdfFrame.hidden = true;
        els.pdfEmpty.hidden = false;
        els.summaryContent.className = 'markdown-body empty-state';
        els.summaryContent.textContent = 'Select a paper with a summary.';
        els.slideContent.className = 'slide-html empty-state';
        els.slideContent.textContent = 'Select a paper with a slide deck.';
        setActionLink(els.openPdf, '');
        return;
      }

      els.viewerTitle.textContent = paper.title;
      els.viewerMeta.textContent = \`\${paper.author} | \${paper.year}\`;

      const pdfUrl = paper.url ? (paper.url.includes('/pdf/') ? paper.url : paper.url.replace('/abs/', '/pdf/') + '.pdf') : '';
      if (pdfUrl) {
        els.pdfFrame.hidden = false;
        els.pdfEmpty.hidden = true;
        els.pdfFrame.src = pdfUrl;
      } else {
        els.pdfFrame.removeAttribute('src');
        els.pdfFrame.hidden = true;
        els.pdfEmpty.hidden = false;
      }
      setActionLink(els.openPdf, pdfUrl);

      if (paper.summary) {
        try {
          const html = await fetch(\`/api/render-summary?topic=\${encodeURIComponent(state.currentTopic.id)}&file=\${encodeURIComponent(paper.summary)}\`).then((res) => {
            if (!res.ok) throw new Error(\`Request failed: \${res.status}\`);
            return res.text();
          });
          els.summaryContent.className = 'markdown-body';
          els.summaryContent.innerHTML = html;
          await renderMath(els.summaryContent);
        } catch (error) {
          els.summaryContent.className = 'markdown-body empty-state';
          els.summaryContent.textContent = \`Failed to load summary: \${error.message}\`;
        }
      } else {
        els.summaryContent.className = 'markdown-body empty-state';
        els.summaryContent.textContent = 'No summary markdown file is linked for this paper.';
      }

      if (paper.slide) {
        try {
          const html = await fetch(\`/api/render-slide?topic=\${encodeURIComponent(state.currentTopic.id)}&file=\${encodeURIComponent(paper.slide)}\`).then((res) => {
            if (!res.ok) throw new Error(\`Request failed: \${res.status}\`);
            return res.text();
          });
          els.slideContent.className = 'slide-html';
          els.slideContent.innerHTML = html;
          scheduleSlideFit();
        } catch (error) {
          els.slideContent.className = 'slide-html empty-state';
          els.slideContent.textContent = \`Failed to render slide deck: \${error.message}\`;
        }
      } else {
        els.slideContent.className = 'slide-html empty-state';
        els.slideContent.textContent = 'No slide markdown file is linked for this paper.';
      }
    }

    function setupTabs() {
      const buttons = document.querySelectorAll('.tab-button');
      const panels = document.querySelectorAll('.tab-panel');
      for (const button of buttons) {
        button.addEventListener('click', () => {
          const tab = button.dataset.tab;
          buttons.forEach((item) => item.classList.toggle('active', item === button));
          panels.forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === tab));
          if (tab === 'slide') scheduleSlideFit();
        });
      }
    }

    function setupSlideFit() {
      const observer = new ResizeObserver(() => {
        scheduleSlideFit();
      });
      observer.observe(els.slideContent);
      window.addEventListener('resize', scheduleSlideFit);
      scheduleSlideFit();
    }

    function setupSplitters() {
      const app = document.getElementById('app');
      const min1 = 180;
      const min2 = 240;
      const min3 = 320;

      function applyResize(which, clientX) {
        if (window.innerWidth <= 960) return;
        const total = app.clientWidth;
        const styles = getComputedStyle(document.documentElement);
        const currentCol1 = parseFloat(styles.getPropertyValue('--col1'));
        if (which === 'left') {
          const newCol1 = Math.max(min1, Math.min(clientX, total - min2 - min3 - 12));
          document.documentElement.style.setProperty('--col1', \`\${newCol1}px\`);
        } else {
          const leftEdge = currentCol1 + 6;
          const newCol2 = Math.max(min2, Math.min(clientX - leftEdge, total - currentCol1 - min3 - 12));
          document.documentElement.style.setProperty('--col2', \`\${newCol2}px\`);
        }
      }

      document.querySelectorAll('.splitter').forEach((splitter) => {
        splitter.addEventListener('pointerdown', (event) => {
          event.preventDefault();
          const which = splitter.dataset.resize;
          const move = (moveEvent) => applyResize(which, moveEvent.clientX);
          const up = () => {
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', up);
          };
          window.addEventListener('pointermove', move);
          window.addEventListener('pointerup', up);
        });
      });
    }

    setupTabs();
    setupSlideFit();
    setupSplitters();
    loadTopics().catch((error) => {
      els.viewerTitle.textContent = 'Failed to load viewer';
      els.viewerMeta.textContent = error.message;
    });
  </script>
</body>
</html>`;

function main() {
  const args = parseArgs(process.argv.slice(2));
  const baseDir = path.resolve(args.dir);
  if (!fs.existsSync(baseDir) || !fs.statSync(baseDir).isDirectory()) {
    throw new Error(`Base directory does not exist: ${baseDir}`);
  }

  const topics = listTopics(baseDir);
  if (topics.length === 0) {
    throw new Error(`No topic folders with metadata.json and paper_list.jsonl were found in: ${baseDir}`);
  }

  const app = express();
  app.use(express.urlencoded({ extended: false }));
  app.use(express.json({ limit: '2mb' }));

  app.get('/', (_req, res) => {
    res.type('html').send(HTML_PAGE);
  });

  app.get('/favicon.ico', (_req, res) => {
    res.sendFile(path.join(__dirname, 'favicon.ico'));
  });

  app.get('/api/topics', (_req, res) => {
    res.json({
      topics: listTopics(baseDir).map(({ id, title, keyword }) => ({ id, title, keyword })),
    });
  });

  app.get('/api/papers', (req, res) => {
    const topicId = String(req.query.topic || '');
    const topic = getTopicMap(baseDir).get(topicId);
    if (!topic) {
      res.status(404).json({ error: 'Unknown topic' });
      return;
    }
    try {
      res.json({ papers: loadPapers(topic.path) });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  app.get('/api/render-summary', (req, res) => {
    const topicId = String(req.query.topic || '');
    const fileName = String(req.query.file || '');
    const target = ensureInside(baseDir, topicId, fileName);
    if (!target) {
      res.status(404).send('Summary file not found');
      return;
    }
    const markdown = fs.readFileSync(target.filePath, 'utf8');
    res.type('html').send(md.render(markdown));
  });

  app.get('/api/render-slide', (req, res) => {
    const topicId = String(req.query.topic || '');
    const fileName = String(req.query.file || '');
    const target = ensureInside(baseDir, topicId, fileName);
    if (!target) {
      res.status(404).send('Slide file not found');
      return;
    }
    try {
      const markdown = fs.readFileSync(target.filePath, 'utf8');
      const marp = new Marp({ html: true, script: false, math: 'mathjax' });
      const rendered = marp.render(markdown);
      const slideHtml = `
        <div class="marp-root">
          <style>${rendered.css}</style>
          ${rendered.html}
        </div>
      `;
      res.type('html').send(slideHtml);
    } catch (error) {
      res.status(500).send(String(error && error.stack ? error.stack : error));
    }
  });

  app.get('/raw/:topic/:file', (req, res) => {
    const target = ensureInside(baseDir, req.params.topic, req.params.file);
    if (!target) {
      res.status(404).send('File not found');
      return;
    }
    res.sendFile(target.filePath);
  });

  app.listen(args.port, '127.0.0.1', () => {
    console.log(`Paper viewer serving ${topics.length} topics from ${baseDir}`);
    console.log(`Open http://127.0.0.1:${args.port} in your browser`);
  });
}

main();
