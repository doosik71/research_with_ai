import argparse
import json
import mimetypes
import posixpath
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Viewer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;500;700&display=swap" rel="stylesheet">
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
      background:
        linear-gradient(to right, transparent, rgba(70, 65, 58, 0.14), transparent);
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
    .slide-frame {
      width: 100%;
      height: 100%;
      border: 0;
      background: white;
    }
    .debug-block {
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      padding: 16px;
      margin-top: 16px;
      border: 1px solid #d9d2c5;
      border-radius: 12px;
      background: #faf7ef;
      color: #3a352f;
      max-height: 42vh;
      overflow: auto;
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
          <button type="button" class="tab-button" data-tab="summary">Summary</button>
          <button type="button" class="tab-button active" data-tab="pdf">PDF</button>
          <a id="open-pdf" class="action-link" target="_blank" rel="noreferrer" aria-disabled="true">Open PDF</a>
        </div>
      </div>
      <div class="viewer-body">
        <div class="tab-panel active" data-panel="pdf">
          <iframe id="pdf-frame" title="PDF viewer"></iframe>
          <div id="pdf-empty" class="empty-state" hidden>No PDF is available for this paper.</div>
        </div>
        <div class="tab-panel" data-panel="summary">
          <div id="summary-content" class="markdown-body empty-state">Select a paper with a summary.</div>
        </div>
        <div class="tab-panel" data-panel="slide">
          <iframe id="slide-frame" class="slide-frame" title="Slide viewer"></iframe>
          <div id="slide-empty" class="empty-state" hidden>Select a paper with a slide deck.</div>
          <pre id="slide-debug" class="debug-block" hidden></pre>
        </div>
      </div>
    </section>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
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
  <script>
    const state = {
      topics: [],
      currentTopic: null,
      papers: [],
      currentPaper: null,
      marpModule: null,
    };

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
      slideFrame: document.getElementById('slide-frame'),
      slideEmpty: document.getElementById('slide-empty'),
      slideDebug: document.getElementById('slide-debug'),
      openPdf: document.getElementById('open-pdf'),
      openSlideSource: document.getElementById('open-slide-source'),
    };

    const MARP_MODULE_URLS = [
      'https://esm.sh/@marp-team/marp-core@latest?bundle&target=es2022',
      'https://esm.sh/@marp-team/marp-core@latest?bundle'
    ];

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

    function normalizePdfUrl(url) {
      if (!url) return '';
      if (url.includes('/pdf/')) return url;
      if (url.includes('arxiv.org/abs/')) {
        return url.replace('/abs/', '/pdf/') + '.pdf';
      }
      return url;
    }

    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }
      return res.json();
    }

    async function fetchText(url) {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }
      return res.text();
    }

    async function postDebug(title, content) {
      try {
        await fetch('/api/debug-srcdoc', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ title, content })
        });
      } catch (error) {
        console.error('Failed to write debug payload', error);
      }
    }

    async function renderMath(container) {
      if (!window.MathJax || !window.MathJax.typesetPromise) return;
      await window.MathJax.typesetPromise([container]);
    }

    function renderTopics() {
      els.topicCount.textContent = `${state.topics.length} topics`;
      els.topicList.innerHTML = '';
      for (const topic of state.topics) {
        const li = document.createElement('li');
        li.className = 'item' + (state.currentTopic && state.currentTopic.id === topic.id ? ' active' : '');
        li.innerHTML = `
          <div class="paper-title">${topic.title}</div>
          <div class="topic-keywords">${topic.keyword.join(', ')}</div>
        `;
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
      els.paperCount.textContent = `${state.papers.length} papers`;
      for (const paper of state.papers) {
        const li = document.createElement('li');
        const isActive = state.currentPaper && state.currentPaper.index === paper.index;
        li.className = `item paper-item ${paperClass(paper)}${isActive ? ' active' : ''}`;
        li.innerHTML = `
          <div class="paper-title">${paper.title}</div>
          <div class="paper-meta">${paper.author}<br>${paper.year}</div>
        `;
        li.addEventListener('click', () => selectPaper(paper.index));
        els.paperList.appendChild(li);
      }
    }

    async function loadTopics() {
      const data = await fetchJson('/api/topics');
      state.topics = data.topics;
      state.currentTopic = data.topics[0] || null;
      renderTopics();
      if (state.currentTopic) {
        await loadPapers(state.currentTopic.id);
      }
    }

    async function loadPapers(topicId) {
      const data = await fetchJson(`/api/papers?topic=${encodeURIComponent(topicId)}`);
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

    async function ensureMarpModule() {
      if (state.marpModule) return state.marpModule;
      let lastError = null;
      for (const url of MARP_MODULE_URLS) {
        try {
          state.marpModule = await import(url);
          return state.marpModule;
        } catch (error) {
          lastError = error;
        }
      }
      throw lastError || new Error('Unable to load Marp module.');
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
        els.slideFrame.hidden = true;
        els.slideEmpty.hidden = false;
        els.slideDebug.hidden = true;
        els.slideDebug.textContent = '';
        setActionLink(els.openPdf, '');
        setActionLink(els.openSlideSource, '');
        return;
      }

      els.viewerTitle.textContent = paper.title;
      els.viewerMeta.textContent = `${paper.author} | ${paper.year}`;

      const pdfUrl = normalizePdfUrl(paper.url);
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
        const summaryUrl = `/raw/${encodeURIComponent(state.currentTopic.id)}/${encodeURIComponent(paper.summary)}`;
        try {
          const markdown = await fetchText(summaryUrl);
          const html = DOMPurify.sanitize(marked.parse(markdown));
          els.summaryContent.className = 'markdown-body';
          els.summaryContent.innerHTML = html;
          await renderMath(els.summaryContent);
        } catch (error) {
          els.summaryContent.className = 'markdown-body empty-state';
          els.summaryContent.textContent = `Failed to load summary: ${error.message}`;
        }
      } else {
        els.summaryContent.className = 'markdown-body empty-state';
        els.summaryContent.textContent = 'No summary markdown file is linked for this paper.';
      }

      if (paper.slide) {
        const slideUrl = `/raw/${encodeURIComponent(state.currentTopic.id)}/${encodeURIComponent(paper.slide)}`;
        setActionLink(els.openSlideSource, slideUrl);
        try {
          const markdown = await fetchText(slideUrl);
          const marpModule = await ensureMarpModule();
          const Marp = marpModule.Marp || marpModule.default?.Marp || marpModule.default;
          if (!Marp) throw new Error('Marp export not found.');
          const marp = new Marp({ html: true, math: false, script: false });
          const rendered = marp.render(markdown);
          const srcdoc = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>html,body{margin:0;padding:0;background:#fff;}svg{display:block;max-width:100%;height:auto;margin:0 auto;}</style><style>${rendered.css}</style></head><body>${rendered.html}</body></html>`;
          els.slideFrame.hidden = false;
          els.slideEmpty.hidden = true;
          els.slideDebug.hidden = false;
          els.slideDebug.textContent = srcdoc;
          await postDebug(`Slide srcdoc: ${paper.title}`, srcdoc);
          els.slideFrame.srcdoc = srcdoc;
        } catch (error) {
          els.slideFrame.hidden = true;
          els.slideFrame.removeAttribute('srcdoc');
          els.slideEmpty.hidden = false;
          els.slideEmpty.textContent = `Failed to render slide deck: ${error.message}`;
          els.slideDebug.hidden = false;
          const debugText = error && error.stack ? error.stack : String(error);
          els.slideDebug.textContent = debugText;
          await postDebug(`Slide error: ${paper.title}`, debugText);
        }
      } else {
        els.slideFrame.hidden = true;
        els.slideFrame.removeAttribute('srcdoc');
        els.slideEmpty.hidden = false;
        els.slideEmpty.textContent = 'No slide markdown file is linked for this paper.';
        els.slideDebug.hidden = true;
        els.slideDebug.textContent = '';
        setActionLink(els.openSlideSource, '');
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
        });
      }
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
        const currentCol2 = parseFloat(styles.getPropertyValue('--col2'));
        if (which === 'left') {
          const newCol1 = Math.max(min1, Math.min(clientX, total - min2 - min3 - 12));
          document.documentElement.style.setProperty('--col1', `${newCol1}px`);
        } else {
          const leftEdge = currentCol1 + 6;
          const newCol2 = Math.max(min2, Math.min(clientX - leftEdge, total - currentCol1 - min3 - 12));
          document.documentElement.style.setProperty('--col2', `${newCol2}px`);
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
    setupSplitters();
    loadTopics().catch((error) => {
      els.viewerTitle.textContent = 'Failed to load viewer';
      els.viewerMeta.textContent = error.message;
    });
  </script>
</body>
</html>
"""


@dataclass
class Topic:
    id: str
    title: str
    keyword: list[str]
    path: Path


class PaperViewerApp:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.debug_path = self.base_dir / "temp.txt"

    def scan_topics(self) -> list[Topic]:
        topics: list[Topic] = []
        for child in sorted(self.base_dir.iterdir()):
            if not child.is_dir():
                continue
            metadata_path = child / "metadata.json"
            paper_list_path = child / "paper_list.jsonl"
            if not metadata_path.is_file() or not paper_list_path.is_file():
                continue
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            title = str(metadata.get("title", child.name)
                        ).strip() or child.name
            keyword = metadata.get("keyword", [])
            if not isinstance(keyword, list):
                keyword = []
            topics.append(
                Topic(
                    id=child.name,
                    title=title,
                    keyword=[str(item) for item in keyword],
                    path=child,
                )
            )
        return topics

    def topic_map(self) -> dict[str, Topic]:
        return {topic.id: topic for topic in self.scan_topics()}

    def load_papers(self, topic_id: str) -> list[dict[str, Any]]:
        topics = self.topic_map()
        topic = topics.get(topic_id)
        if topic is None:
            raise KeyError(topic_id)

        papers: list[dict[str, Any]] = []
        paper_path = topic.path / "paper_list.jsonl"
        for index, raw_line in enumerate(paper_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            papers.append(
                {
                    "index": index,
                    "title": str(obj.get("title", "")).strip(),
                    "author": str(obj.get("author", "")).strip(),
                    "year": str(obj.get("year", "")).strip(),
                    "url": str(obj.get("url", "")).strip(),
                    "summary": str(obj.get("summary", "")).strip(),
                    "slide": str(obj.get("slide", "")).strip(),
                }
            )
        return papers

    def write_debug(self, title: str, content: str) -> None:
        payload = f"{title}\n{'=' * len(title)}\n{content}\n"
        self.debug_path.write_text(payload, encoding="utf-8")


class ViewerRequestHandler(BaseHTTPRequestHandler):
    server_version = "PaperViewer/0.1"

    @property
    def app(self) -> PaperViewerApp:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.respond_html(HTML_PAGE)
            return
        if parsed.path == "/api/topics":
            self.handle_topics()
            return
        if parsed.path == "/api/papers":
            self.handle_papers(parsed.query)
            return
        if parsed.path.startswith("/raw/"):
            self.handle_raw(parsed.path)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/debug-srcdoc":
            self.handle_debug_srcdoc()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def handle_topics(self) -> None:
        topics = [
            {"id": topic.id, "title": topic.title, "keyword": topic.keyword}
            for topic in self.app.scan_topics()
        ]
        self.respond_json({"topics": topics})

    def handle_papers(self, query: str) -> None:
        params = parse_qs(query)
        topic_id = params.get("topic", [""])[0]
        if not topic_id:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing topic parameter")
            return
        try:
            papers = self.app.load_papers(topic_id)
        except KeyError:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown topic")
            return
        except json.JSONDecodeError as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR,
                            f"Invalid JSONL: {exc}")
            return
        self.respond_json({"papers": papers})

    def handle_raw(self, path: str) -> None:
        remainder = path[len("/raw/"):]
        parts = [unquote(part) for part in remainder.split("/") if part]
        if len(parts) != 2:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid raw path")
            return

        topic_id, file_name = parts
        topic = self.app.topic_map().get(topic_id)
        if topic is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown topic")
            return

        safe_name = posixpath.basename(file_name)
        if safe_name != file_name:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid file name")
            return

        file_path = (topic.path / safe_name).resolve()
        if topic.path not in file_path.parents or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        self.respond_bytes(file_path.read_bytes(),
                           content_type or "text/plain; charset=utf-8")

    def handle_debug_srcdoc(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON payload")
            return

        title = str(payload.get("title", "Slide Debug")
                    ).strip() or "Slide Debug"
        content = str(payload.get("content", ""))
        self.app.write_debug(title, content)
        self.respond_json({"ok": True})

    def respond_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def respond_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def respond_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


class PaperViewerServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], app: PaperViewerApp) -> None:
        super().__init__(server_address, handler_class)
        self.app = app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-column paper viewer")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on")
    parser.add_argument(
        "--dir", default=".", help="Base directory that contains research topic folders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.dir)
    if not base_dir.exists() or not base_dir.is_dir():
        raise SystemExit(f"Base directory does not exist: {base_dir}")

    app = PaperViewerApp(base_dir)
    topics = app.scan_topics()
    if not topics:
        raise SystemExit(
            f"No topic folders with metadata.json and paper_list.jsonl were found in: {base_dir.resolve()}")

    server = PaperViewerServer(
        ("127.0.0.1", args.port), ViewerRequestHandler, app)
    url = f"http://127.0.0.1:{args.port}"
    print(
        f"Paper viewer serving {len(topics)} topics from {base_dir.resolve()}")
    print(f"Open {url} in your browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
