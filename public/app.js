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
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
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

  els.slideContent.style.setProperty('--slide-base-width', `${slideWidth}px`);
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
    const isActive = state.currentPaper && state.currentPaper.index === paper.index;
    const li = document.createElement('li');
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
  if (state.currentTopic) await loadPapers(state.currentTopic.id);
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
  els.viewerMeta.textContent = `${paper.author} | ${paper.year}`;

  const pdfUrl = paper.url
    ? (paper.url.includes('/pdf/') ? paper.url : `${paper.url.replace('/abs/', '/pdf/')}.pdf`)
    : '';

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
      const html = await fetch(`/api/render-summary?topic=${encodeURIComponent(state.currentTopic.id)}&file=${encodeURIComponent(paper.summary)}`).then((res) => {
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        return res.text();
      });
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
    try {
      const html = await fetch(`/api/render-slide?topic=${encodeURIComponent(state.currentTopic.id)}&file=${encodeURIComponent(paper.slide)}`).then((res) => {
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        return res.text();
      });
      els.slideContent.className = 'slide-html';
      els.slideContent.innerHTML = html;
      scheduleSlideFit();
    } catch (error) {
      els.slideContent.className = 'slide-html empty-state';
      els.slideContent.textContent = `Failed to render slide deck: ${error.message}`;
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
setupSlideFit();
setupSplitters();
loadTopics().catch((error) => {
  els.viewerTitle.textContent = 'Failed to load viewer';
  els.viewerMeta.textContent = error.message;
});
