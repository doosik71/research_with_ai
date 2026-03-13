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
  const args = { port: 8080, dir: 'docs' };
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
  app.use(express.static(path.join(__dirname, 'public')));

  app.get('/', (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
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
