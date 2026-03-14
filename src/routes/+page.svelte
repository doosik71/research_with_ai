<script>
	import { onMount } from 'svelte';
	import MarkdownIt from 'markdown-it';

	let topics = $state([]);
	let papers = $state([]);
	let selectedTopic = $state(null);
	let selectedPaper = $state(null);

	// Preview State
	let renderHtml = $state('');
	let renderType = $state('none'); // 'none', 'summary', 'slide'
	let isLoading = $state(false);

	// Renderers
	const md = new MarkdownIt({ html: true, linkify: true, typographer: true });

	onMount(async () => {
		try {
			const res = await fetch('/docs/manifest.json');

			if (res.ok) {
				topics = await res.json();
				// Sort topics by title
				topics.sort((a, b) => (a.title || a.id).localeCompare(b.title || b.id));
			} else {
				console.error('Failed to load manifest.json');
			}
		} catch (e) {
			console.error('Error fetching manifest:', e);
		}
	});

	async function selectTopic(topic) {
		selectedTopic = topic;
		selectedPaper = null;
		renderType = 'none';
		renderHtml = '';
		papers = [];
		isLoading = true;

		try {
			const res = await fetch(`/docs/${topic.id}/paper_list.jsonl`);
			if (res.ok) {
				const text = await res.text();
				papers = text
					.split('\n')
					.filter((line) => line.trim())
					.map((line, index) => {
						try {
							return { ...JSON.parse(line), index };
						} catch {
							return null;
						}
					})
					.filter((p) => p);
			}
		} catch (e) {
			console.error('Error fetching paper list:', e);
		} finally {
			isLoading = false;
		}
	}

	async function loadPreview(paper, type) {
		selectedPaper = paper;
		renderType = type;
		isLoading = true;
		renderHtml = '';

		const filename = type === 'summary' ? paper.summary : paper.slide;
		if (!filename) {
			renderHtml = '<p>No file specified.</p>';
			isLoading = false;
			return;
		}

		try {
			const res = await fetch(`/docs/${selectedTopic.id}/${filename}`);
			if (res.ok) {
				const text = await res.text();
				if (type === 'summary') {
					renderHtml = md.render(text);
				} else if (type === 'slide') {
					const renderRes = await fetch('/api/render-slide', {
						method: 'POST',
						headers: {
							'Content-Type': 'application/json'
						},
						body: JSON.stringify({ markdown: text })
					});

					if (renderRes.ok) {
						const { html, css } = await renderRes.json();
						renderHtml = `<style>${css}</style><div class="marp-slide">${html}</div>`;
					} else {
						renderHtml = `<p>Failed to render slide on server.</p>`;
					}
				}
			} else {
				renderHtml = `<p>File not found: ${filename}</p>`;
			}
		} catch (e) {
			renderHtml = `<p>Error loading file: ${e.message}</p>`;
		} finally {
			isLoading = false;
		}
	}
</script>

<div class="app-container">
	<!-- 1. Topic List -->
	<aside class="sidebar">
		<h2>Topics</h2>
		<ul>
			{#each topics as topic (topic.id)}
				<li>
					<button class:active={selectedTopic?.id === topic.id} onclick={() => selectTopic(topic)}>
						{topic.title || topic.id}
					</button>
				</li>
			{/each}
		</ul>
	</aside>

	<!-- 2. Paper List -->
	<div class="paper-list">
		<h2>Papers {selectedTopic ? `(${papers.length})` : ''}</h2>
		{#if isLoading && papers.length === 0}
			<div class="loading">Loading...</div>
		{/if}
		<div class="list-content">
			{#each papers as paper (paper.index)}
				<div class="paper-item" class:selected={selectedPaper?.index === paper.index}>
					<h3>{paper.title}</h3>
					<p class="meta">{paper.author} ({paper.year})</p>
					<div class="actions">
						{#if paper.summary}
							<button onclick={() => loadPreview(paper, 'summary')}>Summary</button>
						{/if}
						{#if paper.slide}
							<button onclick={() => loadPreview(paper, 'slide')}>Slide</button>
						{/if}
						{#if paper.url}
							<a href={paper.url} target="_blank" rel="noopener noreferrer">PDF/Link</a>
						{/if}
					</div>
				</div>
			{/each}
		</div>
	</div>

	<!-- 3. Preview Area -->
	<main class="preview">
		{#if renderType !== 'none'}
			<div class="preview-header">
				<h2>{renderType === 'summary' ? '📝 Summary' : '📊 Slide'}</h2>
			</div>
			<div class="preview-content markdown-body">
				{#if isLoading}
					<p>Loading content...</p>
				{:else}
					{@html renderHtml}
				{/if}
			</div>
		{:else}
			<div class="placeholder">Select a summary or slide to preview</div>
		{/if}
	</main>
</div>

<style>
	:global(body) {
		margin: 0;
		font-family: sans-serif;
		height: 100vh;
		overflow: hidden;
	}
	.app-container {
		display: flex;
		height: 100vh;
	}

	.sidebar {
		width: 250px;
		background: #f4f4f5;
		border-right: 1px solid #ddd;
		overflow-y: auto;
		padding: 1rem;
	}
	.sidebar h2 {
		font-size: 1.2rem;
		margin-top: 0;
	}
	.sidebar ul {
		list-style: none;
		padding: 0;
	}
	.sidebar button {
		display: block;
		width: 100%;
		text-align: left;
		padding: 0.5rem;
		border: none;
		background: none;
		cursor: pointer;
		border-radius: 4px;
	}
	.sidebar button:hover {
		background: #e4e4e7;
	}
	.sidebar button.active {
		background: #e0e7ff;
		color: #3730a3;
		font-weight: bold;
	}

	.paper-list {
		width: 350px;
		border-right: 1px solid #ddd;
		display: flex;
		flex-direction: column;
		background: #fff;
	}
	.paper-list h2 {
		padding: 1rem;
		margin: 0;
		border-bottom: 1px solid #eee;
		font-size: 1.2rem;
	}
	.list-content {
		overflow-y: auto;
		flex: 1;
		padding: 0.5rem;
	}
	.paper-item {
		padding: 0.8rem;
		border-bottom: 1px solid #eee;
	}
	.paper-item.selected {
		background: #f0f9ff;
	}
	.paper-item h3 {
		margin: 0 0 0.3rem 0;
		font-size: 1rem;
	}
	.paper-item .meta {
		margin: 0 0 0.5rem 0;
		color: #666;
		font-size: 0.85rem;
	}
	.actions {
		display: flex;
		gap: 0.5rem;
	}
	.actions button,
	.actions a {
		font-size: 0.8rem;
		padding: 0.2rem 0.6rem;
		border: 1px solid #ccc;
		background: #fff;
		border-radius: 4px;
		text-decoration: none;
		color: #333;
		cursor: pointer;
	}
	.actions button:hover,
	.actions a:hover {
		background: #f4f4f5;
	}

	.preview {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background: #fff;
	}
	.preview-header {
		padding: 0.8rem;
		border-bottom: 1px solid #eee;
		background: #fafafa;
	}
	.preview-header h2 {
		margin: 0;
		font-size: 1.1rem;
	}
	.preview-content {
		flex: 1;
		overflow-y: auto;
		padding: 2rem;
	}
	.placeholder {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100%;
		color: #999;
	}
	.loading {
		padding: 1rem;
		color: #666;
		text-align: center;
	}
</style>
