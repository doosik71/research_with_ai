import { dev } from '$app/environment';
import type { RequestHandler } from './$types';

const ALLOWED_HOSTS = new Set(['arxiv.org', 'www.arxiv.org', 'ar5iv.labs.arxiv.org']);

export const GET: RequestHandler = async ({ url, fetch }) => {
	if (!dev) {
		return new Response('Not found', { status: 404 });
	}

	const target = url.searchParams.get('url');
	if (!target) return new Response('Missing url', { status: 400 });

	let parsed: URL;
	try {
		parsed = new URL(target);
	} catch {
		return new Response('Invalid url', { status: 400 });
	}

	if (!['http:', 'https:'].includes(parsed.protocol)) {
		return new Response('Unsupported protocol', { status: 400 });
	}

	if (!ALLOWED_HOSTS.has(parsed.hostname)) {
		return new Response('Host not allowed', { status: 403 });
	}

	const upstream = await fetch(parsed.toString());
	if (!upstream.ok) {
		return new Response('Upstream error', { status: upstream.status });
	}

	const body = await upstream.arrayBuffer();
	const contentType = upstream.headers.get('content-type') ?? 'application/octet-stream';

	return new Response(body, {
		status: 200,
		headers: {
			'content-type': contentType,
			'cache-control': 'no-store'
		}
	});
};
