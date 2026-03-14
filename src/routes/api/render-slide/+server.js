import { Marp } from '@marp-team/marp-core';
import { json } from '@sveltejs/kit';

const marp = new Marp({ html: true, script: false });

/**
 * Handles POST requests to render markdown slides.
 * @param {object} { request }
 * @returns {Response}
 */
export async function POST({ request }) {
	try {
		const { markdown } = await request.json();
		if (typeof markdown !== 'string') {
			return json({ error: 'Markdown content must be a string.' }, { status: 400 });
		}
		const { html, css } = marp.render(markdown);
		return json({ html, css });
	} catch (error) {
		return json({ error: 'Failed to render slide.', details: error.message }, { status: 500 });
	}
}
