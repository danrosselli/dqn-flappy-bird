export async function recordEpisode(episode, score, epsilon) {
	try {
		await fetch('/api/episode', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ episode, score, epsilon })
		});
	} catch (err) {
		console.warn('[episodeLogger] Failed to record episode:', err.message);
	}
}
