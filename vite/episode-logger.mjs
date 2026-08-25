import fs from 'fs';
import path from 'path';

export function episodeLoggerPlugin() {
	return {
		name: 'episode-logger',
		configureServer(server) {
			server.middlewares.use('/api/episode', (req, res) => {
				if (req.method === 'GET') {
					const { filePath } = resolveRunFile();
					if (fs.existsSync(filePath)) {
						const content = fs.readFileSync(filePath, 'utf-8');
						res.setHeader('Content-Type', 'application/jsonl');
						res.end(content);
					} else {
						res.setHeader('Content-Type', 'application/jsonl');
						res.end('');
					}
					return;
				}

				if (req.method !== 'POST') {
					res.statusCode = 405;
					res.end(JSON.stringify({ error: 'Method not allowed' }));
					return;
				}

				let body = '';
				req.on('data', chunk => { body += chunk; });
				req.on('end', () => {
					try {
						const data = JSON.parse(body);
						const { filePath, dir } = resolveRunFile();

						fs.mkdirSync(dir, { recursive: true });

						const entry = {
							episode: data.episode,
							score: data.score,
							epsilon: Math.round(data.epsilon * 10000) / 10000,
							timestamp: new Date().toISOString()
						};
						fs.appendFileSync(filePath, JSON.stringify(entry) + '\n');

						res.setHeader('Content-Type', 'application/json');
						res.end(JSON.stringify({ ok: true }));
					} catch (err) {
						console.error('[episode-logger] Error:', err.message);
						res.statusCode = 500;
						res.end(JSON.stringify({ error: err.message }));
					}
				});
			});
		}
	};
}

function resolveRunFile() {
	const experimentDir = process.cwd();
	const runsDir = path.join(experimentDir, 'runs');

	let runNumber = 1;
	if (fs.existsSync(runsDir)) {
		const files = fs.readdirSync(runsDir)
			.filter(f => f.endsWith('.json'))
			.sort();
		if (files.length > 0) {
			const last = JSON.parse(fs.readFileSync(path.join(runsDir, files[files.length - 1]), 'utf-8'));
			runNumber = last.run ?? files.length;
		}
	}

	const pad = String(runNumber).padStart(3, '0');
	const dir = path.join(experimentDir, 'data');
	const filePath = path.join(dir, `run-${pad}.jsonl`);

	return { filePath, dir, runNumber };
}
