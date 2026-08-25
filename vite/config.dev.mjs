import { defineConfig } from 'vite';
import { episodeLoggerPlugin } from './episode-logger.mjs';

export default defineConfig({
	base: './',
	publicDir: '../../public',
	plugins: [episodeLoggerPlugin()],
	build: {
		rollupOptions: {
			output: {
				manualChunks: {
					phaser: ['phaser']
				}
			}
		},
	},
	server: {
		port: 8080
	}
});
