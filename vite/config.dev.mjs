import { defineConfig } from 'vite';

export default defineConfig({
	base: './',
	publicDir: '../../public',
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
