import { defineConfig } from 'astro/config';

export default defineConfig({
  server: { port: 4321, strictPort: true },
  vite: {
    server: {
      proxy: {
        '/api': 'http://127.0.0.1:7860',
      },
    },
  },
});
