import { defineConfig } from 'vite';

// House rule: the server binds to 127.0.0.1 only — nothing is served to, or fetched from, the outside.
export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
});
