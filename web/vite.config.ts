import { defineConfig } from 'vite'

export default defineConfig({
  base: './',
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
