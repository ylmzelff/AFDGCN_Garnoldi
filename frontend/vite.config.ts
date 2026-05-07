import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/auth': 'http://localhost:9001',
      '/phases': 'http://localhost:9001',
      '/api': 'http://localhost:9001',
      '/admin': 'http://localhost:9001',
      '/health': 'http://localhost:9001',
      '/ws': {
        target: 'ws://localhost:9001',
        ws: true,
      },
    },
  },
})
