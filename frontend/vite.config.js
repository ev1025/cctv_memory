import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// build → ../ui-dist (FastAPI 가 서빙). dev 시 API 는 8000 으로 프록시.
const API = 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  build: { outDir: '../ui-dist', emptyOutDir: true },
  server: {
    proxy: {
      '/index-video': API,
      '/query': API,
      '/video': API,
      '/thumb': API,
      '/memory-status': API,
    },
  },
})
