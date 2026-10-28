import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发期代理：前端 /api、/steps、/out、/uploads 转发到 FastAPI 后端（127.0.0.1:8000）
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/steps': 'http://127.0.0.1:8000',
      '/out': 'http://127.0.0.1:8000',
      '/uploads': 'http://127.0.0.1:8000',
      '/opening.jpg': 'http://127.0.0.1:8000'
    }
  }
})