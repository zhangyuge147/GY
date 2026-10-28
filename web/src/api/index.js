// 统一 API 封装（原生 fetch，开发期经 Vite proxy 转发到 FastAPI 后端）

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  })
  if (!res.ok) {
    let msg = `请求失败（${res.status}）`
    try {
      const data = await res.json()
      if (data && data.detail) msg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    } catch (_) { /* 忽略非 JSON 响应体 */ }
    throw new Error(msg)
  }
  return res.json()
}

export const api = {
  // 修复
  heroes: () => request('/api/heroes'),
  hero: (id) => request(`/api/heroes/${id}`),
  frames: (id) => request(`/api/heroes/${id}/frames`),
  // 上传照片修复（multipart，不用统一 JSON 封装，避免覆盖浏览器自动生成的 boundary）
  uploadRestore: (file, heroId, prompt) => {
    const fd = new FormData()
    fd.append('file', file)
    if (heroId) fd.append('hero_id', heroId)
    if (prompt) fd.append('prompt', prompt)
    return fetch('/api/restore', { method: 'POST', body: fd }).then(async (res) => {
      if (!res.ok) {
        let msg = `上传失败（${res.status}）`
        try {
          const d = await res.json()
          if (d && d.detail) msg = typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail)
        } catch (_) { /* 忽略非 JSON 响应体 */ }
        throw new Error(msg)
      }
      return res.json()
    })
  },
  // 修复任务状态轮询
  restoreStatus: (taskId) => request(`/api/restore/${taskId}`),
  // 照片背后的故事（仅史料库收录人物，未收录不编造）
  photoStory: (name) => request('/api/story', {
    method: 'POST',
    body: JSON.stringify({ name })
  }),
  // 对话
  deeds: (id) => request(`/api/deeds/${id}`),
  narrate: (payload) => request('/api/dialogue/narrate', {
    method: 'POST',
    body: JSON.stringify(payload)
  }),
  // 问答
  ragAsk: (question, topk = 3) => request('/api/rag/ask', {
    method: 'POST',
    body: JSON.stringify({ question, topk })
  }),
  // 对话（流式 SSE）：onDelta 逐字回调，onError 错误回调，onDone 完成回调
  chatStream: (payload, { onDelta, onError, onDone, signal } = {}) => {
    return fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal
    }).then(async (res) => {
      if (!res.ok) {
        let msg = `请求失败（${res.status}）`
        try {
          const d = await res.json()
          if (d && d.detail) msg = typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail)
        } catch (_) { /* 忽略非 JSON 响应体 */ }
        throw new Error(msg)
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      const handle = (line) => {
        if (!line.startsWith('data:')) return
        const data = line.slice(5).trim()
        if (!data) return
        let obj
        try { obj = JSON.parse(data) } catch (_) { return }
        if (obj.error) { onError && onError(new Error(obj.error)); return }
        if (obj.delta) onDelta && onDelta(obj.delta)
        if (obj.done) onDone && onDone()
      }
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        let idx
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const chunk = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          chunk.split('\n').forEach(handle)
        }
      }
      if (buf.trim()) buf.split('\n').forEach(handle)
    })
  }
}