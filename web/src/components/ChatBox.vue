<script setup>
import { ref, nextTick, onMounted, onUnmounted } from 'vue'
import { api } from '../api'
import assistantAvatar from '../assets/assistant-avatar.jpg'

const personas = ref([{ id: 'general', name: '通用助手' }])
const persona = ref('general')
const messages = ref([])
const input = ref('')
const streaming = ref(false)
const scrollEl = ref(null)
let abortCtrl = null

async function loadPersonas() {
  try {
    const data = await api.heroes()
    for (const h of (data.heroes || [])) {
      if (h.has_portrait) personas.value.push({ id: h.id, name: h.name })
    }
  } catch (_) { /* 后端未就绪时仅保留通用助手 */ }
}

function scrollToBottom() {
  nextTick(() => {
    if (scrollEl.value) scrollEl.value.scrollTop = scrollEl.value.scrollHeight
  })
}

async function send() {
  const text = input.value.trim()
  if (!text || streaming.value) return
  input.value = ''
  messages.value.push({ role: 'user', content: text })
  messages.value.push({ role: 'assistant', content: '' })
  const last = messages.value[messages.value.length - 1]
  const history = messages.value.slice(0, -1)
  streaming.value = true
  abortCtrl = new AbortController()
  scrollToBottom()
  try {
    await api.chatStream(
      { persona: persona.value, messages: history },
      {
        onDelta: (d) => { last.content += d; scrollToBottom() },
        onError: (e) => {
          last.content = last.content || ''
          last.content += (last.content ? '\n' : '') + `（${e.message}）`
          scrollToBottom()
        },
        onDone: () => scrollToBottom(),
        signal: abortCtrl.signal
      }
    )
  } catch (e) {
    if (e.name !== 'AbortError' && !last.content) {
      last.content = e.message || '对话失败，请稍后再试'
    }
  } finally {
    streaming.value = false
    abortCtrl = null
    scrollToBottom()
  }
}

function stop() {
  if (abortCtrl) abortCtrl.abort()
}

function switchPersona(id) {
  if (streaming.value) stop()
  persona.value = id
  messages.value = [
    { role: 'assistant', content: welcomeFor(id) }
  ]
}

function welcomeFor(id) {
  const p = personas.value.find((x) => x.id === id)
  const name = p ? (p.id === 'general' ? '智能问答助手' : p.name) : '智能问答助手'
  return `你好，我是${name}。有什么想了解的，尽管问我。`
}

onMounted(async () => {
  await loadPersonas()
  messages.value = [{ role: 'assistant', content: welcomeFor('general') }]
})
onUnmounted(stop)
</script>

<template>
  <div class="chatbox">
    <!-- 人设切换 -->
    <div class="chatbox__personas">
      <button
        v-for="p in personas"
        :key="p.id"
        class="persona"
        :class="{ 'is-active': p.id === persona }"
        :disabled="streaming"
        @click="switchPersona(p.id)"
      >{{ p.name }}</button>
    </div>

    <!-- 消息列表 -->
    <div ref="scrollEl" class="chatbox__messages">
      <div
        v-for="(m, i) in messages"
        :key="i"
        class="msg"
        :class="`msg--${m.role}`"
      >
        <img
          v-if="m.role === 'assistant'"
          class="msg__avatar"
          :src="assistantAvatar"
          alt="小助手头像"
        />
        <div class="msg__bubble">{{ m.content }}</div>
      </div>
    </div>

    <!-- 输入 -->
    <form class="chatbox__input" @submit.prevent="send">
      <input
        v-model="input"
        class="chatbox__field"
        type="text"
        placeholder="输入问题，Enter 发送"
        :disabled="streaming"
      />
      <button v-if="!streaming" class="btn btn--primary chatbox__send" :disabled="!input.trim()">
        发送
      </button>
      <button v-else type="button" class="btn btn--ghost chatbox__send" @click="stop">停止</button>
    </form>
  </div>
</template>

<style scoped>
.chatbox {
  display: flex;
  flex-direction: column;
  gap: var(--sp-3);
  height: 100%;
}

.chatbox__personas {
  display: flex;
  flex-wrap: wrap;
  gap: var(--sp-2);
  padding-bottom: var(--sp-2);
  border-bottom: 1px solid var(--c-line);
}
.persona {
  padding: var(--sp-1) var(--sp-3);
  border: 1px solid var(--c-line);
  border-radius: 999px;
  background: var(--c-card);
  color: var(--c-ink-soft);
  font-size: var(--fs-sm);
  cursor: pointer;
  transition: border-color var(--t-fast), background var(--t-fast), color var(--t-fast);
}
.persona:hover { border-color: var(--c-red); color: var(--c-red); }
.persona.is-active {
  border-color: var(--c-red);
  background: var(--c-red-weak);
  color: var(--c-red);
  font-weight: 600;
}
.persona:disabled { opacity: 0.5; cursor: not-allowed; }

.chatbox__messages {
  flex: 1;
  height: 340px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: var(--sp-3);
  padding: var(--sp-2) var(--sp-1);
}
.msg { display: flex; }
.msg--user { justify-content: flex-end; }
.msg--assistant { justify-content: flex-start; align-items: flex-start; gap: var(--sp-2); }
.msg__avatar {
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  border-radius: 50%;
  object-fit: cover;
  border: 1px solid var(--c-line);
}
.msg__bubble {
  max-width: 82%;
  padding: var(--sp-2) var(--sp-4);
  border-radius: var(--radius);
  font-size: var(--fs-base);
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}
.msg--user .msg__bubble {
  background: var(--c-red);
  color: #fff;
  border-top-right-radius: 4px;
}
.msg--assistant .msg__bubble {
  background: var(--c-card);
  border: 1px solid var(--c-line);
  color: var(--c-ink);
  border-top-left-radius: 4px;
}

.chatbox__input { display: flex; gap: var(--sp-2); padding-top: var(--sp-2); border-top: 1px solid var(--c-line); }
.chatbox__field {
  flex: 1;
  padding: var(--sp-2) var(--sp-4);
  border: 1px solid var(--c-line);
  border-radius: var(--radius);
  font-size: var(--fs-base);
  font-family: var(--font-body);
  color: var(--c-ink);
  background: var(--c-card);
  transition: border-color var(--t-fast), box-shadow var(--t-fast);
}
.chatbox__field:focus {
  outline: none;
  border-color: var(--c-red);
  box-shadow: 0 0 0 3px var(--c-red-weak);
}
.chatbox__send { flex-shrink: 0; }
</style>