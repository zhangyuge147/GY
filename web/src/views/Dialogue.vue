<script setup>
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { api } from '../api'

const heroes = ref([])
const selected = ref('')
const info = ref('')
const deeds = ref([])
const sources = ref([])

const loading = ref(true)        // 英雄列表加载
const loadingDeeds = ref(false)  // 事迹卡加载
const error = ref('')

const audioUrl = ref('')
const captions = ref('')
const narrating = ref(false)     // 等待 TTS 合成
const playingText = ref('')      // 当前播放事迹条目（用于高亮）
const audioEl = ref(null)

const selectableHeroes = computed(() => heroes.value.filter((h) => h.has_portrait))
const selectedHero = computed(() => heroes.value.find((h) => h.id === selected.value))

function stopAudio() {
  const el = audioEl.value
  if (el) { el.pause(); el.currentTime = 0 }
  audioUrl.value = ''
  captions.value = ''
  playingText.value = ''
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const data = await api.heroes()
    heroes.value = data.heroes || []
    const first = heroes.value.find((h) => h.has_portrait)
    if (first) await selectHero(first.id)
  } catch (e) {
    error.value = e.message || '加载失败，请确认后端已启动'
  } finally {
    loading.value = false
  }
}

async function selectHero(id) {
  if (selected.value === id && deeds.value.length) return
  selected.value = id
  stopAudio()
  loadingDeeds.value = true
  error.value = ''
  try {
    const data = await api.deeds(id)
    info.value = data.info || ''
    deeds.value = data.deeds || []
    sources.value = data.sources || []
  } catch (e) {
    error.value = e.message || '加载史料失败'
    deeds.value = []
    sources.value = []
  } finally {
    loadingDeeds.value = false
  }
}

async function playItem(deedTitle, item) {
  if (!selected.value) return
  playingText.value = item.text
  narrating.value = true
  error.value = ''
  captions.value = ''
  try {
    const res = await api.narrate({
      hero_id: selected.value,
      deed_title: deedTitle,
      item_text: item.text
    })
    audioUrl.value = res.audio_url
    captions.value = res.captions || ''
  } catch (e) {
    error.value = e.message || '讲解生成失败'
    playingText.value = ''
  } finally {
    narrating.value = false
  }
}

// 音频就绪后尝试自动播放（用户点击触发，多数浏览器放行）
watch(audioUrl, async (v) => {
  if (!v) return
  await nextTick()
  const el = audioEl.value
  if (el) el.play().catch(() => { /* 被浏览器拦截则交由用户手动播放 */ })
})

onMounted(load)
</script>

<template>
  <div class="dialogue">
    <!-- 页头 -->
    <section class="page-head">
      <div class="container">
        <p class="page-head__kicker">AI 数字人讲解</p>
        <h1 class="page-head__title">时空对话</h1>
        <p class="page-head__sub">聆听英雄事迹，讲解语音与影像均为 AI 演绎，历史内容以权威史料为准。</p>
      </div>
    </section>

    <!-- 英雄选择 -->
    <section class="picker section">
      <div class="container">
        <div v-if="loading" class="state">
          <div class="spinner"></div>
          <p class="state__hint">正在加载英雄…</p>
        </div>

        <div v-else-if="error && !heroes.length" class="state">
          <p>{{ error }}</p>
          <button class="btn btn--primary" @click="load">重试</button>
        </div>

        <template v-else>
          <div class="picker__list">
            <button
              v-for="h in selectableHeroes"
              :key="h.id"
              class="picker__chip"
              :class="{ 'is-active': h.id === selected }"
              @click="selectHero(h.id)"
            >
              <span class="picker__name">{{ h.name }}</span>
              <span class="picker__years">{{ h.years }}</span>
            </button>
          </div>
        </template>
      </div>
    </section>

    <!-- 讲解画面 + 事迹卡 -->
    <section class="stage section">
      <div class="container stage__grid">
        <!-- 左：讲解画面 + 字幕 -->
        <div class="panel card">
          <header class="panel__head">
            <h2>讲解画面</h2>
            <p class="panel__sub">以下影像为 AI 修复演绎</p>
          </header>

          <div class="portrait">
            <img
              v-if="selectedHero && selectedHero.portrait_url"
              :src="selectedHero.portrait_url"
              :alt="selectedHero.name"
              class="portrait__img"
            />
            <span class="portrait__tag">AI 演绎</span>
          </div>

          <div class="subtitle">
            <div v-if="narrating" class="subtitle__loading">
              <span class="spinner"></span>
              <span>正在合成讲解语音…</span>
            </div>
            <p v-else-if="captions" class="subtitle__text">{{ captions }}</p>
            <p v-else class="subtitle__hint">点击右侧「播放讲解」，聆听英雄的故事。</p>
          </div>

          <audio ref="audioEl" :src="audioUrl" controls class="player" controlslist="nodownload"></audio>
        </div>

        <!-- 右：事迹卡 -->
        <div class="panel card">
          <header class="panel__head">
            <h2>事迹卡</h2>
            <p v-if="selectedHero" class="panel__sub">{{ selectedHero.name }} · {{ selectedHero.title }}</p>
          </header>

          <div v-if="loadingDeeds" class="state state--deeds">
            <div class="spinner"></div>
            <p class="state__hint">正在加载事迹…</p>
          </div>
          <div v-else-if="error && !deeds.length" class="state state--deeds">
            <p>{{ error }}</p>
            <button class="btn btn--primary" @click="selectHero(selected)">重试</button>
          </div>
          <div v-else-if="!deeds.length" class="state state--deeds">
            <p>暂无事迹史料</p>
          </div>

          <div v-else class="deeds">
            <article v-for="deed in deeds" :key="deed.title" class="deed">
              <h3 class="deed__title">{{ deed.title }}</h3>
              <ul class="deed__list">
                <li
                  v-for="item in deed.items"
                  :key="item.text"
                  class="deed__item"
                  :class="{ 'is-playing': playingText === item.text }"
                >
                  <p class="deed__text">{{ item.text }}</p>
                  <button class="btn btn--ghost deed__play" @click="playItem(deed.title, item)">
                    {{ narrating && playingText === item.text ? '合成中…' : '播放讲解' }}
                  </button>
                </li>
              </ul>
            </article>
          </div>
        </div>
      </div>
    </section>

    <!-- 引用出处 -->
    <section class="sources section">
      <div class="container">
        <div class="panel card">
          <header class="panel__head">
            <h2>引用出处</h2>
            <p class="panel__sub">讲解内容均出自以下权威史料</p>
          </header>
          <ol v-if="sources.length" class="sources__list">
            <li v-for="(s, i) in sources" :key="i">{{ s }}</li>
          </ol>
          <p v-else class="state__hint">暂无出处</p>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
/* 页头 */
.page-head {
  padding: var(--sp-12) 0 var(--sp-8);
  background: linear-gradient(180deg, #7f0f22 0%, #8c1424 100%);
  color: #f6efe4;
}
.page-head__kicker { font-size: var(--fs-sm); letter-spacing: 0.18em; color: var(--c-gold); }
.page-head__title { font-size: clamp(28px, 4.5vw, 44px); color: #fff; margin-top: var(--sp-1); }
.page-head__sub { margin-top: var(--sp-2); color: rgba(246, 239, 228, 0.9); max-width: 640px; }

.section { padding: var(--sp-8) 0; }

/* 英雄选择 */
.picker__list { display: flex; flex-wrap: wrap; gap: var(--sp-3); }
.picker__chip {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: var(--sp-1);
  min-width: 96px;
  padding: var(--sp-3) var(--sp-6);
  background: var(--c-card);
  border: 1px solid var(--c-line);
  border-radius: var(--radius);
  cursor: pointer;
  transition: border-color var(--t-fast), background var(--t-fast), box-shadow var(--t-fast);
}
.picker__chip:hover { border-color: var(--c-red); }
.picker__chip.is-active { border-color: var(--c-red); background: var(--c-red-weak); box-shadow: var(--shadow-1); }
.picker__name { font-family: var(--font-title); font-weight: 700; color: var(--c-ink); }
.picker__chip.is-active .picker__name { color: var(--c-red); }
.picker__years { font-size: var(--fs-xs); color: var(--c-ink-faint); }

/* 主体两栏 */
.stage__grid {
  display: grid;
  grid-template-columns: 5fr 7fr;
  gap: var(--sp-6);
  align-items: start;
}
.panel { padding: var(--sp-6); }
.panel__head { margin-bottom: var(--sp-5); }
.panel__head h2 { font-size: var(--fs-xl); }
.panel__sub { color: var(--c-ink-soft); font-size: var(--fs-sm); margin-top: var(--sp-1); }
.state--deeds { min-height: 260px; }

/* 讲解画面 */
.portrait {
  position: relative;
  aspect-ratio: 3 / 4;
  background: #efe8db;
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.portrait__img { width: 100%; height: 100%; object-fit: contain; }
.portrait__tag {
  position: absolute;
  right: var(--sp-3);
  top: var(--sp-3);
  padding: var(--sp-1) var(--sp-3);
  background: rgba(35, 18, 14, 0.72);
  color: #fff;
  font-size: var(--fs-sm);
  border-radius: 999px;
}

.subtitle {
  margin-top: var(--sp-4);
  min-height: 88px;
  padding: var(--sp-4);
  background: var(--c-red-weak);
  border-left: 3px solid var(--c-gold);
  border-radius: var(--radius);
  font-size: var(--fs-sm);
  line-height: 1.8;
  color: var(--c-ink);
}
.subtitle__loading { display: flex; align-items: center; gap: var(--sp-3); color: var(--c-ink-soft); }
.subtitle__hint { color: var(--c-ink-faint); }
.player { width: 100%; margin-top: var(--sp-4); }

/* 事迹卡 */
.deeds { display: flex; flex-direction: column; gap: var(--sp-5); }
.deed__title {
  font-size: var(--fs-lg);
  padding-bottom: var(--sp-2);
  margin-bottom: var(--sp-3);
  border-bottom: 1px solid var(--c-line);
}
.deed__list { list-style: none; display: flex; flex-direction: column; gap: var(--sp-3); }
.deed__item {
  display: flex;
  flex-direction: column;
  gap: var(--sp-2);
  padding: var(--sp-3);
  border-radius: var(--radius);
  transition: background var(--t-fast);
}
.deed__item.is-playing { background: var(--c-red-weak); }
.deed__text { font-size: var(--fs-base); line-height: 1.8; }
.deed__play { align-self: flex-start; padding: var(--sp-2) var(--sp-4); font-size: var(--fs-sm); }

/* 引用出处 */
.sources__list {
  margin: 0;
  padding-left: var(--sp-5);
  display: flex;
  flex-direction: column;
  gap: var(--sp-2);
  color: var(--c-ink-soft);
  font-size: var(--fs-sm);
}

@media (max-width: 768px) {
  .stage__grid { grid-template-columns: 1fr; }
}
</style>