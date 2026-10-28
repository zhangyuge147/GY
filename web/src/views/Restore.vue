<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { api } from '../api'

/* ---------- 模式切换：hero 英雄影像 / mine 我的照片 ---------- */
const mode = ref('hero')

/* ---------- 英雄影像（浏览模式） ---------- */
const heroes = ref([])
const selected = ref('')
const frames = ref([])

const loading = ref(true)       // 英雄列表加载
const loadingFrames = ref(true) // 六步中间图加载
const error = ref('')

const animIndex = ref(0)        // 当前展示帧下标
const playing = ref(false)
const sliderPos = ref(50)
let animTimer = null

const selectableHeroes = computed(() => heroes.value.filter((h) => h.frames_ready))
const pendingHeroes = computed(() => heroes.value.filter((h) => !h.frames_ready))
const selectedHero = computed(() => heroes.value.find((h) => h.id === selected.value))

/* ---------- 展示舞台（hero / mine 两种模式共用同一套动画与对比） ---------- */
// 过滤无 url 的帧：普通照片未选身份时不执行「考据校色」，该帧不产出，不展示空帧
const activeFrames = computed(() => {
  const src = mode.value === 'mine' ? myFrames.value : frames.value
  return src.filter((f) => f.url)
})
const currentFrame = computed(() => activeFrames.value[animIndex.value] || null)
const inputFrame = computed(() => activeFrames.value[0] || null)
const finalFrame = computed(() => activeFrames.value[activeFrames.value.length - 1] || null)

function stopAnim() {
  playing.value = false
  if (animTimer) { clearInterval(animTimer); animTimer = null }
}

function togglePlay() {
  if (playing.value) { stopAnim(); return }
  if (!activeFrames.value.length) return
  if (animIndex.value >= activeFrames.value.length - 1) animIndex.value = 0
  playing.value = true
  animTimer = setInterval(() => {
    if (animIndex.value >= activeFrames.value.length - 1) {
      stopAnim()
    } else {
      animIndex.value++
    }
  }, 650)
}

function jump(i) {
  stopAnim()
  animIndex.value = i
}

function resetAnimState() {
  stopAnim()
  animIndex.value = 0
  sliderPos.value = 50
}

function switchMode(m) {
  if (mode.value === m) return
  mode.value = m
  resetAnimState()
}

async function selectHero(id) {
  if (selected.value === id && frames.value.length) return
  selected.value = id
  stopAnim()
  animIndex.value = 0
  sliderPos.value = 50
  loadingFrames.value = true
  error.value = ''
  try {
    const data = await api.frames(id)
    frames.value = data.frames || []
  } catch (e) {
    error.value = e.message || '加载修复中间图失败'
    frames.value = []
  } finally {
    loadingFrames.value = false
  }
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const data = await api.heroes()
    heroes.value = data.heroes || []
    const first = heroes.value.find((h) => h.frames_ready)
    if (first) {
      await selectHero(first.id)
    } else {
      loadingFrames.value = false
    }
  } catch (e) {
    error.value = e.message || '加载失败，请确认后端已启动'
    loadingFrames.value = false
  } finally {
    loading.value = false
  }
}

/* ---------- 我的照片：上传 + 异步修复轮询 ---------- */
const fileInput = ref(null)
const pickedFile = ref(null)   // 选中的 File
const pickedUrl = ref('')      // 本地预览 URL
const heroPick = ref('')       // 照片人物身份（'' = 不确定/普通照片）
const promptPick = ref('')     // L45 修复偏好（'' = 默认；chip 单选，与后端关键词词典对齐）
const toneOptions = [
  { key: '', label: '默认' },
  { key: '暖调', label: '暖调' },
  { key: '冷调', label: '冷调' },
  { key: '怀旧', label: '怀旧' },
  { key: '鲜艳', label: '鲜艳' },
  { key: '淡雅', label: '淡雅' },
  { key: '保持黑白', label: '保持黑白' }
]
const uploadState = ref('idle') // idle | uploading | queued | running | done | failed
const uploadError = ref('')
const taskId = ref('')
const myFrames = ref([])       // 修复完成的 8 帧中间图
let pollTimer = null

const busy = computed(() =>
  uploadState.value === 'uploading' || uploadState.value === 'queued' || uploadState.value === 'running')

const uploadHint = computed(() => ({
  uploading: '正在上传照片…',
  queued: '已排队，即将开始修复…',
  running: 'AI 正在六步修复您的照片（约 1~3 分钟），请稍候…'
}[uploadState.value] || ''))

function onPickFile(e) {
  const f = e.target.files && e.target.files[0]
  if (!f) return
  resetResult()
  pickedFile.value = f
  if (pickedUrl.value) URL.revokeObjectURL(pickedUrl.value)
  pickedUrl.value = URL.createObjectURL(f)
}

function resetResult() {
  myFrames.value = []
  storyName.value = ''
  storyResult.value = null
  storyError.value = ''
}

function resetUpload() {
  stopPoll()
  uploadState.value = 'idle'
  uploadError.value = ''
  pickedFile.value = null
  promptPick.value = ''
  if (pickedUrl.value) { URL.revokeObjectURL(pickedUrl.value); pickedUrl.value = '' }
  if (fileInput.value) fileInput.value.value = ''
  resetResult()
}

function startUpload() {
  if (!pickedFile.value || busy.value) return
  uploadState.value = 'uploading'
  uploadError.value = ''
  resetResult()
  api.uploadRestore(pickedFile.value, heroPick.value || undefined, promptPick.value || undefined).then((d) => {
    taskId.value = d.task_id
    uploadState.value = d.status || 'queued'
    startPoll()
  }).catch((e) => {
    uploadState.value = 'failed'
    uploadError.value = e.message || '上传失败'
  })
}

function startPoll() {
  stopPoll()
  pollTimer = setInterval(async () => {
    try {
      const t = await api.restoreStatus(taskId.value)
      uploadState.value = t.status
      if (t.status === 'done') {
        stopPoll()
        myFrames.value = (t.result && t.result.frames) || []
        resetAnimState()
      } else if (t.status === 'failed') {
        stopPoll()
        uploadError.value = t.error || '修复失败'
      }
    } catch (_) { /* 轮询瞬时失败忽略，下个周期重试 */ }
  }, 2000)
}

function stopPoll() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

/* ---------- 照片背后的故事（仅史料库收录人物，未收录不编造） ---------- */
const storyName = ref('')
const storyLoading = ref(false)
const storyError = ref('')
const storyResult = ref(null)  // { found, name, story, citations }

function tellStory() {
  const n = storyName.value.trim()
  if (!n) { storyError.value = '请先填写照片人物姓名'; return }
  storyLoading.value = true
  storyError.value = ''
  storyResult.value = null
  api.photoStory(n).then((d) => {
    storyResult.value = d
  }).catch((e) => {
    storyError.value = e.message || '生成失败'
  }).finally(() => {
    storyLoading.value = false
  })
}

onMounted(load)
onUnmounted(() => {
  stopAnim()
  stopPoll()
  if (pickedUrl.value) URL.revokeObjectURL(pickedUrl.value)
})
</script>

<template>
  <div class="restore">
    <!-- 页头 -->
    <section class="page-head">
      <div class="container">
        <p class="page-head__kicker">六步 AI 修复</p>
        <h1 class="page-head__title">修复照片</h1>
        <p class="page-head__sub">选择一位英雄见证「苏醒」全程，或上传您的老照片，让记忆重新清晰。</p>
      </div>
    </section>

    <!-- 模式切换（胶囊按钮组） -->
    <section class="tabsbar">
      <div class="container">
        <div class="tabs" role="tablist">
          <button
            class="tabs__btn"
            :class="{ 'is-active': mode === 'hero' }"
            role="tab"
            :aria-selected="mode === 'hero'"
            @click="switchMode('hero')"
          >英雄影像</button>
          <button
            class="tabs__btn"
            :class="{ 'is-active': mode === 'mine' }"
            role="tab"
            :aria-selected="mode === 'mine'"
            @click="switchMode('mine')"
          >修复我的照片</button>
        </div>
      </div>
    </section>

    <!-- 英雄选择（仅英雄模式） -->
    <section v-if="mode === 'hero'" class="picker section">
      <div class="container">
        <div v-if="loading" class="state">
          <div class="spinner"></div>
          <p class="state__hint">正在加载英雄影像…</p>
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

          <p v-if="pendingHeroes.length" class="picker__pending">
            更多英雄影像整理中：
            <template v-for="(h, i) in pendingHeroes" :key="h.id">
              {{ h.name }}<span v-if="i < pendingHeroes.length - 1">、</span>
            </template>
          </p>
        </template>
      </div>
    </section>

    <!-- 上传（仅我的照片模式） -->
    <section v-if="mode === 'mine'" class="upload section">
      <div class="container">
        <div class="panel card upload__panel">
          <header class="panel__head">
            <h2>上传您的老照片</h2>
            <p class="panel__sub">支持 JPG / PNG，六步 AI 修复：去网纹 → 划痕补全 → 超分 → 人脸精修 → 上色 → 考据校色</p>
          </header>

          <!-- 文件输入框（常驻，供 ref 调用；放最前避免隔断 v-if/v-else 相邻关系） -->
          <input
            ref="fileInput"
            type="file"
            accept=".jpg,.jpeg,.png,.bmp,.webp"
            hidden
            @change="onPickFile"
          />

          <!-- 未选文件：点选区 -->
          <button v-if="!pickedFile" class="upload__drop" @click="fileInput && fileInput.click()">
            <span class="upload__plus">＋</span>
            <span class="upload__drop-text">点击选择照片</span>
            <span class="upload__drop-sub">照片仅用于本次修复演示，不上传至其他服务</span>
          </button>

          <template v-else>
            <div class="upload__preview-row">
              <img v-if="pickedUrl" :src="pickedUrl" alt="待修复照片预览" class="upload__preview" />
              <div class="upload__meta">
                <p class="upload__name">{{ pickedFile.name }}</p>
                <label class="upload__field">
                  <span>照片人物（可选）</span>
                  <select v-model="heroPick" class="upload__select">
                    <option value="">不确定 / 普通老照片</option>
                    <option v-for="h in heroes" :key="h.id" :value="h.id">{{ h.name }}（{{ h.years }}）</option>
                  </select>
                </label>
                <div class="upload__field">
                  <span>修复偏好（可选）</span>
                  <div class="upload__tones" role="radiogroup" aria-label="修复偏好">
                    <button
                      v-for="t in toneOptions"
                      :key="t.key"
                      type="button"
                      class="upload__tone"
                      :class="{ 'is-active': promptPick === t.key }"
                      role="radio"
                      :aria-checked="promptPick === t.key"
                      @click="promptPick = t.key"
                    >{{ t.label }}</button>
                  </div>
                </div>
                <p class="upload__tip">若照片人物是平台收录的英雄，选择身份后修复会按史料考据校色，色彩更准确。</p>
                <p class="upload__tip">偏好只影响整体色调氛围；选择身份修复时，以史料考据颜色为准。「保持黑白」= 只修复不上色。</p>
                <div class="upload__actions">
                  <button class="btn btn--primary" :disabled="busy" @click="startUpload">开始修复</button>
                  <button class="btn" :disabled="busy" @click="resetUpload">换一张</button>
                </div>
              </div>
            </div>

            <!-- 修复进度 -->
            <div v-if="uploadHint" class="upload__progress">
              <div class="spinner"></div>
              <p class="upload__progress-text">{{ uploadHint }}</p>
            </div>
            <div v-if="uploadState === 'done'" class="upload__ok">✓ 修复完成！下方可查看六步「苏醒」过程与前后对比。</div>
            <div v-if="uploadState === 'failed'" class="upload__fail">
              <p>修复失败：{{ uploadError || '未知错误' }}</p>
              <button class="btn btn--primary" @click="startUpload">重试</button>
            </div>
          </template>
        </div>
      </div>
    </section>

    <!-- 展示区：六步动画 + 前后对比（hero 恒显；mine 修复完成后显示） -->
    <section v-if="mode === 'hero' || activeFrames.length" class="stage section">
      <div class="container stage__grid">
        <!-- 左：六步苏醒动画 -->
        <div class="panel card">
          <header class="panel__head">
            <h2>六步修复 · 苏醒</h2>
            <p class="panel__sub">{{ mode === 'mine' ? '您的照片逐帧修复过程' : '逐帧展示 AI 修复的六道工序' }}</p>
          </header>

          <div v-if="mode === 'hero' && loadingFrames" class="state state--frame">
            <div class="spinner"></div>
            <p class="state__hint">正在加载修复中间图…</p>
          </div>
          <div v-else-if="mode === 'hero' && error" class="state state--frame">
            <p>{{ error }}</p>
            <button class="btn btn--primary" @click="selectHero(selected)">重试</button>
          </div>
          <div v-else-if="!currentFrame" class="state state--frame">
            <p>暂无修复中间图</p>
          </div>

          <template v-else>
            <div class="anim">
              <div class="anim__stage">
                <transition name="fade" mode="out-in">
                  <img
                    :key="currentFrame.url"
                    :src="currentFrame.url"
                    :alt="currentFrame.name"
                    class="anim__img"
                  />
                </transition>
                <span class="anim__badge">{{ animIndex + 1 }} / {{ activeFrames.length }} · {{ currentFrame.name }}</span>
              </div>

              <div class="anim__control">
                <button class="btn btn--primary" @click="togglePlay">
                  {{ playing ? '暂停' : '播放修复过程' }}
                </button>
              </div>

              <div class="anim__steps">
                <button
                  v-for="(f, i) in activeFrames"
                  :key="f.key"
                  class="anim__step"
                  :class="{ 'is-active': i === animIndex, 'is-done': i < animIndex }"
                  :aria-label="f.name"
                  @click="jump(i)"
                >
                  <span class="anim__dot"></span>
                  <span class="anim__label">{{ f.name }}</span>
                </button>
              </div>
            </div>
          </template>
        </div>

        <!-- 右：前后对比滑块 -->
        <div class="panel card">
          <header class="panel__head">
            <h2>前后对比</h2>
            <p class="panel__sub">左右拖动滑块，对比修复前与成片</p>
          </header>

          <div v-if="mode === 'hero' && loadingFrames" class="state state--frame">
            <div class="spinner"></div>
            <p class="state__hint">正在加载对比图…</p>
          </div>
          <div v-else-if="!inputFrame || !finalFrame" class="state state--frame">
            <p>暂无对比图</p>
          </div>

          <div v-else class="compare">
            <img :src="finalFrame.url" alt="修复后" class="compare__img" />
            <img
              :src="inputFrame.url"
              alt="修复前"
              class="compare__img compare__img--before"
              :style="{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }"
            />
            <span class="compare__tag compare__tag--before">修复前</span>
            <span class="compare__tag compare__tag--after">修复后</span>
            <div class="compare__divider" :style="{ left: sliderPos + '%' }">
              <span class="compare__grip">‹ ›</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="0.5"
              v-model.number="sliderPos"
              class="compare__range"
              aria-label="拖动滑块对比修复前后"
            />
          </div>

          <p v-if="mode === 'hero' && selectedHero" class="panel__foot">
            {{ selectedHero.name }} · {{ selectedHero.title }}
          </p>
          <p v-else-if="mode === 'mine'" class="panel__foot">
            您的照片 · AI 修复成片
          </p>
        </div>
      </div>
    </section>

    <!-- 照片背后的故事（仅史料库收录人物，未收录不编造） -->
    <section v-if="mode === 'mine' && myFrames.length" class="story section">
      <div class="container">
        <div class="panel card">
          <header class="panel__head">
            <h2>照片背后的故事</h2>
            <p class="panel__sub">告诉我们照片上的人是谁——平台只讲有史料依据的故事，不编造内容</p>
          </header>

          <div class="story__form">
            <input
              v-model="storyName"
              list="hero-names"
              class="story__input"
              placeholder="照片人物姓名，如：杨靖宇"
              @keyup.enter="tellStory"
            />
            <datalist id="hero-names">
              <option v-for="h in heroes" :key="h.id" :value="h.name" />
            </datalist>
            <button class="btn btn--primary" :disabled="storyLoading" @click="tellStory">
              {{ storyLoading ? '正在讲述…' : '讲讲 TA 的故事' }}
            </button>
          </div>

          <p v-if="storyError" class="story__error">{{ storyError }}</p>

          <div v-if="storyLoading" class="state state--story">
            <div class="spinner"></div>
            <p class="state__hint">正在依据史料为您讲述…</p>
          </div>

          <template v-else-if="storyResult">
            <!-- 命中史料库 -->
            <div v-if="storyResult.found" class="story__result">
              <div class="story__notice">以下内容依据平台史料库整理，由 AI 讲述。</div>
              <p class="story__text">{{ storyResult.story }}</p>
              <div v-if="Object.keys(storyResult.citations || {}).length" class="story__sources">
                <p class="story__sources-title">史料出处：</p>
                <p v-for="(s, n) in storyResult.citations" :key="n" class="story__source">[{{ n }}] {{ s }}</p>
              </div>
            </div>
            <!-- 未收录：明确告知，不编造 -->
            <div v-else class="story__miss">
              <p class="story__miss-title">史料库暂未收录「{{ storyResult.name }}」</p>
              <p class="story__miss-sub">平台只讲有史料依据的故事，不编造内容。您可以核对姓名后重试，或前往「史料检索」页了解更多。</p>
            </div>
          </template>
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
.page-head__kicker {
  font-size: var(--fs-sm);
  letter-spacing: 0.18em;
  color: var(--c-gold);
}
.page-head__title {
  font-size: clamp(28px, 4.5vw, 44px);
  color: #fff;
  margin-top: var(--sp-1);
}
.page-head__sub {
  margin-top: var(--sp-2);
  color: rgba(246, 239, 228, 0.9);
  max-width: 640px;
}

.section { padding: var(--sp-8) 0; }

/* 模式切换（胶囊按钮组，居中醒目） */
.tabsbar {
  background: var(--c-card);
  border-bottom: 1px solid var(--c-line);
  padding: var(--sp-4) 0;
}
.tabs {
  display: flex;
  justify-content: center;
  gap: var(--sp-2);
  padding: var(--sp-1);
  background: var(--c-red-weak);
  border-radius: 999px;
  max-width: fit-content;
  margin: 0 auto;
}
.tabs__btn {
  padding: var(--sp-2) var(--sp-6);
  background: transparent;
  border: none;
  border-radius: 999px;
  cursor: pointer;
  font-size: var(--fs-base);
  color: var(--c-ink-soft);
  transition: color var(--t-fast), background var(--t-fast), box-shadow var(--t-fast);
}
.tabs__btn:hover { color: var(--c-red); }
.tabs__btn.is-active {
  background: var(--c-red);
  color: #fff;
  font-weight: 600;
  box-shadow: var(--shadow-1);
}

/* 英雄选择 */
.picker__list {
  display: flex;
  flex-wrap: wrap;
  gap: var(--sp-3);
}
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
  transition: border-color var(--t-fast), background var(--t-fast), color var(--t-fast), box-shadow var(--t-fast);
}
.picker__chip:hover { border-color: var(--c-red); }
.picker__chip.is-active {
  border-color: var(--c-red);
  background: var(--c-red-weak);
  box-shadow: var(--shadow-1);
}
.picker__name { font-family: var(--font-title); font-weight: 700; color: var(--c-ink); }
.picker__chip.is-active .picker__name { color: var(--c-red); }
.picker__years { font-size: var(--fs-xs); color: var(--c-ink-faint); }
.picker__pending {
  margin-top: var(--sp-5);
  font-size: var(--fs-sm);
  color: var(--c-ink-soft);
}

/* 展示区 */
.stage__grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--sp-6);
  align-items: start;
}

.panel { padding: var(--sp-6); }
.panel__head { margin-bottom: var(--sp-5); }
.panel__head h2 { font-size: var(--fs-xl); }
.panel__sub { color: var(--c-ink-soft); font-size: var(--fs-sm); margin-top: var(--sp-1); }
.panel__foot { margin-top: var(--sp-4); font-size: var(--fs-sm); color: var(--c-ink-soft); text-align: center; }

.state--frame { min-height: 320px; }
.state--story { min-height: 120px; }

/* 上传 */
.upload__panel { max-width: 760px; margin: 0 auto; }
.upload__drop {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--sp-2);
  padding: var(--sp-12) var(--sp-6);
  border: 2px dashed var(--c-line);
  border-radius: var(--radius-lg);
  background: var(--c-paper);
  cursor: pointer;
  transition: border-color var(--t-fast), background var(--t-fast);
}
.upload__drop:hover { border-color: var(--c-red); background: var(--c-red-weak); }
.upload__plus { font-size: 40px; line-height: 1; color: var(--c-red); }
.upload__drop-text { font-size: var(--fs-lg); font-weight: 600; color: var(--c-ink); }
.upload__drop-sub { font-size: var(--fs-sm); color: var(--c-ink-faint); }

.upload__preview-row {
  display: flex;
  gap: var(--sp-5);
  align-items: flex-start;
}
.upload__preview {
  width: 180px;
  aspect-ratio: 3 / 4;
  object-fit: contain;
  background: #efe8db;
  border: 1px solid var(--c-line);
  border-radius: var(--radius);
}
.upload__meta {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--sp-3);
  min-width: 0;
}
.upload__name { font-weight: 600; word-break: break-all; }
.upload__field {
  display: flex;
  flex-direction: column;
  gap: var(--sp-1);
  max-width: 320px;
  font-size: var(--fs-sm);
  color: var(--c-ink-soft);
}
.upload__select {
  padding: var(--sp-2) var(--sp-3);
  border: 1px solid var(--c-line);
  border-radius: var(--radius);
  background: #fff;
  font-size: var(--fs-base);
  color: var(--c-ink);
}
.upload__tip { font-size: var(--fs-xs); color: var(--c-ink-faint); }
.upload__actions { display: flex; gap: var(--sp-3); }

/* L45 修复偏好 chip 组（单选，风格与模式切换胶囊一致） */
.upload__tones {
  display: flex;
  flex-wrap: wrap;
  gap: var(--sp-2);
}
.upload__tone {
  padding: var(--sp-1) var(--sp-3);
  background: #fff;
  border: 1px solid var(--c-line);
  border-radius: 999px;
  cursor: pointer;
  font-size: var(--fs-sm);
  color: var(--c-ink-soft);
  transition: color var(--t-fast), background var(--t-fast), border-color var(--t-fast);
}
.upload__tone:hover { border-color: var(--c-red); color: var(--c-red); }
.upload__tone.is-active {
  background: var(--c-red);
  border-color: var(--c-red);
  color: #fff;
  font-weight: 600;
}

.upload__progress {
  display: flex;
  align-items: center;
  gap: var(--sp-3);
  margin-top: var(--sp-5);
  padding: var(--sp-4);
  background: var(--c-red-weak);
  border-radius: var(--radius);
}
.upload__progress-text { color: var(--c-red-dark); font-size: var(--fs-sm); }
.upload__ok {
  margin-top: var(--sp-5);
  padding: var(--sp-3) var(--sp-4);
  background: #f0f7ee;
  border: 1px solid #b7d4ae;
  color: #2f6b2a;
  border-radius: var(--radius);
  font-size: var(--fs-sm);
}
.upload__fail {
  margin-top: var(--sp-5);
  padding: var(--sp-3) var(--sp-4);
  background: #fdf0ee;
  border: 1px solid #e8b9ae;
  color: #8c2f1d;
  border-radius: var(--radius);
  font-size: var(--fs-sm);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--sp-3);
  flex-wrap: wrap;
}

/* 照片背后的故事 */
.story .container { max-width: 900px; }
.story__form {
  display: flex;
  gap: var(--sp-3);
  flex-wrap: wrap;
}
.story__input {
  flex: 1;
  min-width: 220px;
  padding: var(--sp-3) var(--sp-4);
  border: 1px solid var(--c-line);
  border-radius: var(--radius);
  font-size: var(--fs-base);
  color: var(--c-ink);
  background: #fff;
}
.story__input:focus {
  outline: none;
  border-color: var(--c-red);
}
.story__error {
  margin-top: var(--sp-3);
  color: #8c2f1d;
  font-size: var(--fs-sm);
}

.story__notice {
  margin-top: var(--sp-5);
  padding: var(--sp-2) var(--sp-4);
  background: #fbf3dc;
  border-left: 3px solid var(--c-gold);
  color: #7a5b12;
  border-radius: var(--radius);
  font-size: var(--fs-sm);
}
.story__text {
  margin-top: var(--sp-4);
  white-space: pre-wrap;
  line-height: 1.9;
  color: var(--c-ink);
}
.story__sources {
  margin-top: var(--sp-5);
  padding: var(--sp-4);
  background: var(--c-paper);
  border-radius: var(--radius);
}
.story__sources-title {
  font-weight: 600;
  font-size: var(--fs-sm);
  margin-bottom: var(--sp-2);
}
.story__source {
  font-size: var(--fs-sm);
  color: var(--c-ink-soft);
  line-height: 1.7;
  word-break: break-all;
}
.story__miss {
  margin-top: var(--sp-5);
  padding: var(--sp-6) var(--sp-4);
  text-align: center;
  background: var(--c-paper);
  border-radius: var(--radius);
}
.story__miss-title { font-weight: 600; }
.story__miss-sub {
  margin-top: var(--sp-2);
  font-size: var(--fs-sm);
  color: var(--c-ink-soft);
}

/* 六步动画 */
.anim { display: flex; flex-direction: column; gap: var(--sp-4); }
.anim__stage {
  position: relative;
  aspect-ratio: 3 / 4;
  background: #efe8db;
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.anim__img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.anim__badge {
  position: absolute;
  left: var(--sp-3);
  bottom: var(--sp-3);
  padding: var(--sp-1) var(--sp-3);
  background: rgba(35, 18, 14, 0.72);
  color: #fff;
  font-size: var(--fs-sm);
  border-radius: 999px;
  letter-spacing: 0.04em;
  z-index: 2;
}
.anim__control { display: flex; justify-content: center; }

.anim__steps {
  display: flex;
  justify-content: space-between;
  gap: var(--sp-1);
}
.anim__step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--sp-1);
  background: transparent;
  border: none;
  cursor: pointer;
  padding: 0;
  flex: 1;
}
.anim__dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--c-line);
  transition: background var(--t-fast), transform var(--t-fast);
}
.anim__step.is-active .anim__dot { background: var(--c-red); transform: scale(1.25); }
.anim__step.is-done .anim__dot { background: var(--c-gold); }
.anim__label {
  font-size: var(--fs-xs);
  color: var(--c-ink-faint);
  white-space: nowrap;
  transition: color var(--t-fast);
}
.anim__step.is-active .anim__label { color: var(--c-red); font-weight: 600; }

.fade-enter-active, .fade-leave-active { transition: opacity var(--t-base); }
.fade-enter-from, .fade-leave-to { opacity: 0; }

/* 前后对比 */
.compare {
  position: relative;
  aspect-ratio: 3 / 4;
  background: #efe8db;
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.compare__img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.compare__img--before { z-index: 1; }
.compare__tag {
  position: absolute;
  top: var(--sp-3);
  padding: var(--sp-1) var(--sp-3);
  font-size: var(--fs-sm);
  color: #fff;
  background: rgba(35, 18, 14, 0.72);
  border-radius: 999px;
  z-index: 2;
  pointer-events: none;
}
.compare__tag--before { left: var(--sp-3); }
.compare__tag--after { right: var(--sp-3); }
.compare__divider {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #fff;
  z-index: 2;
  transform: translateX(-50%);
  box-shadow: 0 0 8px rgba(0, 0, 0, 0.45);
  pointer-events: none;
}
.compare__grip {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  white-space: nowrap;
  font-size: var(--fs-sm);
  color: #fff;
  background: var(--c-red);
  padding: var(--sp-1) var(--sp-2);
  border-radius: 999px;
  box-shadow: var(--shadow-1);
}
.compare__range {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  opacity: 0;
  cursor: ew-resize;
  z-index: 3;
}

@media (max-width: 768px) {
  .stage__grid { grid-template-columns: 1fr; }
  .anim__label { display: none; }
  .upload__preview-row { flex-direction: column; }
}
</style>
