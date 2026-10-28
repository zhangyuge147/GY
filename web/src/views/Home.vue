<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api'
import heroTexture from '../assets/hero-texture.png'
import archiveBg from '../assets/archive-placeholder.png'

const router = useRouter()
const heroes = ref([])
const loading = ref(true)
const error = ref('')
const current = ref(0)
const featureEl = ref(null)
let timer = null

const featured = computed(() => heroes.value.filter((h) => h.has_portrait))
const pending = computed(() => heroes.value.filter((h) => !h.has_portrait))

async function load() {
  loading.value = true
  error.value = ''
  try {
    const data = await api.heroes()
    heroes.value = data.heroes || []
  } catch (e) {
    error.value = e.message || '加载失败，请确认后端已启动'
  } finally {
    loading.value = false
  }
}

function next() {
  if (featured.value.length) current.value = (current.value + 1) % featured.value.length
}
function prev() {
  if (featured.value.length) current.value = (current.value - 1 + featured.value.length) % featured.value.length
}
function go(i) { current.value = i; restartAuto() }

function startAuto() { if (!timer) timer = setInterval(next, 5000) }
function stopAuto() { if (timer) { clearInterval(timer); timer = null } }
function restartAuto() { stopAuto(); startAuto() }

function scrollToFeature() { featureEl.value?.scrollIntoView({ behavior: 'smooth' }) }

onMounted(() => { load(); startAuto() })
onUnmounted(stopAuto)
</script>

<template>
  <div class="home">
    <!-- 首屏引导 -->
    <section class="hero" :style="{ backgroundImage: `url(${heroTexture})` }">
      <div class="container hero__inner">
        <p class="hero__kicker">AI+教育 · 红色文化数字化</p>
        <h1 class="hero__title">想再看一次<br />英雄的脸吗</h1>
        <p class="hero__sub">用人工智能修复泛黄的老照片，让英雄的影像与故事重新鲜活。</p>
        <div class="hero__actions">
          <button class="btn btn--gold" @click="router.push('/restore')">开始体验</button>
          <button class="btn btn--ghost btn--ghost-light" @click="scrollToFeature">了解英雄</button>
        </div>
      </div>
    </section>

    <!-- 精选案例轮播 -->
    <section ref="featureEl" class="feature" :style="{ backgroundImage: `url(${archiveBg})` }">
      <div class="container">
        <header class="section-head">
          <h2>精选英雄影像</h2>
          <p class="section-head__sub">六步 AI 修复，让历史鲜活可见</p>
        </header>

        <div v-if="loading" class="feature__state">
          <div class="spinner"></div>
          <p class="state__hint">正在加载英雄影像…</p>
        </div>

        <div v-else-if="error" class="state">
          <p>{{ error }}</p>
          <button class="btn btn--primary" @click="load">重试</button>
        </div>

        <div v-else-if="featured.length === 0" class="state">
          <p>暂无可用影像</p>
        </div>

        <div v-else class="carousel">
          <button class="carousel__arrow carousel__arrow--prev" aria-label="上一个" @click="prev(); restartAuto()">‹</button>

          <transition name="fade" mode="out-in">
            <article :key="featured[current].id" class="hero-card card">
              <div class="hero-card__media">
                <img :src="featured[current].portrait_url" :alt="featured[current].name" loading="lazy" />
              </div>
              <div class="hero-card__body">
                <h3 class="hero-card__name">{{ featured[current].name }}</h3>
                <p class="hero-card__meta">{{ featured[current].years }} · {{ featured[current].title }}</p>
                <div class="hero-card__actions">
                  <button class="btn btn--primary" @click="router.push('/dialogue')">进入时空对话</button>
                  <button class="btn btn--ghost" @click="router.push('/restore')">去修复照片</button>
                </div>
              </div>
            </article>
          </transition>

          <button class="carousel__arrow carousel__arrow--next" aria-label="下一个" @click="next(); restartAuto()">›</button>

          <div class="carousel__dots">
            <button
              v-for="(h, i) in featured"
              :key="h.id"
              class="carousel__dot"
              :class="{ 'is-active': i === current }"
              :aria-label="h.name"
              @click="go(i)"
            ></button>
          </div>
        </div>

        <div v-if="pending.length" class="pending">
          <span class="pending__label">更多英雄影像整理中</span>
          <div class="pending__list">
            <div v-for="h in pending" :key="h.id" class="pending__item">
              <span class="pending__name">{{ h.name }}</span>
              <span class="pending__years">{{ h.years }}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 三大特性 -->
    <section class="highlights">
      <div class="container">
        <div class="grid">
          <div class="item card">
            <span class="item__no">01</span>
            <h4>六步 AI 修复</h4>
            <p>去网纹 → 补全 → 超分 → 人脸精修 → 上色 → 考据校色</p>
          </div>
          <div class="item card">
            <span class="item__no">02</span>
            <h4>史实有据可依</h4>
            <p>史料检索问答，答案逐条标注权威出处</p>
          </div>
          <div class="item card">
            <span class="item__no">03</span>
            <h4>红色教育闭环</h4>
            <p>修复影像 → 语音讲解 → 交互追问，让教育更沉浸</p>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
/* 首屏 */
.hero {
  position: relative;
  overflow: hidden;
  background-color: #7f0f22;
  background-size: cover;
  background-position: center;
  color: #f6efe4;
  padding: calc(var(--sp-16) * 0.3) 0;
}
.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(35, 10, 12, 0.26) 0%, rgba(35, 10, 12, 0.16) 100%);
  pointer-events: none;
}
.hero__inner { position: relative; z-index: 1; display: flex; flex-direction: column; gap: var(--sp-5); max-width: 760px; }
.hero__kicker {
  font-size: var(--fs-sm);
  letter-spacing: 0.18em;
  color: var(--c-gold);
}
.hero__title {
  font-size: clamp(36px, 6vw, 64px);
  line-height: 1.15;
  color: #fff;
  text-shadow: 0 2px 16px rgba(0, 0, 0, 0.4);
}
.hero__sub {
  font-size: var(--fs-lg);
  color: rgba(246, 239, 228, 0.92);
  max-width: 560px;
  text-shadow: 0 1px 8px rgba(0, 0, 0, 0.35);
}
.hero__actions { display: flex; gap: var(--sp-4); margin-top: var(--sp-2); flex-wrap: wrap; }
.btn--ghost-light { color: #fff; border-color: rgba(255, 255, 255, 0.6); }
.btn--ghost-light:hover { background: rgba(255, 255, 255, 0.12); }

/* 区块标题 */
.section-head { text-align: center; margin-bottom: var(--sp-8); }
.section-head h2 {
  font-size: var(--fs-2xl);
  position: relative;
  display: inline-block;
  padding-bottom: var(--sp-3);
}
.section-head h2::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: 0;
  transform: translateX(-50%);
  width: 48px;
  height: 3px;
  background: var(--c-gold);
  border-radius: 2px;
}
.section-head__sub { color: var(--c-ink-soft); margin-top: var(--sp-2); }

.feature { padding: var(--sp-8) 0 var(--sp-16); background-size: cover; background-position: center; }
.feature__state { display: flex; flex-direction: column; align-items: center; gap: var(--sp-3); padding: var(--sp-12) 0; }

/* 轮播 */
.carousel { position: relative; }
.hero-card {
  display: grid;
  grid-template-columns: 42% 58%;
  overflow: hidden;
  max-width: 960px;
  margin: 0 auto;
}
.hero-card__media {
  aspect-ratio: 3 / 4;
  background: #eee5d9;
  overflow: hidden;
}
.hero-card__media img { width: 100%; height: 100%; object-fit: cover; }
.hero-card__body {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: var(--sp-3);
  padding: var(--sp-8);
}
.hero-card__name { font-size: var(--fs-2xl); }
.hero-card__meta { color: var(--c-ink-soft); font-size: var(--fs-sm); }
.hero-card__actions { display: flex; gap: var(--sp-3); margin-top: var(--sp-3); flex-wrap: wrap; }

.carousel__arrow {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  z-index: 2;
  width: 44px;
  height: 44px;
  display: grid;
  place-items: center;
  font-size: 28px;
  line-height: 1;
  color: var(--c-ink);
  background: #fff;
  border: 1px solid var(--c-line);
  border-radius: 50%;
  box-shadow: var(--shadow-1);
  cursor: pointer;
  transition: color var(--t-fast), background var(--t-fast), border-color var(--t-fast);
}
.carousel__arrow:hover { color: #fff; background: var(--c-red); border-color: var(--c-red); }
.carousel__arrow--prev { left: -8px; }
.carousel__arrow--next { right: -8px; }

.carousel__dots { display: flex; justify-content: center; gap: var(--sp-2); margin-top: var(--sp-3); }
.carousel__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  border: none;
  background: var(--c-line);
  cursor: pointer;
  transition: background var(--t-fast), width var(--t-base);
}
.carousel__dot.is-active { width: 22px; border-radius: 4px; background: var(--c-red); }

/* 待补充 */
.pending {
  margin-top: var(--sp-4);
  padding: var(--sp-6) var(--sp-4);
  text-align: center;
  border-radius: var(--radius-lg);
}
.pending__label { font-size: var(--fs-sm); color: var(--c-ink-soft); letter-spacing: 0.08em; }
.pending__list { display: flex; justify-content: center; gap: var(--sp-3); margin-top: var(--sp-4); flex-wrap: wrap; }
.pending__item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: var(--sp-4) var(--sp-6);
  background: rgba(255, 255, 255, 0.7);
  border: 1px dashed var(--c-line);
  border-radius: var(--radius);
  backdrop-filter: blur(2px);
}
.pending__name { font-family: var(--font-title); font-weight: 700; }
.pending__years { font-size: var(--fs-xs); color: var(--c-ink-faint); }

/* 三大特性 */
.highlights { padding: 0 0 var(--sp-16); }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--sp-6); }
.item { padding: var(--sp-8); display: flex; flex-direction: column; gap: var(--sp-3); }
.item__no { font-family: var(--font-title); font-size: var(--fs-xl); color: var(--c-gold); }
.item h4 { font-size: var(--fs-lg); }
.item p { color: var(--c-ink-soft); font-size: var(--fs-sm); }

/* 过渡动画 */
.fade-enter-active, .fade-leave-active { transition: opacity var(--t-base); }
.fade-enter-from, .fade-leave-to { opacity: 0; }

/* 响应式 */
@media (max-width: 768px) {
  .hero { padding: calc(var(--sp-12) * 0.3) 0; }
  .hero-card { grid-template-columns: 1fr; }
  .hero-card__media { aspect-ratio: 4 / 5; }
  .hero-card__body { padding: var(--sp-6); }
  .carousel__arrow--prev { left: var(--sp-2); }
  .carousel__arrow--next { right: var(--sp-2); }
  .grid { grid-template-columns: 1fr; }
}
</style>