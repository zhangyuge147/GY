<script setup>
import { ref } from 'vue'
import { api } from '../api'

const question = ref('')
const asking = ref(false)
const error = ref('')
const result = ref(null)

const samples = [
  '吉鸿昌是怎么牺牲的？',
  '杨靖宇的主要事迹是什么？',
  '彭雪枫是谁？',
  '焦裕禄有什么贡献？'
]

async function ask(preset) {
  const text = (preset ?? question.value).trim()
  if (!text || asking.value) return
  question.value = text
  asking.value = true
  error.value = ''
  result.value = null
  try {
    result.value = await api.ragAsk(text, 3)
  } catch (e) {
    error.value = e.message || '检索失败，请稍后再试'
  } finally {
    asking.value = false
  }
}

function onKeyup(e) {
  if (e.key === 'Enter') ask()
}
</script>

<template>
  <div class="archive">
    <!-- 页头 -->
    <section class="page-head">
      <div class="container">
        <p class="page-head__kicker">史料检索 · 权威问答</p>
        <h1 class="page-head__title">史料检索</h1>
        <p class="page-head__sub">向英雄提问，答案逐条标注权威出处，历史内容有据可依。</p>
      </div>
    </section>

    <!-- 提问区 -->
    <section class="ask section">
      <div class="container">
        <div class="ask__box">
          <input
            v-model="question"
            class="ask__input"
            type="text"
            placeholder="输入你想了解的历史问题，如：吉鸿昌是怎么牺牲的？"
            :disabled="asking"
            @keyup="onKeyup"
          />
          <button class="btn btn--primary ask__btn" :disabled="asking || !question.trim()" @click="ask()">
            {{ asking ? '检索中…' : '提问' }}
          </button>
        </div>

        <div class="ask__samples">
          <span class="ask__samples-label">试试问：</span>
          <button
            v-for="s in samples"
            :key="s"
            class="ask__sample"
            :disabled="asking"
            @click="ask(s)"
          >{{ s }}</button>
        </div>

        <p v-if="error" class="ask__error">{{ error }}</p>
      </div>
    </section>

    <!-- 结果区 -->
    <section class="result section">
      <div class="container">
        <div v-if="asking" class="state">
          <div class="spinner"></div>
          <p class="state__hint">正在检索史料并生成答案…（首次加载检索模型可能稍慢）</p>
        </div>

        <template v-else-if="result">
          <!-- 答案 -->
          <article class="card answer">
            <header class="answer__head">
              <h2>答案</h2>
              <p class="answer__q">问：{{ result.question }}</p>
            </header>
            <div v-if="result.web" class="answer__notice">以下内容来自联网检索，未经人工核验，仅供了解参考。</div>
            <div v-else-if="result.fallback" class="answer__notice">以下内容由 AI 基于自身知识生成，未经权威史料核验，仅供了解参考。</div>
            <p class="answer__text">{{ result.answer }}</p>
          </article>

          <!-- 出处 -->
          <article v-if="result.citations && Object.keys(result.citations).length" class="card cites">
            <h2 class="cites__title">引用出处</h2>
            <ol class="cites__list">
              <li v-for="(src, n) in result.citations" :key="n">
                <span class="cites__no">[{{ n }}]</span>
                <span>{{ src }}</span>
              </li>
            </ol>
          </article>

          <!-- 联网来源（素材库检索不到时的受限联网兜底） -->
          <article v-if="result.web && result.web_sources && result.web_sources.length" class="card webs">
            <h2 class="webs__title">联网来源（未经人工核验）</h2>
            <ul class="webs__list">
              <li v-for="(w, i) in result.web_sources" :key="i" class="webs__item">
                <a :href="w.url" target="_blank" rel="noopener" class="webs__link">{{ w.title }}</a>
                <p class="webs__snippet">{{ w.snippet }}</p>
              </li>
            </ul>
          </article>

          <!-- 命中史料 -->
          <article v-if="result.retrieved && result.retrieved.length" class="card hits">
            <h2 class="hits__title">命中史料</h2>
            <ul class="hits__list">
              <li v-for="(h, i) in result.retrieved" :key="i" class="hits__item">
                <div class="hits__meta">
                  <span class="hits__person">{{ h.person }}</span>
                  <span class="hits__section">{{ h.section }}</span>
                  <span class="hits__score">相似度 {{ (h.score * 100).toFixed(0) }}%</span>
                </div>
                <p class="hits__text">{{ h.text }}</p>
              </li>
            </ul>
          </article>
        </template>

        <div v-else class="empty">
          <p class="empty__text">输入问题，开始检索权威史料。</p>
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

/* 提问区 */
.ask__box { display: flex; gap: var(--sp-3); max-width: 760px; }
.ask__input {
  flex: 1;
  padding: var(--sp-3) var(--sp-4);
  border: 1px solid var(--c-line);
  border-radius: var(--radius);
  font-size: var(--fs-base);
  font-family: var(--font-body);
  color: var(--c-ink);
  background: var(--c-card);
  transition: border-color var(--t-fast), box-shadow var(--t-fast);
}
.ask__input:focus {
  outline: none;
  border-color: var(--c-red);
  box-shadow: 0 0 0 3px var(--c-red-weak);
}
.ask__btn { min-width: 88px; }
.ask__samples {
  margin-top: var(--sp-4);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--sp-2);
}
.ask__samples-label { font-size: var(--fs-sm); color: var(--c-ink-soft); }
.ask__sample {
  padding: var(--sp-1) var(--sp-3);
  background: var(--c-red-weak);
  color: var(--c-red);
  border: 1px solid transparent;
  border-radius: 999px;
  font-size: var(--fs-sm);
  cursor: pointer;
  transition: border-color var(--t-fast), background var(--t-fast);
}
.ask__sample:hover { border-color: var(--c-red); }
.ask__error { margin-top: var(--sp-4); color: var(--c-red-dark); font-size: var(--fs-sm); }

/* 结果区 */
.result .container { display: flex; flex-direction: column; gap: var(--sp-5); max-width: 880px; }

.answer { padding: var(--sp-6); }
.answer__head { margin-bottom: var(--sp-4); }
.answer__head h2 { font-size: var(--fs-xl); }
.answer__q { margin-top: var(--sp-1); color: var(--c-ink-soft); font-size: var(--fs-sm); }
.answer__text {
  font-size: var(--fs-base);
  line-height: 1.9;
  white-space: pre-wrap;
  color: var(--c-ink);
}
.answer__notice {
  margin-bottom: var(--sp-3);
  padding: var(--sp-2) var(--sp-4);
  background: var(--c-red-weak);
  border-left: 3px solid var(--c-gold);
  border-radius: var(--radius);
  color: var(--c-red-dark);
  font-size: var(--fs-sm);
}

.cites { padding: var(--sp-6); }
.cites__title { font-size: var(--fs-xl); margin-bottom: var(--sp-4); }
.cites__list { margin: 0; padding-left: 0; list-style: none; display: flex; flex-direction: column; gap: var(--sp-2); }
.cites__list li { display: flex; gap: var(--sp-3); color: var(--c-ink-soft); font-size: var(--fs-sm); }
.cites__no { color: var(--c-gold); font-weight: 700; flex-shrink: 0; }

.webs { padding: var(--sp-6); }
.webs__title { font-size: var(--fs-xl); margin-bottom: var(--sp-4); }
.webs__list { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: var(--sp-4); }
.webs__item { padding: var(--sp-3) var(--sp-4); border: 1px solid var(--c-line); border-radius: var(--radius); background: var(--c-paper); }
.webs__link { font-family: var(--font-title); font-weight: 700; color: var(--c-red); font-size: var(--fs-base); }
.webs__link:hover { color: var(--c-red-dark); }
.webs__snippet { margin-top: var(--sp-2); font-size: var(--fs-sm); color: var(--c-ink-soft); line-height: 1.7; }

.hits { padding: var(--sp-6); }
.hits__title { font-size: var(--fs-xl); margin-bottom: var(--sp-4); }
.hits__list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--sp-4); }
.hits__item { padding: var(--sp-4); border: 1px solid var(--c-line); border-radius: var(--radius); background: var(--c-paper); }
.hits__meta { display: flex; align-items: center; gap: var(--sp-3); margin-bottom: var(--sp-2); flex-wrap: wrap; }
.hits__person { font-family: var(--font-title); font-weight: 700; color: var(--c-red); }
.hits__section { font-size: var(--fs-sm); color: var(--c-ink-soft); }
.hits__score { font-size: var(--fs-xs); color: var(--c-ink-faint); margin-left: auto; }
.hits__text { font-size: var(--fs-sm); line-height: 1.8; color: var(--c-ink); }

.empty { padding: var(--sp-12) 0; text-align: center; }
.empty__text { color: var(--c-ink-faint); }
</style>