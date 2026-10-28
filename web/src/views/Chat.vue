<script setup>
import ChatBox from '../components/ChatBox.vue'
</script>

<template>
  <div class="chat-page">
    <!-- 页头 -->
    <section class="page-head">
      <div class="container">
        <p class="page-head__kicker">DeepSeek 智能问答</p>
        <h1 class="page-head__title">智能对话</h1>
        <p class="page-head__sub">可切换「通用助手」或某位英雄的第一人称口吻对话，回答由 AI 生成，历史内容以权威史料为准。</p>
      </div>
    </section>

    <!-- 对话区 -->
    <section class="section">
      <div class="container">
        <div class="panel card">
          <ChatBox />
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.page-head {
  padding: var(--sp-12) 0 var(--sp-8);
  background: linear-gradient(180deg, #7f0f22 0%, #8c1424 100%);
  color: #f6efe4;
}
.page-head__kicker { font-size: var(--fs-sm); letter-spacing: 0.18em; color: var(--c-gold); }
.page-head__title { font-size: clamp(28px, 4.5vw, 44px); color: #fff; margin-top: var(--sp-1); }
.page-head__sub { margin-top: var(--sp-2); color: rgba(246, 239, 228, 0.9); max-width: 640px; }

.section { padding: var(--sp-8) 0; }
.panel { padding: var(--sp-6); height: 540px; max-width: 880px; }
</style>