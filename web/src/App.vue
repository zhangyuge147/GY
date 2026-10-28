<script setup>
import { ref } from 'vue'
import { useRoute } from 'vue-router'
import ChatBox from './components/ChatBox.vue'

const nav = [
  { path: '/', label: '首页' },
  { path: '/restore', label: '修复照片' },
  { path: '/dialogue', label: '时空对话' },
  { path: '/archive', label: '史料检索' },
  { path: '/chat', label: '对话' }
]

const route = useRoute()
const menuOpen = ref(false)
const chatOpen = ref(false)

function isActive(path) {
  return route.path === path
}
</script>

<template>
  <div class="app">
    <header class="topbar">
      <div class="container topbar__inner">
        <router-link to="/" class="brand" @click="menuOpen = false">
          <span class="brand__badge">影</span>
          <span class="brand__text">
            <em>光影重生</em>
            <small>红色历史影像修复平台</small>
          </span>
        </router-link>

        <button class="nav-toggle" :aria-expanded="menuOpen" aria-label="打开菜单" @click="menuOpen = !menuOpen">
          <span></span><span></span><span></span>
        </button>

        <nav class="nav" :class="{ 'nav--open': menuOpen }">
          <router-link
            v-for="n in nav"
            :key="n.path"
            :to="n.path"
            class="nav__link"
            :class="{ 'is-active': isActive(n.path) }"
            @click="menuOpen = false"
          >{{ n.label }}</router-link>
        </nav>
      </div>
    </header>

    <main class="main">
      <router-view />
    </main>

    <footer class="footer">
      <div class="container footer__inner">
        <p class="footer__disclaimer">
          本平台影像均为 AI 修复与演绎，历史内容以权威史料为准；人物形象标注「AI 演绎」，页面背景与装饰由 AI 生成，仅供红色教育体验。
        </p>
        <p class="footer__meta">光影重生 · 红色历史影像修复与数字人交互平台</p>
      </div>
    </footer>

    <!-- 全局浮动对话气泡 -->
    <button class="chat-fab" :aria-expanded="chatOpen" aria-label="打开智能对话" @click="chatOpen = !chatOpen">
      <span v-if="!chatOpen">对话</span>
      <span v-else>×</span>
    </button>
    <transition name="chat-pop">
      <div v-if="chatOpen" class="chat-float">
        <header class="chat-float__head">
          <span>智能对话</span>
          <button class="chat-float__close" aria-label="关闭" @click="chatOpen = false">×</button>
        </header>
        <div class="chat-float__body">
          <ChatBox />
        </div>
      </div>
    </transition>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 50;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--c-line);
}
.topbar__inner {
  display: flex;
  align-items: center;
  gap: var(--sp-6);
  height: 68px;
}

.brand {
  display: flex;
  align-items: center;
  gap: var(--sp-3);
  color: var(--c-ink);
}
.brand__badge {
  width: 40px;
  height: 40px;
  display: grid;
  place-items: center;
  border-radius: var(--radius);
  background: linear-gradient(135deg, var(--c-red), var(--c-red-dark));
  color: var(--c-gold);
  font-family: var(--font-title);
  font-size: var(--fs-lg);
  font-weight: 700;
}
.brand__text { display: flex; flex-direction: column; line-height: 1.25; }
.brand__text em { font-family: var(--font-title); font-style: normal; font-size: var(--fs-lg); font-weight: 700; }
.brand__text small { font-size: var(--fs-xs); color: var(--c-ink-soft); }

.nav {
  display: flex;
  gap: var(--sp-2);
  margin-left: auto;
}
.nav__link {
  position: relative;
  padding: var(--sp-2) var(--sp-4);
  border-radius: var(--radius);
  color: var(--c-ink-soft);
  font-size: var(--fs-base);
  transition: color var(--t-fast), background var(--t-fast);
}
.nav__link:hover { color: var(--c-red); background: var(--c-red-weak); }
.nav__link.is-active { color: var(--c-red); font-weight: 600; }
.nav__link.is-active::after {
  content: "";
  position: absolute;
  left: var(--sp-4);
  right: var(--sp-4);
  bottom: 2px;
  height: 2px;
  background: var(--c-red);
  border-radius: 1px;
}

.nav-toggle {
  display: none;
  flex-direction: column;
  gap: 4px;
  width: 40px;
  height: 40px;
  padding: 9px;
  margin-left: auto;
  background: transparent;
  border: 1px solid var(--c-line);
  border-radius: var(--radius);
  cursor: pointer;
}
.nav-toggle span { display: block; height: 2px; background: var(--c-ink); border-radius: 1px; }

.main { flex: 1; }

.footer {
  margin-top: var(--sp-16);
  border-top: 3px solid var(--c-red);
  background: #241713;
  color: #d8cfc6;
}
.footer__inner { padding: var(--sp-8) var(--sp-4); text-align: center; }
.footer__disclaimer { font-size: var(--fs-sm); }
.footer__meta { margin-top: var(--sp-2); font-size: var(--fs-xs); color: #8f8378; }

/* 浮动对话气泡 */
.chat-fab {
  position: fixed;
  right: var(--sp-6);
  bottom: var(--sp-6);
  z-index: 90;
  width: 56px;
  height: 56px;
  border: none;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--c-red), var(--c-red-dark));
  color: #fff;
  font-size: var(--fs-base);
  cursor: pointer;
  box-shadow: var(--shadow-2);
  transition: transform var(--t-fast), filter var(--t-fast);
}
.chat-fab:hover { filter: brightness(0.95); transform: translateY(-2px); }

.chat-float {
  position: fixed;
  right: var(--sp-6);
  bottom: 88px;
  z-index: 90;
  width: 380px;
  max-width: calc(100vw - 32px);
  background: var(--c-paper);
  border: 1px solid var(--c-line);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-2);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.chat-float__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--sp-3) var(--sp-4);
  background: linear-gradient(135deg, var(--c-red), var(--c-red-dark));
  color: #fff;
  font-family: var(--font-title);
  font-weight: 700;
}
.chat-float__close {
  background: transparent;
  border: none;
  color: #fff;
  font-size: var(--fs-lg);
  line-height: 1;
  cursor: pointer;
}
.chat-float__body { height: 460px; padding: var(--sp-4); }

.chat-pop-enter-active, .chat-pop-leave-active { transition: opacity var(--t-base), transform var(--t-base); }
.chat-pop-enter-from, .chat-pop-leave-to { opacity: 0; transform: translateY(12px); }

@media (max-width: 768px) {
  .nav-toggle { display: flex; }
  .nav {
    position: absolute;
    top: 68px;
    left: 0;
    right: 0;
    flex-direction: column;
    gap: 0;
    margin: 0;
    padding: var(--sp-2);
    background: #fff;
    border-bottom: 1px solid var(--c-line);
    box-shadow: var(--shadow-2);
    display: none;
  }
  .nav--open { display: flex; }
  .nav__link { padding: var(--sp-3) var(--sp-4); border-radius: var(--radius); }
}
</style>