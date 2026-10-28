import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Restore from '../views/Restore.vue'
import Dialogue from '../views/Dialogue.vue'
import Archive from '../views/Archive.vue'
import Chat from '../views/Chat.vue'
import ComingSoon from '../components/ComingSoon.vue'

const routes = [
  { path: '/', name: 'home', component: Home, meta: { title: '首页' } },
  // 6.4 / 6.4b 修复照片（左：英雄影像展厅 / 右：用户照片上传修复）
  { path: '/restore', name: 'restore', component: Restore, meta: { title: '修复照片' } },
  // 6.5 时空对话
  { path: '/dialogue', name: 'dialogue', component: Dialogue, meta: { title: '时空对话' } },
  // 6.6 史料检索
  { path: '/archive', name: 'archive', component: Archive, meta: { title: '史料检索' } },
  // 智能对话
  { path: '/chat', name: 'chat', component: Chat, meta: { title: '智能对话' } }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior() {
    return { top: 0 }
  }
})

router.afterEach((to) => {
  document.title = to.meta.title
    ? `${to.meta.title} · 光影重生`
    : '光影重生 · 红色历史影像修复平台'
})

export default router