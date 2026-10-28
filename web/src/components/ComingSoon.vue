<script setup>
import { useRoute } from 'vue-router'
const route = useRoute()
</script>

<template>
  <div class="container">
    <div class="state">
      <div class="state__icon">·</div>
      <h2>{{ route.meta.title }} · 建设中</h2>
      <p class="state__hint">该模块正在开发中，敬请期待。</p>
    </div>
  </div>
</template>