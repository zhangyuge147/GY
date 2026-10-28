# -*- coding: utf-8 -*-
"""
阶段 6.2 后端异步任务管理（任务 A4）。

极简线程池 + 内存状态表：task_id → queued / running / done / failed。
用于两类耗时任务：
  1. 自由上传的图片修复流水线（A5，重 GPU，走子进程）；
  2. 讲解视频后台合成（A3，音频/字幕先行后异步出视频）。
"""
import threading
import uuid

_TASKS = {}
_LOCK = threading.Lock()


def submit(fn, *args, **kwargs):
    """提交任务，返回 task_id；fn 的返回值作为任务 result。"""
    task_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _TASKS[task_id] = {"status": "queued", "result": None, "error": None}

    def _run():
        with _LOCK:
            _TASKS[task_id]["status"] = "running"
        try:
            result = fn(*args, **kwargs)
            with _LOCK:
                _TASKS[task_id]["status"] = "done"
                _TASKS[task_id]["result"] = result
        except Exception as e:  # noqa: BLE001 —— 捕获一切异常写入 status，供轮询读取
            with _LOCK:
                _TASKS[task_id]["status"] = "failed"
                _TASKS[task_id]["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return task_id


def get(task_id: str) -> dict | None:
    with _LOCK:
        t = _TASKS.get(task_id)
        return dict(t) if t else None