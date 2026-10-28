# -*- coding: utf-8 -*-
"""
任务 5.4：数字人口型同步（avatar.py）——占位骨架。
====================================================
输入：修复后的历史人物照片（阶段 3.12 样片）+ 配音 wav（5.2 tts.py 生成）
输出：口型同步讲述视频

数字人模型（EchoMimic / Hallo，装不动退回 SadTalker）待 5.3 部署完成后接入。
保底方案（5.7）：若数字人效果差，降级为「修复照片 + 语音 + 字幕」轮播方案，
不阻塞后续阶段。

伦理约束：视频片头强制标注「AI 演绎」，不使用逝者声音复刻。
"""
from pathlib import Path


def generate(photo, audio, out_video, model="echomimic"):
    """占位：生成数字人口型同步视频。"""
    raise NotImplementedError(
        "数字人模型尚未部署（任务 5.3）。完成后在此接入 EchoMimic/Hallo/SadTalker，"
        "或按 5.7 降级为照片+语音+字幕。"
    )


if __name__ == "__main__":
    raise SystemExit("avatar.py 为占位骨架，待 5.3/5.4 实现。")