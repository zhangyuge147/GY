# -*- coding: utf-8 -*-
"""L46 临时诊断（跑完即删）：DeepSeek 连通性诊断——key 存在性（不打印内容）、
直连成功率、代理环境。不输出任何密钥。"""
import os
import time
import urllib.request

# 1) .env key 存在性（不打印内容）
key = ""
env_path = r"c:\Users\yu\Desktop\XF\guangying\.env"
if os.path.exists(env_path):
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("DEEPSEEK_API_KEY") and "=" in line:
            key = line.split("=", 1)[1].strip()
            break
print(f"[1] .env 中 DEEPSEEK_API_KEY：{'已配置，长度 ' + str(len(key)) if key else '未配置/为空'}")
print(f"    .env 文件存在：{os.path.exists(env_path)}")

# 2) 代理环境变量
proxies = {k: os.environ.get(k) for k in
           ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
           if os.environ.get(k)}
print(f"[2] 进程代理环境变量：{proxies if proxies else '无'}")

# 3) 直连 api.deepseek.com 成功率（GET /models，5 次）
url = "https://api.deepseek.com/models"
ok = fail = 0
errs = []
for i in range(5):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
        ok += 1
    except Exception as e:
        fail += 1
        errs.append(f"{type(e).__name__}: {str(e)[:90]}")
    time.sleep(1)
print(f"[3] api.deepseek.com/models 直连：成功 {ok}/5 次")
for e in errs:
    print(f"    失败详情：{e}")
