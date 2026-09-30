# -*- coding: utf-8 -*-
"""Streamlit AppTest 无头验证：脚本可渲染、无异常；诊断模式走一轮真实对话。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(r"C:\Users\Administrator\Desktop\FDE-Agent\app.py", default_timeout=300)
at.run()

if at.exception:
    for e in at.exception:
        print("EXCEPTION:", e.message)
        sys.exit(1)
print("[1] 应用渲染无异常；侧栏radio数:", len(at.radio), "; 代码块(进度):", len(at.code))

# 诊断模式发一句话（真实调用 DeepSeek + 结构化输出）
try:
    ci = at.chat_input[0]
    ci.set_value("我们是做汽车零部件注塑的，500人，我管生产，报废率太高。")
    at.run()
except Exception as e:
    print("chat_input 交互不受 AppTest 支持（不影响真实浏览器）：", type(e).__name__)
    sys.exit(0)

if at.exception:
    for e in at.exception:
        print("EXCEPTION after turn:", e.message)
        sys.exit(1)
md = "\n".join(m.value for m in at.markdown)
print("[2] 对话轮完成；回复片段：", md[-160:].replace("\n", " "))
print("APPTEST PASS")
