# -*- coding: utf-8 -*-
"""真实链路冒烟：本地向量检索质量 + DeepSeek 问答 + 诊断访谈结构化输出。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config as C
from fde_agent.knowledge import load_vectorstore, format_docs
from fde_agent.llm import get_embeddings, get_llm

cfg = C.Config()
print("配置：", cfg.describe())

print("\n[1] 检索质量（BGE 语义向量）...")
store = load_vectorstore(cfg, get_embeddings(cfg))
for q in ("怎么判断一个FDE团队是不是高级外包", "AI项目验收标准怎么定义", "客户拖欠尾款怎么应对"):
    hits = store.similarity_search(q, k=2)
    print(f"  Q: {q}")
    for h in hits:
        print(f"    → {h.metadata.get('doc','?')}｜{h.metadata.get('section','')[:36]}")

print("\n[2] RAG 问答（DeepSeek 真实调用）...")
from fde_agent.advisor import FDEAdvisor
advisor = FDEAdvisor(get_llm(cfg, temperature=0.3), store)
ans, docs = advisor.answer("我们公司想上一个AI客服，预算一年50万，你觉得可行吗？怎么做第一步？",
                           return_docs=True)
print(ans[:600])
print(f"  （引用片段 {len(docs)} 条）")

print("\n[3] 诊断访谈（DeepSeek + 结构化输出 function_calling）...")
from fde_agent.diagnose import DiagnoseAgent
agent = DiagnoseAgent(get_llm(cfg, temperature=0.5), store)
reply = agent.turn("我们是做汽车零部件注塑的，大概500人，我管生产。今年最头疼的是报废率太高。")
print("顾问>", reply[:400])
print("已采集字段：", [k for k, v in agent.info.model_dump().items() if isinstance(v, str) and v.strip()])
print("\nSMOKE PASS")
