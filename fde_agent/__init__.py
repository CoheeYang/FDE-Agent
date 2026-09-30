# -*- coding: utf-8 -*-
"""FDE 售前智能体：基于 LangChain + RAG 的主动诊断访谈与方法论问答。

模块结构：
  config    配置（提供方/知识源路径/离线模式）
  llm       模型工厂（OpenAI 兼容端点 / 离线自检）
  knowledge 知识库构建与检索（md+PDF → 分块 → 向量库）
  schemas   诊断访谈结构化状态（立项四问+客户四问+场景化）
  prompts   提示词层（FDE 方法论编码）
  diagnose  诊断访谈智能体（阶段推进 + 诊断报告）
  advisor   RAG 问答顾问（检索增强 + 敢说不）
"""
__version__ = "0.1.0"
